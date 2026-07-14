from __future__ import annotations

import math
import numpy as np
import os
import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    quat_apply,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
    yaw_quat,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class MotionLoader:
    def __init__(self, motion_file: str, body_indexes: Sequence[int], device: str = "cpu"):
        self.motion_file = motion_file
        self._body_indexes = body_indexes
        self.motion_files = self._resolve_motion_files(motion_file)
        loaded = [self._load_npz(path) for path in self.motion_files]

        self.fps = loaded[0]["fps"]
        reference_joint_dim = loaded[0]["joint_pos"].shape[1]
        reference_body_count = loaded[0]["body_pos_w"].shape[1]
        for item in loaded[1:]:
            if abs(item["fps"] - self.fps) > 1e-5:
                raise ValueError(
                    f"All motions in {motion_file} must use the same fps. "
                    f"Got {self.fps} and {item['fps']}."
                )
            if item["joint_pos"].shape[1] != reference_joint_dim:
                raise ValueError(
                    f"{item['file']}: joint dimension is {item['joint_pos'].shape[1]}, "
                    f"but the first motion has {reference_joint_dim}."
                )
            if item["body_pos_w"].shape[1] != reference_body_count:
                raise ValueError(
                    f"{item['file']}: body count is {item['body_pos_w'].shape[1]}, "
                    f"but the first motion has {reference_body_count}."
                )

        self.joint_pos = torch.cat(
            [torch.tensor(item["joint_pos"], dtype=torch.float32, device=device) for item in loaded], dim=0
        )
        self.joint_vel = torch.cat(
            [torch.tensor(item["joint_vel"], dtype=torch.float32, device=device) for item in loaded], dim=0
        )
        self._body_pos_w = torch.cat(
            [torch.tensor(item["body_pos_w"], dtype=torch.float32, device=device) for item in loaded], dim=0
        )
        self._body_quat_w = torch.cat(
            [torch.tensor(item["body_quat_w"], dtype=torch.float32, device=device) for item in loaded], dim=0
        )
        self._body_lin_vel_w = torch.cat(
            [torch.tensor(item["body_lin_vel_w"], dtype=torch.float32, device=device) for item in loaded], dim=0
        )
        self._body_ang_vel_w = torch.cat(
            [torch.tensor(item["body_ang_vel_w"], dtype=torch.float32, device=device) for item in loaded], dim=0
        )
        self.motion_lengths = torch.tensor([item["length"] for item in loaded], dtype=torch.long, device=device)
        self.motion_offsets = torch.zeros_like(self.motion_lengths)
        if len(self.motion_lengths) > 1:
            self.motion_offsets[1:] = torch.cumsum(self.motion_lengths[:-1], dim=0)
        self.num_motions = len(self.motion_files)
        self.total_frames = int(self.motion_lengths.sum().item())
        self.time_step_total = int(self.motion_lengths.max().item())

        self._validate_shapes()

    def _resolve_motion_files(self, motion_path: str) -> list[str]:
        if os.path.isdir(motion_path):
            files = sorted(
                os.path.join(root, name)
                for root, _, names in os.walk(motion_path)
                for name in names
                if name.endswith(".npz")
            )
        elif os.path.isfile(motion_path) and motion_path.endswith(".txt"):
            base_dir = os.path.dirname(os.path.abspath(motion_path))
            files = []
            with open(motion_path) as f:
                for line in f:
                    item = line.strip()
                    if not item or item.startswith("#"):
                        continue
                    files.append(item if os.path.isabs(item) else os.path.join(base_dir, item))
        elif os.path.isfile(motion_path):
            files = [motion_path]
        else:
            raise FileNotFoundError(f"Invalid motion path: {motion_path}")
        if not files:
            raise ValueError(f"No WBT .npz motion files found in {motion_path}")
        return files

    def _load_npz(self, motion_file: str) -> dict[str, np.ndarray | float | int | str]:
        loaded = np.load(motion_file, allow_pickle=True)
        try:
            if not hasattr(loaded, "files"):
                raise ValueError(
                    f"{motion_file} is not a whole_body_tracking motion .npz file. "
                    "PHUMA .npy files must be converted first with scripts/phuma_to_npz.py."
                )
            required_keys = {
                "fps",
                "joint_pos",
                "joint_vel",
                "body_pos_w",
                "body_quat_w",
                "body_lin_vel_w",
                "body_ang_vel_w",
            }
            missing_keys = required_keys.difference(loaded.files)
            if missing_keys:
                raise ValueError(
                    f"{motion_file} is missing WBT motion keys: {sorted(missing_keys)}. "
                    "If this is PHUMA data, convert it first with scripts/phuma_to_npz.py."
                )
            item = {
                "file": motion_file,
                "fps": float(np.asarray(loaded["fps"]).squeeze()),
                "joint_pos": np.asarray(loaded["joint_pos"], dtype=np.float32),
                "joint_vel": np.asarray(loaded["joint_vel"], dtype=np.float32),
                "body_pos_w": np.asarray(loaded["body_pos_w"], dtype=np.float32),
                "body_quat_w": np.asarray(loaded["body_quat_w"], dtype=np.float32),
                "body_lin_vel_w": np.asarray(loaded["body_lin_vel_w"], dtype=np.float32),
                "body_ang_vel_w": np.asarray(loaded["body_ang_vel_w"], dtype=np.float32),
            }
        finally:
            if hasattr(loaded, "close"):
                loaded.close()

        self._validate_item_shapes(motion_file, item)
        item["length"] = item["joint_pos"].shape[0]
        return item

    @staticmethod
    def _validate_item_shapes(motion_file: str, item: dict[str, np.ndarray | float | str]) -> None:
        joint_pos = item["joint_pos"]
        joint_vel = item["joint_vel"]
        if joint_pos.ndim != 2:
            raise ValueError(f"{motion_file}: joint_pos must have shape (T, num_joints), got {joint_pos.shape}")
        if joint_pos.shape[0] < 2:
            raise ValueError(f"{motion_file}: motion must contain at least 2 frames, got {joint_pos.shape[0]}")
        if joint_vel.shape != joint_pos.shape:
            raise ValueError(
                f"{motion_file}: joint_vel shape {joint_vel.shape} does not match joint_pos shape {joint_pos.shape}"
            )
        for name, width in (
            ("body_pos_w", 3),
            ("body_quat_w", 4),
            ("body_lin_vel_w", 3),
            ("body_ang_vel_w", 3),
        ):
            tensor = item[name]
            if tensor.ndim != 3 or tensor.shape[0] != joint_pos.shape[0] or tensor.shape[2] != width:
                raise ValueError(f"{motion_file}: {name} must have shape (T, num_bodies, {width}), got {tensor.shape}")

    def frame_indices(self, motion_ids: torch.Tensor, time_steps: torch.Tensor) -> torch.Tensor:
        clamped_steps = torch.minimum(time_steps, self.motion_lengths[motion_ids] - 1)
        return self.motion_offsets[motion_ids] + clamped_steps

    def _validate_shapes(self) -> None:
        if self.joint_pos.ndim != 2:
            raise ValueError(f"{self.motion_file}: joint_pos must have shape (T, num_joints), got {self.joint_pos.shape}")
        if self.joint_pos.shape[0] != self.total_frames:
            raise ValueError(
                f"{self.motion_file}: concatenated joint_pos has {self.joint_pos.shape[0]} frames, "
                f"expected {self.total_frames}"
            )
        if self.joint_vel.shape != self.joint_pos.shape:
            raise ValueError(
                f"{self.motion_file}: joint_vel shape {self.joint_vel.shape} does not match "
                f"joint_pos shape {self.joint_pos.shape}"
            )
        expected_body_frame_count = self.total_frames
        for name, tensor, width in (
            ("body_pos_w", self._body_pos_w, 3),
            ("body_quat_w", self._body_quat_w, 4),
            ("body_lin_vel_w", self._body_lin_vel_w, 3),
            ("body_ang_vel_w", self._body_ang_vel_w, 3),
        ):
            if tensor.ndim != 3 or tensor.shape[0] != expected_body_frame_count or tensor.shape[2] != width:
                raise ValueError(
                    f"{self.motion_file}: {name} must have shape (T, num_bodies, {width}), got {tensor.shape}"
                )
        if len(self._body_indexes) > 0 and int(torch.max(self._body_indexes).item()) >= self._body_pos_w.shape[1]:
            raise ValueError(
                f"{self.motion_file}: body tensors contain {self._body_pos_w.shape[1]} bodies, but the current "
                f"robot config requested body index {int(torch.max(self._body_indexes).item())}. "
                "Regenerate the motion with scripts/phuma_to_npz.py using the same robot/task config."
            )

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._body_pos_w[:, self._body_indexes]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._body_quat_w[:, self._body_indexes]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._body_lin_vel_w[:, self._body_indexes]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._body_ang_vel_w[:, self._body_indexes]


class MotionCommand(CommandTerm):
    cfg: MotionCommandCfg

    def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.robot_anchor_body_index = self.robot.body_names.index(self.cfg.anchor_body_name)
        self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0], dtype=torch.long, device=self.device
        )

        self.motion = MotionLoader(self.cfg.motion_file, self.body_indexes, device=self.device)
        expected_joint_count = len(self.robot.joint_names)
        if self.motion.joint_pos.shape[1] != expected_joint_count:
            raise ValueError(
                f"{self.cfg.motion_file}: motion joint dimension is {self.motion.joint_pos.shape[1]}, "
                f"but task robot '{self.cfg.asset_name}' has {expected_joint_count} joints. "
                "This would change the policy observation size. For PHUMA G1 data, convert it with "
                "scripts/phuma_to_npz.py so dof_pos is mapped to the same Isaac G1 joint order."
            )
        print(
            f"[INFO]: Loaded {self.motion.num_motions} motion(s), "
            f"{self.motion.total_frames} total frames, fps={self.motion.fps:g}"
        )
        self.motion_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.body_pos_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 3, device=self.device)
        self.body_quat_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 4, device=self.device)
        self.body_quat_relative_w[:, :, 0] = 1.0

        self.bin_count = int(self.motion.time_step_total // (1 / (env.cfg.decimation * env.cfg.sim.dt))) + 1
        self.bin_failed_count = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self._current_bin_failed = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self.kernel = torch.tensor(
            [self.cfg.adaptive_lambda**i for i in range(self.cfg.adaptive_kernel_size)], device=self.device
        )
        self.kernel = self.kernel / self.kernel.sum()

        self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_lin_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_ang_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_entropy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_prob"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_bin"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:  # TODO Consider again if this is the best observation
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)

    @property
    def frame_indices(self) -> torch.Tensor:
        return self.motion.frame_indices(self.motion_ids, self.time_steps)

    @property
    def joint_pos(self) -> torch.Tensor:
        return self.motion.joint_pos[self.frame_indices]

    @property
    def joint_vel(self) -> torch.Tensor:
        return self.motion.joint_vel[self.frame_indices]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self.frame_indices] + self._env.scene.env_origins[:, None, :]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.frame_indices]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.frame_indices]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.frame_indices]

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self.frame_indices, self.motion_anchor_body_index] + self._env.scene.env_origins

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.frame_indices, self.motion_anchor_body_index]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.frame_indices, self.motion_anchor_body_index]

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.frame_indices, self.motion_anchor_body_index]

    @property
    def robot_joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self.robot.data.joint_vel

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.body_indexes]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.body_indexes]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.body_indexes]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.body_indexes]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.robot_anchor_body_index]

    def _update_metrics(self):
        self.metrics["error_anchor_pos"] = torch.norm(self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1)
        self.metrics["error_anchor_rot"] = quat_error_magnitude(self.anchor_quat_w, self.robot_anchor_quat_w)
        self.metrics["error_anchor_lin_vel"] = torch.norm(self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1)
        self.metrics["error_anchor_ang_vel"] = torch.norm(self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1)

        self.metrics["error_body_pos"] = torch.norm(self.body_pos_relative_w - self.robot_body_pos_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_rot"] = quat_error_magnitude(self.body_quat_relative_w, self.robot_body_quat_w).mean(
            dim=-1
        )

        self.metrics["error_body_lin_vel"] = torch.norm(self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_ang_vel"] = torch.norm(self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1).mean(
            dim=-1
        )

        self.metrics["error_joint_pos"] = torch.norm(self.joint_pos - self.robot_joint_pos, dim=-1)
        self.metrics["error_joint_vel"] = torch.norm(self.joint_vel - self.robot_joint_vel, dim=-1)

    def _adaptive_sampling(self, env_ids: Sequence[int]):
        if self.motion.num_motions > 1:
            sampled_motion_ids = torch.randint(self.motion.num_motions, (len(env_ids),), device=self.device)
            sampled_lengths = self.motion.motion_lengths[sampled_motion_ids]
            sampled_phases = sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device)
            self.motion_ids[env_ids] = sampled_motion_ids
            self.time_steps[env_ids] = (sampled_phases * (sampled_lengths - 1).float()).long()
            self.metrics["sampling_entropy"][:] = 1.0
            self.metrics["sampling_top1_prob"][:] = 1.0 / float(self.motion.num_motions)
            self.metrics["sampling_top1_bin"][env_ids] = sampled_motion_ids.float() / max(self.motion.num_motions - 1, 1)
            return

        episode_failed = self._env.termination_manager.terminated[env_ids]
        if torch.any(episode_failed):
            current_bin_index = torch.clamp(
                (self.time_steps * self.bin_count) // max(self.motion.time_step_total, 1), 0, self.bin_count - 1
            )
            fail_bins = current_bin_index[env_ids][episode_failed]
            self._current_bin_failed[:] = torch.bincount(fail_bins, minlength=self.bin_count)

        # Sample
        sampling_probabilities = self.bin_failed_count + self.cfg.adaptive_uniform_ratio / float(self.bin_count)
        sampling_probabilities = torch.nn.functional.pad(
            sampling_probabilities.unsqueeze(0).unsqueeze(0),
            (0, self.cfg.adaptive_kernel_size - 1),  # Non-causal kernel
            mode="replicate",
        )
        sampling_probabilities = torch.nn.functional.conv1d(sampling_probabilities, self.kernel.view(1, 1, -1)).view(-1)

        sampling_probabilities = sampling_probabilities / sampling_probabilities.sum()

        sampled_bins = torch.multinomial(sampling_probabilities, len(env_ids), replacement=True)

        self.time_steps[env_ids] = (
            (sampled_bins + sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device))
            / self.bin_count
            * (self.motion.time_step_total - 1)
        ).long()

        # Metrics
        H = -(sampling_probabilities * (sampling_probabilities + 1e-12).log()).sum()
        H_norm = H / math.log(self.bin_count)
        pmax, imax = sampling_probabilities.max(dim=0)
        self.metrics["sampling_entropy"][:] = H_norm
        self.metrics["sampling_top1_prob"][:] = pmax
        self.metrics["sampling_top1_bin"][:] = imax.float() / self.bin_count

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        self._adaptive_sampling(env_ids)

        root_pos = self.body_pos_w[:, 0].clone()
        root_ori = self.body_quat_w[:, 0].clone()
        root_lin_vel = self.body_lin_vel_w[:, 0].clone()
        root_ang_vel = self.body_ang_vel_w[:, 0].clone()

        range_list = [self.cfg.pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_pos[env_ids] += rand_samples[:, 0:3]
        orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        root_ori[env_ids] = quat_mul(orientations_delta, root_ori[env_ids])
        range_list = [self.cfg.velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_lin_vel[env_ids] += rand_samples[:, :3]
        root_ang_vel[env_ids] += rand_samples[:, 3:]

        joint_pos = self.joint_pos.clone()
        joint_vel = self.joint_vel.clone()

        joint_pos += sample_uniform(*self.cfg.joint_position_range, joint_pos.shape, joint_pos.device)
        soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
        joint_pos[env_ids] = torch.clip(
            joint_pos[env_ids], soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1]
        )
        self.robot.write_joint_state_to_sim(joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids)
        self.robot.write_root_state_to_sim(
            torch.cat([root_pos[env_ids], root_ori[env_ids], root_lin_vel[env_ids], root_ang_vel[env_ids]], dim=-1),
            env_ids=env_ids,
        )

    def _update_command(self):
        self.time_steps += 1
        env_ids = torch.where(self.time_steps >= self.motion.motion_lengths[self.motion_ids])[0]
        self._resample_command(env_ids)

        anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)

        delta_pos_w = robot_anchor_pos_w_repeat
        delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
        delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat)))

        self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
        self.body_pos_relative_w = delta_pos_w + quat_apply(delta_ori_w, self.body_pos_w - anchor_pos_w_repeat)

        self.bin_failed_count = (
            self.cfg.adaptive_alpha * self._current_bin_failed + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
        )
        self._current_bin_failed.zero_()

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/current/anchor")
                )
                self.goal_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/anchor")
                )

                self.current_body_visualizers = []
                self.goal_body_visualizers = []
                for name in self.cfg.body_names:
                    self.current_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/current/" + name)
                        )
                    )
                    self.goal_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/" + name)
                        )
                    )

            self.current_anchor_visualizer.set_visibility(True)
            self.goal_anchor_visualizer.set_visibility(True)
            for i in range(len(self.cfg.body_names)):
                self.current_body_visualizers[i].set_visibility(True)
                self.goal_body_visualizers[i].set_visibility(True)

        else:
            if hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer.set_visibility(False)
                self.goal_anchor_visualizer.set_visibility(False)
                for i in range(len(self.cfg.body_names)):
                    self.current_body_visualizers[i].set_visibility(False)
                    self.goal_body_visualizers[i].set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return

        self.current_anchor_visualizer.visualize(self.robot_anchor_pos_w, self.robot_anchor_quat_w)
        self.goal_anchor_visualizer.visualize(self.anchor_pos_w, self.anchor_quat_w)

        for i in range(len(self.cfg.body_names)):
            self.current_body_visualizers[i].visualize(self.robot_body_pos_w[:, i], self.robot_body_quat_w[:, i])
            self.goal_body_visualizers[i].visualize(self.body_pos_relative_w[:, i], self.body_quat_relative_w[:, i])


@configclass
class MotionCommandCfg(CommandTermCfg):
    """Configuration for the motion command."""

    class_type: type = MotionCommand

    asset_name: str = MISSING

    motion_file: str = MISSING
    anchor_body_name: str = MISSING
    body_names: list[str] = MISSING

    pose_range: dict[str, tuple[float, float]] = {}
    velocity_range: dict[str, tuple[float, float]] = {}

    joint_position_range: tuple[float, float] = (-0.52, 0.52)

    adaptive_kernel_size: int = 1
    adaptive_lambda: float = 0.8
    adaptive_uniform_ratio: float = 0.1
    adaptive_alpha: float = 0.001

    anchor_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    anchor_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)

    body_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    body_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
