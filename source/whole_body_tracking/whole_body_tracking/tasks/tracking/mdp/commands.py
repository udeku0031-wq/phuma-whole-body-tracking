from __future__ import annotations

import math
import os
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import numpy as np
import torch

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

from whole_body_tracking.utils.difficulty import DEFAULT_ALGORITHM_SCHEMA_VERSION
from whole_body_tracking.utils.difficulty_metadata import SegmentDifficultyMetadata
from whole_body_tracking.utils.quality_metadata import (
    QUALITY_STATUS_TO_CODE,
    SegmentQualityMetadata,
    canonical_manifest_entries,
)
from whole_body_tracking.utils.sampling import (
    AssignmentTraceRecorder,
    SAMPLING_STATE_VERSION,
    FixedLengthSegmentIndex,
    QualityGatedStartIndex,
    SamplingStatistics,
    motion_pool_fingerprint,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class MotionLoader:
    def __init__(self, motion_file: str, body_indexes: Sequence[int], device: str = "cpu"):
        self.motion_file = motion_file
        self._body_indexes = body_indexes
        self.motion_files = self._resolve_motion_files(motion_file)
        self.motion_keys = (
            canonical_manifest_entries(motion_file)
            if os.path.isfile(motion_file) and motion_file.endswith(".txt")
            else [os.path.basename(path) for path in self.motion_files]
        )
        loaded = [self._load_npz(path) for path in self.motion_files]

        self.fps = loaded[0]["fps"]
        motion_fps = [float(item["fps"]) for item in loaded]
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
        self.motion_fps = torch.tensor(motion_fps, dtype=torch.float64, device=device)
        self.motion_offsets = torch.zeros_like(self.motion_lengths)
        if len(self.motion_lengths) > 1:
            self.motion_offsets[1:] = torch.cumsum(self.motion_lengths[:-1], dim=0)
        self.num_motions = len(self.motion_files)
        self.total_frames = int(self.motion_lengths.sum().item())
        self.time_step_total = int(self.motion_lengths.max().item())
        self.pool_fingerprint = motion_pool_fingerprint(
            self.motion_files,
            [int(item["length"]) for item in loaded],
            motion_fps,
        )

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
                    files.append(self._resolve_manifest_entry(item, base_dir))
        elif os.path.isfile(motion_path):
            files = [motion_path]
        else:
            raise FileNotFoundError(f"Invalid motion path: {motion_path}")
        if not files:
            raise ValueError(f"No WBT .npz motion files found in {motion_path}")
        return files

    @staticmethod
    def _resolve_manifest_entry(item: str, base_dir: str) -> str:
        if os.path.isabs(item):
            return item

        candidates = [
            os.path.join(base_dir, item),
            os.path.abspath(item),
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
        return candidates[0]

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


def _validate_research_config(cfg: ResearchExperimentCfg) -> None:
    """Validate the implemented M0/M1 combinations and load-only difficulty metadata."""

    if cfg.method_name not in {"M0", "M1"}:
        raise NotImplementedError(
            f"Research method '{cfg.method_name}' is not implemented yet; use 'M0' or 'M1'."
        )
    if cfg.method_name == "M0" and cfg.quality_gate.enabled:
        raise ValueError("method_name='M0' requires quality_gate.enabled=False.")
    if cfg.method_name == "M1" and not cfg.quality_gate.enabled:
        raise ValueError("method_name='M1' requires quality_gate.enabled=True.")
    if cfg.diversity_constraint.enabled:
        raise NotImplementedError("diversity_constraint.enabled=True is not implemented in module 1.")
    if cfg.motion_sampling.mode != "uniform":
        raise NotImplementedError(
            f"motion_sampling mode '{cfg.motion_sampling.mode}' is not implemented in module 1; use 'uniform'."
        )
    if cfg.segment_sampling.mode != "uniform":
        raise NotImplementedError(
            f"segment_sampling mode '{cfg.segment_sampling.mode}' is not implemented in module 1; use 'uniform'."
        )
    if not math.isfinite(cfg.segment.length_seconds) or cfg.segment.length_seconds <= 0.0:
        raise ValueError("segment.length_seconds must be finite and greater than zero.")
    if cfg.assignment_trace.enabled:
        if not cfg.segment.enabled:
            raise ValueError("assignment_trace requires segment.enabled=True.")
        if not cfg.assignment_trace.output_path:
            raise ValueError("assignment_trace.output_path is required when assignment_trace.enabled=True.")
        if cfg.assignment_trace.max_entries < 1:
            raise ValueError("assignment_trace.max_entries must be at least 1.")
    if cfg.sampling_statistics.enabled and not cfg.segment.enabled:
        raise ValueError("sampling_statistics requires segment.enabled=True in stage 0.")
    if cfg.sampling_statistics.log_interval < 1:
        raise ValueError("sampling_statistics.log_interval must be at least 1.")
    if not math.isfinite(cfg.probability_validation.epsilon) or cfg.probability_validation.epsilon <= 0.0:
        raise ValueError("probability_validation.epsilon must be finite and greater than zero.")
    if cfg.difficulty_calibration.enabled:
        if not cfg.segment.enabled:
            raise ValueError("difficulty_calibration requires segment.enabled=True.")
        if not cfg.difficulty_calibration.metadata_path:
            raise ValueError(
                "difficulty_calibration.metadata_path is required when difficulty calibration is enabled."
            )
        if cfg.difficulty_calibration.expected_num_bins < 2:
            raise ValueError("difficulty_calibration.expected_num_bins must be at least 2.")
    if cfg.quality_gate.enabled:
        if not cfg.segment.enabled:
            raise ValueError("quality_gate requires segment.enabled=True.")
        if not cfg.sampling_statistics.enabled:
            raise ValueError("quality_gate requires sampling_statistics.enabled=True for audit counters.")
        if not cfg.quality_gate.metadata_path:
            raise ValueError("quality_gate.metadata_path is required when the gate is enabled.")
        reject_statuses = tuple(cfg.quality_gate.reject_statuses)
        if len(set(reject_statuses)) != len(reject_statuses):
            raise ValueError("quality_gate.reject_statuses must not contain duplicates.")
        unknown_statuses = set(reject_statuses).difference(QUALITY_STATUS_TO_CODE)
        if unknown_statuses:
            raise ValueError(f"Unknown quality_gate.reject_statuses: {sorted(unknown_statuses)}")
        if "reject" not in reject_statuses or "pass" in reject_statuses:
            raise ValueError("M1 must reject status 'reject' and must never reject status 'pass'.")
        if cfg.quality_gate.include_borderline and "borderline" in reject_statuses:
            raise ValueError(
                "quality_gate.include_borderline=True conflicts with rejecting status 'borderline'."
            )
        if cfg.quality_gate.gate_scope != "assignment_start":
            raise NotImplementedError("Only quality_gate.gate_scope='assignment_start' is implemented.")
        if cfg.quality_gate.empty_motion_policy not in QualityGatedStartIndex._EMPTY_MOTION_POLICIES:
            raise ValueError(
                "quality_gate.empty_motion_policy must be one of "
                f"{sorted(QualityGatedStartIndex._EMPTY_MOTION_POLICIES)}."
            )


def _canonical_quality_gate_config(config: Mapping[str, object]) -> dict[str, object]:
    """Normalize quality-gate config identity across module-1 checkpoint revisions."""

    canonical = dict(config)
    if "empty_motion_policy" not in canonical and "fail_on_empty_motion" in canonical:
        canonical["empty_motion_policy"] = "error" if canonical["fail_on_empty_motion"] else "exclude"
    canonical.pop("fail_on_empty_motion", None)
    return canonical


def _canonical_difficulty_calibration_config(config: Mapping[str, object]) -> dict[str, object]:
    """Normalize difficulty config semantics while allowing metadata relocation."""

    canonical = dict(config)
    canonical.pop("metadata_path", None)
    return canonical


class MotionCommand(CommandTerm):
    cfg: MotionCommandCfg

    def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        _validate_research_config(cfg.research)

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
        self.segment_index: FixedLengthSegmentIndex | None = None
        self.sampling_statistics: SamplingStatistics | None = None
        self.assignment_trace: AssignmentTraceRecorder | None = None
        self.quality_metadata: SegmentQualityMetadata | None = None
        self.quality_gate_index: QualityGatedStartIndex | None = None
        self.quality_status_codes: torch.Tensor | None = None
        self.quality_metadata_match_ok = False
        self.quality_reference_frame_count = torch.zeros((), dtype=torch.long, device=self.device)
        self.quality_reject_reference_frame_count = torch.zeros((), dtype=torch.long, device=self.device)
        self.difficulty_metadata: SegmentDifficultyMetadata | None = None
        self.difficulty_scores: torch.Tensor | None = None
        self.difficulty_bins: torch.Tensor | None = None
        self.difficulty_metadata_match_ok = False
        if self.cfg.research.segment.enabled:
            self.segment_index = FixedLengthSegmentIndex(
                self.motion.motion_lengths,
                self.motion.motion_fps,
                self.cfg.research.segment.length_seconds,
                device=self.device,
            )
            if self.cfg.research.sampling_statistics.enabled:
                self.sampling_statistics = SamplingStatistics(
                    self.segment_index, pool_fingerprint=self.motion.pool_fingerprint
                )
            print(
                f"[INFO]: Built {self.segment_index.num_segments} fixed-length segment(s), "
                f"duration={self.segment_index.segment_length_seconds:g}s"
            )
            if self.cfg.research.assignment_trace.enabled:
                self.assignment_trace = AssignmentTraceRecorder(
                    self.cfg.research.assignment_trace.output_path,
                    self.cfg.research.assignment_trace.max_entries,
                    pool_fingerprint=self.motion.pool_fingerprint,
                    run_label=self.cfg.research.method_name,
                )
                print(
                    "[INFO]: Assignment trace enabled; writing first "
                    f"{self.cfg.research.assignment_trace.max_entries} assignment(s) to "
                    f"{self.assignment_trace.output_path}"
                )
        if self.cfg.research.quality_gate.enabled:
            self._initialize_quality_gate()
        if self.cfg.research.difficulty_calibration.enabled:
            self._initialize_difficulty_calibration()
        self.motion_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.assigned_local_segment_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.assigned_global_segment_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
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

    def _initialize_quality_gate(self) -> None:
        """Load frozen quality labels and bind them to the exact segment layout."""

        if self.segment_index is None:
            raise RuntimeError("Quality gate initialization requires a segment index.")
        if not self.cfg.motion_file.endswith(".txt") or not os.path.isfile(self.cfg.motion_file):
            raise ValueError("Quality-gated training requires a local .txt motion manifest.")

        metadata = SegmentQualityMetadata.load(self.cfg.research.quality_gate.metadata_path)
        metadata_match_ok = metadata.validate_against(
            manifest_path=self.cfg.motion_file,
            motion_keys=self.motion.motion_keys,
            motion_lengths=self.motion.motion_lengths.detach().cpu().tolist(),
            motion_fps=self.motion.motion_fps.detach().cpu().tolist(),
            motion_segment_offsets=self.segment_index.motion_segment_offsets.detach().cpu().tolist(),
            segment_start_frames=self.segment_index.segment_start_frames.detach().cpu().tolist(),
            segment_end_frames=self.segment_index.segment_end_frames.detach().cpu().tolist(),
            segment_length_seconds=self.segment_index.segment_length_seconds,
            segment_schema_version=SAMPLING_STATE_VERSION,
            pool_fingerprint=self.motion.pool_fingerprint,
            strict=self.cfg.research.quality_gate.strict_metadata_match,
        )
        if not metadata_match_ok:
            warnings.warn(
                "Quality metadata mismatch was explicitly allowed by strict_metadata_match=False; "
                "do not use this run as a formal M1 result.",
                RuntimeWarning,
                stacklevel=2,
            )

        status_masks = {
            "pass": metadata.pass_mask,
            "borderline": metadata.borderline_mask,
            "reject": metadata.reject_mask,
        }
        rejected_mask = np.zeros(metadata.num_segments, dtype=bool)
        for status in self.cfg.research.quality_gate.reject_statuses:
            rejected_mask |= status_masks[status]
        allowed_mask = ~rejected_mask
        if not self.cfg.research.quality_gate.include_borderline:
            allowed_mask &= ~metadata.borderline_mask
        self.quality_metadata = metadata
        self.quality_status_codes = torch.as_tensor(
            metadata.quality_status, dtype=torch.int8, device=self.device
        )
        self.quality_gate_index = QualityGatedStartIndex(
            self.segment_index,
            torch.as_tensor(allowed_mask, dtype=torch.bool, device=self.device),
            empty_motion_policy=self.cfg.research.quality_gate.empty_motion_policy,
        )
        if self.quality_gate_index.eligible_motion_ids.numel() == 0:
            raise ValueError("Quality gate excluded every motion; no eligible start frame remains.")
        self.quality_metadata_match_ok = metadata_match_ok
        gate_summary = self.quality_gate_index.summary()
        if gate_summary["num_empty_motions"]:
            warnings.warn(
                f"Quality gate found {gate_summary['num_empty_motions']} motion(s) with no eligible start frame; "
                f"empty_motion_policy='{self.cfg.research.quality_gate.empty_motion_policy}'. "
                "The manifest and motion loader pool are unchanged; empty motions have zero sampling probability "
                "only when policy='exclude'.",
                RuntimeWarning,
                stacklevel=2,
            )
        print(
            f"[INFO]: Loaded quality metadata '{metadata.path}' "
            f"(pass={int(np.count_nonzero(metadata.pass_mask))}, "
            f"borderline={int(np.count_nonzero(metadata.borderline_mask))}, "
            f"reject={int(np.count_nonzero(metadata.reject_mask))}, "
            f"eligible_starts={gate_summary['num_eligible_start_frames']}, "
            f"empty_motions={gate_summary['num_empty_motions']}, "
            f"effective_motions={gate_summary['num_eligible_motions']})"
        )

    def _initialize_difficulty_calibration(self) -> None:
        """Load frozen difficulty labels without changing assignment sampling."""

        if self.segment_index is None:
            raise RuntimeError("Difficulty calibration initialization requires a segment index.")
        if not self.cfg.motion_file.endswith(".txt") or not os.path.isfile(self.cfg.motion_file):
            raise ValueError("Difficulty calibration requires a local .txt motion manifest.")

        metadata = SegmentDifficultyMetadata.load(self.cfg.research.difficulty_calibration.metadata_path)
        if metadata.algorithm_schema_version != DEFAULT_ALGORITHM_SCHEMA_VERSION:
            raise ValueError(
                f"Difficulty metadata algorithm schema {metadata.algorithm_schema_version!r} is unsupported; "
                f"expected {DEFAULT_ALGORITHM_SCHEMA_VERSION!r}."
            )
        motion_lengths = np.asarray(
            self.motion.motion_lengths.detach().cpu().tolist(), dtype=np.int64
        )
        motion_fps = np.asarray(
            self.motion.motion_fps.detach().cpu().tolist(), dtype=np.float64
        )
        motion_segment_offsets = np.asarray(
            self.segment_index.motion_segment_offsets.detach().cpu().tolist(), dtype=np.int64
        )
        segment_global_ids = np.arange(self.segment_index.num_segments, dtype=np.int64)
        segment_motion_ids = np.asarray(
            self.segment_index.segment_motion_ids.detach().cpu().tolist(), dtype=np.int64
        )
        segment_local_ids = np.asarray(
            self.segment_index.segment_local_ids.detach().cpu().tolist(), dtype=np.int64
        )
        segment_start_frames = np.asarray(
            self.segment_index.segment_start_frames.detach().cpu().tolist(), dtype=np.int64
        )
        segment_end_frames = np.asarray(
            self.segment_index.segment_end_frames.detach().cpu().tolist(), dtype=np.int64
        )
        segment_duration_seconds = (
            (
                self.segment_index.segment_end_frames - self.segment_index.segment_start_frames
            ).to(torch.float64)
            / self.motion.motion_fps[self.segment_index.segment_motion_ids]
        ).detach().cpu().numpy()

        # Even an explicitly non-strict diagnostic run must preserve the exact
        # row-to-segment mapping.  Non-strict mode may relax provenance hashes,
        # but it must never expose scores under different global segment IDs.
        layout_checks = (
            ("motion count", metadata.num_motions == self.motion.num_motions),
            ("global segment count", metadata.num_segments == self.segment_index.num_segments),
            (
                "motion keys/order",
                np.array_equal(metadata.motion_keys, np.asarray(self.motion.motion_keys, dtype=str)),
            ),
            ("motion frame counts", np.array_equal(metadata.motion_lengths, motion_lengths)),
            (
                "motion FPS",
                metadata.motion_fps.shape == motion_fps.shape
                and np.allclose(metadata.motion_fps, motion_fps, rtol=0.0, atol=1.0e-12),
            ),
            (
                "segment offsets",
                np.array_equal(metadata.motion_segment_offsets, motion_segment_offsets),
            ),
            ("global segment IDs", np.array_equal(metadata.global_segment_id, segment_global_ids)),
            ("segment motion IDs", np.array_equal(metadata.motion_id, segment_motion_ids)),
            ("local segment IDs", np.array_equal(metadata.local_segment_id, segment_local_ids)),
            ("segment start frames", np.array_equal(metadata.start_frame, segment_start_frames)),
            ("segment end frames", np.array_equal(metadata.end_frame_exclusive, segment_end_frames)),
            (
                "segment durations",
                metadata.duration_seconds.shape == segment_duration_seconds.shape
                and np.allclose(
                    metadata.duration_seconds,
                    segment_duration_seconds,
                    rtol=1.0e-7,
                    atol=1.0e-9,
                ),
            ),
            (
                "segment length",
                math.isclose(
                    metadata.segment_length_seconds,
                    self.segment_index.segment_length_seconds,
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                ),
            ),
            ("segment schema version", metadata.segment_schema_version == SAMPLING_STATE_VERSION),
        )
        layout_mismatches = [label for label, matches in layout_checks if not matches]
        if layout_mismatches:
            raise ValueError(
                "Difficulty metadata has an incompatible segment layout that cannot be relaxed by "
                "strict_metadata_match=False: "
                + ", ".join(layout_mismatches)
                + "."
            )

        metadata_match_ok = metadata.validate_against(
            manifest_path=self.cfg.motion_file,
            motion_keys=self.motion.motion_keys,
            motion_lengths=motion_lengths,
            motion_fps=motion_fps,
            motion_segment_offsets=motion_segment_offsets,
            segment_start_frames=segment_start_frames,
            segment_end_frames=segment_end_frames,
            segment_length_seconds=self.segment_index.segment_length_seconds,
            segment_schema_version=SAMPLING_STATE_VERSION,
            pool_fingerprint=self.motion.pool_fingerprint,
            segment_global_ids=segment_global_ids,
            segment_motion_ids=segment_motion_ids,
            segment_local_ids=segment_local_ids,
            segment_duration_seconds=segment_duration_seconds,
            expected_num_bins=self.cfg.research.difficulty_calibration.expected_num_bins,
            strict=self.cfg.research.difficulty_calibration.strict_metadata_match,
        )
        if metadata.num_bins != self.cfg.research.difficulty_calibration.expected_num_bins:
            raise ValueError(
                f"Difficulty metadata contains {metadata.num_bins} bins, but "
                "difficulty_calibration.expected_num_bins is "
                f"{self.cfg.research.difficulty_calibration.expected_num_bins}."
            )
        if not metadata_match_ok:
            warnings.warn(
                "Difficulty metadata mismatch was explicitly allowed by strict_metadata_match=False; "
                "do not use this run as a formal module-2 result.",
                RuntimeWarning,
                stacklevel=2,
            )

        self.difficulty_metadata = metadata
        self.difficulty_scores = torch.as_tensor(
            metadata.difficulty_score, dtype=torch.float32, device=self.device
        )
        self.difficulty_bins = torch.as_tensor(
            metadata.difficulty_bin, dtype=torch.long, device=self.device
        )
        self.difficulty_metadata_match_ok = metadata_match_ok
        print(
            f"[INFO]: Loaded difficulty metadata '{metadata.path}' "
            f"(segments={self.segment_index.num_segments}, motions={self.motion.num_motions}, "
            f"bins={metadata.num_bins}, sampling=uniform load-only)"
        )

    @property
    def command(self) -> torch.Tensor:  # TODO Consider again if this is the best observation
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)

    @property
    def frame_indices(self) -> torch.Tensor:
        return self.motion.frame_indices(self.motion_ids, self.time_steps)

    @property
    def current_local_segment_ids(self) -> torch.Tensor | None:
        if self.segment_index is None:
            return None
        local_ids, _ = self.segment_index.motion_frame_to_segment(self.motion_ids, self.time_steps)
        return local_ids

    @property
    def current_global_segment_ids(self) -> torch.Tensor | None:
        if self.segment_index is None:
            return None
        _, global_ids = self.segment_index.motion_frame_to_segment(self.motion_ids, self.time_steps)
        return global_ids

    @property
    def current_difficulty_scores(self) -> torch.Tensor | None:
        """Return difficulty scores for the currently referenced segments."""

        current_ids = self.current_global_segment_ids
        if self.difficulty_scores is None or current_ids is None:
            return None
        return self.difficulty_scores[current_ids]

    @property
    def current_difficulty_bins(self) -> torch.Tensor | None:
        """Return difficulty bins for the currently referenced segments."""

        current_ids = self.current_global_segment_ids
        if self.difficulty_bins is None or current_ids is None:
            return None
        return self.difficulty_bins[current_ids]

    @property
    def assigned_difficulty_scores(self) -> torch.Tensor | None:
        """Return difficulty scores for the latest assignment-start segments."""

        if self.difficulty_scores is None:
            return None
        return self.difficulty_scores[self.assigned_global_segment_ids]

    @property
    def assigned_difficulty_bins(self) -> torch.Tensor | None:
        """Return difficulty bins for the latest assignment-start segments."""

        if self.difficulty_bins is None:
            return None
        return self.difficulty_bins[self.assigned_global_segment_ids]

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

    def _write_current_motion_state_to_sim(self, env_ids: Sequence[int], randomize: bool = True) -> None:
        if len(env_ids) == 0:
            return

        root_pos = self.body_pos_w[:, 0].clone()
        root_ori = self.body_quat_w[:, 0].clone()
        root_lin_vel = self.body_lin_vel_w[:, 0].clone()
        root_ang_vel = self.body_ang_vel_w[:, 0].clone()

        if randomize:
            range_list = [self.cfg.pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
            ranges = torch.tensor(range_list, device=self.device)
            rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
            root_pos[env_ids] += rand_samples[:, 0:3]
            orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
            root_ori[env_ids] = quat_mul(orientations_delta, root_ori[env_ids])
            range_list = [
                self.cfg.velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]
            ]
            ranges = torch.tensor(range_list, device=self.device)
            rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
            root_lin_vel[env_ids] += rand_samples[:, :3]
            root_ang_vel[env_ids] += rand_samples[:, 3:]

        joint_pos = self.joint_pos.clone()
        joint_vel = self.joint_vel.clone()

        if randomize:
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

    def _update_relative_body_targets(self) -> None:
        anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)

        delta_pos_w = robot_anchor_pos_w_repeat
        delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
        delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat)))

        self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
        self.body_pos_relative_w = delta_pos_w + quat_apply(delta_ori_w, self.body_pos_w - anchor_pos_w_repeat)

    def set_eval_motion_state(
        self,
        env_ids: Sequence[int] | torch.Tensor,
        motion_ids: Sequence[int] | torch.Tensor,
        time_steps: Sequence[int] | torch.Tensor | None = None,
    ) -> None:
        """Assign exact motion frames for deterministic evaluation.

        This bypasses adaptive sampling and reset randomization so an evaluator
        can run each motion once from a known frame, usually frame zero.
        """
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        motion_ids = torch.as_tensor(motion_ids, dtype=torch.long, device=self.device)
        if time_steps is None:
            time_steps = torch.zeros_like(motion_ids)
        else:
            time_steps = torch.as_tensor(time_steps, dtype=torch.long, device=self.device)

        if env_ids.numel() != motion_ids.numel() or env_ids.numel() != time_steps.numel():
            raise ValueError("env_ids, motion_ids, and time_steps must have the same length.")
        if torch.any(motion_ids < 0) or torch.any(motion_ids >= self.motion.num_motions):
            raise ValueError(f"motion_ids must be in [0, {self.motion.num_motions}).")

        max_time_steps = self.motion.motion_lengths[motion_ids] - 1
        self.motion_ids[env_ids] = motion_ids
        self.time_steps[env_ids] = torch.clamp(time_steps, min=0)
        self.time_steps[env_ids] = torch.minimum(self.time_steps[env_ids], max_time_steps)
        self.time_left[env_ids] = 1.0e9
        self.command_counter[env_ids] = 0

        self._write_current_motion_state_to_sim(env_ids, randomize=False)
        self._env.scene.write_data_to_sim()
        self._env.sim.forward()
        self._update_relative_body_targets()
        self._update_metrics()

    def _record_sampling_assignments(self, env_ids: Sequence[int]) -> None:
        """Observe legacy sampler output without consuming random numbers."""

        if self.segment_index is None or len(env_ids) == 0:
            return
        motion_ids = self.motion_ids[env_ids]
        start_frames = self.time_steps[env_ids]
        if self.sampling_statistics is None:
            local_ids, global_ids = self.segment_index.motion_frame_to_segment(motion_ids, start_frames)
        else:
            local_ids, global_ids = self.sampling_statistics.record_assignments(motion_ids, start_frames)
        self.assigned_local_segment_ids[env_ids] = local_ids
        self.assigned_global_segment_ids[env_ids] = global_ids
        if self.assignment_trace is not None:
            self.assignment_trace.record_assignments(env_ids, motion_ids, start_frames, local_ids, global_ids)

    def _record_quality_reference_exposure(self) -> None:
        """Count current reference frames that lie inside reject segments.

        This is a deterministic observation of ``motion_ids`` and
        ``time_steps`` after the usual reference clock update.  It does not
        draw random numbers and it deliberately does not alter reset behavior:
        ``assignment_start`` still gates only the sampled start frame.
        """

        if (
            not self.cfg.research.quality_gate.enabled
            or self.segment_index is None
            or self.quality_status_codes is None
        ):
            return
        current_ids = self.current_global_segment_ids
        if current_ids is None or current_ids.numel() == 0:
            return
        reject_code = QUALITY_STATUS_TO_CODE["reject"]
        self.quality_reference_frame_count += int(current_ids.numel())
        self.quality_reject_reference_frame_count += int(
            torch.count_nonzero(self.quality_status_codes[current_ids] == reject_code).item()
        )

    def _quality_gated_uniform_sampling(self, env_ids: Sequence[int]) -> None:
        """Uniformly sample motions and quality-eligible legacy start frames."""

        if self.quality_gate_index is None:
            raise RuntimeError("Quality-gated sampling requested before quality metadata initialization.")
        num_samples = len(env_ids)
        eligible_motion_ids = self.quality_gate_index.eligible_motion_ids
        if eligible_motion_ids.numel() == self.motion.num_motions:
            sampled_motion_ids = torch.randint(self.motion.num_motions, (num_samples,), device=self.device)
        else:
            eligible_indexes = torch.randint(eligible_motion_ids.numel(), (num_samples,), device=self.device)
            sampled_motion_ids = eligible_motion_ids[eligible_indexes]
        sampled_phases = sample_uniform(0.0, 1.0, (num_samples,), device=self.device)
        sampled_start_frames, _, _ = self.quality_gate_index.map_uniform_samples(
            sampled_motion_ids, sampled_phases
        )
        self.motion_ids[env_ids] = sampled_motion_ids
        self.time_steps[env_ids] = sampled_start_frames

        num_eligible_motions = int(eligible_motion_ids.numel())
        self.metrics["sampling_entropy"][:] = 1.0
        self.metrics["sampling_top1_prob"][:] = 1.0 / float(num_eligible_motions)
        self.metrics["sampling_top1_bin"][env_ids] = sampled_motion_ids.float() / max(
            self.motion.num_motions - 1, 1
        )

    def _sample_motion_and_start_frame(self, env_ids: Sequence[int]) -> None:
        """Dispatch sampling modes while preserving the legacy uniform RNG path."""

        motion_mode = self.cfg.research.motion_sampling.mode
        segment_mode = self.cfg.research.segment_sampling.mode
        if motion_mode == "uniform" and segment_mode == "uniform":
            if self.cfg.research.quality_gate.enabled:
                self._quality_gated_uniform_sampling(env_ids)
            else:
                self._adaptive_sampling(env_ids)
            return
        raise NotImplementedError(
            f"Sampling modes motion='{motion_mode}', segment='{segment_mode}' are not implemented in module 1."
        )

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        self._sample_motion_and_start_frame(env_ids)
        self._record_sampling_assignments(env_ids)
        self._write_current_motion_state_to_sim(env_ids, randomize=True)

    def _update_command(self):
        self.time_steps += 1
        env_ids = torch.where(self.time_steps >= self.motion.motion_lengths[self.motion_ids])[0]
        self._resample_command(env_ids)
        self._record_quality_reference_exposure()
        self._update_relative_body_targets()

        self.bin_failed_count = (
            self.cfg.adaptive_alpha * self._current_bin_failed + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
        )
        self._current_bin_failed.zero_()

    def research_config_dict(self) -> dict[str, object]:
        """Return low-cardinality research configuration for logs and checkpoints."""

        quality_gate_config: dict[str, object] = {"enabled": self.cfg.research.quality_gate.enabled}
        if self.cfg.research.quality_gate.enabled:
            if self.quality_metadata is None:
                raise RuntimeError("Enabled quality gate has no loaded metadata.")
            quality_gate_config.update(
                {
                    "metadata_path": self.quality_metadata.path,
                    "metadata_sha256": self.quality_metadata.metadata_sha256,
                    "quality_config_sha256": self.quality_metadata.quality_config_sha256,
                    "manifest_sha256": self.quality_metadata.manifest_sha256,
                    "schema_version": self.quality_metadata.schema_version,
                    "reject_statuses": list(self.cfg.research.quality_gate.reject_statuses),
                    "include_borderline": self.cfg.research.quality_gate.include_borderline,
                    "strict_metadata_match": self.cfg.research.quality_gate.strict_metadata_match,
                    "empty_motion_policy": self.cfg.research.quality_gate.empty_motion_policy,
                    "gate_scope": self.cfg.research.quality_gate.gate_scope,
                }
            )
        difficulty_config: dict[str, object] = {
            "enabled": self.cfg.research.difficulty_calibration.enabled
        }
        if self.cfg.research.difficulty_calibration.enabled:
            if self.difficulty_metadata is None:
                raise RuntimeError("Enabled difficulty calibration has no loaded metadata.")
            difficulty_config.update(
                {
                    "metadata_path": self.difficulty_metadata.path,
                    "metadata_sha256": self.difficulty_metadata.metadata_sha256,
                    "profile_sha256": self.difficulty_metadata.profile_sha256,
                    "difficulty_config_sha256": self.difficulty_metadata.difficulty_config_sha256,
                    "manifest_sha256": self.difficulty_metadata.manifest_sha256,
                    "schema_version": self.difficulty_metadata.schema_version,
                    "strict_metadata_match": self.cfg.research.difficulty_calibration.strict_metadata_match,
                    "expected_num_bins": self.cfg.research.difficulty_calibration.expected_num_bins,
                    "num_bins": self.difficulty_metadata.num_bins,
                    "sampling_still_uniform": True,
                }
            )
        return {
            "method_name": self.cfg.research.method_name,
            "segment": {
                "enabled": self.cfg.research.segment.enabled,
                "length_seconds": self.cfg.research.segment.length_seconds,
            },
            "quality_gate": quality_gate_config,
            "difficulty_calibration": difficulty_config,
            "motion_sampling": {"mode": self.cfg.research.motion_sampling.mode},
            "segment_sampling": {"mode": self.cfg.research.segment_sampling.mode},
            "diversity_constraint": {"enabled": self.cfg.research.diversity_constraint.enabled},
            "sampling_statistics": {
                "enabled": self.cfg.research.sampling_statistics.enabled,
                "log_interval": self.cfg.research.sampling_statistics.log_interval,
            },
            "assignment_trace": {
                "enabled": self.cfg.research.assignment_trace.enabled,
                "output_path": self.cfg.research.assignment_trace.output_path,
                "max_entries": self.cfg.research.assignment_trace.max_entries,
            },
            "probability_validation": {"epsilon": self.cfg.research.probability_validation.epsilon},
        }

    def sampling_metrics(self) -> dict[str, float | int] | None:
        if self.sampling_statistics is None:
            return None
        metrics = self.sampling_statistics.summary()
        metrics["quality_gate_enabled"] = int(self.cfg.research.quality_gate.enabled)
        if self.assignment_trace is not None:
            metrics["assignment_trace_recorded_entries"] = int(self.assignment_trace.recorded_entries)
        if self.quality_gate_index is None:
            return metrics
        if self.quality_status_codes is None or self.quality_metadata is None:
            raise RuntimeError("Quality gate metrics requested without status metadata.")

        segment_counts = self.sampling_statistics.segment_sample_count
        pass_code = QUALITY_STATUS_TO_CODE["pass"]
        borderline_code = QUALITY_STATUS_TO_CODE["borderline"]
        reject_code = QUALITY_STATUS_TO_CODE["reject"]
        metrics.update(
            {
                "eligible_segment_coverage": float(
                    torch.count_nonzero(
                        (segment_counts > 0) & self.quality_gate_index.eligible_segment_mask
                    ).item()
                )
                / max(int(torch.count_nonzero(self.quality_gate_index.eligible_segment_mask).item()), 1),
                "pass_segment_sample_count": int(
                    segment_counts[self.quality_status_codes == pass_code].sum().item()
                ),
                "borderline_segment_sample_count": int(
                    segment_counts[self.quality_status_codes == borderline_code].sum().item()
                ),
                "reject_segment_sample_count": int(
                    segment_counts[self.quality_status_codes == reject_code].sum().item()
                ),
                "reject_start_assignment_count": int(
                    segment_counts[self.quality_status_codes == reject_code].sum().item()
                ),
            }
        )
        current_ids = self.current_global_segment_ids
        metrics["current_reject_reference_fraction"] = (
            float((self.quality_status_codes[current_ids] == reject_code).float().mean().item())
            if current_ids is not None and current_ids.numel()
            else 0.0
        )
        return metrics

    def dataset_metrics(self) -> dict[str, float | int]:
        """Return runtime dataset-pool sizes shared by M0/M1 logging."""

        manifest_motion_count = int(self.motion.num_motions)
        if self.quality_gate_index is None:
            effective_motion_count = manifest_motion_count
        else:
            effective_motion_count = int(self.quality_gate_index.eligible_motion_ids.numel())
        excluded_motion_count = manifest_motion_count - effective_motion_count
        return {
            "manifest_motion_count": manifest_motion_count,
            "effective_motion_count": effective_motion_count,
            "excluded_motion_count": excluded_motion_count,
            "eligible_motion_ratio": effective_motion_count / manifest_motion_count
            if manifest_motion_count
            else 0.0,
        }

    def quality_metrics(self) -> dict[str, float | int] | None:
        """Return static quality-pool metrics without uploading segment arrays."""

        if self.quality_metadata is None or self.quality_gate_index is None:
            return None
        metrics = self.quality_metadata.quality_metrics()
        manifest_motion_count = int(self.motion.num_motions)
        effective_motion_count = int(self.quality_gate_index.eligible_motion_ids.numel())
        excluded_motion_count = manifest_motion_count - effective_motion_count
        metrics["manifest_motion_count"] = manifest_motion_count
        metrics["effective_motion_count"] = effective_motion_count
        metrics["num_empty_eligible_motions"] = int(self.quality_gate_index.empty_motion_ids.numel())
        metrics["excluded_motion_count"] = excluded_motion_count
        metrics["eligible_motion_ratio"] = (
            effective_motion_count / manifest_motion_count if manifest_motion_count else 0.0
        )
        metrics["metadata_match_ok"] = int(self.quality_metadata_match_ok)
        reject_code = QUALITY_STATUS_TO_CODE["reject"]
        reject_start_count = 0
        if self.sampling_statistics is not None and self.quality_status_codes is not None:
            reject_start_count = int(
                self.sampling_statistics.segment_sample_count[
                    self.quality_status_codes == reject_code
                ].sum().item()
            )
        reference_count = int(self.quality_reference_frame_count.item())
        reject_reference_count = int(self.quality_reject_reference_frame_count.item())
        metrics["reject_start_assignment_count"] = reject_start_count
        metrics["reject_reference_frame_count"] = reject_reference_count
        metrics["reference_frame_count"] = reference_count
        metrics["reject_rollout_exposure_ratio"] = (
            float(reject_reference_count) / reference_count if reference_count else 0.0
        )
        return metrics

    def difficulty_metrics(self) -> dict[str, float | int] | None:
        """Return static difficulty summaries without affecting sampling."""

        if self.difficulty_metadata is None:
            return None
        metrics = dict(self.difficulty_metadata.difficulty_metrics())
        metrics["enabled"] = 1
        metrics["metadata_match_ok"] = int(self.difficulty_metadata_match_ok)
        metrics["sampling_still_uniform"] = int(
            self.cfg.research.motion_sampling.mode == "uniform"
            and self.cfg.research.segment_sampling.mode == "uniform"
        )
        return metrics

    def wandb_research_metadata(self) -> dict[str, object]:
        """Return flat static-metadata identity fields for the W&B run config."""

        metadata: dict[str, object] = {}
        if self.quality_metadata is not None:
            metadata.update(
                {
                    "quality_metadata_file": os.path.basename(self.quality_metadata.path),
                    "quality_metadata_sha256": self.quality_metadata.metadata_sha256,
                    "quality_config_sha256": self.quality_metadata.quality_config_sha256,
                    "quality_manifest_sha256": self.quality_metadata.manifest_sha256,
                    "quality_schema_version": self.quality_metadata.schema_version,
                    "quality_gate_scope": self.cfg.research.quality_gate.gate_scope,
                    "quality_include_borderline": self.cfg.research.quality_gate.include_borderline,
                    "quality_empty_motion_policy": self.cfg.research.quality_gate.empty_motion_policy,
                }
            )
        if self.difficulty_metadata is not None:
            metadata.update(
                {
                    "difficulty_metadata_file": os.path.basename(self.difficulty_metadata.path),
                    "difficulty_metadata_sha256": self.difficulty_metadata.metadata_sha256,
                    "difficulty_profile_sha256": self.difficulty_metadata.profile_sha256,
                    "difficulty_config_sha256": self.difficulty_metadata.difficulty_config_sha256,
                    "difficulty_manifest_sha256": self.difficulty_metadata.manifest_sha256,
                    "difficulty_schema_version": self.difficulty_metadata.schema_version,
                    "difficulty_num_bins": self.difficulty_metadata.num_bins,
                    "difficulty_metadata_match_ok": self.difficulty_metadata_match_ok,
                    "difficulty_sampling_still_uniform": True,
                }
            )
        return metadata

    def sampling_state_dict(self) -> dict[str, object]:
        """Return segment layout and assignment counters for a training checkpoint."""

        state = {
            "version": SAMPLING_STATE_VERSION,
            "research_config": self.research_config_dict(),
            "segment_index": self.segment_index.state_dict() if self.segment_index is not None else None,
            "statistics": self.sampling_statistics.state_dict() if self.sampling_statistics is not None else None,
        }
        if self.quality_metadata is not None:
            state["quality_gate"] = self.quality_metadata.identity_state()
            if self.quality_gate_index is not None:
                state["quality_gate"].update(self.quality_gate_index.identity_state())
            state["quality_exposure"] = self._quality_exposure_state_dict()
        if self.difficulty_metadata is not None:
            state["difficulty_calibration"] = self._difficulty_identity_state()
        return state

    def _difficulty_identity_state(self) -> dict[str, object]:
        """Return the static difficulty identity persisted in checkpoints."""

        if self.difficulty_metadata is None:
            raise RuntimeError("Difficulty identity requested while difficulty calibration is disabled.")
        identity = dict(self.difficulty_metadata.identity_state())
        identity.update(
            {
                "metadata_path": self.difficulty_metadata.path,
                "metadata_sha256": self.difficulty_metadata.metadata_sha256,
                "profile_sha256": self.difficulty_metadata.profile_sha256,
                "difficulty_config_sha256": self.difficulty_metadata.difficulty_config_sha256,
                "manifest_sha256": self.difficulty_metadata.manifest_sha256,
                "schema_version": self.difficulty_metadata.schema_version,
                "num_bins": self.difficulty_metadata.num_bins,
            }
        )
        return identity

    def _quality_exposure_state_dict(self) -> dict[str, object]:
        return {
            "version": SAMPLING_STATE_VERSION,
            "reference_frame_count": self.quality_reference_frame_count.detach().clone(),
            "reject_reference_frame_count": self.quality_reject_reference_frame_count.detach().clone(),
        }

    def _load_quality_exposure_state_dict(self, state: Mapping[str, object] | None) -> None:
        if state is None:
            self.quality_reference_frame_count.zero_()
            self.quality_reject_reference_frame_count.zero_()
            return
        if int(state.get("version", -1)) != SAMPLING_STATE_VERSION:
            raise ValueError(
                f"Unsupported quality exposure state version {state.get('version')}; "
                f"expected {SAMPLING_STATE_VERSION}."
            )
        required = {"reference_frame_count", "reject_reference_frame_count"}
        missing = required.difference(state)
        if missing:
            raise ValueError(f"Quality exposure state is missing fields: {sorted(missing)}")
        reference_count = torch.as_tensor(
            state.get("reference_frame_count"), dtype=torch.long, device=self.device
        ).reshape(())
        reject_reference_count = torch.as_tensor(
            state.get("reject_reference_frame_count"), dtype=torch.long, device=self.device
        ).reshape(())
        if reference_count.item() < 0 or reject_reference_count.item() < 0:
            raise ValueError("Checkpoint quality exposure counters must be non-negative.")
        if reject_reference_count.item() > reference_count.item():
            raise ValueError("Checkpoint reject_reference_frame_count cannot exceed reference_frame_count.")
        self.quality_reference_frame_count.copy_(reference_count)
        self.quality_reject_reference_frame_count.copy_(reject_reference_count)

    def load_sampling_state_dict(self, state: Mapping[str, object]) -> None:
        """Restore stage-0 statistics after validating the current motion pool."""

        if int(state.get("version", -1)) != SAMPLING_STATE_VERSION:
            raise ValueError(
                f"Unsupported MotionCommand sampling state version {state.get('version')}; "
                f"expected {SAMPLING_STATE_VERSION}."
            )
        saved_config = state.get("research_config")
        if not isinstance(saved_config, Mapping):
            raise ValueError("Checkpoint research_config is missing or invalid.")
        current_config = self.research_config_dict()
        semantic_keys = (
            "method_name",
            "segment",
            "motion_sampling",
            "segment_sampling",
            "diversity_constraint",
            "probability_validation",
        )
        for key in semantic_keys:
            if saved_config.get(key) != current_config[key]:
                raise ValueError(f"Checkpoint research config field '{key}' does not match the current configuration.")
        saved_quality_config = saved_config.get("quality_gate")
        current_quality_config = current_config["quality_gate"]
        if not isinstance(saved_quality_config, Mapping) or not isinstance(current_quality_config, Mapping):
            raise ValueError("Checkpoint quality_gate configuration is invalid.")
        saved_quality_semantics = {
            key: value
            for key, value in _canonical_quality_gate_config(saved_quality_config).items()
            if key != "metadata_path"
        }
        current_quality_semantics = {
            key: value
            for key, value in _canonical_quality_gate_config(current_quality_config).items()
            if key != "metadata_path"
        }
        if saved_quality_semantics != current_quality_semantics:
            raise ValueError(
                "Checkpoint research config field 'quality_gate' does not match the current configuration."
            )
        saved_difficulty_config = saved_config.get("difficulty_calibration")
        current_difficulty_config = current_config["difficulty_calibration"]
        if not isinstance(saved_difficulty_config, Mapping) or not isinstance(
            current_difficulty_config, Mapping
        ):
            raise ValueError("Checkpoint difficulty_calibration configuration is invalid.")
        if _canonical_difficulty_calibration_config(
            saved_difficulty_config
        ) != _canonical_difficulty_calibration_config(current_difficulty_config):
            raise ValueError(
                "Checkpoint research config field 'difficulty_calibration' does not match the current configuration."
            )
        saved_statistics_config = saved_config.get("sampling_statistics")
        current_statistics_config = current_config["sampling_statistics"]
        if not isinstance(current_statistics_config, Mapping):
            raise RuntimeError("Current sampling_statistics configuration is invalid.")
        if not isinstance(saved_statistics_config, Mapping) or (
            saved_statistics_config.get("enabled") != current_statistics_config.get("enabled")
        ):
            raise ValueError("Checkpoint sampling_statistics.enabled does not match the current configuration.")

        saved_index = state.get("segment_index")
        if (saved_index is None) != (self.segment_index is None):
            raise ValueError("Checkpoint segment.enabled setting does not match the current configuration.")
        if self.segment_index is not None:
            if not isinstance(saved_index, dict):
                raise ValueError("Checkpoint segment index state is invalid.")
            self.segment_index.load_state_dict(saved_index)

        saved_statistics = state.get("statistics")
        if (saved_statistics is None) != (self.sampling_statistics is None):
            raise ValueError("Checkpoint sampling_statistics.enabled setting does not match the current configuration.")
        if self.sampling_statistics is not None:
            if not isinstance(saved_statistics, dict):
                raise ValueError("Checkpoint sampling statistics state is invalid.")
            self.sampling_statistics.load_state_dict(saved_statistics)

        saved_quality_state = state.get("quality_gate")
        if self.quality_metadata is None:
            if saved_quality_state is not None:
                raise ValueError("Checkpoint enables a quality gate but the current configuration does not.")
        else:
            if not isinstance(saved_quality_state, Mapping):
                raise ValueError("Quality-gated resume requires checkpoint quality metadata identity state.")
            current_identity = self.quality_metadata.identity_state()
            if self.quality_gate_index is not None:
                current_identity.update(self.quality_gate_index.identity_state())
            for key in (
                "schema_version",
                "segment_schema_version",
                "metadata_sha256",
                "quality_config_sha256",
                "manifest_sha256",
                "pool_fingerprint",
                "empty_motion_policy",
                "manifest_motion_count",
                "effective_motion_count",
                "excluded_motion_count",
                "eligible_motion_mask_sha256",
            ):
                if saved_quality_state.get(key) != current_identity[key]:
                    raise ValueError(f"Checkpoint quality metadata field '{key}' does not match the current run.")
            saved_exposure_state = state.get("quality_exposure")
            if saved_exposure_state is not None and not isinstance(saved_exposure_state, Mapping):
                raise ValueError("Checkpoint quality_exposure state is invalid.")
            self._load_quality_exposure_state_dict(saved_exposure_state)

        saved_difficulty_state = state.get("difficulty_calibration")
        if self.difficulty_metadata is None:
            if saved_difficulty_state is not None:
                raise ValueError(
                    "Checkpoint enables difficulty calibration but the current configuration does not."
                )
        else:
            if not isinstance(saved_difficulty_state, Mapping):
                raise ValueError(
                    "Difficulty-enabled resume requires checkpoint difficulty metadata identity state."
                )
            current_difficulty_identity = self._difficulty_identity_state()
            saved_identity = {
                key: value for key, value in saved_difficulty_state.items() if key != "metadata_path"
            }
            current_identity = {
                key: value
                for key, value in current_difficulty_identity.items()
                if key != "metadata_path"
            }
            identity_keys = sorted(set(saved_identity) | set(current_identity))
            for key in identity_keys:
                if saved_identity.get(key) != current_identity.get(key):
                    raise ValueError(
                        f"Checkpoint difficulty metadata field '{key}' does not match the current run."
                    )

    def reset_sampling_statistics(self) -> None:
        if self.sampling_statistics is not None:
            self.sampling_statistics.reset_statistics()
        if self.assignment_trace is not None:
            self.assignment_trace.reset()
        self.quality_reference_frame_count.zero_()
        self.quality_reject_reference_frame_count.zero_()

    def record_current_sampling_assignments(self) -> None:
        """Count the active assignments created while a resume environment was initialized."""

        env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        self._record_sampling_assignments(env_ids)

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
class SegmentInfrastructureCfg:
    """Fixed-duration segment indexing settings."""

    enabled: bool = True
    length_seconds: float = 1.0


@configclass
class ResearchFeatureToggleCfg:
    """Common switch for research algorithms introduced after stage 0."""

    enabled: bool = False


@configclass
class QualityGateCfg:
    """Frozen offline quality metadata and M1 start-frame gating settings."""

    enabled: bool = False
    metadata_path: str = ""
    reject_statuses: tuple[str, ...] = ("reject",)
    include_borderline: bool = True
    strict_metadata_match: bool = True
    empty_motion_policy: str = "error"
    gate_scope: str = "assignment_start"


@configclass
class DifficultyCalibrationCfg:
    """Frozen policy-independent difficulty metadata loaded without resampling."""

    enabled: bool = False
    metadata_path: str = ""
    strict_metadata_match: bool = True
    expected_num_bins: int = 10


@configclass
class SamplingModeCfg:
    """Sampling mode selection; stage 0 implements only the legacy path."""

    mode: str = "uniform"


@configclass
class SamplingStatisticsCfg:
    """Shared assignment statistics and runner logging settings."""

    enabled: bool = True
    log_interval: int = 100


@configclass
class AssignmentTraceCfg:
    """Optional bounded CSV trace for RNG-equivalence smoke tests."""

    enabled: bool = False
    output_path: str = ""
    max_entries: int = 2048


@configclass
class ProbabilityValidationCfg:
    """Numerical tolerance for future adaptive sampling probabilities."""

    epsilon: float = 1.0e-8


@configclass
class ResearchExperimentCfg:
    """Unified M0--M7 research configuration."""

    method_name: str = "M0"
    segment: SegmentInfrastructureCfg = SegmentInfrastructureCfg()
    quality_gate: QualityGateCfg = QualityGateCfg()
    difficulty_calibration: DifficultyCalibrationCfg = DifficultyCalibrationCfg()
    motion_sampling: SamplingModeCfg = SamplingModeCfg()
    segment_sampling: SamplingModeCfg = SamplingModeCfg()
    diversity_constraint: ResearchFeatureToggleCfg = ResearchFeatureToggleCfg()
    sampling_statistics: SamplingStatisticsCfg = SamplingStatisticsCfg()
    assignment_trace: AssignmentTraceCfg = AssignmentTraceCfg()
    probability_validation: ProbabilityValidationCfg = ProbabilityValidationCfg()


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

    research: ResearchExperimentCfg = ResearchExperimentCfg()

    anchor_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    anchor_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)

    body_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    body_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
