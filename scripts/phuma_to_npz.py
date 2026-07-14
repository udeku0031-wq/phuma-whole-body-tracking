"""Convert PHUMA G1 motions to whole_body_tracking motion npz files.

PHUMA stores each motion as a .npy dictionary:
    root_trans: (T, 3)
    root_ori:   (T, 4), xyzw quaternion
    dof_pos:    (T, 29) for G1
    fps:        input fps

whole_body_tracking expects a .npz containing full robot joint/body states.
This script plays PHUMA root/dof poses through the same Isaac Lab G1 asset and
records the tensors consumed by MotionCommand.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

from isaaclab.app import AppLauncher


G1_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert PHUMA G1 .npy motions to WBT .npz motions.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input_file", type=Path, help="Single PHUMA .npy motion file.")
    source.add_argument("--input_dir", type=Path, help="Directory containing PHUMA .npy motion files.")
    parser.add_argument("--pattern", type=str, default="*.npy", help="Glob pattern used with --input_dir.")
    parser.add_argument("--output_dir", type=Path, default=Path("PHUMA_wbt_motions/g1"), help="Output directory.")
    parser.add_argument("--output_fps", type=int, default=50, help="Output fps expected by WBT training.")
    parser.add_argument("--start", type=int, default=0, help="Start index after sorting matched files.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of files to convert.")
    parser.add_argument(
        "--frame_range",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        help="Optional 1-based inclusive frame range to convert from each PHUMA motion.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output .npz files.")
    parser.add_argument("--compressed", action="store_true", help="Use np.savez_compressed for smaller files.")
    parser.add_argument("--progress_interval", type=int, default=500, help="Print progress every N output frames.")
    parser.add_argument(
        "--close_app",
        action="store_true",
        help="Close Isaac Sim normally after conversion. By default the script exits immediately after saving.",
    )
    parser.add_argument(
        "--upload_wandb",
        action="store_true",
        help="Upload each converted motion to the W&B motion registry after saving it locally.",
    )
    parser.add_argument("--wandb_project", type=str, default="phuma_to_npz", help="W&B project for conversion runs.")
    parser.add_argument(
        "--registry_prefix",
        type=str,
        default="phuma",
        help="Prefix used for W&B artifact names when --upload_wandb is set.",
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


args_cli = parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_mul, quat_slerp

from whole_body_tracking.robots.g1 import G1_CYLINDER_CFG


@configclass
class ConversionSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )
    robot: ArticulationCfg = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


class PhumaMotion:
    def __init__(
        self,
        motion_file: Path,
        output_fps: int,
        device: torch.device,
        frame_range: tuple[int, int] | None = None,
    ):
        self.motion_file = motion_file
        self.output_fps = output_fps
        self.output_dt = 1.0 / output_fps
        self.device = device
        self.frame_range = frame_range
        self._load()
        self._interpolate()
        self._compute_velocities()

    def _load(self) -> None:
        raw = np.load(self.motion_file, allow_pickle=True)
        if not isinstance(raw, np.ndarray) or raw.shape != ():
            raise ValueError(f"{self.motion_file} is not a PHUMA dictionary .npy file.")
        data = raw.item()
        required = {"root_trans", "root_ori", "dof_pos", "fps"}
        missing = required.difference(data)
        if missing:
            raise ValueError(f"{self.motion_file} is missing PHUMA keys: {sorted(missing)}")

        root_trans = np.asarray(data["root_trans"], dtype=np.float32)
        root_ori = np.asarray(data["root_ori"], dtype=np.float32)
        dof_pos = np.asarray(data["dof_pos"], dtype=np.float32)
        if root_trans.ndim != 2 or root_trans.shape[1] != 3:
            raise ValueError(f"{self.motion_file}: root_trans must have shape (T, 3), got {root_trans.shape}")
        if root_ori.ndim != 2 or root_ori.shape[1] != 4:
            raise ValueError(f"{self.motion_file}: root_ori must have shape (T, 4), got {root_ori.shape}")
        if dof_pos.ndim != 2:
            raise ValueError(f"{self.motion_file}: dof_pos must have shape (T, D), got {dof_pos.shape}")
        if dof_pos.shape[1] == 23:
            # Some raw PHUMA G1 files omit the six wrist joints. Match WBT's 29-DOF G1 ordering.
            dof_pos = np.concatenate(
                [dof_pos[:, :19], np.zeros((dof_pos.shape[0], 3), dtype=np.float32), dof_pos[:, 19:], np.zeros((dof_pos.shape[0], 3), dtype=np.float32)],
                axis=1,
            )
        if dof_pos.shape[1] != len(G1_JOINT_NAMES):
            raise ValueError(
                f"{self.motion_file}: PHUMA G1 dof_pos has {dof_pos.shape[1]} DOFs, "
                f"but WBT G1 expects {len(G1_JOINT_NAMES)}. Use PHUMA/data/g1, not h1_2."
            )
        if not (root_trans.shape[0] == root_ori.shape[0] == dof_pos.shape[0]):
            raise ValueError(
                f"{self.motion_file}: frame counts differ: root_trans={root_trans.shape[0]}, "
                f"root_ori={root_ori.shape[0]}, dof_pos={dof_pos.shape[0]}"
            )

        if self.frame_range is not None:
            start = self.frame_range[0] - 1
            end = self.frame_range[1]
            root_trans = root_trans[start:end]
            root_ori = root_ori[start:end]
            dof_pos = dof_pos[start:end]

        if root_trans.shape[0] < 3:
            raise ValueError(f"{self.motion_file}: need at least 3 frames, got {root_trans.shape[0]}")

        self.input_fps = float(np.asarray(data["fps"]).squeeze())
        self.input_dt = 1.0 / self.input_fps
        self.root_pos_input = torch.tensor(root_trans, dtype=torch.float32, device=self.device)
        # PHUMA stores xyzw, Isaac/WBT root state uses wxyz.
        self.root_quat_input = torch.tensor(root_ori[:, [3, 0, 1, 2]], dtype=torch.float32, device=self.device)
        self.root_quat_input = self.root_quat_input / torch.linalg.norm(self.root_quat_input, dim=1, keepdim=True)
        self.dof_pos_input = torch.tensor(dof_pos, dtype=torch.float32, device=self.device)
        self.input_frames = root_trans.shape[0]
        self.duration = (self.input_frames - 1) * self.input_dt

    def _interpolate(self) -> None:
        times = torch.arange(0.0, self.duration, self.output_dt, device=self.device, dtype=torch.float32)
        self.output_frames = int(times.shape[0])
        phase = times / self.duration
        index_0 = (phase * (self.input_frames - 1)).floor().long()
        index_1 = torch.minimum(index_0 + 1, torch.tensor(self.input_frames - 1, device=self.device))
        blend = phase * (self.input_frames - 1) - index_0

        self.root_pos = self._lerp(self.root_pos_input[index_0], self.root_pos_input[index_1], blend[:, None])
        self.root_quat = self._slerp(self.root_quat_input[index_0], self.root_quat_input[index_1], blend)
        self.dof_pos = self._lerp(self.dof_pos_input[index_0], self.dof_pos_input[index_1], blend[:, None])

    def _compute_velocities(self) -> None:
        self.root_lin_vel = torch.gradient(self.root_pos, spacing=self.output_dt, dim=0)[0]
        self.dof_vel = torch.gradient(self.dof_pos, spacing=self.output_dt, dim=0)[0]
        self.root_ang_vel = self._so3_derivative(self.root_quat, self.output_dt)

    @staticmethod
    def _lerp(a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        return a * (1.0 - blend) + b * blend

    @staticmethod
    def _slerp(a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(a)
        for i in range(a.shape[0]):
            out[i] = quat_slerp(a[i], b[i], blend[i])
        return out

    @staticmethod
    def _so3_derivative(rotations: torch.Tensor, dt: float) -> torch.Tensor:
        q_prev, q_next = rotations[:-2], rotations[2:]
        q_rel = quat_mul(q_next, quat_conjugate(q_prev))
        omega = axis_angle_from_quat(q_rel) / (2.0 * dt)
        return torch.cat([omega[:1], omega, omega[-1:]], dim=0)


def collect_input_files(args: argparse.Namespace) -> list[Path]:
    if args.input_file is not None:
        return [args.input_file]
    files = sorted(args.input_dir.rglob(args.pattern))
    if args.limit is None:
        return files[args.start :]
    return files[args.start : args.start + args.limit]


def output_path_for(input_file: Path, args: argparse.Namespace) -> Path:
    if args.input_dir is not None:
        rel = input_file.relative_to(args.input_dir)
        return args.output_dir / rel.with_suffix(".npz")
    return args.output_dir / input_file.with_suffix(".npz").name


def upload_motion(output_file: Path, artifact_name: str, args: argparse.Namespace) -> None:
    import wandb

    run = wandb.init(project=args.wandb_project, name=artifact_name)
    print(f"[INFO]: Logging PHUMA motion to wandb: {artifact_name}", flush=True)
    artifact = run.log_artifact(artifact_or_path=str(output_file), name=artifact_name, type="motions")
    run.link_artifact(artifact=artifact, target_path=f"wandb-registry-motions/{artifact_name}")
    run.finish()


def artifact_name_for(output_file: Path, args: argparse.Namespace) -> str:
    rel = output_file.relative_to(args.output_dir).with_suffix("")
    safe_name = rel.as_posix().replace("/", "__").replace(" ", "_")
    return f"{args.registry_prefix}_{safe_name}"


def convert_one(
    input_file: Path,
    output_file: Path,
    sim: sim_utils.SimulationContext,
    scene: InteractiveScene,
    joint_indexes: list[int],
    args: argparse.Namespace,
) -> None:
    motion = PhumaMotion(input_file, output_fps=args.output_fps, device=sim.device, frame_range=args.frame_range)
    robot = scene["robot"]
    output_file.parent.mkdir(parents=True, exist_ok=True)

    log = {
        "fps": np.array([args.output_fps], dtype=np.float32),
        "joint_pos": [],
        "joint_vel": [],
        "body_pos_w": [],
        "body_quat_w": [],
        "body_lin_vel_w": [],
        "body_ang_vel_w": [],
    }

    for frame in range(motion.output_frames):
        root_states = robot.data.default_root_state.clone()
        root_states[:, :3] = motion.root_pos[frame : frame + 1]
        root_states[:, 3:7] = motion.root_quat[frame : frame + 1]
        root_states[:, 7:10] = motion.root_lin_vel[frame : frame + 1]
        root_states[:, 10:] = motion.root_ang_vel[frame : frame + 1]
        robot.write_root_state_to_sim(root_states)

        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()
        joint_pos[:, joint_indexes] = motion.dof_pos[frame : frame + 1]
        joint_vel[:, joint_indexes] = motion.dof_vel[frame : frame + 1]
        robot.write_joint_state_to_sim(joint_pos, joint_vel)

        sim.render()
        scene.update(sim.get_physics_dt())

        log["joint_pos"].append(robot.data.joint_pos[0].cpu().numpy().copy())
        log["joint_vel"].append(robot.data.joint_vel[0].cpu().numpy().copy())
        log["body_pos_w"].append(robot.data.body_pos_w[0].cpu().numpy().copy())
        log["body_quat_w"].append(robot.data.body_quat_w[0].cpu().numpy().copy())
        log["body_lin_vel_w"].append(robot.data.body_lin_vel_w[0].cpu().numpy().copy())
        log["body_ang_vel_w"].append(robot.data.body_ang_vel_w[0].cpu().numpy().copy())

        if args.progress_interval > 0 and (frame + 1) % args.progress_interval == 0:
            print(f"[INFO]: {input_file} converted {frame + 1}/{motion.output_frames} frames", flush=True)

    for key in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
        log[key] = np.stack(log[key], axis=0)
    log["joint_names"] = np.asarray(robot.joint_names)
    log["body_names"] = np.asarray(robot.body_names)
    log["source_file"] = np.asarray(str(input_file))
    log["source_format"] = np.asarray("PHUMA_G1")

    if args.compressed:
        np.savez_compressed(output_file, **log)
    else:
        np.savez(output_file, **log)


def main() -> None:
    files = collect_input_files(args_cli)
    if not files:
        raise SystemExit("No PHUMA .npy files matched the input.")

    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / args_cli.output_fps
    sim = SimulationContext(sim_cfg)
    scene = InteractiveScene(ConversionSceneCfg(num_envs=1, env_spacing=2.0))
    sim.reset()

    robot = scene["robot"]
    joint_indexes = robot.find_joints(G1_JOINT_NAMES, preserve_order=True)[0]
    if len(joint_indexes) != len(G1_JOINT_NAMES):
        raise RuntimeError(f"Could not find all G1 joints in Isaac robot. Found {len(joint_indexes)}.")

    print(f"[INFO]: Found {len(files)} PHUMA motion(s). Output directory: {args_cli.output_dir}", flush=True)
    for index, input_file in enumerate(files, start=1):
        output_file = output_path_for(input_file, args_cli)
        if output_file.exists() and not args_cli.overwrite:
            print(f"[{index}/{len(files)}] skip existing {output_file}", flush=True)
            continue
        print(f"[{index}/{len(files)}] converting {input_file} -> {output_file}", flush=True)
        convert_one(input_file, output_file, sim, scene, joint_indexes, args_cli)
        print(f"[INFO]: Saved {output_file}", flush=True)
        if args_cli.upload_wandb:
            artifact_name = artifact_name_for(output_file, args_cli)
            upload_motion(output_file, artifact_name, args_cli)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    if args_cli.close_app:
        simulation_app.close()
    else:
        os._exit(0)
