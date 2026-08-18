"""Collect frozen-policy per-segment joint-error snapshots on the train set.

This is an offline diagnostic collector.  It loads frozen policy checkpoints,
runs eval/inference only, and writes per-segment mean absolute joint error
snapshots for STEP 7 lambda replay.  It does not train, update PPO, or touch
optimizer state.
"""

from __future__ import annotations

"""Launch Isaac Sim before importing simulator-dependent modules."""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Sequence

from isaaclab.app import AppLauncher

import cli_args  # isort: skip
import evaluation_utils as eval_utils  # isort: skip
import joint_diagnostics as joint_diag  # isort: skip


DEFAULT_RUN_DIR = "logs/rsl_rl/g1_flat/2026-08-05_15-31-20_formal_v1_ablation_M7Raw_n6000_trainseed42"
DEFAULT_TRAIN_MANIFEST = "PHUMA_wbt_motions/manifests/experiments/random_seed42/random6000_seed42.txt"
DEFAULT_OUTPUT_DIR = "outputs/joint_gap_stage3_lambda_replay/snapshots"
DEFAULT_QUALITY_METADATA = "outputs/module1_quality_random6000_seed42_original_v1/segment_quality_metadata.npz"
DEFAULT_DIFFICULTY_METADATA = "outputs/module2_difficulty_random6000_seed42_v1/segment_difficulty_metadata.npz"
DEFAULT_CLUSTER_METADATA = "outputs/module4_clusters_random6000_seed42_v1/motion_cluster_metadata.npz"


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
parser.add_argument("--num_envs", type=int, default=3072, help="Parallel diagnostic environments.")
parser.add_argument("--task", type=str, default="Tracking-Flat-G1-v0", help="Task name.")
parser.add_argument("--motion_file", type=str, default=DEFAULT_TRAIN_MANIFEST, help="Training motion manifest.")
parser.add_argument("--run_dir", type=str, default=DEFAULT_RUN_DIR, help="M7-Raw run directory.")
parser.add_argument("--checkpoints", default="10000,20000,30000,final", help="Checkpoint targets.")
parser.add_argument("--checkpoint_pattern", default="model_*.pt", help="Checkpoint glob inside --run_dir.")
parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Snapshot output directory.")
parser.add_argument("--quality_metadata", type=str, default=DEFAULT_QUALITY_METADATA, help="Frozen quality metadata.")
parser.add_argument("--difficulty_metadata", type=str, default=DEFAULT_DIFFICULTY_METADATA, help="Frozen difficulty metadata.")
parser.add_argument("--cluster_metadata", type=str, default=DEFAULT_CLUSTER_METADATA, help="Frozen cluster metadata.")
parser.add_argument("--seed", type=int, default=42, help="Diagnostic seed.")
parser.add_argument("--episode_length_s", type=float, default=60.0, help="Safety cap for the diagnostic env.")
parser.add_argument("--deterministic", action="store_true", help="Use deterministic inference policy.")
parser.add_argument("--disable_randomization", action="store_true", default=True, help="Disable reset/domain randomization.")
parser.add_argument("--include_start_frame", action="store_true", help="Include the reset/reference start frame in the proxy mean.")
parser.add_argument("--max_segments", type=int, default=None, help="Debug only: collect the first N eligible segments.")
parser.add_argument("--progress_interval", type=int, default=1, help="Print every N completed segment batches.")
parser.add_argument("--resume_snapshots", action="store_true", help="Skip snapshots that already exist.")
parser.add_argument("--dry_run", action="store_true", help="Print selected checkpoints and exit before launching Isaac.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.dry_run:
    project_root = Path.cwd().resolve()
    run_dir = (project_root / args_cli.run_dir).resolve() if not Path(args_cli.run_dir).is_absolute() else Path(args_cli.run_dir)
    selected = eval_utils.select_checkpoints(
        eval_utils.list_checkpoints(run_dir, args_cli.checkpoint_pattern),
        args_cli.checkpoints,
    )
    print("[DRY-RUN] collect_joint_gap_training_snapshot.py")
    print(f"  task: {args_cli.task}")
    print(f"  motion_file: {args_cli.motion_file}")
    print(f"  run_dir: {run_dir}")
    print(f"  output_dir: {args_cli.output_dir}")
    print(f"  num_envs: {args_cli.num_envs}")
    print(f"  selected_checkpoints: {[path.name for path in selected]}")
    sys.exit(0)

if hasattr(args_cli, "headless"):
    args_cli.headless = True

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
sys.argv = [sys.argv[0]] + hydra_args

import gymnasium as gym
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

import whole_body_tracking.tasks  # noqa: F401
from whole_body_tracking.utils.cluster_metadata import MotionClusterMetadata
from whole_body_tracking.utils.difficulty_metadata import SegmentDifficultyMetadata
from whole_body_tracking.utils.quality_metadata import SegmentQualityMetadata
from whole_body_tracking.utils.sampling import SAMPLING_STATE_VERSION


def _project_root() -> Path:
    return Path.cwd().resolve()


def _absolute(path_text: str | os.PathLike[str], project_root: Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else project_root / path


def _checkpoint_iteration(path: Path) -> int:
    return eval_utils.parse_checkpoint_iteration(path)


def _snapshot_path(output_dir: Path, checkpoint: Path) -> Path:
    return output_dir / f"m7raw_ckpt_{_checkpoint_iteration(checkpoint)}_joint_snapshot.npz"


def _disable_randomization(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg) -> list[str]:
    warnings: list[str] = []
    motion_cfg = getattr(getattr(env_cfg, "commands", None), "motion", None)
    if motion_cfg is not None:
        zero_pose_keys = ("x", "y", "z", "roll", "pitch", "yaw")
        motion_cfg.pose_range = {key: (0.0, 0.0) for key in zero_pose_keys}
        motion_cfg.velocity_range = {key: (0.0, 0.0) for key in zero_pose_keys}
        motion_cfg.joint_position_range = (0.0, 0.0)
        motion_cfg.debug_vis = False
    else:
        warnings.append("env_cfg.commands.motion was not found; motion reset randomization could not be disabled.")

    observations_cfg = getattr(env_cfg, "observations", None)
    if observations_cfg is not None:
        for group_name in ("policy", "critic"):
            group_cfg = getattr(observations_cfg, group_name, None)
            if group_cfg is None:
                continue
            if hasattr(group_cfg, "enable_corruption"):
                group_cfg.enable_corruption = False
            for term_cfg in getattr(group_cfg, "__dict__", {}).values():
                if hasattr(term_cfg, "noise"):
                    term_cfg.noise = None
    else:
        warnings.append("env_cfg.observations was not found; observation corruption could not be checked.")

    events_cfg = getattr(env_cfg, "events", None)
    if events_cfg is not None:
        for name in ("physics_material", "add_joint_default_pos", "base_com", "push_robot"):
            if hasattr(events_cfg, name):
                setattr(events_cfg, name, None)
    else:
        warnings.append("env_cfg.events was not found; domain randomization events could not be checked.")
    return warnings


def _force_deterministic_eval_research_config(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg) -> None:
    """Use evaluator-safe uniform config; STEP 7 metadata is loaded offline."""

    motion_cfg = getattr(getattr(env_cfg, "commands", None), "motion", None)
    research = getattr(motion_cfg, "research", None)
    if research is None:
        return
    research.method_name = "M0"
    research.motion_sampling.mode = "uniform"
    research.segment_sampling.mode = "uniform"
    research.quality_gate.enabled = False
    research.difficulty_calibration.enabled = False
    research.diversity_constraint.enabled = False
    research.online_learning.enabled = False
    research.online_learning.statistics_enabled = False
    research.joint_gap.enabled = False
    research.assignment_trace.enabled = False
    research.sampling_statistics.enabled = False


def _segment_layout(command) -> dict[str, np.ndarray]:
    segment_index = command.segment_index
    if segment_index is None:
        raise RuntimeError("Segment index is required for STEP 7 snapshots.")
    motion_ids = segment_index.segment_motion_ids.detach().cpu().numpy().astype(np.int64, copy=False)
    starts = segment_index.segment_start_frames.detach().cpu().numpy().astype(np.int64, copy=False)
    ends = segment_index.segment_end_frames.detach().cpu().numpy().astype(np.int64, copy=False)
    fps = command.motion.motion_fps[segment_index.segment_motion_ids].detach().cpu().numpy().astype(np.float64, copy=False)
    durations = (ends - starts) / fps
    return {
        "motion_keys": np.asarray(command.motion.motion_keys, dtype=str),
        "motion_lengths": command.motion.motion_lengths.detach().cpu().numpy().astype(np.int64, copy=False),
        "motion_fps": command.motion.motion_fps.detach().cpu().numpy().astype(np.float64, copy=False),
        "motion_segment_offsets": segment_index.motion_segment_offsets.detach().cpu().numpy().astype(np.int64, copy=False),
        "global_segment_id": np.arange(int(segment_index.num_segments), dtype=np.int64),
        "motion_id": motion_ids,
        "local_segment_id": segment_index.segment_local_ids.detach().cpu().numpy().astype(np.int64, copy=False),
        "start_frame": starts,
        "end_frame_exclusive": ends,
        "duration_seconds": durations,
        "segment_length_seconds": float(segment_index.segment_length_seconds),
        "pool_fingerprint": str(command.motion.pool_fingerprint),
    }


def _load_and_validate_offline_metadata(command, project_root: Path) -> tuple[
    SegmentQualityMetadata,
    SegmentDifficultyMetadata,
    MotionClusterMetadata,
]:
    layout = _segment_layout(command)
    quality = SegmentQualityMetadata.load(_absolute(args_cli.quality_metadata, project_root))
    difficulty = SegmentDifficultyMetadata.load(_absolute(args_cli.difficulty_metadata, project_root))
    cluster = MotionClusterMetadata.load(_absolute(args_cli.cluster_metadata, project_root))

    validate_common = {
        "manifest_path": args_cli.motion_file,
        "motion_keys": layout["motion_keys"],
        "motion_lengths": layout["motion_lengths"],
        "motion_fps": layout["motion_fps"],
        "motion_segment_offsets": layout["motion_segment_offsets"],
        "segment_start_frames": layout["start_frame"],
        "segment_end_frames": layout["end_frame_exclusive"],
        "segment_length_seconds": layout["segment_length_seconds"],
        "segment_schema_version": SAMPLING_STATE_VERSION,
        "pool_fingerprint": layout["pool_fingerprint"],
        "strict": True,
    }
    quality.validate_against(**validate_common)
    difficulty.validate_against(
        **validate_common,
        segment_global_ids=layout["global_segment_id"],
        segment_motion_ids=layout["motion_id"],
        segment_local_ids=layout["local_segment_id"],
        segment_duration_seconds=layout["duration_seconds"],
        expected_num_bins=10,
    )
    cluster.validate_against(
        manifest_path=args_cli.motion_file,
        motion_keys=layout["motion_keys"],
        motion_lengths=layout["motion_lengths"],
        motion_fps=layout["motion_fps"],
        motion_segment_offsets=layout["motion_segment_offsets"],
        pool_fingerprint=layout["pool_fingerprint"],
        difficulty_metadata_sha256=difficulty.metadata_sha256,
        difficulty_profile_sha256=difficulty.profile_sha256,
        expected_num_clusters=8,
        strict=True,
    )
    if quality.manifest_sha256 != difficulty.manifest_sha256 or quality.pool_fingerprint != difficulty.pool_fingerprint:
        raise RuntimeError("Quality and difficulty metadata do not describe the same frozen training pool.")
    if cluster.manifest_sha256 != difficulty.manifest_sha256 or cluster.pool_fingerprint != difficulty.pool_fingerprint:
        raise RuntimeError("Cluster and difficulty metadata do not describe the same frozen training pool.")
    return quality, difficulty, cluster


def _offline_eligible_mask(command, quality: SegmentQualityMetadata) -> torch.Tensor:
    segment_index = command.segment_index
    if segment_index is None:
        raise RuntimeError("Segment index is required for STEP 7 snapshots.")
    segment_motion_ids = segment_index.segment_motion_ids
    legal_end = torch.minimum(
        segment_index.segment_end_frames,
        command.motion.motion_lengths[segment_motion_ids] - 1,
    )
    eligible = legal_end > segment_index.segment_start_frames
    quality_mask = torch.as_tensor(
        quality.accepted_mask(include_borderline=True),
        dtype=torch.bool,
        device=eligible.device,
    )
    eligible &= quality_mask
    return eligible.to(torch.bool)


def _write_manifest(path: Path, entries: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "kind": "joint_gap_training_snapshot_manifest",
        "proxy_limitation": (
            "Snapshots are deterministic frozen-policy diagnostic rollouts on the training "
            "motion library. They are not reconstructed M7-Raw historical per-joint EMA state."
        ),
        "snapshots": entries,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _checkpoint_snapshot(
    *,
    env,
    base_env,
    ppo_runner: OnPolicyRunner,
    checkpoint: Path,
    output_path: Path,
    selected_segments: torch.Tensor,
    eligible_mask: torch.Tensor,
    joint_names: Sequence[str],
    joint_mapping_hash: str,
    motion_paths: Sequence[str],
    categories: Sequence[str],
    source_groups: Sequence[str],
    difficulty_bins: torch.Tensor,
    motion_cluster_ids: torch.Tensor,
    metadata_identity: dict[str, str],
    collector_config: dict[str, object],
) -> dict[str, object]:
    ppo_runner.load(str(checkpoint))
    actor_critic = getattr(ppo_runner.alg, "policy", getattr(ppo_runner.alg, "actor_critic", None))
    if actor_critic is not None and hasattr(actor_critic, "eval"):
        actor_critic.eval()
    policy = ppo_runner.get_inference_policy(device=base_env.device)

    command = base_env.command_manager.get_term("motion")
    segment_index = command.segment_index
    if segment_index is None:
        raise RuntimeError("Segment index is required for STEP 7 snapshots.")
    num_segments = int(segment_index.num_segments)
    joint_count = len(joint_names)
    abs_sum = torch.zeros(num_segments, joint_count, dtype=torch.float64, device=base_env.device)
    obs_count = torch.zeros(num_segments, dtype=torch.long, device=base_env.device)
    interrupted = 0
    started_at = time.perf_counter()

    for batch_number, batch_start in enumerate(range(0, int(selected_segments.numel()), env.num_envs), start=1):
        if not simulation_app.is_running():
            interrupted += int(selected_segments.numel()) - batch_start
            break
        batch_segments = selected_segments[batch_start : batch_start + env.num_envs].to(base_env.device)
        batch_size = int(batch_segments.numel())
        env_ids = torch.arange(batch_size, dtype=torch.long, device=base_env.device)
        motion_ids = segment_index.segment_motion_ids[batch_segments]
        start_frames = segment_index.segment_start_frames[batch_segments]
        end_frames = torch.minimum(
            segment_index.segment_end_frames[batch_segments],
            command.motion.motion_lengths[motion_ids] - 1,
        )
        target_steps = torch.clamp(end_frames - start_frames - (0 if args_cli.include_start_frame else 1), min=1)

        env.reset()
        command = base_env.command_manager.get_term("motion")
        command.set_eval_motion_state(env_ids, motion_ids, start_frames)
        base_env.episode_length_buf[env_ids] = 0
        obs, _ = env.get_observations()
        active = torch.zeros(env.num_envs, dtype=torch.bool, device=base_env.device)
        active[:batch_size] = True
        recorded_steps = torch.zeros(env.num_envs, dtype=torch.long, device=base_env.device)

        if args_cli.include_start_frame:
            command._update_metrics(record_online=False)
            joint_error = torch.abs(command.joint_pos[env_ids] - command.robot_joint_pos[env_ids]).to(torch.float64)
            abs_sum.index_add_(0, batch_segments, joint_error)
            obs_count.index_add_(0, batch_segments, torch.ones(batch_size, dtype=torch.long, device=base_env.device))
            recorded_steps[:batch_size] += 1

        while bool(torch.any(active).item()) and simulation_app.is_running():
            active_ids = torch.nonzero(active, as_tuple=False).flatten()
            with torch.no_grad():
                actions = policy(obs)
                obs, _, dones, _ = env.step(actions)
            command = base_env.command_manager.get_term("motion")
            command._update_metrics(record_online=False)
            current_ids = command.current_global_segment_ids
            if current_ids is None:
                raise RuntimeError("Current segment IDs are required for STEP 7 snapshots.")

            expected_segments = batch_segments[active_ids]
            same_segment = current_ids[active_ids] == expected_segments
            record_ids = active_ids[same_segment]
            if record_ids.numel():
                record_segments = batch_segments[record_ids]
                joint_error = torch.abs(command.joint_pos[record_ids] - command.robot_joint_pos[record_ids]).to(torch.float64)
                abs_sum.index_add_(0, record_segments, joint_error)
                obs_count.index_add_(0, record_segments, torch.ones(record_ids.numel(), dtype=torch.long, device=base_env.device))
                recorded_steps[record_ids] += 1

            done_now = torch.zeros_like(active_ids, dtype=torch.bool)
            done_now |= dones[active_ids].to(torch.bool)
            done_now |= ~same_segment
            done_now |= recorded_steps[active_ids] >= target_steps[active_ids]
            active[active_ids[done_now]] = False

        if args_cli.progress_interval > 0 and batch_number % args_cli.progress_interval == 0:
            observed = int(torch.count_nonzero(obs_count[selected_segments.to(base_env.device)] > 0).item())
            print(
                f"[INFO]: {checkpoint.name} batch {batch_number} "
                f"observed_segments={observed}/{int(selected_segments.numel())}",
                flush=True,
            )

    counts = obs_count.detach().cpu()
    mean_abs = torch.zeros_like(abs_sum)
    positive = obs_count > 0
    mean_abs[positive] = abs_sum[positive] / obs_count[positive].to(torch.float64).unsqueeze(1)
    command = base_env.command_manager.get_term("motion")
    segment_index = command.segment_index
    checkpoint_sha = eval_utils.sha256_file(checkpoint)
    manifest_sha = eval_utils.sha256_file(args_cli.motion_file)
    snapshot_payload = {
        "schema_version": "wbt.joint_gap.proxy_snapshot.v1",
        "proxy_statistic": "per_segment_mean_absolute_joint_error",
        "proxy_limitation": (
            "frozen-policy deterministic diagnostic rollout; not reconstructed historical online EMA"
        ),
        "checkpoint_iteration": _checkpoint_iteration(checkpoint),
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha,
        "git_sha": eval_utils.git_commit(_project_root()),
        "train_manifest_path": str(Path(args_cli.motion_file).resolve()),
        "train_manifest_sha256": manifest_sha,
        "quality_metadata_path": metadata_identity["quality_metadata_path"],
        "quality_metadata_sha256": metadata_identity["quality_metadata_sha256"],
        "difficulty_metadata_path": metadata_identity["difficulty_metadata_path"],
        "difficulty_metadata_sha256": metadata_identity["difficulty_metadata_sha256"],
        "cluster_metadata_path": metadata_identity["cluster_metadata_path"],
        "cluster_metadata_sha256": metadata_identity["cluster_metadata_sha256"],
        "seed": int(args_cli.seed),
        "deterministic": bool(args_cli.deterministic),
        "randomization_disabled": bool(args_cli.disable_randomization),
        "num_motions": int(command.motion.num_motions),
        "num_segments": num_segments,
        "joint_names": np.asarray(list(joint_names), dtype=str),
        "joint_mapping_hash": joint_mapping_hash,
        "collector_config_json": json.dumps(collector_config, sort_keys=True, separators=(",", ":")),
        "segment_joint_error": mean_abs.detach().cpu().to(torch.float32).numpy(),
        "segment_observation_count": counts.numpy(),
        "eligible_mask": eligible_mask.detach().cpu().numpy(),
        "observed_mask": counts.numpy() > 0,
        "difficulty_bin": difficulty_bins.detach().cpu().numpy(),
        "motion_cluster_id": motion_cluster_ids.detach().cpu().numpy(),
        "motion_segment_offsets": segment_index.motion_segment_offsets.detach().cpu().numpy(),
        "motion_id": segment_index.segment_motion_ids.detach().cpu().numpy(),
        "local_segment_id": segment_index.segment_local_ids.detach().cpu().numpy(),
        "start_frame": segment_index.segment_start_frames.detach().cpu().numpy(),
        "end_frame_exclusive": segment_index.segment_end_frames.detach().cpu().numpy(),
        "motion_path": np.asarray(list(motion_paths), dtype=str),
        "category": np.asarray(list(categories), dtype=str),
        "source_group": np.asarray(list(source_groups), dtype=str),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp.npz")
    np.savez_compressed(tmp_path, **snapshot_payload)
    os.replace(tmp_path, output_path)
    elapsed = time.perf_counter() - started_at
    return {
        "checkpoint": checkpoint.name,
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_iteration": _checkpoint_iteration(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "snapshot_path": str(output_path.resolve()),
        "snapshot_sha256": eval_utils.sha256_file(output_path),
        "eligible_segments": int(torch.count_nonzero(eligible_mask).item()),
        "selected_segments": int(selected_segments.numel()),
        "observed_segments": int(torch.count_nonzero(counts > 0).item()),
        "cold_segments": int(torch.count_nonzero((eligible_mask.detach().cpu()) & (counts == 0)).item()),
        "interrupted_segments": interrupted,
        "elapsed_seconds": elapsed,
    }


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg) -> int:
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env_cfg.scene.num_envs = int(args_cli.num_envs)
    env_cfg.episode_length_s = float(args_cli.episode_length_s)
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
        agent_cfg.device = args_cli.device
    if hasattr(env_cfg, "seed"):
        env_cfg.seed = int(args_cli.seed)
    _force_deterministic_eval_research_config(env_cfg)
    randomization_warnings = _disable_randomization(env_cfg) if args_cli.disable_randomization else []

    random.seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    torch.manual_seed(args_cli.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args_cli.seed)

    project_root = _project_root()
    run_dir = _absolute(args_cli.run_dir, project_root)
    output_dir = _absolute(args_cli.output_dir, project_root)
    selected_checkpoints = eval_utils.select_checkpoints(
        eval_utils.list_checkpoints(run_dir, args_cli.checkpoint_pattern),
        args_cli.checkpoints,
    )
    if not selected_checkpoints:
        raise RuntimeError(f"No checkpoints selected from {run_dir}.")
    if not any(checkpoint.name == "model_33999.pt" for checkpoint in selected_checkpoints):
        raise RuntimeError("STEP 7 checkpoint selection must include model_33999.pt.")

    motion_cfg = getattr(getattr(env_cfg, "commands", None), "motion", None)
    if motion_cfg is None:
        raise RuntimeError("Task config has no motion command.")
    motion_cfg.motion_file = args_cli.motion_file

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env)
    base_env = env.unwrapped
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)

    command = base_env.command_manager.get_term("motion")
    if command.online_learning is not None:
        raise RuntimeError("Collector requires evaluator-safe online_learning.enabled=False.")
    if command.segment_index is None:
        raise RuntimeError("Collector requires env.commands.motion.research.segment.enabled=true.")
    quality_metadata, difficulty_metadata, cluster_metadata = _load_and_validate_offline_metadata(command, project_root)
    metadata_identity = {
        "quality_metadata_path": str(Path(quality_metadata.path).resolve()),
        "quality_metadata_sha256": quality_metadata.metadata_sha256,
        "difficulty_metadata_path": str(Path(difficulty_metadata.path).resolve()),
        "difficulty_metadata_sha256": difficulty_metadata.metadata_sha256,
        "cluster_metadata_path": str(Path(cluster_metadata.path).resolve()),
        "cluster_metadata_sha256": cluster_metadata.metadata_sha256,
    }
    difficulty_bins = torch.as_tensor(difficulty_metadata.difficulty_bin, dtype=torch.long, device=base_env.device)
    motion_cluster_ids = torch.as_tensor(cluster_metadata.cluster_id, dtype=torch.long, device=base_env.device)

    motion_paths = [eval_utils.project_relative(Path(path), project_root) for path in command.motion.motion_files]
    metadata_lookup = eval_utils.load_metadata_lookup(project_root)
    categories: list[str] = []
    source_groups: list[str] = []
    for path in command.motion.motion_files:
        category, source_group = eval_utils.motion_info(Path(path), project_root, metadata_lookup)
        categories.append(category)
        source_groups.append(source_group)
    joint_names = [str(name) for name in command.robot.joint_names]
    mapping_rows = joint_diag.joint_mapping_rows(joint_names)
    joint_mapping_hash = joint_diag.mapping_hash(mapping_rows)
    eligible_mask = _offline_eligible_mask(command, quality_metadata).detach().cpu()
    selected_segments = torch.where(eligible_mask)[0]
    if args_cli.max_segments is not None:
        selected_segments = selected_segments[: int(args_cli.max_segments)]

    collector_config = {
        "script": "collect_joint_gap_training_snapshot.py",
        "motion_file": args_cli.motion_file,
        "run_dir": str(run_dir.resolve()),
        "checkpoints": args_cli.checkpoints,
        "num_envs": env.num_envs,
        "seed": int(args_cli.seed),
        "deterministic": bool(args_cli.deterministic),
        "disable_randomization": bool(args_cli.disable_randomization),
        "include_start_frame": bool(args_cli.include_start_frame),
        "max_segments": args_cli.max_segments,
        "proxy_statistic": "per_segment_mean_absolute_joint_error",
        "quality_metadata_sha256": quality_metadata.metadata_sha256,
        "difficulty_metadata_sha256": difficulty_metadata.metadata_sha256,
        "cluster_metadata_sha256": cluster_metadata.metadata_sha256,
        "randomization_warnings": randomization_warnings,
    }
    print(f"[INFO]: selected checkpoints: {[path.name for path in selected_checkpoints]}")
    print(f"[INFO]: selected eligible segments: {int(selected_segments.numel())}/{int(eligible_mask.numel())}")

    entries: list[dict[str, object]] = []
    for checkpoint in selected_checkpoints:
        snapshot_path = _snapshot_path(output_dir, checkpoint)
        if args_cli.resume_snapshots and snapshot_path.exists():
            print(f"[INFO]: skipping existing snapshot {snapshot_path}")
            entries.append(
                {
                    "checkpoint": checkpoint.name,
                    "checkpoint_iteration": _checkpoint_iteration(checkpoint),
                    "checkpoint_path": str(checkpoint.resolve()),
                    "checkpoint_sha256": eval_utils.sha256_file(checkpoint),
                    "snapshot_path": str(snapshot_path.resolve()),
                    "snapshot_sha256": eval_utils.sha256_file(snapshot_path),
                    "skipped_existing": True,
                }
            )
            continue
        print(f"[INFO]: collecting {checkpoint.name} -> {snapshot_path}")
        entries.append(
            _checkpoint_snapshot(
                env=env,
                base_env=base_env,
                ppo_runner=ppo_runner,
                checkpoint=checkpoint,
                output_path=snapshot_path,
                selected_segments=selected_segments,
                eligible_mask=eligible_mask,
                joint_names=joint_names,
                joint_mapping_hash=joint_mapping_hash,
                motion_paths=motion_paths,
                categories=categories,
                source_groups=source_groups,
                difficulty_bins=difficulty_bins,
                motion_cluster_ids=motion_cluster_ids,
                metadata_identity=metadata_identity,
                collector_config=collector_config,
            )
        )
        _write_manifest(output_dir.parent / "snapshot_manifest.json", entries)

    _write_manifest(output_dir.parent / "snapshot_manifest.json", entries)
    env.close()
    print(f"[INFO]: wrote {output_dir.parent / 'snapshot_manifest.json'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()
