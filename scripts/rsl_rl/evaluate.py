"""Deterministic checkpoint evaluation for WBT motion tracking."""

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip
import evaluation_utils as eval_utils  # isort: skip


parser = argparse.ArgumentParser(description="Evaluate an RSL-RL WBT checkpoint on a fixed motion manifest.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of parallel evaluation environments.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--motion_file", type=str, required=True, help="Path to a WBT .npz file, directory, or manifest.")
parser.add_argument("--registry_name", type=str, default=None, help="Optional wandb motion registry name.")
parser.add_argument("--output_dir", type=str, required=True, help="Directory for per-motion CSV and summary JSON.")
parser.add_argument("--seed", type=int, default=42, help="Evaluation seed.")
parser.add_argument("--max_motions", type=int, default=None, help="Evaluate only the first N motions.")
parser.add_argument("--progress_interval", type=int, default=20, help="Print progress every N completed motions.")
parser.add_argument("--episode_length_s", type=float, default=60.0, help="Evaluation episode length cap in seconds.")
parser.add_argument("--deterministic", action="store_true", help="Use deterministic inference policy.")
parser.add_argument("--confirm_final_test", "--confirm-final-test", action="store_true", help="Allow evaluation on the frozen final test manifest.")
parser.add_argument("--dry_run", "--dry-run", action="store_true", help="Print evaluation configuration without launching Isaac Sim.")
parser.add_argument(
    "--disable_randomization",
    "--disable-randomization",
    action="store_true",
    help="Disable reset randomization, pushes, and observation corruption for evaluation.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if eval_utils.should_reject_final_test(args_cli.motion_file, args_cli.confirm_final_test):
    parser.error("Test is reserved for the frozen final checkpoint. Re-run with --confirm_final_test.")

if args_cli.dry_run:
    print("[DRY-RUN] evaluate.py configuration")
    print(f"  task: {args_cli.task}")
    print(f"  motion_file: {args_cli.motion_file}")
    print(f"  output_dir: {args_cli.output_dir}")
    print(f"  load_run: {args_cli.load_run}")
    print(f"  checkpoint: {args_cli.checkpoint}")
    print(f"  num_envs: {args_cli.num_envs}")
    print(f"  seed: {args_cli.seed}")
    print(f"  deterministic: {args_cli.deterministic}")
    print(f"  disable_randomization: {args_cli.disable_randomization}")
    print(f"  resume: {args_cli.resume}")
    sys.exit(0)

if hasattr(args_cli, "headless"):
    args_cli.headless = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
# AppLauncher can append internal Kit/livestream flags to sys.argv. Restore
# the Hydra-only argv so those flags are not parsed as task overrides.
sys.argv = [sys.argv[0]] + hydra_args

"""Rest everything follows."""

import csv
import json
import os
import pathlib
import random
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import whole_body_tracking.tasks  # noqa: F401


CSV_COLUMNS = (
    "motion_path",
    "category",
    "source_group",
    "num_frames",
    "completed_frames",
    "success",
    "completion_ratio",
    "body_pos_error_m",
    "joint_pos_error_rad",
    "termination_reason",
)

SLICE_SUFFIX_RE = re.compile(r"(?i)(?:_chunk_\d+|_chunk\d+|-chunk-\d+)$")


def _project_root() -> Path:
    return Path.cwd().resolve()


def _project_relative(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _infer_category(path: Path) -> str:
    parts = path.as_posix().split("/")
    for marker in ("g1_all", "g1_single", "g1_subset20"):
        if marker in parts:
            index = parts.index(marker)
            if index + 1 < len(parts):
                return parts[index + 1]
    return path.parent.name or "__unknown__"


def _infer_source_group(path: Path) -> str:
    stem = SLICE_SUFFIX_RE.sub("", path.stem)
    category = _infer_category(path)
    parts = path.as_posix().split("/")
    if "g1_all" in parts:
        rel_parts = parts[parts.index("g1_all") + 1 : -1]
        return "/".join(rel_parts + [stem])
    return f"{category}/{stem}"


def _load_metadata_lookup(project_root: Path) -> dict[str, dict[str, str]]:
    metadata_path = project_root / "PHUMA_wbt_motions" / "manifests" / "splits_v1" / "metadata.csv"
    lookup: dict[str, dict[str, str]] = {}
    if not metadata_path.exists():
        return lookup

    with metadata_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data = {
                "category": row.get("category", ""),
                "source_group": row.get("source_group", ""),
            }
            rel = row.get("relative_path", "")
            abs_text = row.get("path", "")
            if rel:
                lookup[rel] = data
                lookup[(project_root / rel).resolve().as_posix()] = data
            if abs_text:
                lookup[Path(abs_text).resolve().as_posix()] = data
    return lookup


def _motion_info(path: Path, project_root: Path, metadata_lookup: dict[str, dict[str, str]]) -> tuple[str, str]:
    abs_key = path.resolve().as_posix()
    rel_key = _project_relative(path, project_root)
    item = metadata_lookup.get(abs_key) or metadata_lookup.get(rel_key)
    if item:
        return item.get("category") or _infer_category(path), item.get("source_group") or _infer_source_group(path)
    return _infer_category(path), _infer_source_group(path)


def _disable_eval_randomization(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg) -> list[str]:
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


def _termination_reason(env, env_idx: int) -> str:
    manager = env.termination_manager
    reasons: list[str] = []
    for name in manager.active_terms:
        try:
            if bool(manager.get_term(name)[env_idx].item()):
                reasons.append(name)
        except Exception:
            continue
    return "+".join(reasons) if reasons else "done"


def _summary(results: list[dict[str, object]]) -> dict[str, object]:
    if not results:
        return {
            "num_motions": 0,
            "success_rate": 0.0,
            "completion_ratio": 0.0,
            "body_position_error": 0.0,
            "joint_position_error": 0.0,
            "termination_reason": {},
        }

    return {
        "num_motions": len(results),
        "success_rate": float(np.mean([bool(row["success"]) for row in results])),
        "completion_ratio": float(np.mean([float(row["completion_ratio"]) for row in results])),
        "body_position_error": float(np.mean([float(row["body_pos_error_m"]) for row in results])),
        "joint_position_error": float(np.mean([float(row["joint_pos_error_rad"]) for row in results])),
        "termination_reason": dict(Counter(str(row["termination_reason"]) for row in results)),
    }


def _write_outputs(output_dir: Path, results: list[dict[str, object]], extra_summary: dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    per_motion_path = output_dir / "per_motion.csv"
    with per_motion_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(results)

    summary = _summary(results)
    summary.update(extra_summary)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")


def _make_result_row(
    motion_path: Path,
    category: str,
    source_group: str,
    num_frames: int,
    completed_steps: int,
    body_error_sum: float,
    joint_error_sum: float,
    metric_count: int,
    success: bool,
    termination_reason: str,
    project_root: Path,
) -> dict[str, object]:
    completed_frames = min(completed_steps + 1, num_frames)
    completion_ratio = min(completed_frames / max(num_frames, 1), 1.0)
    denom = max(metric_count, 1)
    return {
        "motion_path": _project_relative(motion_path, project_root),
        "category": category,
        "source_group": source_group,
        "num_frames": num_frames,
        "completed_frames": completed_frames,
        "success": int(success),
        "completion_ratio": f"{completion_ratio:.6f}",
        "body_pos_error_m": f"{body_error_sum / denom:.6f}",
        "joint_pos_error_rad": f"{joint_error_sum / denom:.6f}",
        "termination_reason": termination_reason,
    }


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Evaluate an RSL-RL checkpoint on fixed motions."""
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.episode_length_s = args_cli.episode_length_s
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
        agent_cfg.device = args_cli.device
    if hasattr(env_cfg, "seed"):
        env_cfg.seed = args_cli.seed
    randomization_warnings: list[str] = []
    if args_cli.disable_randomization:
        randomization_warnings = _disable_eval_randomization(env_cfg)

    random.seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    torch.manual_seed(args_cli.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args_cli.seed)

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    checkpoint_name = Path(resume_path).name

    if args_cli.registry_name is not None:
        import wandb

        registry_name = args_cli.registry_name
        if ":" not in registry_name:
            registry_name += ":latest"
        artifact = wandb.Api().artifact(registry_name)
        env_cfg.commands.motion.motion_file = str(pathlib.Path(artifact.download()) / "motion.npz")
        print(f"[INFO]: Using motion file from registry: {registry_name}")
    else:
        env_cfg.commands.motion.motion_file = args_cli.motion_file
        print(f"[INFO]: Using motion file from CLI: {args_cli.motion_file}")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env)

    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    actor_critic = getattr(ppo_runner.alg, "policy", getattr(ppo_runner.alg, "actor_critic", None))
    if actor_critic is not None and hasattr(actor_critic, "eval"):
        actor_critic.eval()
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    base_env = env.unwrapped
    command = base_env.command_manager.get_term("motion")
    if not hasattr(command, "set_eval_motion_state"):
        raise RuntimeError("MotionCommand.set_eval_motion_state is required for deterministic evaluation.")

    project_root = Path.cwd().resolve()
    metadata_lookup = eval_utils.load_metadata_lookup(project_root)
    output_dir = Path(args_cli.output_dir)
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir

    motion_paths = [Path(path) for path in command.motion.motion_files]
    motion_lengths = [int(item) for item in command.motion.motion_lengths.detach().cpu().tolist()]
    total_motions = len(motion_paths)
    if args_cli.max_motions is not None:
        total_motions = min(total_motions, args_cli.max_motions)
        motion_paths = motion_paths[:total_motions]
        motion_lengths = motion_lengths[:total_motions]

    expected_motion_paths = [eval_utils.project_relative(path, project_root) for path in motion_paths]
    expected_set = set(expected_motion_paths)
    results: list[dict[str, object]] = []
    if args_cli.resume:
        results = eval_utils.load_per_motion_csv(output_dir / "per_motion.csv")
        observed = [str(row.get("motion_path", "")) for row in results]
        duplicates = sorted(path for path, count in Counter(observed).items() if count > 1)
        unexpected = sorted(set(observed).difference(expected_set))
        if duplicates:
            raise RuntimeError(f"Existing per_motion.csv contains duplicate motion rows: {duplicates[:5]}")
        if unexpected:
            raise RuntimeError(f"Existing per_motion.csv contains motions outside this manifest: {unexpected[:5]}")

    completed_paths = {str(row.get("motion_path", "")) for row in results}
    pending_indices = [index for index, path in enumerate(expected_motion_paths) if path not in completed_paths]

    print(f"[INFO]: Evaluating {total_motions} motion(s) with {env.num_envs} env(s).")
    if args_cli.resume and results:
        print(f"[INFO]: Resuming from {output_dir / 'per_motion.csv'}; skipping {len(results)} completed motion(s).")
    print("[INFO]: Each motion starts from frame 0 and is evaluated once.")
    if args_cli.disable_randomization and randomization_warnings:
        for warning in randomization_warnings:
            print(f"[WARN]: {warning}", flush=True)

    manifest_path = Path(args_cli.motion_file).resolve() if args_cli.motion_file is not None else None
    evaluation_config = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task": args_cli.task,
        "manifest": args_cli.motion_file,
        "manifest_sha256": eval_utils.sha256_file(args_cli.motion_file) if args_cli.motion_file and Path(args_cli.motion_file).exists() else "",
        "load_run": agent_cfg.load_run,
        "checkpoint": checkpoint_name,
        "checkpoint_path": str(Path(resume_path).resolve()),
        "checkpoint_sha256": eval_utils.sha256_file(resume_path),
        "git_commit": eval_utils.git_commit(project_root),
        "seed": args_cli.seed,
        "num_envs": env.num_envs,
        "episode_length_s": args_cli.episode_length_s,
        "deterministic": bool(args_cli.deterministic),
        "disable_randomization": bool(args_cli.disable_randomization),
        "randomization_warnings": randomization_warnings,
        "confirm_final_test": bool(args_cli.confirm_final_test),
        "max_motions": args_cli.max_motions,
        "joint_count": int(command.joint_pos.shape[1]),
        "metric_definitions": {
            "success": "1 only when the evaluator reaches the final motion frame before early termination.",
            "completion_ratio": "completed_frames / num_frames, clamped to [0, 1].",
            "body_position_error_m": (
                "Mean over evaluation steps of MotionCommand.metrics['error_body_pos']; "
                "that metric is the mean Euclidean distance over configured tracked bodies after yaw/root alignment "
                "in MotionCommand._update_relative_body_targets."
            ),
            "joint_position_error_l2_rad": "Mean over evaluation steps of the L2 norm over all robot joints.",
            "joint_position_error_rms_rad": "joint_position_error_l2_rad / sqrt(joint_count).",
        },
    }
    if manifest_path is not None:
        evaluation_config["manifest_path"] = str(manifest_path)

    completed_since_print = 0
    for batch_start in range(0, len(pending_indices), env.num_envs):
        if not simulation_app.is_running():
            break

        batch_motion_ids = pending_indices[batch_start : batch_start + env.num_envs]
        batch_size = len(batch_motion_ids)
        env_ids = torch.arange(batch_size, dtype=torch.long, device=base_env.device)
        motion_ids = torch.tensor(batch_motion_ids, dtype=torch.long, device=base_env.device)

        env.reset()
        command = base_env.command_manager.get_term("motion")
        command.set_eval_motion_state(env_ids, motion_ids, torch.zeros_like(motion_ids))
        base_env.episode_length_buf[env_ids] = 0
        obs, _ = env.get_observations()

        active = torch.zeros(env.num_envs, dtype=torch.bool, device=base_env.device)
        active[:batch_size] = True
        completed_steps = torch.zeros(env.num_envs, dtype=torch.long, device=base_env.device)
        metric_counts = torch.zeros(env.num_envs, dtype=torch.long, device=base_env.device)
        body_error_sums = torch.zeros(env.num_envs, dtype=torch.float32, device=base_env.device)
        joint_error_sums = torch.zeros(env.num_envs, dtype=torch.float32, device=base_env.device)

        while bool(torch.any(active).item()) and simulation_app.is_running():
            active_ids = torch.nonzero(active, as_tuple=False).flatten()
            command._update_metrics()
            body_error_sums[active_ids] += command.metrics["error_body_pos"][active_ids]
            joint_error_sums[active_ids] += command.metrics["error_joint_pos"][active_ids]
            metric_counts[active_ids] += 1

            with torch.no_grad():
                actions = policy(obs)
                obs, _, dones, _ = env.step(actions)

            completed_steps[active_ids] += 1

            for env_idx in active_ids.detach().cpu().tolist():
                motion_index = batch_motion_ids[env_idx]
                target_steps = max(motion_lengths[motion_index] - 1, 1)
                reached_motion_end = int(completed_steps[env_idx].item()) >= target_steps
                done = bool(dones[env_idx].item())
                if not reached_motion_end and not done:
                    continue

                success = reached_motion_end
                reason = "completed" if success else _termination_reason(base_env, env_idx)
                category, source_group = eval_utils.motion_info(motion_paths[motion_index], project_root, metadata_lookup)
                results.append(
                    eval_utils.make_result_row(
                        motion_path=motion_paths[motion_index],
                        category=category,
                        source_group=source_group,
                        num_frames=motion_lengths[motion_index],
                        completed_steps=int(completed_steps[env_idx].item()),
                        body_error_sum=float(body_error_sums[env_idx].item()),
                        joint_error_sum=float(joint_error_sums[env_idx].item()),
                        metric_count=int(metric_counts[env_idx].item()),
                        success=success,
                        termination_reason=reason,
                        checkpoint=checkpoint_name,
                        project_root=project_root,
                        joint_count=int(command.joint_pos.shape[1]),
                    )
                )
                active[env_idx] = False
                completed_since_print += 1
                eval_utils.write_evaluation_outputs(
                    output_dir,
                    results,
                    checkpoint=checkpoint_name,
                    evaluation_config=evaluation_config,
                    expected_motion_paths=expected_motion_paths,
                )

        if bool(torch.any(active).item()):
            for env_idx in torch.nonzero(active, as_tuple=False).flatten().detach().cpu().tolist():
                motion_index = batch_motion_ids[env_idx]
                category, source_group = eval_utils.motion_info(motion_paths[motion_index], project_root, metadata_lookup)
                results.append(
                    eval_utils.make_result_row(
                        motion_path=motion_paths[motion_index],
                        category=category,
                        source_group=source_group,
                        num_frames=motion_lengths[motion_index],
                        completed_steps=int(completed_steps[env_idx].item()),
                        body_error_sum=float(body_error_sums[env_idx].item()),
                        joint_error_sum=float(joint_error_sums[env_idx].item()),
                        metric_count=int(metric_counts[env_idx].item()),
                        success=False,
                        termination_reason="interrupted",
                        checkpoint=checkpoint_name,
                        project_root=project_root,
                        joint_count=int(command.joint_pos.shape[1]),
                    )
                )

        eval_utils.write_evaluation_outputs(
            output_dir,
            results,
            checkpoint=checkpoint_name,
            evaluation_config=evaluation_config,
            expected_motion_paths=expected_motion_paths,
        )
        if args_cli.progress_interval > 0 and completed_since_print >= args_cli.progress_interval:
            summary = eval_utils.summarize_rows(results, checkpoint=checkpoint_name)
            print(
                f"[INFO]: Evaluated {len(results)}/{total_motions} motions, "
                f"success_rate={summary['micro_success_rate']:.3f}, "
                f"completion={summary['mean_completion_ratio']:.3f}",
                flush=True,
            )
            completed_since_print = 0

    summary = eval_utils.write_evaluation_outputs(
        output_dir,
        results,
        checkpoint=checkpoint_name,
        evaluation_config=evaluation_config,
        expected_motion_paths=expected_motion_paths,
    )
    print("[INFO]: Evaluation complete.")
    print(f"[INFO]: per_motion.csv: {output_dir / 'per_motion.csv'}")
    print(f"[INFO]: summary.json: {output_dir / 'summary.json'}")
    print(
        f"[INFO]: success_rate={summary['micro_success_rate']:.4f}, "
        f"macro_success_rate={summary['macro_success_rate']:.4f}, "
        f"completion_ratio={summary['mean_completion_ratio']:.4f}, "
        f"body_position_error_m={summary['mean_body_position_error_m']:.4f}, "
        f"joint_position_error_l2_rad={summary['mean_joint_position_error_l2_rad']:.4f}, "
        f"joint_position_error_rms_rad={summary['mean_joint_position_error_rms_rad']:.4f}"
    )

    env.close()
    integrity = summary.get("manifest_integrity", {})
    if integrity and not integrity.get("ok", False):
        print(f"[ERROR]: Manifest integrity check failed: {json.dumps(integrity, indent=2)}", flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
    simulation_app.close()
