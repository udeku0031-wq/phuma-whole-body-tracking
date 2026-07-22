# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import json
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
motion_source = parser.add_mutually_exclusive_group(required=True)
motion_source.add_argument("--registry_name", type=str, help="The name of the wandb motion registry.")
motion_source.add_argument(
    "--motion_file", type=str, help="Path to a local WBT motion .npz file, directory, or .txt manifest."
)

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import gymnasium as gym
from datetime import datetime

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import whole_body_tracking.tasks  # noqa: F401
from whole_body_tracking.utils.my_on_policy_runner import MotionOnPolicyRunner as OnPolicyRunner

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def _count_manifest_entries(path: str) -> int | None:
    try:
        with open(path) as f:
            return sum(1 for line in f if line.strip() and not line.lstrip().startswith("#"))
    except OSError:
        return None


def _read_sampling_config(motion_file: str) -> dict:
    config_path = Path(motion_file).resolve().parent / "sampling_config.json"
    if not config_path.exists():
        return {}
    try:
        return json.loads(config_path.read_text())
    except json.JSONDecodeError:
        return {}


def _build_research_metadata(env_cfg) -> dict:
    """Flatten the low-cardinality research config for runner and W&B metadata."""

    research = env_cfg.commands.motion.research
    return {
        "method_name": research.method_name,
        "segment_enabled": research.segment.enabled,
        "segment_length_seconds": research.segment.length_seconds,
        "quality_gate_enabled": research.quality_gate.enabled,
        "quality_gate_metadata_path": research.quality_gate.metadata_path or None,
        "quality_gate_reject_statuses": list(research.quality_gate.reject_statuses),
        "quality_gate_include_borderline": research.quality_gate.include_borderline,
        "quality_gate_strict_metadata_match": research.quality_gate.strict_metadata_match,
        "quality_gate_empty_motion_policy": research.quality_gate.empty_motion_policy,
        "quality_gate_scope": research.quality_gate.gate_scope,
        "difficulty_calibration_enabled": research.difficulty_calibration.enabled,
        "difficulty_calibration_metadata_path": research.difficulty_calibration.metadata_path or None,
        "difficulty_calibration_strict_metadata_match": research.difficulty_calibration.strict_metadata_match,
        "difficulty_calibration_expected_num_bins": research.difficulty_calibration.expected_num_bins,
        "online_learning_enabled": research.online_learning.enabled,
        "online_statistics_enabled": research.online_learning.statistics_enabled,
        "online_warmup_iterations": research.online_learning.warmup_iterations,
        "online_probability_update_interval": research.online_learning.probability_update_interval,
        "online_ema_decay": research.online_learning.ema_decay,
        "online_min_segment_observations": research.online_learning.min_segment_observations,
        "online_min_motion_episodes": research.online_learning.min_motion_episodes,
        "online_min_bin_valid_segments": research.online_learning.min_bin_valid_segments,
        "online_minimum_segment_observed_fraction": (
            research.online_learning.minimum_segment_observed_fraction
        ),
        "online_sigma_floor": research.online_learning.sigma_floor,
        "online_gap_clip": research.online_learning.gap_clip,
        "online_score_clip": research.online_learning.score_clip,
        "online_bin_observation_weighted": research.online_learning.bin_observation_weighted,
        "adaptive_sampler_seed": research.online_learning.sampler_seed,
        "online_provisional": research.online_learning.provisional,
        "error_body_position_scale_m": research.error_definition.body_position_scale_m,
        "error_joint_position_scale_rad": research.error_definition.joint_position_scale_rad,
        "error_orientation_scale_rad": research.error_definition.orientation_scale_rad,
        "error_component_clip": research.error_definition.component_clip,
        "error_body_weight": research.error_definition.body_weight,
        "error_joint_weight": research.error_definition.joint_weight,
        "error_orientation_weight": research.error_definition.orientation_weight,
        "error_termination_weight": research.error_definition.termination_weight,
        "error_completion_weight": research.error_definition.completion_weight,
        "error_success_weight": research.error_definition.success_weight,
        "motion_error_segment_mean_weight": research.motion_error.segment_mean_weight,
        "motion_error_segment_p90_weight": research.motion_error.segment_p90_weight,
        "motion_error_termination_weight": research.motion_error.termination_weight,
        "motion_error_completion_weight": research.motion_error.completion_weight,
        "motion_error_success_weight": research.motion_error.success_weight,
        "motion_gap_positive_mean_weight": research.motion_gap.positive_mean_weight,
        "motion_gap_positive_p90_weight": research.motion_gap.positive_p90_weight,
        "motion_gap_termination_weight": research.motion_gap.termination_weight,
        "motion_gap_completion_weight": research.motion_gap.completion_weight,
        "motion_gap_success_weight": research.motion_gap.success_weight,
        "adaptive_uniform_mix": research.adaptive_sampling.uniform_mix,
        "adaptive_temperature": research.adaptive_sampling.temperature,
        "adaptive_under_sampling_weight": research.adaptive_sampling.under_sampling_weight,
        "adaptive_motion_probability_cap": research.adaptive_sampling.motion_probability_cap,
        "adaptive_segment_probability_cap": research.adaptive_sampling.segment_probability_cap,
        "adaptive_fallback": research.adaptive_sampling.fallback,
        "motion_sampling_mode": research.motion_sampling.mode,
        "segment_sampling_mode": research.segment_sampling.mode,
        "diversity_constraint_enabled": research.diversity_constraint.enabled,
        "sampling_statistics_enabled": research.sampling_statistics.enabled,
        "sampling_statistics_log_interval": research.sampling_statistics.log_interval,
        "assignment_trace_enabled": research.assignment_trace.enabled,
        "assignment_trace_output_path": research.assignment_trace.output_path or None,
        "assignment_trace_max_entries": research.assignment_trace.max_entries,
        "probability_validation_epsilon": research.probability_validation.epsilon,
    }


def _build_training_metadata(args_cli: argparse.Namespace, agent_cfg, env_cfg) -> dict:
    motion_file = os.path.abspath(args_cli.motion_file) if args_cli.motion_file is not None else None
    sampling_config = _read_sampling_config(motion_file) if motion_file is not None else {}
    manifest_name = Path(motion_file).name if motion_file is not None else None
    train_size = _count_manifest_entries(motion_file) if motion_file is not None and motion_file.endswith(".txt") else None
    experiment_label = agent_cfg.run_name or args_cli.run_name or agent_cfg.experiment_name
    metadata = {
        "experiment_name": experiment_label,
        "train_manifest": manifest_name,
        "train_size": train_size,
        "training_seed": agent_cfg.seed,
        "num_envs": env_cfg.scene.num_envs,
        "max_iterations": agent_cfg.max_iterations,
        "resume": bool(agent_cfg.resume),
        "curriculum": False,
    }
    metadata.update(_build_research_metadata(env_cfg))
    if sampling_config.get("sampling_method") == "uniform_random_file_sampling":
        metadata.update(
            {
                "training_strategy": "direct_mixed",
                "sampling_method": sampling_config.get("sampling_method"),
                "split_version": sampling_config.get("split_version"),
                "train_pool_size": sampling_config.get("train_pool_size"),
                "sampling_seed": sampling_config.get("seed"),
            }
        )
    return {key: value for key, value in metadata.items() if value is not None}


def _prepare_wandb_resume_env(args_cli: argparse.Namespace, agent_cfg) -> None:
    """Expose stable W&B run identity to rsl_rl's WandbSummaryWriter."""
    if agent_cfg.logger != "wandb":
        return
    if args_cli.wandb_run_id:
        os.environ["WANDB_RUN_ID"] = args_cli.wandb_run_id
    if args_cli.wandb_resume:
        os.environ["WANDB_RESUME"] = args_cli.wandb_resume
    run_name = args_cli.wandb_run_name or args_cli.run_name or agent_cfg.run_name
    if run_name:
        os.environ["WANDB_NAME"] = run_name


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.disable_fabric:
        env_cfg.sim.use_fabric = False

    registry_name = None
    if args_cli.registry_name is not None:
        registry_name = args_cli.registry_name
        if ":" not in registry_name:  # Check if the registry name includes alias, if not, append ":latest"
            registry_name += ":latest"
        import pathlib

        import wandb

        api = wandb.Api()
        artifact = api.artifact(registry_name)
        env_cfg.commands.motion.motion_file = str(pathlib.Path(artifact.download()) / "motion.npz")
        print(f"[INFO]: Using motion file from registry: {registry_name}")
    else:
        env_cfg.commands.motion.motion_file = os.path.abspath(args_cli.motion_file)
        print(f"[INFO]: Using local motion file: {env_cfg.commands.motion.motion_file}")

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    training_metadata = _build_training_metadata(args_cli, agent_cfg, env_cfg)
    train_cfg = agent_cfg.to_dict()
    train_cfg.update(training_metadata)
    train_cfg["wandb_metadata"] = training_metadata
    if args_cli.wandb_run_name is not None:
        train_cfg["wandb_run_name"] = args_cli.wandb_run_name
    if args_cli.wandb_run_id is not None:
        train_cfg["wandb_run_id"] = args_cli.wandb_run_id
    if args_cli.wandb_resume is not None:
        train_cfg["wandb_resume"] = args_cli.wandb_resume

    # create runner from rsl-rl
    _prepare_wandb_resume_env(args_cli, agent_cfg)
    runner = OnPolicyRunner(
        env, train_cfg, log_dir=log_dir, device=agent_cfg.device, registry_name=registry_name
    )
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # save resume path before creating a new log_dir
    if agent_cfg.resume:
        # get path to previous checkpoint
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # run training
    try:
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    finally:
        writer = getattr(runner, "writer", None)
        try:
            if writer is not None:
                if hasattr(writer, "flush"):
                    writer.flush()
                if hasattr(writer, "stop"):
                    writer.stop()
                if hasattr(writer, "close"):
                    writer.close()
        finally:
            # close the simulator
            env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
