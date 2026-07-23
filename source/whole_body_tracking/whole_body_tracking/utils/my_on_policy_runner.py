import os
import warnings
from collections.abc import Mapping
from copy import deepcopy
from importlib.util import find_spec

from rsl_rl.env import VecEnv
from rsl_rl.runners.on_policy_runner import OnPolicyRunner

from isaaclab_rl.rsl_rl import export_policy_as_onnx

import wandb
from whole_body_tracking.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx


SAMPLING_STATE_INFO_KEY = "sampling_state"


def _adapt_rsl_rl_cfg(train_cfg: dict) -> dict:
    """Adapt Isaac Lab 2.1 RSL-RL configs to newer rsl_rl model/algorithm configs."""
    if find_spec("rsl_rl.models") is None:
        return train_cfg

    cfg = deepcopy(train_cfg)
    if "actor" in cfg and "critic" in cfg:
        return cfg

    policy_cfg = cfg.pop("policy")
    algorithm_cfg = cfg.setdefault("algorithm", {})
    algorithm_cfg["class_name"] = "rsl_rl.algorithms:PPO"
    algorithm_cfg.setdefault("rnd_cfg", None)
    algorithm_cfg.setdefault("symmetry_cfg", None)

    obs_normalization = cfg.get("empirical_normalization", False)
    activation = policy_cfg.get("activation", "elu")
    noise_std_type = policy_cfg.get("noise_std_type", "scalar")

    cfg["actor"] = {
        "class_name": "rsl_rl.models:MLPModel",
        "hidden_dims": policy_cfg["actor_hidden_dims"],
        "activation": activation,
        "obs_normalization": obs_normalization,
        "distribution_cfg": {
            "class_name": "rsl_rl.modules:GaussianDistribution",
            "init_std": policy_cfg["init_noise_std"],
            "std_type": noise_std_type,
        },
    }
    cfg["critic"] = {
        "class_name": "rsl_rl.models:MLPModel",
        "hidden_dims": policy_cfg["critic_hidden_dims"],
        "activation": activation,
        "obs_normalization": obs_normalization,
    }
    cfg.setdefault("obs_groups", {"actor": ["policy"], "critic": ["critic"]})
    cfg.setdefault("torch_compile_mode", None)
    cfg.setdefault("check_for_nan", True)
    return cfg


def _is_wandb_logger(runner: OnPolicyRunner) -> bool:
    return (
        getattr(runner, "logger_type", "").lower() == "wandb"
        or getattr(getattr(runner, "logger", None), "logger_type", "").lower() == "wandb"
    )


def _disable_onnx_on_save() -> bool:
    return os.environ.get("WBT_DISABLE_ONNX_ON_SAVE", "").strip().lower() in {"1", "true", "yes", "on"}


def _enable_wandb_model_upload() -> bool:
    return os.environ.get("WBT_ENABLE_WANDB_MODEL_SAVE", "").strip().lower() in {"1", "true", "yes", "on"}


def _save_checkpoint_without_wandb_upload(runner: OnPolicyRunner, path: str, infos=None) -> None:
    """Save locally while keeping W&B scalar logging, but skip cloud checkpoint files by default."""
    if getattr(runner, "logger_type", "").lower() != "wandb" or _enable_wandb_model_upload():
        OnPolicyRunner.save(runner, path, infos)
        return

    logger_type = runner.logger_type
    runner.logger_type = "_wandb_no_model_upload"
    try:
        OnPolicyRunner.save(runner, path, infos)
    finally:
        runner.logger_type = logger_type


def _actor_critic(runner: OnPolicyRunner):
    if hasattr(runner.alg, "policy"):
        return runner.alg.policy
    return getattr(runner.alg, "actor_critic", None)


def _motion_command(runner: OnPolicyRunner):
    """Return the motion command without coupling the runner to its concrete class."""

    base_env = getattr(runner.env, "unwrapped", None)
    command_manager = getattr(base_env, "command_manager", None)
    if command_manager is None or "motion" not in getattr(command_manager, "active_terms", ()):
        return None
    command = command_manager.get_term("motion")
    if not hasattr(command, "sampling_state_dict"):
        return None
    return command


def _checkpoint_infos_with_sampling_state(runner: OnPolicyRunner, infos):
    command = _motion_command(runner)
    if command is None:
        return infos
    if infos is None:
        checkpoint_infos = {}
    elif isinstance(infos, Mapping):
        checkpoint_infos = dict(infos)
    else:
        raise TypeError("Checkpoint infos must be a mapping when sampling state persistence is enabled.")
    checkpoint_infos[SAMPLING_STATE_INFO_KEY] = command.sampling_state_dict()
    return checkpoint_infos


def _wandb_step_offset(current_wandb_step: int, current_training_iteration: int, resume: bool = False) -> int:
    """Return the smallest fixed offset that prevents W&B step regression."""

    min_offset = 2 if resume else 1
    return max(min_offset, int(current_wandb_step) - int(current_training_iteration))


def _install_wandb_step_guard(runner: OnPolicyRunner) -> None:
    """Translate writer steps when W&B initialization or resume is already ahead."""

    if (
        getattr(runner, "_wandb_step_guard_installed", False)
        or not _is_wandb_logger(runner)
        or wandb.run is None
        or getattr(runner, "writer", None) is None
    ):
        return

    offset = _wandb_step_offset(
        wandb.run.step,
        runner.current_learning_iteration,
        resume=bool(runner.cfg.get("resume", False)),
    )
    original_add_scalar = runner.writer.add_scalar

    def add_scalar_with_monotonic_step(tag, scalar_value, global_step=None, walltime=None, new_style=False):
        mapped_step = None if global_step is None else int(global_step) + offset
        return original_add_scalar(
            tag,
            scalar_value,
            global_step=mapped_step,
            walltime=walltime,
            new_style=new_style,
        )

    runner.writer.add_scalar = add_scalar_with_monotonic_step
    runner._wandb_step_offset = offset
    runner._wandb_step_guard_installed = True


def _sync_wandb_metadata(runner: OnPolicyRunner) -> None:
    """Apply the requested name and flat metadata after RSL-RL initializes W&B."""

    if not _is_wandb_logger(runner) or wandb.run is None:
        return
    _install_wandb_step_guard(runner)
    if getattr(runner, "_wandb_metadata_synced", False):
        return
    run_name = runner.cfg.get("wandb_run_name") or runner.cfg.get("run_name")
    if run_name:
        wandb.run.name = run_name
    metadata = runner.cfg.get("wandb_metadata", {})
    if not isinstance(metadata, Mapping):
        raise TypeError("wandb_metadata must be a mapping.")
    resolved_metadata = dict(metadata)
    command = _motion_command(runner)
    if command is not None and hasattr(command, "wandb_research_metadata"):
        command_metadata = command.wandb_research_metadata()
        if not isinstance(command_metadata, Mapping):
            raise TypeError("Motion command W&B metadata must be a mapping.")
        resolved_metadata.update(command_metadata)
    wandb.config.update(resolved_metadata, allow_val_change=True)
    runner._wandb_metadata_synced = True


def _restore_sampling_state(runner: OnPolicyRunner, infos, checkpoint_path: str) -> None:
    command = _motion_command(runner)
    if command is None:
        return
    if not isinstance(infos, Mapping) or SAMPLING_STATE_INFO_KEY not in infos:
        if (
            command.cfg.research.difficulty_calibration.enabled
            or command.cfg.research.online_learning.enabled
        ):
            raise ValueError(
                "Difficulty-enabled resume requires checkpoint sampling state; online-learning resume has the "
                "same requirement for metadata identity, shared statistics and adaptive RNG state."
            )
        warnings.warn(
            f"Checkpoint '{checkpoint_path}' has no sampling state; initializing statistics from the active "
            "environment assignments.",
            RuntimeWarning,
            stacklevel=2,
        )
        command.reset_sampling_statistics()
        command.record_current_sampling_assignments()
        return
    sampling_state = infos[SAMPLING_STATE_INFO_KEY]
    if not isinstance(sampling_state, Mapping):
        raise ValueError(f"Checkpoint field '{SAMPLING_STATE_INFO_KEY}' must be a mapping.")
    command.load_sampling_state_dict(sampling_state)
    # RslRlVecEnvWrapper resets the environment before runner.load().  Those
    # active assignments are real work for the resumed process, so add them on
    # top of the restored cumulative counters after the overwrite.
    online = getattr(command, "online_learning", None)
    if online is not None and getattr(online, "sampler", None) is not None:
        command.reassign_after_online_resume()
    else:
        command.record_current_sampling_assignments()


class MyOnPolicyRunner(OnPolicyRunner):
    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu"):
        super().__init__(env, _adapt_rsl_rl_cfg(train_cfg), log_dir, device)

    def save(self, path: str, infos=None):
        """Save the model and training information."""
        _save_checkpoint_without_wandb_upload(self, path, infos)
        actor_critic = _actor_critic(self)
        if (
            _is_wandb_logger(self)
            and not _disable_onnx_on_save()
            and actor_critic is not None
            and hasattr(self, "obs_normalizer")
        ):
            policy_path = path.split("model")[0]
            filename = policy_path.split("/")[-2] + ".onnx"
            export_policy_as_onnx(actor_critic, normalizer=self.obs_normalizer, path=policy_path, filename=filename)
            attach_onnx_metadata(self.env.unwrapped, wandb.run.name, path=policy_path, filename=filename)
            wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))


class MotionOnPolicyRunner(OnPolicyRunner):
    def __init__(
        self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu", registry_name: str = None
    ):
        super().__init__(env, _adapt_rsl_rl_cfg(train_cfg), log_dir, device)
        self.registry_name = registry_name
        self._wandb_metadata_synced = False
        self._wandb_step_guard_installed = False
        self._wandb_step_offset = 0
        command = _motion_command(self)
        if (
            command is not None
            and command.cfg.research.online_learning.enabled
            and log_dir is None
        ):
            raise ValueError(
                "Online learning requires a runner log_dir because the PPO-iteration boundary hook "
                "is dispatched from MotionOnPolicyRunner.log()."
            )

    def save(self, path: str, infos=None):
        """Save the model and training information."""
        _sync_wandb_metadata(self)
        checkpoint_infos = _checkpoint_infos_with_sampling_state(self, infos)
        _save_checkpoint_without_wandb_upload(self, path, checkpoint_infos)
        if _is_wandb_logger(self):
            policy_path = path.split("model")[0]
            filename = policy_path.split("/")[-2] + ".onnx"
            actor_critic = _actor_critic(self)
            if actor_critic is not None and hasattr(self, "obs_normalizer") and not _disable_onnx_on_save():
                export_motion_policy_as_onnx(
                    self.env.unwrapped,
                    actor_critic,
                    normalizer=self.obs_normalizer,
                    path=policy_path,
                    filename=filename,
                )
                attach_onnx_metadata(self.env.unwrapped, wandb.run.name, path=policy_path, filename=filename)
                wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))

            # link the artifact registry to this run
            if self.registry_name is not None:
                wandb.run.use_artifact(self.registry_name)
                self.registry_name = None

    def load(self, path: str, load_optimizer: bool = True):
        """Load the model, then restore compatible shared sampling statistics."""

        infos = super().load(path, load_optimizer=load_optimizer)
        _restore_sampling_state(self, infos, path)
        return infos

    def log(self, locs: dict, width: int = 80, pad: int = 35):
        """Log normal RSL-RL values plus rate-limited research summaries."""

        _sync_wandb_metadata(self)
        command = _motion_command(self)
        iteration = int(locs["it"])
        if command is not None and hasattr(command, "on_learning_iteration_end"):
            command.on_learning_iteration_end(iteration)
        super().log(locs, width=width, pad=pad)
        if command is None:
            return
        statistics_enabled = bool(command.cfg.research.sampling_statistics.enabled)
        difficulty_enabled = bool(command.cfg.research.difficulty_calibration.enabled)
        online_enabled = bool(command.cfg.research.online_learning.enabled)
        if not statistics_enabled and not difficulty_enabled and not online_enabled:
            return
        interval = int(command.cfg.research.sampling_statistics.log_interval)
        if iteration % interval != 0:
            return
        if statistics_enabled:
            metrics = command.sampling_metrics()
            if metrics is not None:
                for name, value in metrics.items():
                    self.writer.add_scalar(f"sampling/{name}", value, iteration)
            if hasattr(command, "quality_metrics"):
                quality_metrics = command.quality_metrics()
                if quality_metrics is not None:
                    for name, value in quality_metrics.items():
                        self.writer.add_scalar(f"quality/{name}", value, iteration)
            if hasattr(command, "dataset_metrics"):
                dataset_metrics = command.dataset_metrics()
                if dataset_metrics is not None:
                    for name, value in dataset_metrics.items():
                        self.writer.add_scalar(f"dataset/{name}", value, iteration)
        if difficulty_enabled and hasattr(command, "difficulty_metrics"):
            difficulty_metrics = command.difficulty_metrics()
            if difficulty_metrics is not None:
                for name, value in difficulty_metrics.items():
                    self.writer.add_scalar(f"difficulty/{name}", value, iteration)
        if online_enabled and hasattr(command, "online_learning_metrics"):
            online_metrics = command.online_learning_metrics()
            if online_metrics is not None:
                for name, value in online_metrics.items():
                    self.writer.add_scalar(name, value, iteration)
