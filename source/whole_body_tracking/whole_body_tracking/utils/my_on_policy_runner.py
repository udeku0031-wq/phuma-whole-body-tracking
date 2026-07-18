import os
from copy import deepcopy
from importlib.util import find_spec

from rsl_rl.env import VecEnv
from rsl_rl.runners.on_policy_runner import OnPolicyRunner

from isaaclab_rl.rsl_rl import export_policy_as_onnx

import wandb
from whole_body_tracking.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx


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

    def save(self, path: str, infos=None):
        """Save the model and training information."""
        _save_checkpoint_without_wandb_upload(self, path, infos)
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
