from __future__ import annotations

import ast
import math
import unittest
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMMANDS_PATH = (
    PROJECT_ROOT
    / "source"
    / "whole_body_tracking"
    / "whole_body_tracking"
    / "tasks"
    / "tracking"
    / "mdp"
    / "commands.py"
)
RUNNER_PATH = (
    PROJECT_ROOT / "source" / "whole_body_tracking" / "whole_body_tracking" / "utils" / "my_on_policy_runner.py"
)
ONLINE_ENV_PATH = (
    PROJECT_ROOT
    / "source"
    / "whole_body_tracking"
    / "whole_body_tracking"
    / "tasks"
    / "tracking"
    / "online_learning_env.py"
)
G1_REGISTRY_PATH = (
    PROJECT_ROOT
    / "source"
    / "whole_body_tracking"
    / "whole_body_tracking"
    / "tasks"
    / "tracking"
    / "config"
    / "g1"
    / "__init__.py"
)


def _parsed(path: Path = COMMANDS_PATH) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _command_method(name: str) -> ast.FunctionDef:
    command = next(node for node in _parsed().body if isinstance(node, ast.ClassDef) and node.name == "MotionCommand")
    return next(node for node in command.body if isinstance(node, ast.FunctionDef) and node.name == name)


def _validator():
    function = next(
        node for node in _parsed().body if isinstance(node, ast.FunctionDef) and node.name == "_validate_research_config"
    )
    namespace = {
        "math": math,
        "QUALITY_STATUS_TO_CODE": {"pass": 0, "borderline": 1, "reject": 2},
        "QualityGatedStartIndex": SimpleNamespace(_EMPTY_MOTION_POLICIES=frozenset({"error", "exclude"})),
    }
    exec(
        compile("from __future__ import annotations\n" + ast.unparse(function), str(COMMANDS_PATH), "exec"),
        namespace,
    )
    return namespace["_validate_research_config"]


def _config(method: str) -> SimpleNamespace:
    modes = {
        "M0": ("uniform", "uniform"),
        "M1": ("uniform", "uniform"),
        "M2": ("raw_error", "uniform"),
        "M3": ("uniform", "raw_error"),
        "M4": ("raw_error", "raw_error"),
        "M5": ("learning_gap", "relative_learning_gap"),
        "M6": ("learning_gap", "relative_learning_gap"),
        "GLOBAL_BIN_RAW_ERROR": ("uniform", "global_bin_raw_error"),
    }
    motion_mode, segment_mode = modes[method]
    adaptive = method not in {"M0", "M1"}
    quality = method in {"M1", "M6"}
    difficulty = method in {"M5", "M6"}
    return SimpleNamespace(
        method_name=method,
        segment=SimpleNamespace(enabled=True, length_seconds=1.0),
        quality_gate=SimpleNamespace(
            enabled=quality,
            metadata_path="quality.npz" if quality else "",
            reject_statuses=("reject",),
            include_borderline=True,
            strict_metadata_match=True,
            empty_motion_policy="exclude" if method == "M6" else "error",
            gate_scope="assignment_start",
        ),
        difficulty_calibration=SimpleNamespace(
            enabled=difficulty,
            metadata_path="difficulty.npz" if difficulty else "",
            strict_metadata_match=True,
            expected_num_bins=10,
        ),
        online_learning=SimpleNamespace(
            enabled=adaptive,
            statistics_enabled=adaptive,
            warmup_iterations=1000,
            probability_update_interval=50,
            ema_decay=0.95,
            min_segment_observations=32,
            min_motion_episodes=8,
            min_bin_valid_segments=32,
            minimum_segment_observed_fraction=0.2,
            sigma_floor=0.1,
            gap_clip=5,
            score_clip=10,
        ),
        adaptive_sampling=SimpleNamespace(
            uniform_mix=0.15,
            temperature=1,
            under_sampling_weight=0.25,
            motion_probability_cap=0.02,
            segment_probability_cap=1.0,
            fallback="uniform",
        ),
        error_definition=SimpleNamespace(
            body_position_scale_m=0.3,
            joint_position_scale_rad=0.5,
            orientation_scale_rad=0.4,
            body_weight=1.0,
            joint_weight=1.0,
            orientation_weight=1.0,
            termination_weight=1.0,
            completion_weight=0.5,
            success_weight=0.5,
            component_clip=5.0,
        ),
        motion_error=SimpleNamespace(
            segment_mean_weight=1.0,
            segment_p90_weight=0.25,
            termination_weight=0.5,
            completion_weight=0.5,
            success_weight=0.5,
        ),
        motion_gap=SimpleNamespace(
            positive_mean_weight=1.0,
            positive_p90_weight=0.5,
            termination_weight=0.5,
            completion_weight=0.5,
            success_weight=0.5,
        ),
        online_snapshot=SimpleNamespace(enabled=False),
        diversity_constraint=SimpleNamespace(enabled=False),
        motion_sampling=SimpleNamespace(mode=motion_mode),
        segment_sampling=SimpleNamespace(mode=segment_mode),
        sampling_statistics=SimpleNamespace(enabled=True, log_interval=100),
        assignment_trace=SimpleNamespace(enabled=False, output_path="", max_entries=2048),
        probability_validation=SimpleNamespace(epsilon=1.0e-8),
    )


class ModeValidationTest(unittest.TestCase):
    def test_m0_through_m6_and_global_bin_contracts(self) -> None:
        validate = _validator()
        for method in ("M0", "M1", "M2", "M3", "M4", "M5", "M6", "GLOBAL_BIN_RAW_ERROR"):
            with self.subTest(method=method):
                validate(_config(method))

    def test_learning_gap_requires_difficulty_and_m6_requires_quality_exclude(self) -> None:
        validate = _validator()
        cfg = _config("M5")
        cfg.difficulty_calibration.enabled = False
        with self.assertRaisesRegex(ValueError, "requires difficulty"):
            validate(cfg)
        cfg = _config("M6")
        cfg.quality_gate.empty_motion_policy = "error"
        with self.assertRaisesRegex(ValueError, "empty_motion_policy='exclude'"):
            validate(cfg)

    def test_unknown_mode_fails_loudly(self) -> None:
        cfg = _config("M2")
        cfg.motion_sampling.mode = "cluster"
        with self.assertRaisesRegex(NotImplementedError, "not implemented"):
            _validator()(cfg)

    def test_non_finite_exploration_and_zero_active_weights_fail_fast(self) -> None:
        validate = _validator()
        cfg = _config("M3")
        cfg.adaptive_sampling.under_sampling_weight = float("nan")
        with self.assertRaisesRegex(ValueError, "finite and non-negative"):
            validate(cfg)

        cfg = _config("M2")
        for name in vars(cfg.motion_error):
            setattr(cfg.motion_error, name, 0.0)
        with self.assertRaisesRegex(ValueError, "motion_error requires at least one positive"):
            validate(cfg)

        cfg = _config("M5")
        for name in vars(cfg.motion_gap):
            setattr(cfg.motion_gap, name, 0.0)
        with self.assertRaisesRegex(ValueError, "motion_gap requires at least one positive"):
            validate(cfg)


class LifecycleIntegrationTest(unittest.TestCase):
    def test_disabled_uniform_path_still_dispatches_directly_to_legacy_sampler(self) -> None:
        source = ast.unparse(_command_method("_sample_motion_and_start_frame"))
        uniform_branch = source.split("if motion_mode == 'uniform' and segment_mode == 'uniform':", 1)[1]
        self.assertIn("self._adaptive_sampling(env_ids)", uniform_branch)
        self.assertIn("self._quality_gated_uniform_sampling(env_ids)", uniform_branch)
        self.assertNotIn("online_learning.sample", uniform_branch.split("return", 1)[0])

    def test_pre_reset_environment_hook_and_current_segment_are_used(self) -> None:
        online_env = ONLINE_ENV_PATH.read_text(encoding="utf-8")
        registry = G1_REGISTRY_PATH.read_text(encoding="utf-8")
        self.assertIn("class OnlineLearningManagerBasedRLEnv", online_env)
        self.assertIn("command.record_online_learning_step(env_ids)", online_env)
        self.assertIn("env_ids: Sequence[int] | None = None", online_env)
        self.assertIn("_online_external_reset_in_progress", online_env)
        self.assertIn("close_online_assignments_for_external_reset", online_env)
        self.assertLess(
            online_env.index("command.record_online_learning_step(env_ids)"),
            online_env.index("super()._reset_idx(env_ids)"),
        )
        self.assertIn("OnlineLearningManagerBasedRLEnv", registry)
        self.assertIn("refresh_online_state_after_external_reset", online_env)
        capture = ast.unparse(_command_method("record_online_learning_step"))
        helper = ast.unparse(_command_method("_record_online_learning_components"))
        self.assertIn("self._record_online_learning_components", capture)
        self.assertIn("self._trusted_current_global_segment_ids()", helper)
        self.assertIn("quat_error_magnitude", capture)
        update_metrics = ast.unparse(_command_method("_update_metrics"))
        self.assertIn("active_env_ids = torch.where(~reset_mask)[0]", update_metrics)
        external_reset = ast.unparse(_command_method("refresh_online_state_after_external_reset"))
        self.assertIn("self._update_metrics(record_online=False)", external_reset)

    def test_segment_crossing_and_terminal_completion_finalize_before_resampling(self) -> None:
        update = ast.unparse(_command_method("_update_command"))
        resample = ast.unparse(_command_method("_resample_command"))
        self.assertIn("cross_segment_boundaries", update)
        self.assertLess(update.index("cross_segment_boundaries"), update.index("self._resample_command"))
        self.assertIn("finish_assignments", resample)
        self.assertIn("segment_natural_completion=segment_natural", resample)
        self.assertLess(resample.index("finish_assignments"), resample.index("_sample_motion_and_start_frame"))

    def test_iteration_commit_occurs_before_rate_limited_metric_branch(self) -> None:
        runner = _parsed(RUNNER_PATH)
        motion_runner = next(
            node for node in runner.body if isinstance(node, ast.ClassDef) and node.name == "MotionOnPolicyRunner"
        )
        log = next(node for node in motion_runner.body if isinstance(node, ast.FunctionDef) and node.name == "log")
        source = ast.unparse(log)
        self.assertIn("command.on_learning_iteration_end(iteration)", source)
        self.assertLess(source.index("on_learning_iteration_end"), source.index("iteration % interval"))

    def test_checkpoint_contains_online_stats_probability_and_rng_and_resume_reassigns(self) -> None:
        state = ast.unparse(_command_method("sampling_state_dict"))
        restore = ast.unparse(_command_method("load_sampling_state_dict"))
        runner = RUNNER_PATH.read_text(encoding="utf-8")
        self.assertIn("self.online_learning.state_dict()", state)
        self.assertIn("'motion_pool'", state)
        self.assertIn("self.online_learning.load_state_dict", restore)
        self.assertIn("reassign_after_online_resume", runner)

    def test_learning_gap_segment_sampler_uses_signed_local_gap(self) -> None:
        controller_path = (
            PROJECT_ROOT
            / "source"
            / "whole_body_tracking"
            / "whole_body_tracking"
            / "utils"
            / "online_learning.py"
        )
        source = controller_path.read_text(encoding="utf-8")
        self.assertIn("segment_score = self.gap_result.local_gap", source)
        self.assertNotIn("torch.clamp(self.gap_result.local_gap, min=0.0)", source)


if __name__ == "__main__":
    unittest.main()
