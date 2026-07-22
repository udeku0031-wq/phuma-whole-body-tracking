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


def _parsed() -> ast.Module:
    return ast.parse(COMMANDS_PATH.read_text(encoding="utf-8"), filename=str(COMMANDS_PATH))


def _command_method(name: str) -> ast.FunctionDef:
    command = next(
        node for node in _parsed().body if isinstance(node, ast.ClassDef) and node.name == "MotionCommand"
    )
    return next(node for node in command.body if isinstance(node, ast.FunctionDef) and node.name == name)


def _runner_function(name: str) -> ast.FunctionDef:
    module = ast.parse(RUNNER_PATH.read_text(encoding="utf-8"), filename=str(RUNNER_PATH))
    return next(
        node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _validator():
    function = next(
        node
        for node in _parsed().body
        if isinstance(node, ast.FunctionDef) and node.name == "_validate_research_config"
    )
    namespace = {
        "math": math,
        "QUALITY_STATUS_TO_CODE": {"pass": 0, "borderline": 1, "reject": 2},
        "QualityGatedStartIndex": SimpleNamespace(_EMPTY_MOTION_POLICIES=frozenset({"error", "exclude"})),
    }
    source = "from __future__ import annotations\n" + ast.unparse(function)
    exec(compile(source, filename=str(COMMANDS_PATH), mode="exec"), namespace)
    return namespace["_validate_research_config"]


def _config(*, difficulty_enabled: bool) -> SimpleNamespace:
    return SimpleNamespace(
        method_name="M0",
        segment=SimpleNamespace(enabled=True, length_seconds=1.0),
        quality_gate=SimpleNamespace(
            enabled=False,
            metadata_path="",
            reject_statuses=("reject",),
            include_borderline=True,
            strict_metadata_match=True,
            empty_motion_policy="error",
            gate_scope="assignment_start",
        ),
        difficulty_calibration=SimpleNamespace(
            enabled=difficulty_enabled,
            metadata_path="difficulty.npz" if difficulty_enabled else "",
            strict_metadata_match=True,
            expected_num_bins=10,
        ),
        diversity_constraint=SimpleNamespace(enabled=False),
        motion_sampling=SimpleNamespace(mode="uniform"),
        segment_sampling=SimpleNamespace(mode="uniform"),
        sampling_statistics=SimpleNamespace(enabled=True, log_interval=100),
        assignment_trace=SimpleNamespace(enabled=False, output_path="", max_entries=2048),
        probability_validation=SimpleNamespace(epsilon=1.0e-8),
    )


class DifficultyRuntimeIntegrationTest(unittest.TestCase):
    def test_enabled_is_load_only_and_validates_required_identity(self) -> None:
        validate = _validator()
        validate(_config(difficulty_enabled=False))
        validate(_config(difficulty_enabled=True))

        cfg = _config(difficulty_enabled=True)
        cfg.difficulty_calibration.metadata_path = ""
        with self.assertRaisesRegex(ValueError, "metadata_path"):
            validate(cfg)

        cfg = _config(difficulty_enabled=True)
        cfg.difficulty_calibration.expected_num_bins = 1
        with self.assertRaisesRegex(ValueError, "expected_num_bins"):
            validate(cfg)

        cfg = _config(difficulty_enabled=True)
        cfg.segment.enabled = False
        with self.assertRaisesRegex(ValueError, "segment.enabled"):
            validate(cfg)

    def test_disabled_does_not_load_and_sampler_has_no_difficulty_branch(self) -> None:
        initializer = ast.unparse(_command_method("__init__"))
        self.assertIn("if self.cfg.research.difficulty_calibration.enabled:", initializer)
        self.assertIn("self._initialize_difficulty_calibration()", initializer)
        self.assertNotIn("SegmentDifficultyMetadata.load", initializer)

        loader = ast.unparse(_command_method("_initialize_difficulty_calibration"))
        self.assertIn("SegmentDifficultyMetadata.load", loader)
        self.assertIn("DEFAULT_ALGORITHM_SCHEMA_VERSION", loader)
        self.assertIn("layout_mismatches", loader)
        self.assertIn("cannot be relaxed", loader)
        sampler = ast.unparse(_command_method("_sample_motion_and_start_frame"))
        self.assertNotIn("difficulty", sampler.lower())
        self.assertNotIn("difficulty_scores", sampler)
        self.assertNotIn("difficulty_bins", sampler)

    def test_static_arrays_are_exposed_but_not_used_as_probabilities(self) -> None:
        loader = ast.unparse(_command_method("_initialize_difficulty_calibration"))
        self.assertIn("metadata.difficulty_score", loader)
        self.assertIn("metadata.difficulty_bin", loader)
        self.assertNotIn("multinomial", loader)
        self.assertNotIn("rand", loader.lower())

        class_node = next(
            node for node in _parsed().body if isinstance(node, ast.ClassDef) and node.name == "MotionCommand"
        )
        properties = {
            node.name
            for node in class_node.body
            if isinstance(node, ast.FunctionDef)
            and any(isinstance(decorator, ast.Name) and decorator.id == "property" for decorator in node.decorator_list)
        }
        self.assertTrue(
            {
                "current_difficulty_scores",
                "current_difficulty_bins",
                "assigned_difficulty_scores",
                "assigned_difficulty_bins",
            }.issubset(properties)
        )

    def test_checkpoint_and_wandb_store_identity_not_segment_arrays(self) -> None:
        state_source = ast.unparse(_command_method("sampling_state_dict"))
        self.assertIn("state['difficulty_calibration']", state_source)
        identity_source = ast.unparse(_command_method("_difficulty_identity_state"))
        for field in (
            "metadata_sha256",
            "profile_sha256",
            "difficulty_config_sha256",
            "manifest_sha256",
            "num_bins",
        ):
            self.assertIn(field, identity_source)
        self.assertNotIn("difficulty_score", state_source)
        self.assertNotIn("difficulty_bin", state_source)

        restore_source = ast.unparse(_command_method("load_sampling_state_dict"))
        self.assertIn("saved_difficulty_state", restore_source)
        self.assertIn("current_difficulty_identity", restore_source)
        runner_source = RUNNER_PATH.read_text(encoding="utf-8")
        self.assertIn('f"difficulty/{name}"', runner_source)

        runner_restore = ast.unparse(_runner_function("_restore_sampling_state"))
        self.assertIn("difficulty_calibration.enabled", runner_restore)
        self.assertIn("Difficulty-enabled resume requires checkpoint sampling state", runner_restore)


if __name__ == "__main__":
    unittest.main()
