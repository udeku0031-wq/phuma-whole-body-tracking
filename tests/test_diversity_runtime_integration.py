from __future__ import annotations

import ast
import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from tests.test_online_learning_runtime_integration import _config, _validator


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
SAMPLING_PATH = (
    PROJECT_ROOT
    / "source"
    / "whole_body_tracking"
    / "whole_body_tracking"
    / "utils"
    / "sampling.py"
)


def _m7_config() -> SimpleNamespace:
    cfg = _config("M6")
    cfg.method_name = "M7"
    cfg.diversity_constraint = SimpleNamespace(
        enabled=True,
        metadata_path="clusters.npz",
        strict_metadata_match=True,
        expected_num_clusters=8,
        budget_mode="sqrt_size_with_floor",
        minimum_budget_fraction_of_uniform=0.5,
        cluster_size_exponent=0.5,
        diversity_during_warmup=True,
        count_aware_correction=False,
    )
    return cfg


def _command_method(name: str) -> ast.FunctionDef:
    module = ast.parse(COMMANDS_PATH.read_text(encoding="utf-8"), filename=str(COMMANDS_PATH))
    command = next(
        node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "MotionCommand"
    )
    return next(
        node for node in command.body if isinstance(node, ast.FunctionDef) and node.name == name
    )


class DiversityConfigContractTest(unittest.TestCase):
    def test_m7_contract_and_diversity_only_debug_contract(self) -> None:
        validate = _validator()
        validate(_m7_config())

        diagnostic = _m7_config()
        diagnostic.method_name = "DIVERSITY_ONLY"
        diagnostic.motion_sampling.mode = "uniform"
        diagnostic.segment_sampling.mode = "uniform"
        diagnostic.quality_gate.enabled = False
        diagnostic.quality_gate.metadata_path = ""
        diagnostic.quality_gate.empty_motion_policy = "error"
        diagnostic.difficulty_calibration.enabled = False
        diagnostic.difficulty_calibration.metadata_path = ""
        validate(diagnostic)

        diagnostic.difficulty_calibration.enabled = True
        diagnostic.difficulty_calibration.metadata_path = "/tmp/difficulty.npz"
        with self.assertRaisesRegex(ValueError, "must not use difficulty"):
            validate(diagnostic)

    def test_m7_requires_all_four_layers_and_valid_budget(self) -> None:
        validate = _validator()
        for field in ("quality_gate", "difficulty_calibration", "diversity_constraint"):
            cfg = _m7_config()
            getattr(cfg, field).enabled = False
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate(cfg)

        cfg = _m7_config()
        cfg.diversity_constraint.minimum_budget_fraction_of_uniform = 1.1
        with self.assertRaisesRegex(ValueError, "minimum_budget_fraction"):
            validate(cfg)

    def test_m0_through_m6_keep_diversity_disabled(self) -> None:
        validate = _validator()
        for method in ("M0", "M1", "M2", "M3", "M4", "M5", "M6"):
            with self.subTest(method=method):
                cfg = _config(method)
                validate(cfg)
                cfg.diversity_constraint.enabled = True
                with self.assertRaises(ValueError):
                    validate(cfg)


class DiversityWiringTest(unittest.TestCase):
    def test_cluster_metadata_is_loaded_only_behind_enabled_guard(self) -> None:
        initializer = ast.unparse(_command_method("__init__"))
        self.assertIn("if self.cfg.research.diversity_constraint.enabled:", initializer)
        self.assertEqual(initializer.count("self._initialize_cluster_metadata()"), 1)

    def test_online_controller_receives_cluster_ids_and_settings(self) -> None:
        source = ast.unparse(_command_method("_initialize_online_learning"))
        self.assertIn("motion_cluster_ids=self.motion_cluster_ids", source)
        self.assertIn("self._diversity_sampling_settings()", source)

    def test_diversity_only_dispatches_to_online_sampler_not_legacy_uniform(self) -> None:
        source = ast.unparse(_command_method("_sample_motion_and_start_frame"))
        self.assertIn("not self.cfg.research.diversity_constraint.enabled", source)
        self.assertIn("self.online_learning.sample(len(env_ids))", source)

    def test_checkpoint_contains_cluster_identity_and_sampler_state(self) -> None:
        save_source = ast.unparse(_command_method("sampling_state_dict"))
        load_source = ast.unparse(_command_method("load_sampling_state_dict"))
        self.assertIn("state['diversity_constraint']", save_source)
        self.assertIn("motion_cluster_ids", load_source)
        self.assertIn("saved_online_state", load_source)


class DiversityAssignmentTraceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        spec = importlib.util.spec_from_file_location("wbt_sampling_diversity_trace", SAMPLING_PATH)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Unable to import sampling utilities from {SAMPLING_PATH}")
        cls.sampling = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.sampling)

    def test_disabled_header_remains_byte_compatible(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.csv"
            recorder = self.sampling.AssignmentTraceRecorder(str(path), 2)
            recorder.record_assignments([0], [1], [2], [0], [3])
            with path.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.reader(stream))
            self.assertEqual(tuple(rows[0]), recorder.HEADER)
            self.assertEqual(len(rows[0]), 8)

    def test_m7_trace_adds_factorized_probability_observations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "m7.csv"
            recorder = self.sampling.AssignmentTraceRecorder(
                str(path), 2, include_diversity=True
            )
            recorder.record_assignments(
                torch.tensor([0]),
                torch.tensor([1]),
                torch.tensor([2]),
                torch.tensor([0]),
                torch.tensor([3]),
                cluster_ids=torch.tensor([4]),
                cluster_probabilities=torch.tensor([0.2]),
                conditional_motion_probabilities=torch.tensor([0.1]),
                global_motion_probabilities=torch.tensor([0.02]),
                conditional_segment_probabilities=torch.tensor([0.5]),
            )
            with path.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[0]["cluster_id"], "4")
            self.assertAlmostEqual(float(rows[0]["motion_probability_global"]), 0.02)


if __name__ == "__main__":
    unittest.main()
