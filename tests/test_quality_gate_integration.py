from __future__ import annotations

import ast
import math
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


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


def _parsed(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _command_method(name: str) -> ast.FunctionDef:
    command_class = next(
        node for node in _parsed(COMMANDS_PATH).body if isinstance(node, ast.ClassDef) and node.name == "MotionCommand"
    )
    return next(node for node in command_class.body if isinstance(node, ast.FunctionDef) and node.name == name)


def _standalone_command_method(name: str):
    function = _command_method(name)
    namespace = {
        "Mapping": dict,
        "SAMPLING_STATE_VERSION": 1,
        "QUALITY_STATUS_TO_CODE": {"pass": 0, "borderline": 1, "reject": 2},
        "torch": torch,
    }
    source = "from __future__ import annotations\n" + ast.unparse(function)
    exec(compile(source, filename=str(COMMANDS_PATH), mode="exec"), namespace)
    return namespace[name]


def _call_name(call: ast.Call) -> str:
    parts: list[str] = []
    node: ast.expr = call.func
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _validator():
    function = next(
        node
        for node in _parsed(COMMANDS_PATH).body
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


def _config(*, method: str = "M0", quality_enabled: bool = False, **quality_overrides):
    quality = dict(
        enabled=quality_enabled,
        metadata_path="quality.npz" if quality_enabled else "",
        reject_statuses=("reject",),
        include_borderline=True,
        strict_metadata_match=True,
        empty_motion_policy="error",
        gate_scope="assignment_start",
    )
    quality.update(quality_overrides)
    return SimpleNamespace(
        method_name=method,
        segment=SimpleNamespace(enabled=True, length_seconds=1.0),
        quality_gate=SimpleNamespace(**quality),
        difficulty_calibration=SimpleNamespace(enabled=False),
        diversity_constraint=SimpleNamespace(enabled=False),
        motion_sampling=SimpleNamespace(mode="uniform"),
        segment_sampling=SimpleNamespace(mode="uniform"),
        sampling_statistics=SimpleNamespace(enabled=True, log_interval=100),
        assignment_trace=SimpleNamespace(enabled=False, output_path="", max_entries=2048),
        probability_validation=SimpleNamespace(epsilon=1.0e-8),
    )


class ResearchConfigValidationTest(unittest.TestCase):
    def test_valid_m0_and_m1(self) -> None:
        validate = _validator()
        validate(_config())
        validate(_config(method="M1", quality_enabled=True))

    def test_method_and_gate_must_agree(self) -> None:
        validate = _validator()
        with self.assertRaisesRegex(ValueError, "M0"):
            validate(_config(method="M0", quality_enabled=True))
        with self.assertRaisesRegex(ValueError, "M1"):
            validate(_config(method="M1", quality_enabled=False))

    def test_m1_requires_metadata_statistics_and_assignment_start_scope(self) -> None:
        validate = _validator()
        cfg = _config(method="M1", quality_enabled=True, metadata_path="")
        with self.assertRaisesRegex(ValueError, "metadata_path"):
            validate(cfg)

        cfg = _config(method="M1", quality_enabled=True)
        cfg.sampling_statistics.enabled = False
        with self.assertRaisesRegex(ValueError, "sampling_statistics"):
            validate(cfg)

        with self.assertRaisesRegex(NotImplementedError, "assignment_start"):
            validate(_config(method="M1", quality_enabled=True, gate_scope="whole_rollout"))

        with self.assertRaisesRegex(ValueError, "empty_motion_policy"):
            validate(_config(method="M1", quality_enabled=True, empty_motion_policy="ignore"))

    def test_reject_status_semantics_are_conservative(self) -> None:
        validate = _validator()
        with self.assertRaisesRegex(ValueError, "must reject"):
            validate(_config(method="M1", quality_enabled=True, reject_statuses=()))
        with self.assertRaisesRegex(ValueError, "must never reject"):
            validate(_config(method="M1", quality_enabled=True, reject_statuses=("pass", "reject")))
        with self.assertRaisesRegex(ValueError, "conflicts"):
            validate(
                _config(
                    method="M1",
                    quality_enabled=True,
                    reject_statuses=("reject", "borderline"),
                    include_borderline=True,
                )
            )

    def test_assignment_trace_requires_segment_path_and_positive_limit(self) -> None:
        validate = _validator()
        cfg = _config()
        cfg.assignment_trace.enabled = True
        cfg.assignment_trace.output_path = "/tmp/wbt_assignment_trace.csv"
        validate(cfg)

        cfg.segment.enabled = False
        with self.assertRaisesRegex(ValueError, "assignment_trace"):
            validate(cfg)

        cfg = _config()
        cfg.assignment_trace.enabled = True
        with self.assertRaisesRegex(ValueError, "output_path"):
            validate(cfg)

        cfg.assignment_trace.output_path = "/tmp/wbt_assignment_trace.csv"
        cfg.assignment_trace.max_entries = 0
        with self.assertRaisesRegex(ValueError, "max_entries"):
            validate(cfg)


class QualitySamplingDispatchTest(unittest.TestCase):
    def test_gate_off_still_calls_legacy_sampler_and_gate_on_uses_separate_path(self) -> None:
        method = _command_method("_sample_motion_and_start_frame")
        call_names = [_call_name(call) for call in ast.walk(method) if isinstance(call, ast.Call)]
        self.assertEqual(call_names.count("self._adaptive_sampling"), 1)
        self.assertEqual(call_names.count("self._quality_gated_uniform_sampling"), 1)

    def test_quality_sampler_has_fixed_random_calls_and_no_rejection_loop(self) -> None:
        method = _command_method("_quality_gated_uniform_sampling")
        call_names = [_call_name(call) for call in ast.walk(method) if isinstance(call, ast.Call)]
        self.assertEqual(call_names.count("sample_uniform"), 1)
        self.assertEqual(call_names.count("self.quality_gate_index.map_uniform_samples"), 1)
        self.assertGreaterEqual(call_names.count("torch.randint"), 1)
        self.assertNotIn("torch.multinomial", call_names)
        self.assertFalse(any(isinstance(node, (ast.For, ast.While)) for node in ast.walk(method)))

    def test_gate_disabled_does_not_load_quality_metadata(self) -> None:
        init_method = _command_method("__init__")
        source = ast.unparse(init_method)
        self.assertIn("if self.cfg.research.quality_gate.enabled:", source)
        self.assertIn("self._initialize_quality_gate()", source)
        self.assertNotIn("SegmentQualityMetadata.load", source)

        init_gate_method = _command_method("_initialize_quality_gate")
        init_gate_source = ast.unparse(init_gate_method)
        self.assertIn("SegmentQualityMetadata.load", init_gate_source)

    def test_quality_exposure_counts_current_reject_reference_frames(self) -> None:
        method = _standalone_command_method("_record_quality_reference_exposure")

        dummy = SimpleNamespace(
            cfg=SimpleNamespace(research=SimpleNamespace(quality_gate=SimpleNamespace(enabled=True))),
            segment_index=object(),
            quality_status_codes=torch.tensor([0, 2, 2, 1], dtype=torch.int8),
            current_global_segment_ids=torch.tensor([0, 1, 3, 2], dtype=torch.long),
            quality_reference_frame_count=torch.zeros((), dtype=torch.long),
            quality_reject_reference_frame_count=torch.zeros((), dtype=torch.long),
        )
        method(dummy)
        self.assertEqual(int(dummy.quality_reference_frame_count.item()), 4)
        self.assertEqual(int(dummy.quality_reject_reference_frame_count.item()), 2)

        dummy.cfg.research.quality_gate.enabled = False
        method(dummy)
        self.assertEqual(int(dummy.quality_reference_frame_count.item()), 4)
        self.assertEqual(int(dummy.quality_reject_reference_frame_count.item()), 2)

    def test_quality_exposure_state_round_trip_and_legacy_missing_state(self) -> None:
        state_method = _standalone_command_method("_quality_exposure_state_dict")
        load_method = _standalone_command_method("_load_quality_exposure_state_dict")

        source = SimpleNamespace(
            quality_reference_frame_count=torch.tensor(10, dtype=torch.long),
            quality_reject_reference_frame_count=torch.tensor(3, dtype=torch.long),
        )
        state = state_method(source)
        restored = SimpleNamespace(
            device="cpu",
            quality_reference_frame_count=torch.zeros((), dtype=torch.long),
            quality_reject_reference_frame_count=torch.zeros((), dtype=torch.long),
        )
        load_method(restored, state)
        self.assertEqual(int(restored.quality_reference_frame_count.item()), 10)
        self.assertEqual(int(restored.quality_reject_reference_frame_count.item()), 3)

        load_method(restored, None)
        self.assertEqual(int(restored.quality_reference_frame_count.item()), 0)
        self.assertEqual(int(restored.quality_reject_reference_frame_count.item()), 0)

        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            load_method(
                restored,
                {
                    "version": 1,
                    "reference_frame_count": torch.tensor(2),
                    "reject_reference_frame_count": torch.tensor(3),
                },
            )

    def test_quality_identity_and_exposure_are_checkpointed_and_logged_as_scalars(self) -> None:
        state_method = _command_method("sampling_state_dict")
        state_source = ast.unparse(state_method)
        self.assertIn("state['quality_gate']", state_source)
        self.assertIn("state['quality_exposure']", state_source)
        self.assertIn("identity_state", state_source)
        self.assertIn("quality_gate_index.identity_state", state_source)

        load_source = ast.unparse(_command_method("load_sampling_state_dict"))
        self.assertIn("_load_quality_exposure_state_dict", load_source)
        self.assertIn("eligible_motion_mask_sha256", load_source)

        runner_source = RUNNER_PATH.read_text(encoding="utf-8")
        self.assertIn('f"quality/{name}"', runner_source)
        self.assertIn('f"sampling/{name}"', runner_source)
        self.assertIn('f"dataset/{name}"', runner_source)


if __name__ == "__main__":
    unittest.main()
