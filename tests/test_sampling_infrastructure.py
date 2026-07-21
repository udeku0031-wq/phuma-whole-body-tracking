from __future__ import annotations

import ast
import csv
import importlib.util
import tempfile
import unittest
import warnings
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLING_PATH = (
    PROJECT_ROOT / "source" / "whole_body_tracking" / "whole_body_tracking" / "utils" / "sampling.py"
)
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

_SAMPLING_SPEC = importlib.util.spec_from_file_location("wbt_sampling_for_tests", SAMPLING_PATH)
if _SAMPLING_SPEC is None or _SAMPLING_SPEC.loader is None:
    raise RuntimeError(f"Unable to load sampling utilities from {SAMPLING_PATH}")
sampling = importlib.util.module_from_spec(_SAMPLING_SPEC)
_SAMPLING_SPEC.loader.exec_module(sampling)


def _command_method(name: str) -> ast.FunctionDef:
    module = ast.parse(COMMANDS_PATH.read_text(encoding="utf-8"), filename=str(COMMANDS_PATH))
    motion_command = next(
        node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "MotionCommand"
    )
    return next(
        node for node in motion_command.body if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _module_function(path: Path, name: str) -> ast.FunctionDef:
    module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == name)


def _standalone_module_functions(
    path: Path, names: tuple[str, ...], namespace: dict[str, object] | None = None
) -> tuple[object, ...]:
    parsed = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    functions = [
        next(node for node in parsed.body if isinstance(node, ast.FunctionDef) and node.name == name)
        for name in names
    ]
    module = ast.Module(body=functions, type_ignores=[])
    namespace = {} if namespace is None else namespace
    exec(compile(module, filename=str(path), mode="exec"), namespace)
    return tuple(namespace[name] for name in names)


def _standalone_module_function(path: Path, name: str):
    return _standalone_module_functions(path, (name,))[0]


def _call_name(call: ast.Call) -> str:
    parts: list[str] = []
    node: ast.expr = call.func
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


class FixedLengthSegmentIndexTest(unittest.TestCase):
    def test_segment_boundaries_at_50_fps(self) -> None:
        expected = {
            1: ([0], [1]),
            49: ([0], [49]),
            50: ([0], [50]),
            51: ([0, 50], [50, 51]),
            99: ([0, 50], [50, 99]),
            100: ([0, 50], [50, 100]),
            101: ([0, 50, 100], [50, 100, 101]),
        }

        for num_frames, (expected_starts, expected_ends) in expected.items():
            with self.subTest(num_frames=num_frames):
                index = sampling.FixedLengthSegmentIndex([num_frames], [50.0])
                metadata = index.metadata()

                self.assertEqual(index.segment_frames.tolist(), [50])
                self.assertEqual(index.num_segments, len(expected_starts))
                self.assertEqual(index.motion_num_segments.tolist(), [len(expected_starts)])
                self.assertEqual(metadata["start_frame"].tolist(), expected_starts)
                self.assertEqual(metadata["end_frame_exclusive"].tolist(), expected_ends)
                self.assertEqual(
                    metadata["local_segment_id"].tolist(), list(range(len(expected_starts)))
                )
                self.assertTrue(torch.all(metadata["start_frame"] < metadata["end_frame_exclusive"]))
                self.assertTrue(torch.all(metadata["end_frame_exclusive"] <= num_frames))

    def test_non_50_and_mixed_fps(self) -> None:
        index = sampling.FixedLengthSegmentIndex(
            motion_lengths=[61, 91, 1],
            motion_fps=[30.0, 60.0, 24.0],
        )
        metadata = index.metadata()

        self.assertEqual(index.segment_frames.tolist(), [30, 60, 24])
        self.assertEqual(index.motion_num_segments.tolist(), [3, 2, 1])
        self.assertEqual(index.motion_segment_offsets.tolist(), [0, 3, 5, 6])
        self.assertEqual(metadata["motion_id"].tolist(), [0, 0, 0, 1, 1, 2])
        self.assertEqual(metadata["start_frame"].tolist(), [0, 30, 60, 0, 60, 0])
        self.assertEqual(metadata["end_frame_exclusive"].tolist(), [30, 60, 61, 60, 91, 1])
        self.assertEqual(metadata["fps"].tolist(), [30.0, 30.0, 30.0, 60.0, 60.0, 24.0])

    def test_three_way_mappings(self) -> None:
        index = sampling.FixedLengthSegmentIndex([101, 49], [50.0, 24.0])
        motion_ids = torch.tensor([0, 0, 0, 1, 1, 1])
        frame_ids = torch.tensor([0, 50, 100, 0, 24, 48])

        local_ids, global_ids = index.motion_frame_to_segment(motion_ids, frame_ids)
        self.assertEqual(local_ids.tolist(), [0, 1, 2, 0, 1, 2])
        self.assertEqual(global_ids.tolist(), [0, 1, 2, 3, 4, 5])
        self.assertTrue(torch.equal(index.motion_local_to_global(motion_ids, local_ids), global_ids))

        recovered_motion_ids, recovered_local_ids = index.global_to_motion_local(global_ids)
        self.assertTrue(torch.equal(recovered_motion_ids, motion_ids))
        self.assertTrue(torch.equal(recovered_local_ids, local_ids))

    def test_mapping_rejects_invalid_ids_and_shapes(self) -> None:
        index = sampling.FixedLengthSegmentIndex([51, 49], [50.0, 24.0])

        with self.assertRaisesRegex(ValueError, "same shape"):
            index.motion_frame_to_segment([0, 1], [0])
        for motion_id in (-1, 2):
            with self.subTest(motion_id=motion_id), self.assertRaisesRegex(ValueError, "motion_ids"):
                index.motion_frame_to_segment([motion_id], [0])
        for frame_id in (-1, 51):
            with self.subTest(frame_id=frame_id), self.assertRaisesRegex(ValueError, "frame_ids"):
                index.motion_frame_to_segment([0], [frame_id])
        for local_id in (-1, 2):
            with self.subTest(local_id=local_id), self.assertRaisesRegex(ValueError, "local_segment_ids"):
                index.motion_local_to_global([0], [local_id])
        for global_id in (-1, index.num_segments):
            with self.subTest(global_id=global_id), self.assertRaisesRegex(
                ValueError, "global_segment_ids"
            ):
                index.global_to_motion_local([global_id])

    def test_empty_mapping_is_safe(self) -> None:
        index = sampling.FixedLengthSegmentIndex([50], [50.0])
        local_ids, global_ids = index.motion_frame_to_segment([], [])
        self.assertEqual(local_ids.numel(), 0)
        self.assertEqual(global_ids.numel(), 0)
        motion_ids, recovered_local_ids = index.global_to_motion_local([])
        self.assertEqual(motion_ids.numel(), 0)
        self.assertEqual(recovered_local_ids.numel(), 0)


class SamplingStatisticsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.index = sampling.FixedLengthSegmentIndex([100, 60], [50.0, 30.0])
        self.fingerprint = sampling.motion_pool_fingerprint(
            ["motions/a.npz", "motions/b.npz"], [100, 60], [50.0, 30.0]
        )
        self.statistics = sampling.SamplingStatistics(
            self.index,
            pool_fingerprint=self.fingerprint,
        )

    def _record_fixture(self) -> None:
        local_ids, global_ids = self.statistics.record_assignments(
            torch.tensor([[0, 0, 1], [1, 0, 1]]),
            torch.tensor([[0, 75, 0], [45, 5, 45]]),
        )
        self.assertEqual(local_ids.tolist(), [[0, 1, 0], [1, 0, 1]])
        self.assertEqual(global_ids.tolist(), [[0, 1, 2], [3, 0, 3]])

    def test_batch_and_duplicate_counts_and_coverage(self) -> None:
        self._record_fixture()
        self.statistics.record_probability_fallback(2)

        self.assertEqual(self.statistics.get_motion_sample_counts().tolist(), [3, 3])
        self.assertEqual(self.statistics.get_segment_sample_counts().tolist(), [2, 1, 1, 2])
        summary = self.statistics.summary()
        self.assertEqual(summary["total_assignments"], 6)
        self.assertEqual(summary["motion_coverage"], 1.0)
        self.assertEqual(summary["segment_coverage"], 1.0)
        self.assertEqual(summary["max_motion_sample_fraction"], 0.5)
        self.assertAlmostEqual(summary["max_segment_sample_fraction"], 1.0 / 3.0)
        self.assertEqual(summary["mean_motion_sample_count"], 3.0)
        self.assertEqual(summary["mean_segment_sample_count"], 1.5)
        self.assertEqual(summary["invalid_probability_fallbacks"], 2)

    def test_empty_batch_does_not_change_counts(self) -> None:
        local_ids, global_ids = self.statistics.record_assignments([], [])
        self.assertEqual(local_ids.numel(), 0)
        self.assertEqual(global_ids.numel(), 0)
        self.assertEqual(int(self.statistics.total_assignments.item()), 0)

    def test_state_round_trip_and_reset(self) -> None:
        self._record_fixture()
        self.statistics.record_probability_fallback(3)
        state = self.statistics.state_dict()

        restored = sampling.SamplingStatistics(
            sampling.FixedLengthSegmentIndex([100, 60], [50.0, 30.0]),
            pool_fingerprint=self.fingerprint,
        )
        restored.load_state_dict(state)

        self.assertTrue(
            torch.equal(restored.get_motion_sample_counts(), self.statistics.get_motion_sample_counts())
        )
        self.assertTrue(
            torch.equal(restored.get_segment_sample_counts(), self.statistics.get_segment_sample_counts())
        )
        self.assertEqual(restored.summary(), self.statistics.summary())

        restored.reset_statistics()
        self.assertEqual(restored.get_motion_sample_counts().tolist(), [0, 0])
        self.assertEqual(restored.get_segment_sample_counts().tolist(), [0, 0, 0, 0])
        self.assertEqual(int(restored.total_assignments.item()), 0)
        self.assertEqual(int(restored.invalid_probability_fallback_count.item()), 0)

    def test_state_rejects_motion_pool_fingerprint_mismatch(self) -> None:
        self._record_fixture()
        reordered_fingerprint = sampling.motion_pool_fingerprint(
            ["motions/b.npz", "motions/a.npz"], [60, 100], [30.0, 50.0]
        )
        self.assertNotEqual(self.fingerprint, reordered_fingerprint)
        restored = sampling.SamplingStatistics(
            sampling.FixedLengthSegmentIndex([100, 60], [50.0, 30.0]),
            pool_fingerprint=reordered_fingerprint,
        )

        with self.assertRaisesRegex(ValueError, "fingerprint"):
            restored.load_state_dict(self.statistics.state_dict())

    def test_pool_fingerprint_is_stable_when_the_data_tree_moves(self) -> None:
        first = sampling.motion_pool_fingerprint(
            ["/old/root/motions/a.npz", "/old/root/motions/sub/b.npz"], [100, 60], [50.0, 30.0]
        )
        moved = sampling.motion_pool_fingerprint(
            ["/new/root/motions/a.npz", "/new/root/motions/sub/b.npz"], [100, 60], [50.0, 30.0]
        )
        self.assertEqual(first, moved)


class AssignmentTraceRecorderTest(unittest.TestCase):
    def test_records_bounded_csv_without_consuming_rng(self) -> None:
        index = sampling.FixedLengthSegmentIndex([100, 60], [50.0, 30.0])
        motion_ids = torch.tensor([0, 0, 1, 1], dtype=torch.long)
        start_frames = torch.tensor([0, 75, 0, 45], dtype=torch.long)
        local_ids, global_ids = index.motion_frame_to_segment(motion_ids, start_frames)

        with tempfile.TemporaryDirectory() as directory:
            trace_path = Path(directory) / "assignment_trace.csv"
            recorder = sampling.AssignmentTraceRecorder(
                str(trace_path),
                max_entries=3,
                pool_fingerprint="pool-fixture",
                run_label="M0",
            )

            before = torch.random.get_rng_state()
            written = recorder.record_assignments(
                torch.tensor([3, 2, 1, 0], dtype=torch.long),
                motion_ids,
                start_frames,
                local_ids,
                global_ids,
            )
            after = torch.random.get_rng_state()

            self.assertEqual(written, 3)
            self.assertTrue(torch.equal(before, after))
            self.assertEqual(recorder.summary()["recorded_entries"], 3)

            # The recorder is bounded; later calls must not append beyond max_entries.
            self.assertEqual(
                recorder.record_assignments([9], [0], [0], [0], [0]),
                0,
            )

            with trace_path.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 3)
            self.assertEqual(
                list(rows[0]),
                [
                    "assignment_index",
                    "env_id",
                    "motion_id",
                    "start_frame",
                    "local_segment_id",
                    "global_segment_id",
                    "pool_fingerprint",
                    "run_label",
                ],
            )
            self.assertEqual(rows[0]["assignment_index"], "0")
            self.assertEqual(rows[0]["env_id"], "3")
            self.assertEqual(rows[1]["motion_id"], "0")
            self.assertEqual(rows[1]["start_frame"], "75")
            self.assertEqual(rows[2]["global_segment_id"], "2")
            self.assertEqual(rows[2]["pool_fingerprint"], "pool-fixture")
            self.assertEqual(rows[2]["run_label"], "M0")

            recorder.reset()
            with trace_path.open(newline="", encoding="utf-8") as stream:
                self.assertEqual(list(csv.DictReader(stream)), [])
            self.assertEqual(recorder.summary()["recorded_entries"], 0)

    def test_rejects_invalid_trace_configuration_and_shapes(self) -> None:
        with self.assertRaisesRegex(ValueError, "output_path"):
            sampling.AssignmentTraceRecorder("", max_entries=1)
        with self.assertRaisesRegex(ValueError, "max_entries"):
            sampling.AssignmentTraceRecorder("/tmp/trace.csv", max_entries=0)

        with tempfile.TemporaryDirectory() as directory:
            recorder = sampling.AssignmentTraceRecorder(str(Path(directory) / "trace.csv"), 4)
            with self.assertRaisesRegex(ValueError, "same number"):
                recorder.record_assignments([0, 1], [0], [0], [0], [0])


class ProbabilityValidationTest(unittest.TestCase):
    def setUp(self) -> None:
        sampling._probability_warning_counts.clear()

    def test_normal_and_unnormalized_probabilities(self) -> None:
        normalized, used_fallback = sampling.normalize_and_validate_probabilities(
            torch.tensor([0.2, 0.3, 0.5]), expected_size=3
        )
        self.assertFalse(used_fallback)
        self.assertTrue(torch.allclose(normalized, torch.tensor([0.2, 0.3, 0.5])))

        normalized, used_fallback = sampling.normalize_and_validate_probabilities(
            torch.tensor([2.0, 3.0, 5.0])
        )
        self.assertFalse(used_fallback)
        self.assertTrue(torch.allclose(normalized, torch.tensor([0.2, 0.3, 0.5])))
        self.assertAlmostEqual(float(normalized.sum().item()), 1.0)

    def test_invalid_probabilities_use_uniform_fallback(self) -> None:
        invalid_vectors = {
            "zero": torch.zeros(3),
            "negative": torch.tensor([0.5, -0.1, 0.6]),
            "nan": torch.tensor([0.5, float("nan"), 0.5]),
            "positive_inf": torch.tensor([0.5, float("inf"), 0.5]),
            "negative_inf": torch.tensor([0.5, -float("inf"), 0.5]),
        }
        statistics = sampling.SamplingStatistics(
            sampling.FixedLengthSegmentIndex([50], [50.0])
        )

        for name, vector in invalid_vectors.items():
            with self.subTest(name=name), warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                normalized, used_fallback = sampling.normalize_and_validate_probabilities(
                    vector, fallback_statistics=statistics
                )
                self.assertTrue(used_fallback)
                self.assertTrue(torch.equal(normalized, torch.full((3,), 1.0 / 3.0)))
                self.assertTrue(any(item.category is RuntimeWarning for item in caught))

        self.assertEqual(statistics.summary()["invalid_probability_fallbacks"], len(invalid_vectors))

    def test_probability_shape_validation(self) -> None:
        invalid_shapes = (torch.empty(0), torch.ones(2, 2))
        for probabilities in invalid_shapes:
            with self.subTest(shape=tuple(probabilities.shape)), self.assertRaisesRegex(
                ValueError, "non-empty one-dimensional"
            ):
                sampling.normalize_and_validate_probabilities(probabilities)

        with self.assertRaisesRegex(ValueError, "Expected 4 probabilities"):
            sampling.normalize_and_validate_probabilities(torch.ones(3), expected_size=4)

    def test_invalid_probability_can_raise_instead_of_fallback(self) -> None:
        with self.assertRaisesRegex(ValueError, "Invalid probability vector"):
            sampling.normalize_and_validate_probabilities(torch.zeros(3), fallback="raise")

    def test_uniform_fallback_warning_is_rate_limited(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for _ in range(sampling._PROBABILITY_WARNING_LIMIT + 2):
                sampling.normalize_and_validate_probabilities(torch.zeros(3))

        runtime_warnings = [item for item in caught if item.category is RuntimeWarning]
        self.assertEqual(len(runtime_warnings), sampling._PROBABILITY_WARNING_LIMIT)


class BaselineCompatibilityTest(unittest.TestCase):
    def test_observation_helpers_do_not_advance_torch_rng(self) -> None:
        torch.manual_seed(20250719)
        before = torch.random.get_rng_state().clone()

        index = sampling.FixedLengthSegmentIndex([101, 49], [50.0, 24.0])
        statistics = sampling.SamplingStatistics(index, pool_fingerprint="fixture")
        statistics.record_assignments([0, 1, 0], [50, 48, 100])
        statistics.summary()
        sampling.normalize_and_validate_probabilities(torch.tensor([2.0, 3.0, 5.0]))

        after = torch.random.get_rng_state()
        self.assertTrue(torch.equal(before, after))

    def test_resample_observes_assignments_after_legacy_sampling(self) -> None:
        method = _command_method("_resample_command")
        calls = [call for call in ast.walk(method) if isinstance(call, ast.Call)]
        dispatch = [call for call in calls if _call_name(call).endswith("._sample_motion_and_start_frame")]
        record = [call for call in calls if _call_name(call).endswith("._record_sampling_assignments")]
        state_write = [
            call for call in calls if _call_name(call).endswith("._write_current_motion_state_to_sim")
        ]

        self.assertEqual(len(dispatch), 1)
        self.assertEqual(len(record), 1)
        self.assertEqual(len(state_write), 1)
        self.assertLess(dispatch[0].lineno, record[0].lineno)
        self.assertLess(record[0].lineno, state_write[0].lineno)

    def test_uniform_dispatch_calls_the_legacy_sampler(self) -> None:
        method = _command_method("_sample_motion_and_start_frame")
        call_names = [_call_name(call) for call in ast.walk(method) if isinstance(call, ast.Call)]
        self.assertEqual(call_names.count("self._adaptive_sampling"), 1)
        random_leaf_names = {"rand", "randint", "randn", "random", "sample_uniform", "multinomial"}
        observed_random_calls = {
            name for name in call_names if name.rsplit(".", maxsplit=1)[-1] in random_leaf_names
        }
        self.assertEqual(observed_random_calls, set())

    def test_assignment_observer_contains_no_random_calls(self) -> None:
        method = _command_method("_record_sampling_assignments")
        call_names = {_call_name(call) for call in ast.walk(method) if isinstance(call, ast.Call)}
        random_leaf_names = {
            "rand",
            "randint",
            "randn",
            "random",
            "sample_uniform",
            "multinomial",
        }
        observed_random_calls = {
            name for name in call_names if name.rsplit(".", maxsplit=1)[-1] in random_leaf_names
        }
        self.assertEqual(observed_random_calls, set())

    def test_legacy_sampler_retains_original_random_calls(self) -> None:
        method = _command_method("_adaptive_sampling")
        call_names = {_call_name(call) for call in ast.walk(method) if isinstance(call, ast.Call)}
        self.assertIn("torch.randint", call_names)
        self.assertIn("sample_uniform", call_names)

    def test_resume_counts_assignments_created_during_environment_initialization(self) -> None:
        function = _module_function(RUNNER_PATH, "_restore_sampling_state")
        call_names = [_call_name(call) for call in ast.walk(function) if isinstance(call, ast.Call)]
        self.assertEqual(call_names.count("command.record_current_sampling_assignments"), 2)

    def test_wandb_step_offset_prevents_fresh_and_resume_regression(self) -> None:
        step_offset = _standalone_module_function(RUNNER_PATH, "_wandb_step_offset")
        self.assertEqual(step_offset(1, 0), 1)
        self.assertEqual(step_offset(102, 100), 2)
        self.assertEqual(step_offset(1, 100), 1)
        self.assertEqual(step_offset(1, 100, resume=True), 2)
        self.assertEqual(step_offset(102, 100, resume=True), 2)

        sync_function = _module_function(RUNNER_PATH, "_sync_wandb_metadata")
        call_names = [_call_name(call) for call in ast.walk(sync_function) if isinstance(call, ast.Call)]
        self.assertEqual(call_names.count("_install_wandb_step_guard"), 1)

    def test_wandb_step_guard_wraps_writer_once(self) -> None:
        class FakeWriter:
            def __init__(self) -> None:
                self.calls: list[tuple[str, int | None]] = []

            def add_scalar(self, tag, scalar_value, global_step=None, walltime=None, new_style=False):
                self.calls.append((tag, global_step))

        class FakeRunner:
            def __init__(self) -> None:
                self.writer = FakeWriter()
                self.current_learning_iteration = 100
                self._wandb_step_guard_installed = False
                self.cfg = {"resume": True}

        class FakeRun:
            step = 102

        class FakeWandb:
            run = FakeRun()

        namespace = {
            "OnPolicyRunner": object,
            "_is_wandb_logger": lambda runner: True,
            "wandb": FakeWandb(),
        }
        _, install_guard = _standalone_module_functions(
            RUNNER_PATH,
            ("_wandb_step_offset", "_install_wandb_step_guard"),
            namespace,
        )
        runner = FakeRunner()
        install_guard(runner)
        install_guard(runner)
        runner.writer.add_scalar("metric", 1.0, global_step=100)
        runner.writer.add_scalar("no_step", 2.0)

        self.assertEqual(runner._wandb_step_offset, 2)
        self.assertEqual(runner.writer.calls, [("metric", 102), ("no_step", None)])


if __name__ == "__main__":
    unittest.main()
