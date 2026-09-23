from __future__ import annotations

import ast
import importlib.util
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLING_PATH = (
    PROJECT_ROOT / "source" / "whole_body_tracking" / "whole_body_tracking" / "utils" / "sampling.py"
)

_SAMPLING_SPEC = importlib.util.spec_from_file_location("wbt_sampling_quality_gate_tests", SAMPLING_PATH)
if _SAMPLING_SPEC is None or _SAMPLING_SPEC.loader is None:
    raise RuntimeError(f"Unable to load sampling utilities from {SAMPLING_PATH}")
sampling = importlib.util.module_from_spec(_SAMPLING_SPEC)
_SAMPLING_SPEC.loader.exec_module(sampling)


def _call_leaf_name(call: ast.Call) -> str | None:
    node: ast.expr = call.func
    while isinstance(node, ast.Attribute):
        if isinstance(node.value, ast.Name):
            return node.attr
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


class QualityGatedStartIndexTest(unittest.TestCase):
    def test_all_allowed_matches_legacy_start_mapping(self) -> None:
        index = sampling.FixedLengthSegmentIndex([101, 51, 2], [50.0, 50.0, 50.0])
        gate = sampling.QualityGatedStartIndex(
            index, torch.ones(index.num_segments, dtype=torch.bool)
        )
        motion_ids = torch.tensor(
            [
                [0, 0, 0, 0],
                [1, 1, 1, 1],
                [2, 2, 2, 2],
            ]
        )
        uniform_samples = torch.tensor(
            [
                [0.0, 0.01, 0.5, 0.999999],
                [0.0, 0.01, 0.5, 0.999999],
                [0.0, 0.01, 0.5, 0.999999],
            ],
            dtype=torch.float32,
        )

        start_frames, local_segment_ids, global_segment_ids = gate.map_uniform_samples(
            motion_ids, uniform_samples
        )
        legacy_starts = (
            uniform_samples * (index.motion_lengths[motion_ids] - 1).to(torch.float32)
        ).long()
        expected_local_ids, expected_global_ids = index.motion_frame_to_segment(
            motion_ids, legacy_starts
        )

        self.assertTrue(torch.equal(start_frames, legacy_starts))
        self.assertTrue(torch.equal(local_segment_ids, expected_local_ids))
        self.assertTrue(torch.equal(global_segment_ids, expected_global_ids))
        self.assertEqual(gate.motion_eligible_start_counts.tolist(), [100, 50, 1])

    def test_rejected_middle_segment_is_skipped_and_one_frame_tail_has_no_start(self) -> None:
        index = sampling.FixedLengthSegmentIndex([151], [50.0])
        # Segments are [0,50), [50,100), [100,150), [150,151).  The final
        # segment is allowed but contains only legacy-invalid frame T-1.
        gate = sampling.QualityGatedStartIndex(index, [True, False, True, True])

        start_frames, local_segment_ids, global_segment_ids = gate.map_uniform_samples(
            [0, 0, 0, 0], torch.tensor([0.0, 0.49, 0.5, 0.999999])
        )

        self.assertEqual(start_frames.tolist(), [0, 49, 100, 149])
        self.assertEqual(local_segment_ids.tolist(), [0, 0, 2, 2])
        self.assertEqual(global_segment_ids.tolist(), [0, 0, 2, 2])
        self.assertEqual(gate.segment_eligible_start_counts.tolist(), [50, 0, 50, 0])
        self.assertEqual(gate.motion_eligible_segment_counts.tolist(), [2])
        self.assertEqual(
            gate.summary(),
            {
                "num_motions": 1,
                "num_segments": 4,
                "num_allowed_segments": 3,
                "num_eligible_segments": 2,
                "num_eligible_motions": 1,
                "num_empty_motions": 0,
                "num_excluded_motions": 0,
                "num_legacy_start_frames": 150,
                "num_eligible_start_frames": 100,
                "eligible_motion_fraction": 1.0,
                "eligible_start_fraction": 2.0 / 3.0,
            },
        )

    def test_global_prefix_mapping_is_independent_for_each_motion(self) -> None:
        index = sampling.FixedLengthSegmentIndex([61, 91], [30.0, 60.0])
        # Motion 0 keeps [30,60); motion 1 keeps [0,60).
        gate = sampling.QualityGatedStartIndex(index, [False, True, True, True, False])

        start_frames, local_ids, global_ids = gate.map_uniform_samples(
            torch.tensor([0, 0, 1, 1]),
            torch.tensor([0.0, 0.999999, 0.0, 0.999999]),
        )

        self.assertEqual(start_frames.tolist(), [30, 59, 0, 59])
        self.assertEqual(local_ids.tolist(), [1, 1, 0, 0])
        self.assertEqual(global_ids.tolist(), [1, 1, 3, 3])
        self.assertEqual(gate.motion_eligible_start_counts.tolist(), [30, 60])
        self.assertEqual(gate.motion_eligible_start_offsets.tolist(), [0, 30, 90])

    def test_empty_motion_raises_by_default(self) -> None:
        index = sampling.FixedLengthSegmentIndex([51, 51], [50.0, 50.0])

        with self.assertRaisesRegex(ValueError, r"empty motion IDs: \[0\]"):
            sampling.QualityGatedStartIndex(index, [False, False, True, True])

        one_frame_index = sampling.FixedLengthSegmentIndex([1], [50.0])
        with self.assertRaisesRegex(ValueError, r"empty motion IDs: \[0\]"):
            sampling.QualityGatedStartIndex(one_frame_index, [True])

    def test_exclude_policy_exposes_eligible_motions_and_rejects_empty_mapping(self) -> None:
        index = sampling.FixedLengthSegmentIndex([51, 51], [50.0, 50.0])
        gate = sampling.QualityGatedStartIndex(
            index,
            [False, False, True, True],
            empty_motion_policy="exclude",
        )

        self.assertEqual(gate.eligible_motion_ids.tolist(), [1])
        self.assertEqual(gate.empty_motion_ids.tolist(), [0])
        self.assertEqual(gate.summary()["num_empty_motions"], 1)
        self.assertEqual(gate.summary()["num_excluded_motions"], 1)
        self.assertEqual(gate.summary()["num_eligible_motions"], 1)
        self.assertEqual(gate.summary()["eligible_motion_fraction"], 0.5)
        self.assertEqual(gate.identity_state()["empty_motion_policy"], "exclude")
        self.assertEqual(gate.identity_state()["effective_motion_count"], 1)
        self.assertEqual(len(gate.identity_state()["eligible_motion_mask_sha256"]), 64)
        start_frames, local_ids, global_ids = gate.map_uniform_samples([1], [0.5])
        self.assertEqual(start_frames.tolist(), [25])
        self.assertEqual(local_ids.tolist(), [0])
        self.assertEqual(global_ids.tolist(), [2])

        with self.assertRaisesRegex(ValueError, r"motion IDs: \[0\]"):
            gate.map_uniform_samples([0], [0.5])

    def test_mask_and_policy_validation(self) -> None:
        index = sampling.FixedLengthSegmentIndex([51], [50.0])

        with self.assertRaisesRegex(ValueError, "boolean"):
            sampling.QualityGatedStartIndex(index, [1, 0])
        with self.assertRaisesRegex(ValueError, "exactly 2 values"):
            sampling.QualityGatedStartIndex(index, [True])
        with self.assertRaisesRegex(ValueError, "one-dimensional"):
            sampling.QualityGatedStartIndex(index, torch.ones((1, 2), dtype=torch.bool))
        with self.assertRaisesRegex(ValueError, "empty_motion_policy"):
            sampling.QualityGatedStartIndex(index, [True, True], empty_motion_policy="ignore")

    def test_uniform_input_validation_and_empty_batch(self) -> None:
        index = sampling.FixedLengthSegmentIndex([51], [50.0])
        gate = sampling.QualityGatedStartIndex(index, [True, True])

        with self.assertRaisesRegex(ValueError, "same shape"):
            gate.map_uniform_samples([0, 0], [0.5])
        with self.assertRaisesRegex(ValueError, "motion_ids"):
            gate.map_uniform_samples([1], [0.5])
        with self.assertRaisesRegex(ValueError, "floating-point"):
            gate.map_uniform_samples([0], [0])
        for value in (-0.1, 1.0, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, r"\[0, 1\)"):
                gate.map_uniform_samples([0], [value])

        starts, local_ids, global_ids = gate.map_uniform_samples(
            torch.empty((0, 2), dtype=torch.long),
            torch.empty((0, 2), dtype=torch.float32),
        )
        self.assertEqual(starts.shape, (0, 2))
        self.assertEqual(local_ids.shape, (0, 2))
        self.assertEqual(global_ids.shape, (0, 2))

    def test_construction_and_mapping_do_not_consume_rng_or_loop(self) -> None:
        torch.manual_seed(20260719)
        before = torch.random.get_rng_state().clone()
        index = sampling.FixedLengthSegmentIndex([151, 91], [50.0, 60.0])
        gate = sampling.QualityGatedStartIndex(index, [True, False, True, True, False, True])
        gate.map_uniform_samples([0, 1], [0.25, 0.75])
        gate.summary()
        after = torch.random.get_rng_state()
        self.assertTrue(torch.equal(before, after))

        module = ast.parse(SAMPLING_PATH.read_text(encoding="utf-8"), filename=str(SAMPLING_PATH))
        gate_class = next(
            node
            for node in module.body
            if isinstance(node, ast.ClassDef) and node.name == "QualityGatedStartIndex"
        )
        self.assertFalse(any(isinstance(node, (ast.For, ast.While)) for node in ast.walk(gate_class)))
        random_names = {"rand", "randint", "randn", "random", "multinomial", "sample_uniform"}
        observed_calls = {
            name
            for call in ast.walk(gate_class)
            if isinstance(call, ast.Call)
            for name in [_call_leaf_name(call)]
            if name in random_names
        }
        self.assertEqual(observed_calls, set())


if __name__ == "__main__":
    unittest.main()
