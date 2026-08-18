from __future__ import annotations

import csv
import math
import sys
import tempfile
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "rsl_rl"))

import evaluation_utils as eval_utils  # noqa: E402
import joint_diagnostics as joint_diag  # noqa: E402


class JointDiagnosticsTest(unittest.TestCase):
    def test_joint_mapping_has_29_ordered_rows(self):
        names = [f"joint_{index}" for index in range(29)]
        rows = joint_diag.joint_mapping_rows(
            names,
            reference_joint_names=names,
            action_joint_names=names,
            evaluator_joint_names=names,
        )
        self.assertEqual(len(rows), 29)
        self.assertEqual([row["joint_index"] for row in rows], list(range(29)))
        self.assertTrue(all(row["reference_matches_robot"] == 1 for row in rows))
        self.assertTrue(all(row["action_matches_robot"] == 1 for row in rows))
        self.assertTrue(all(row["evaluator_matches_robot"] == 1 for row in rows))

    def test_aggregate_l2_matches_per_joint_vector(self):
        joint_error = torch.zeros(2, 29, dtype=torch.float64)
        joint_error[0, 0] = 3.0
        joint_error[0, 1] = 4.0
        joint_error[1, :] = 1.0
        aggregate = torch.tensor([5.0, math.sqrt(29.0)], dtype=torch.float64)
        self.assertEqual(joint_diag.aggregate_l2_max_abs_error(joint_error, aggregate), 0.0)

    def test_accumulator_writes_per_joint_and_segment_outputs(self):
        names = [f"joint_{index}" for index in range(29)]
        layout = joint_diag.SegmentLayout(
            segment_motion_ids=torch.tensor([0, 0, 1], dtype=torch.long),
            segment_local_ids=torch.tensor([0, 1, 0], dtype=torch.long),
            segment_start_frames=torch.tensor([0, 50, 0], dtype=torch.long),
            segment_end_frames=torch.tensor([50, 100, 50], dtype=torch.long),
        )
        accumulator = joint_diag.JointDiagnosticsAccumulator(
            motion_paths=["a.npz", "b.npz"],
            categories=["cat_a", "cat_b"],
            source_groups=["group_a", "group_b"],
            motion_lengths=[100, 50],
            joint_names=names,
            device="cpu",
            segment_layout=layout,
        )
        errors = torch.zeros(3, 29, dtype=torch.float64)
        errors[0, 0] = 1.0
        errors[1, 1] = 2.0
        errors[2, 2] = 3.0
        accumulator.record_step(
            motion_ids=torch.tensor([0, 0, 1]),
            segment_ids=torch.tensor([0, 1, 2]),
            joint_error=errors,
            aggregate_l2=torch.linalg.vector_norm(errors, dim=-1),
        )
        mapping = joint_diag.joint_mapping_rows(names)
        with tempfile.TemporaryDirectory() as tmp:
            summary = accumulator.write_outputs(tmp, mapping_rows=mapping, evaluation_config={})
            self.assertEqual(summary["joint_count"], 29)
            self.assertEqual(summary["max_l2_consistency_error"], 0.0)
            per_joint_path = Path(tmp) / "per_joint_summary.csv"
            with per_joint_path.open(newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 29)
            self.assertAlmostEqual(float(rows[0]["rms_error"]), math.sqrt(1.0 / 3.0), places=6)
            self.assertTrue((Path(tmp) / "per_segment_joint_diagnostics.csv").exists())

    def test_default_evaluator_per_motion_columns_do_not_include_diagnostics(self):
        self.assertEqual(
            eval_utils.PER_MOTION_COLUMNS,
            (
                "motion_path",
                "category",
                "source_group",
                "num_frames",
                "completed_frames",
                "episode_steps",
                "success",
                "completion_ratio",
                "body_position_error_m",
                "joint_position_error_l2_rad",
                "joint_position_error_rms_rad",
                "termination_reason",
                "checkpoint",
            ),
        )
        self.assertFalse(any(column.startswith("joint_00_") for column in eval_utils.PER_MOTION_COLUMNS))


if __name__ == "__main__":
    unittest.main()
