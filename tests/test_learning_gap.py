from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    PROJECT_ROOT
    / "source"
    / "whole_body_tracking"
    / "whole_body_tracking"
    / "utils"
    / "learning_gap.py"
)
SPEC = importlib.util.spec_from_file_location("wbt_learning_gap_tests", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Unable to load {MODULE_PATH}")
gap = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = gap
SPEC.loader.exec_module(gap)


class ErrorFormulaTest(unittest.TestCase):
    def test_segment_error_matches_hand_calculation_and_ignores_missing_outcome(self) -> None:
        values = {
            "body": torch.tensor([0.3]),
            "joint": torch.tensor([1.0]),
            "orientation": torch.tensor([0.2]),
            "termination": torch.tensor([0.25]),
            "completion": torch.tensor([0.5]),
            "success": torch.tensor([0.0]),
        }
        initialized = {name: torch.ones(1, dtype=torch.bool) for name in values}
        initialized["success"][:] = False
        result = gap.compute_segment_error(
            values,
            initialized,
            torch.tensor([32]),
            min_segment_observations=32,
            body_position_scale_m=0.3,
            joint_position_scale_rad=0.5,
            orientation_scale_rad=0.4,
            weights={name: 1.0 for name in values},
            component_clip=5.0,
        )
        # (1 + 2 + .5 + .25 + .5) / five active components.
        self.assertAlmostEqual(result.error.item(), 0.85)
        self.assertTrue(result.valid.item())
        self.assertEqual(result.contributions["success"].item(), 0.0)

    def test_cold_segment_is_invalid_but_finite(self) -> None:
        values = {name: torch.ones(2) for name in ("body", "joint", "orientation", "termination", "completion", "success")}
        initialized = {name: torch.ones(2, dtype=torch.bool) for name in values}
        result = gap.compute_segment_error(
            values,
            initialized,
            torch.tensor([31, 32]),
            min_segment_observations=32,
            body_position_scale_m=1,
            joint_position_scale_rad=1,
            orientation_scale_rad=1,
            weights={name: 1 for name in values},
            component_clip=5,
        )
        self.assertEqual(result.valid.tolist(), [False, True])
        self.assertTrue(torch.all(torch.isfinite(result.error)))

    def test_motion_error_uses_mean_p90_and_outcomes(self) -> None:
        result = gap.compute_motion_error(
            torch.tensor([1.0, 3.0, 2.0]),
            torch.tensor([True, True, True]),
            torch.tensor([0, 0, 1]),
            torch.tensor([1, 1, 1]),
            {
                "termination": torch.tensor([0.0, 0.0]),
                "completion": torch.tensor([1.0, 1.0]),
                "success": torch.tensor([1.0, 1.0]),
            },
            {name: torch.ones(2, dtype=torch.bool) for name in ("termination", "completion", "success")},
            torch.tensor([8, 8]),
            min_motion_episodes=8,
            weights={"segment_mean": 1, "segment_p90": 1, "termination": 0, "completion": 0, "success": 0},
        )
        # Linear P90 of [1,3] is 2.8; mean is 2.
        self.assertAlmostEqual(result.error[0].item(), 2.4)
        self.assertEqual(result.error[1].item(), 2.0)
        self.assertTrue(torch.all(result.valid))


class DifficultyCalibrationTest(unittest.TestCase):
    def test_bin_mean_std_sparse_backoff_and_zero_sigma_floor(self) -> None:
        result = gap.estimate_difficulty_bin_expectation(
            torch.tensor([1.0, 1.0, 3.0, 5.0]),
            torch.tensor([True, True, True, True]),
            torch.tensor([0, 0, 1, 2]),
            num_bins=3,
            min_bin_valid_segments=2,
            sigma_floor=0.1,
        )
        self.assertEqual(result.valid_segment_count.tolist(), [2, 1, 1])
        self.assertEqual(result.mean[0].item(), 1.0)
        self.assertEqual(result.sigma[0].item(), 0.1)
        self.assertEqual(result.fallback_mask.tolist(), [False, True, True])
        self.assertTrue(torch.all(torch.isfinite(result.mean)))
        self.assertTrue(torch.all(result.sigma >= 0.1))

    def test_high_difficulty_high_error_can_have_small_gap(self) -> None:
        calibration = gap.BinCalibrationResult(
            mean=torch.tensor([1.0, 5.0], dtype=torch.float64),
            sigma=torch.tensor([0.2, 0.5], dtype=torch.float64),
            valid_segment_count=torch.tensor([100, 100]),
            fallback_mask=torch.tensor([False, False]),
            reliable_mask=torch.tensor([True, True]),
            global_mean=torch.tensor(3.0),
            global_sigma=torch.tensor(2.0),
        )
        result = gap.compute_learning_gaps(
            torch.tensor([2.0, 5.0]),
            torch.tensor([True, True]),
            torch.tensor([0, 1]),
            torch.tensor([50, 50]),
            torch.tensor([0, 1]),
            calibration,
            {
                "termination": torch.zeros(2),
                "completion": torch.ones(2),
                "success": torch.ones(2),
            },
            {name: torch.ones(2, dtype=torch.bool) for name in ("termination", "completion", "success")},
            torch.tensor([8, 8]),
            min_motion_episodes=8,
            gap_clip=5,
            motion_gap_weights={"positive_mean": 1, "positive_p90": 0, "termination": 0, "completion": 0, "success": 0},
        )
        self.assertEqual(result.global_gap.tolist(), [5.0, 0.0])
        self.assertGreater(result.global_gap[0], result.global_gap[1])


class RelativeGapTest(unittest.TestCase):
    def test_linear_median_and_translation_invariance(self) -> None:
        values = torch.tensor([-1.0, 1.0, 4.0, 8.0])
        motion_ids = torch.tensor([0, 0, 1, 1])
        valid = torch.ones(4, dtype=torch.bool)
        local = gap.compute_local_relative_gap(
            values, valid, motion_ids, num_motions=2, gap_clip=20
        )
        translated = gap.compute_local_relative_gap(
            values + torch.tensor([7.0, 7.0, -3.0, -3.0]),
            valid,
            motion_ids,
            num_motions=2,
            gap_clip=20,
        )
        self.assertEqual(local.tolist(), [-1.0, 1.0, -2.0, 2.0])
        self.assertTrue(torch.equal(local, translated))

    def test_single_segment_motion_local_gap_is_zero(self) -> None:
        local = gap.compute_local_relative_gap(
            torch.tensor([4.0]),
            torch.tensor([True]),
            torch.tensor([0]),
            num_motions=1,
            gap_clip=5,
        )
        self.assertEqual(local.item(), 0.0)


if __name__ == "__main__":
    unittest.main()
