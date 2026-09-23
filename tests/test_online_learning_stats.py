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
    / "online_learning_stats.py"
)
SPEC = importlib.util.spec_from_file_location("wbt_online_learning_stats_tests", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Unable to load {MODULE_PATH}")
stats_module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = stats_module
SPEC.loader.exec_module(stats_module)


class OnlineLearningStatisticsTest(unittest.TestCase):
    def _stats(self, decay: float = 0.5):
        return stats_module.OnlineLearningStatistics(2, 3, ema_decay=decay, config_hash="fixture")

    def test_batch_aggregation_is_permutation_invariant_with_duplicate_ids(self) -> None:
        fixture = {
            "motion_ids": torch.tensor([0, 1, 0, 0, 1]),
            "segment_ids": torch.tensor([0, 2, 0, 1, 2]),
            "body_error": torch.tensor([1.0, 8.0, 3.0, 5.0, 6.0]),
            "joint_error": torch.tensor([2.0, 7.0, 4.0, 6.0, 5.0]),
            "orientation_error": torch.tensor([0.1, 0.8, 0.3, 0.5, 0.6]),
        }
        first = self._stats()
        first.record_step_observations(**fixture)
        first.commit_window()

        order = torch.tensor([4, 2, 0, 3, 1])
        second = self._stats()
        second.record_step_observations(**{name: value[order] for name, value in fixture.items()})
        second.commit_window()

        for name in (
            "segment_body_error_ema",
            "segment_joint_error_ema",
            "segment_orientation_error_ema",
            "motion_body_error_ema",
            "segment_step_count",
        ):
            self.assertTrue(torch.equal(getattr(first, name), getattr(second, name)), name)
        self.assertEqual(first.segment_body_error_ema.tolist(), [2.0, 5.0, 7.0])

    def test_duplicate_id_gets_one_window_ema_and_first_observation_has_no_zero_bias(self) -> None:
        stats = self._stats(decay=0.5)
        for values in ([1.0, 3.0], [5.0, 7.0]):
            stats.record_step_observations(
                [0, 0],
                [0, 0],
                body_error=values,
                joint_error=values,
                orientation_error=values,
            )
            stats.commit_window()
        # First window initializes to mean=2; second updates once with mean=6.
        self.assertEqual(float(stats.segment_body_error_ema[0]), 4.0)
        self.assertEqual(int(stats.ema_update_count), 2)

    def test_assignment_step_and_outcome_counts_are_distinct(self) -> None:
        stats = self._stats()
        stats.record_assignments([0, 0], [0, 1])
        stats.record_step_observations(
            [0, 0, 0],
            [0, 0, 1],
            body_error=[1, 1, 1],
            joint_error=[1, 1, 1],
            orientation_error=[1, 1, 1],
        )
        stats.record_segment_outcomes(
            [0], termination=[0], completion=[1], success=[1]
        )
        self.assertEqual(stats.segment_sample_count.tolist(), [1, 1, 0])
        self.assertEqual(stats.segment_step_count.tolist(), [2, 1, 0])
        self.assertEqual(stats.segment_outcome_count.tolist(), [1, 0, 0])

    def test_pending_window_round_trip_is_exact(self) -> None:
        source = self._stats()
        source.record_step_observations(
            [0, 1],
            [0, 2],
            body_error=[2.0, 8.0],
            joint_error=[3.0, 7.0],
            orientation_error=[0.2, 0.8],
        )
        restored = self._stats()
        restored.load_state_dict(source.state_dict())
        source.commit_window()
        restored.commit_window()
        self.assertTrue(torch.equal(source.segment_body_error_ema, restored.segment_body_error_ema))
        self.assertTrue(torch.equal(source.segment_step_count, restored.segment_step_count))

    def test_restore_rejects_nonfinite_ema_and_inconsistent_totals(self) -> None:
        source = self._stats()
        source.record_assignments([0], [0])
        state = source.state_dict()
        state["segment_body_error_ema"][0] = float("nan")
        with self.assertRaisesRegex(ValueError, "must be finite"):
            self._stats().load_state_dict(state)

        state = source.state_dict()
        state["total_assignments"] += 1
        with self.assertRaisesRegex(ValueError, "counts disagree"):
            self._stats().load_state_dict(state)


class TraversalOutcomeTest(unittest.TestCase):
    def test_normal_boundary_and_termination_from_middle(self) -> None:
        termination, completion, success = stats_module.segment_traversal_outcomes(
            [30, 10],
            [30, 20],
            [50, 50],
            terminated=[False, True],
            natural_completion=[True, False],
            timed_out=[False, False],
            minimum_observed_fraction=0.2,
        )
        self.assertEqual(termination.tolist(), [0.0, 1.0])
        self.assertEqual(completion.tolist(), [1.0, 0.5])
        self.assertEqual(success.tolist(), [1.0, 0.0])

    def test_too_little_of_segment_does_not_create_success(self) -> None:
        termination, completion, success = stats_module.segment_traversal_outcomes(
            [2],
            [2],
            [50],
            terminated=[False],
            natural_completion=[True],
            timed_out=[False],
            minimum_observed_fraction=0.2,
        )
        self.assertEqual(termination.item(), 0.0)
        self.assertTrue(torch.isnan(completion).item())
        self.assertTrue(torch.isnan(success).item())

    def test_timeout_is_censored_not_failure(self) -> None:
        termination, completion, success = stats_module.motion_episode_outcomes(
            [40],
            [100],
            terminated=[False],
            natural_completion=[False],
            timed_out=[True],
        )
        self.assertEqual(termination.item(), 0.0)
        self.assertTrue(torch.isnan(completion).item())
        self.assertTrue(torch.isnan(success).item())

        termination, completion, success = stats_module.segment_traversal_outcomes(
            [20],
            [50],
            [50],
            terminated=[False],
            natural_completion=[False],
            timed_out=[True],
            minimum_observed_fraction=0.2,
        )
        self.assertEqual(termination.item(), 0.0)
        self.assertTrue(torch.isnan(completion).item())
        self.assertTrue(torch.isnan(success).item())

    def test_motion_natural_completion_and_physical_termination(self) -> None:
        termination, completion, success = stats_module.motion_episode_outcomes(
            [20, 5],
            [20, 20],
            terminated=[False, True],
            natural_completion=[True, False],
            timed_out=[False, False],
        )
        self.assertEqual(termination.tolist(), [0.0, 1.0])
        self.assertEqual(completion.tolist(), [1.0, 0.25])
        self.assertEqual(success.tolist(), [1.0, 0.0])


if __name__ == "__main__":
    unittest.main()
