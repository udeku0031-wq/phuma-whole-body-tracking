from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "source" / "whole_body_tracking"
PACKAGE_DIR = PACKAGE_ROOT / "whole_body_tracking"
package = types.ModuleType("whole_body_tracking")
package.__path__ = [str(PACKAGE_DIR)]
utils_package = types.ModuleType("whole_body_tracking.utils")
utils_package.__path__ = [str(PACKAGE_DIR / "utils")]
sys.modules["whole_body_tracking"] = package
sys.modules["whole_body_tracking.utils"] = utils_package

from whole_body_tracking.utils.online_learning import OnlineLearningController  # noqa: E402


def _settings(*, interval: int = 2) -> dict[str, object]:
    return {
        "ema_decay": 0.5,
        "warmup_iterations": 0,
        "probability_update_interval": interval,
        "min_segment_observations": 1,
        "min_motion_episodes": 1,
        "min_bin_valid_segments": 1,
        "minimum_segment_observed_fraction": 0.2,
        "sigma_floor": 0.1,
        "gap_clip": 5.0,
        "score_clip": 10.0,
        "sampler_seed": 42,
        "bin_observation_weighted": False,
        "num_difficulty_bins": 2,
        "body_position_scale_m": 1.0,
        "joint_position_scale_rad": 1.0,
        "orientation_scale_rad": 1.0,
        "component_clip": 5.0,
        "error_weights": {
            "body": 1.0,
            "joint": 1.0,
            "orientation": 1.0,
            "termination": 1.0,
            "completion": 0.5,
            "success": 0.5,
        },
        "motion_error_weights": {
            "segment_mean": 1.0,
            "segment_p90": 0.25,
            "termination": 0.5,
            "completion": 0.5,
            "success": 0.5,
        },
        "motion_gap_weights": {
            "positive_mean": 1.0,
            "positive_p90": 0.5,
            "termination": 0.5,
            "completion": 0.5,
            "success": 0.5,
        },
        "uniform_mix": 0.15,
        "temperature": 1.0,
        "under_sampling_weight": 0.25,
        "motion_probability_cap": 0.9,
        "segment_probability_cap": 0.9,
        "fallback": "uniform",
        "provisional": True,
        "statistics_enabled": True,
    }


def _controller(*, interval: int = 2) -> OnlineLearningController:
    return OnlineLearningController(
        num_envs=2,
        motion_lengths=torch.tensor([10, 10]),
        segment_motion_ids=torch.tensor([0, 0, 1, 1]),
        segment_start_frames=torch.tensor([0, 5, 0, 5]),
        segment_end_frames=torch.tensor([5, 10, 5, 10]),
        motion_mode="raw_error",
        segment_mode="raw_error",
        settings=_settings(interval=interval),
        motion_eligible_mask=torch.ones(2, dtype=torch.bool),
        segment_eligible_mask=torch.ones(4, dtype=torch.bool),
        difficulty_bins=None,
        device="cpu",
    )


class OnlineLearningControllerTest(unittest.TestCase):
    def _record_complete_assignments(self, controller: OnlineLearningController) -> None:
        env_ids = torch.tensor([0, 1])
        controller.begin_assignments(
            env_ids,
            torch.tensor([0, 1]),
            torch.tensor([8, 8]),
            torch.tensor([1, 3]),
        )
        controller.observe_steps(
            torch.tensor([0, 1]),
            torch.tensor([1, 3]),
            body_error=torch.tensor([1.0, 4.0]),
            joint_error=torch.tensor([1.0, 4.0]),
            orientation_error=torch.tensor([1.0, 4.0]),
        )
        controller.finish_assignments(
            env_ids,
            terminated=torch.tensor([False, False]),
            natural_completion=torch.tensor([True, True]),
            timed_out=torch.tensor([False, False]),
        )

    def test_iteration_cadence_updates_ema_every_time_but_formulas_low_frequency(self) -> None:
        controller = _controller(interval=2)
        self._record_complete_assignments(controller)
        self.assertTrue(controller.on_iteration_end(0))
        first_formula_iteration = controller.last_formula_update_iteration
        first_update_count = controller.sampler.probability_update_count

        controller.statistics.record_step_observations(
            [0], [1], body_error=[3.0], joint_error=[3.0], orientation_error=[3.0]
        )
        self.assertFalse(controller.on_iteration_end(1))
        self.assertEqual(controller.last_formula_update_iteration, first_formula_iteration)
        self.assertEqual(controller.sampler.probability_update_count, first_update_count)
        self.assertEqual(float(controller.statistics.segment_body_error_ema[1]), 2.0)

        self.assertTrue(controller.on_iteration_end(2))
        self.assertEqual(controller.last_formula_update_iteration, 2)
        self.assertEqual(controller.sampler.probability_update_count, first_update_count + 1)

    def test_full_state_and_sampler_rng_round_trip(self) -> None:
        controller = _controller()
        self._record_complete_assignments(controller)
        controller.on_iteration_end(0)
        state = controller.state_dict()
        expected = controller.sample(64)

        restored = _controller()
        restored.load_state_dict(state)
        actual = restored.sample(64)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(expected, actual)))
        self.assertTrue(
            torch.equal(
                controller.statistics.segment_step_count,
                restored.statistics.segment_step_count,
            )
        )
        self.assertTrue(
            torch.equal(
                controller.sampler.motion_probability,
                restored.sampler.motion_probability,
            )
        )
        self.assertEqual(
            restored.last_formula_update_iteration,
            controller.last_formula_update_iteration,
        )
        self.assertEqual(restored.completed_window_count, controller.completed_window_count)

    def test_resume_with_repeated_runner_iteration_keeps_window_cadence(self) -> None:
        uninterrupted = _controller(interval=2)
        self._record_complete_assignments(uninterrupted)
        uninterrupted.on_iteration_end(0)
        state = uninterrupted.state_dict()
        uninterrupted.on_iteration_end(1)

        resumed = _controller(interval=2)
        resumed.load_state_dict(state)
        resumed.on_iteration_end(0)
        self.assertEqual(resumed.completed_window_count, uninterrupted.completed_window_count)
        self.assertEqual(
            resumed.sampler.probability_update_count,
            uninterrupted.sampler.probability_update_count,
        )
        self.assertEqual(
            resumed.last_formula_update_iteration,
            uninterrupted.last_formula_update_iteration,
        )

    def test_checkpoint_rejects_corrupt_derived_cache(self) -> None:
        controller = _controller()
        self._record_complete_assignments(controller)
        controller.on_iteration_end(0)
        state = controller.state_dict()
        state["segment_error_result"]["error"] = torch.zeros(3)
        with self.assertRaisesRegex(ValueError, "wrong shape"):
            _controller().load_state_dict(state)

    def test_timeout_on_segment_last_frame_completes_segment_but_censors_motion(self) -> None:
        controller = _controller()
        controller.begin_assignments(
            torch.tensor([0]), torch.tensor([0]), torch.tensor([4]), torch.tensor([0])
        )
        controller.observe_steps(
            torch.tensor([0, 0]),
            torch.tensor([0, 0]),
            body_error=torch.tensor([1.0, 0.0]),
            joint_error=torch.tensor([1.0, 0.0]),
            orientation_error=torch.tensor([1.0, 0.0]),
            env_ids=torch.tensor([0]),
        )
        controller.finish_assignments(
            torch.tensor([0]),
            terminated=torch.tensor([False]),
            natural_completion=torch.tensor([False]),
            timed_out=torch.tensor([True]),
            segment_natural_completion=torch.tensor([True]),
        )
        controller.statistics.commit_window()
        self.assertEqual(controller.statistics.segment_completion_ema[0].item(), 1.0)
        self.assertEqual(controller.statistics.segment_success_ema[0].item(), 1.0)
        self.assertFalse(controller.statistics.motion_completion_initialized[0].item())
        self.assertFalse(controller.statistics.motion_success_initialized[0].item())

    def test_selected_terminal_components_need_not_allocate_full_env_vectors(self) -> None:
        controller = _controller()
        controller.begin_assignments(
            torch.tensor([0, 1]),
            torch.tensor([0, 1]),
            torch.tensor([8, 8]),
            torch.tensor([1, 3]),
        )
        controller.observe_steps(
            torch.tensor([0, 1]),
            torch.tensor([1, 3]),
            body_error=torch.tensor([2.0]),
            joint_error=torch.tensor([3.0]),
            orientation_error=torch.tensor([4.0]),
            env_ids=torch.tensor([1]),
        )
        self.assertEqual(controller.statistics.total_step_observations.item(), 1)
        self.assertEqual(controller.statistics.segment_step_count.tolist(), [0, 0, 0, 1])


if __name__ == "__main__":
    unittest.main()
