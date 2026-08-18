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
sys.modules.setdefault("whole_body_tracking", package)
sys.modules.setdefault("whole_body_tracking.utils", utils_package)

from whole_body_tracking.utils.joint_gap import (  # noqa: E402
    aggregate_topk_joint_gap,
    compute_joint_bin_statistics,
    compute_joint_gap_correction,
    compute_joint_global_gap,
    compute_joint_local_gap,
    compute_raw_gate,
)
from whole_body_tracking.utils.online_learning import (  # noqa: E402
    OnlineLearningController,
    canonical_joint_mapping_hash,
)
from whole_body_tracking.utils.online_learning_stats import JointGapStatistics  # noqa: E402


def _settings(*, lambda_joint: float = 0.2) -> dict[str, object]:
    joint_names = ("hip", "knee", "ankle")
    return {
        "ema_decay": 0.5,
        "warmup_iterations": 0,
        "probability_update_interval": 1,
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
        "joint_gap": {
            "enabled": True,
            "lambda_joint": lambda_joint,
            "num_joints": 3,
            "ema_decay": 0.5,
            "update_interval": 1,
            "min_observations": 1,
            "num_difficulty_bins": 2,
            "positive_only": True,
            "local_center": "motion_median",
            "aggregation": "topk_mean",
            "top_k": 2,
            "raw_gate": {"enabled": True, "mode": "within_motion_median"},
            "transform": "tanh",
            "joint_names": list(joint_names),
            "joint_mapping_hash": canonical_joint_mapping_hash(joint_names),
            "difficulty_metadata_identity": {"metadata_sha256": "fixture"},
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


def _controller(*, lambda_joint: float = 0.2) -> OnlineLearningController:
    return OnlineLearningController(
        num_envs=2,
        motion_lengths=torch.tensor([10, 10]),
        segment_motion_ids=torch.tensor([0, 0, 1, 1]),
        segment_start_frames=torch.tensor([0, 5, 0, 5]),
        segment_end_frames=torch.tensor([5, 10, 5, 10]),
        motion_mode="raw_error",
        segment_mode="raw_error_joint_gap",
        settings=_settings(lambda_joint=lambda_joint),
        motion_eligible_mask=torch.ones(2, dtype=torch.bool),
        segment_eligible_mask=torch.ones(4, dtype=torch.bool),
        difficulty_bins=torch.tensor([0, 0, 1, 1]),
        device="cpu",
    )


def _seed_controller(controller: OnlineLearningController) -> None:
    stats = controller.statistics
    stats.segment_body_error_ema.copy_(torch.tensor([0.1, 0.6, 0.2, 0.8]))
    stats.segment_joint_error_ema.copy_(torch.tensor([0.1, 0.6, 0.2, 0.8]))
    stats.segment_orientation_error_ema.copy_(torch.tensor([0.1, 0.6, 0.2, 0.8]))
    for name in ("body_error", "joint_error", "orientation_error"):
        getattr(stats, f"segment_{name}_initialized").fill_(True)
    stats.segment_step_count.fill_(4)
    stats.motion_step_count.copy_(torch.tensor([8, 8]))
    stats.total_step_observations.fill_(16)
    stats.motion_episode_count.fill_(2)
    stats.total_motion_episodes.fill_(4)
    for name in ("termination", "completion", "success"):
        getattr(stats, f"motion_{name}_ema").fill_(0.0)
        getattr(stats, f"motion_{name}_initialized").fill_(True)
    joint_stats = stats.joint_gap
    assert joint_stats is not None
    joint_stats.segment_joint_error_ema.copy_(
        torch.tensor(
            [
                [0.1, 1.0, 1.0],
                [0.9, 1.2, 1.1],
                [0.2, 0.5, 3.0],
                [1.0, 0.7, 3.8],
            ]
        )
    )
    joint_stats.segment_joint_error_initialized.fill_(True)


class JointGapFormulaTest(unittest.TestCase):
    def test_bin_statistics_are_per_bin_per_joint(self) -> None:
        result = compute_joint_bin_statistics(
            torch.tensor([[1.0, 10.0], [3.0, 30.0], [5.0, 50.0], [7.0, 70.0]]),
            torch.ones(4, dtype=torch.bool),
            torch.tensor([0, 0, 1, 1]),
            num_bins=2,
            min_bin_valid_segments=1,
            sigma_floor=0.1,
        )
        self.assertEqual(result.mean.shape, (2, 2))
        self.assertEqual(result.sigma.shape, (2, 2))
        self.assertTrue(torch.allclose(result.mean, torch.tensor([[2.0, 20.0], [6.0, 60.0]], dtype=torch.float64)))

    def test_bin_statistics_keep_joint_calibration_independent(self) -> None:
        result = compute_joint_bin_statistics(
            torch.tensor([[1.0, 100.0], [3.0, 300.0]]),
            torch.ones(2, dtype=torch.bool),
            torch.tensor([0, 0]),
            num_bins=1,
            min_bin_valid_segments=1,
            sigma_floor=0.1,
        )
        self.assertAlmostEqual(result.mean[0, 0].item(), 2.0)
        self.assertAlmostEqual(result.mean[0, 1].item(), 200.0)
        self.assertNotAlmostEqual(result.sigma[0, 0].item(), result.sigma[0, 1].item())

    def test_sparse_bin_falls_back_per_joint_and_floors_sigma(self) -> None:
        result = compute_joint_bin_statistics(
            torch.tensor([[1.0, 1.0], [1.05, 2.0], [5.0, 8.0]]),
            torch.ones(3, dtype=torch.bool),
            torch.tensor([0, 0, 1]),
            num_bins=2,
            min_bin_valid_segments=2,
            sigma_floor=0.1,
        )
        self.assertTrue(result.fallback_mask[1, 0].item())
        self.assertTrue(torch.all(result.sigma >= 0.1))
        self.assertEqual(result.global_valid_count.tolist(), [3, 3])

    def test_global_gap_uses_matching_difficulty_bin_and_joint(self) -> None:
        errors = torch.tensor([[2.0, 20.0], [7.0, 70.0]])
        calibration = compute_joint_bin_statistics(
            torch.tensor([[1.0, 10.0], [3.0, 30.0], [5.0, 50.0], [9.0, 90.0]]),
            torch.ones(4, dtype=torch.bool),
            torch.tensor([0, 0, 1, 1]),
            num_bins=2,
            min_bin_valid_segments=1,
            sigma_floor=0.1,
        )
        gap, valid, clipped = compute_joint_global_gap(
            errors,
            torch.ones(2, dtype=torch.bool),
            torch.tensor([0, 1]),
            calibration,
            gap_clip=5.0,
        )
        self.assertTrue(torch.all(valid))
        self.assertFalse(torch.any(clipped))
        self.assertAlmostEqual(gap[0, 0].item(), 0.0)
        self.assertAlmostEqual(gap[1, 1].item(), 0.0)

    def test_local_gap_is_centered_by_motion_median(self) -> None:
        result = compute_joint_local_gap(
            torch.tensor([[1.0, 0.0], [3.0, 2.0], [5.0, 4.0], [10.0, 0.0]]),
            torch.ones(4, 2, dtype=torch.bool),
            torch.tensor([0, 0, 0, 1]),
            num_motions=2,
            gap_clip=5.0,
        )
        self.assertTrue(torch.allclose(result.motion_center[0], torch.tensor([3.0, 2.0], dtype=torch.float64)))
        self.assertEqual(result.motion_valid_count[0].tolist(), [3, 3])
        self.assertFalse(torch.any(result.local_valid[3]))

    def test_negative_local_gap_contributes_zero_to_topk(self) -> None:
        result = aggregate_topk_joint_gap(
            torch.tensor([[-1.0, 0.0, 2.0], [0.5, 0.25, -3.0]]),
            torch.ones(2, 3, dtype=torch.bool),
            top_k=2,
        )
        self.assertAlmostEqual(result.segment_gap_score[0].item(), 1.0)
        self.assertAlmostEqual(result.segment_gap_score[1].item(), 0.375)

    def test_topk_aggregation_is_joint_name_agnostic(self) -> None:
        first = aggregate_topk_joint_gap(torch.tensor([[1.0, 4.0, 2.0]]), torch.ones(1, 3, dtype=torch.bool), top_k=2)
        second = aggregate_topk_joint_gap(torch.tensor([[4.0, 2.0, 1.0]]), torch.ones(1, 3, dtype=torch.bool), top_k=2)
        self.assertAlmostEqual(first.segment_gap_score.item(), second.segment_gap_score.item())

    def test_raw_gate_keeps_ties_at_within_motion_median(self) -> None:
        gate, median, count = compute_raw_gate(
            torch.tensor([1.0, 2.0, 2.0, 9.0]),
            torch.ones(4, dtype=torch.bool),
            torch.tensor([0, 0, 0, 1]),
            num_motions=2,
        )
        self.assertEqual(median[0].item(), 2.0)
        self.assertEqual(count.tolist(), [3, 1])
        self.assertEqual(gate.tolist(), [False, True, True, True])

    def test_correction_is_bounded_and_never_lowers_raw_priority(self) -> None:
        raw = torch.tensor([1.0, 2.0, 3.0, 4.0])
        calibration = compute_joint_bin_statistics(
            torch.tensor([[0.1, 0.1], [0.1, 0.2], [0.1, 0.1], [0.1, 0.2]]),
            torch.ones(4, dtype=torch.bool),
            torch.tensor([0, 0, 1, 1]),
            num_bins=2,
            min_bin_valid_segments=1,
            sigma_floor=0.1,
        )
        corrected = compute_joint_gap_correction(
            raw_priority=raw,
            raw_valid=torch.ones(4, dtype=torch.bool),
            segment_joint_error=torch.tensor([[0.1, 0.1], [3.0, 3.0], [0.1, 0.1], [4.0, 4.0]]),
            segment_joint_valid=torch.ones(4, dtype=torch.bool),
            segment_motion_ids=torch.tensor([0, 0, 1, 1]),
            difficulty_bin=torch.tensor([0, 0, 1, 1]),
            calibration=calibration,
            num_motions=2,
            top_k=2,
            lambda_joint=0.5,
            gap_clip=5.0,
        )
        self.assertTrue(torch.all(corrected.correction >= 0.0))
        self.assertTrue(torch.all(corrected.correction < 1.0))
        self.assertTrue(torch.all(corrected.corrected_priority >= raw))

    def test_lambda_zero_is_exact_raw_identity(self) -> None:
        raw = torch.tensor([1.0, 2.0], dtype=torch.float64)
        calibration = compute_joint_bin_statistics(
            torch.tensor([[0.1, 0.1], [0.2, 0.2]]),
            torch.ones(2, dtype=torch.bool),
            torch.tensor([0, 0]),
            num_bins=1,
            min_bin_valid_segments=1,
            sigma_floor=0.1,
        )
        result = compute_joint_gap_correction(
            raw_priority=raw,
            raw_valid=torch.ones(2, dtype=torch.bool),
            segment_joint_error=torch.tensor([[0.1, 0.1], [3.0, 3.0]]),
            segment_joint_valid=torch.ones(2, dtype=torch.bool),
            segment_motion_ids=torch.tensor([0, 0]),
            difficulty_bin=torch.tensor([0, 0]),
            calibration=calibration,
            num_motions=1,
            top_k=2,
            lambda_joint=0.0,
            gap_clip=5.0,
        )
        self.assertTrue(torch.equal(result.corrected_priority, raw))

    def test_cold_invalid_and_nan_segments_get_zero_correction(self) -> None:
        raw = torch.tensor([1.0, 2.0])
        calibration = compute_joint_bin_statistics(
            torch.tensor([[0.1, 0.1], [0.2, 0.2]]),
            torch.ones(2, dtype=torch.bool),
            torch.tensor([0, 0]),
            num_bins=1,
            min_bin_valid_segments=1,
            sigma_floor=0.1,
        )
        result = compute_joint_gap_correction(
            raw_priority=raw,
            raw_valid=torch.tensor([True, False]),
            segment_joint_error=torch.tensor([[float("nan"), 0.1], [3.0, 3.0]]),
            segment_joint_valid=torch.tensor([False, True]),
            segment_motion_ids=torch.tensor([0, 0]),
            difficulty_bin=torch.tensor([0, 0]),
            calibration=calibration,
            num_motions=1,
            top_k=2,
            lambda_joint=0.5,
            gap_clip=5.0,
        )
        self.assertEqual(result.correction.tolist(), [0.0, 0.0])

    def test_unreliable_bin_and_inf_segments_get_zero_correction(self) -> None:
        calibration = compute_joint_bin_statistics(
            torch.tensor([[0.1, 0.1], [0.2, 0.2]]),
            torch.tensor([False, False]),
            torch.tensor([0, 0]),
            num_bins=1,
            min_bin_valid_segments=1,
            sigma_floor=0.1,
        )
        result = compute_joint_gap_correction(
            raw_priority=torch.tensor([1.0, 2.0]),
            raw_valid=torch.ones(2, dtype=torch.bool),
            segment_joint_error=torch.tensor([[float("inf"), 0.1], [3.0, 3.0]]),
            segment_joint_valid=torch.ones(2, dtype=torch.bool),
            segment_motion_ids=torch.tensor([0, 0]),
            difficulty_bin=torch.tensor([0, 0]),
            calibration=calibration,
            num_motions=1,
            top_k=2,
            lambda_joint=0.5,
            gap_clip=5.0,
        )
        self.assertEqual(result.correction.tolist(), [0.0, 0.0])


class JointGapStatisticsTest(unittest.TestCase):
    def test_per_joint_ema_shape_and_shared_segment_count(self) -> None:
        stats = JointGapStatistics(3, 2, ema_decay=0.5, joint_names=("a", "b"))
        stats.record_step_observations([0, 0, 2], [[1.0, 2.0], [3.0, 6.0], [5.0, 10.0]])
        self.assertEqual(stats.segment_joint_error_ema.shape, (3, 2))
        self.assertEqual(stats.segment_joint_pending_count.tolist(), [2, 0, 1])

    def test_per_joint_ema_math_uses_one_update_per_segment(self) -> None:
        stats = JointGapStatistics(2, 2, ema_decay=0.5, joint_names=("a", "b"))
        stats.record_step_observations([0, 0], [[1.0, 3.0], [3.0, 5.0]])
        self.assertTrue(stats.commit_window())
        self.assertTrue(torch.allclose(stats.segment_joint_error_ema[0], torch.tensor([2.0, 4.0])))
        stats.record_step_observations([0], [[4.0, 8.0]])
        stats.commit_window()
        self.assertTrue(torch.allclose(stats.segment_joint_error_ema[0], torch.tensor([3.0, 6.0])))

    def test_joint_gap_statistics_round_trip_and_mapping_mismatch_fail(self) -> None:
        stats = JointGapStatistics(2, 2, ema_decay=0.5, config_hash="cfg", joint_names=("a", "b"))
        stats.record_step_observations([1], [[2.0, 4.0]])
        stats.commit_window()
        state = stats.state_dict()
        restored = JointGapStatistics(2, 2, ema_decay=0.5, config_hash="cfg", joint_names=("a", "b"))
        restored.load_state_dict(state)
        self.assertTrue(torch.equal(restored.segment_joint_error_ema, stats.segment_joint_error_ema))
        mismatched = JointGapStatistics(2, 2, ema_decay=0.5, config_hash="cfg", joint_names=("b", "a"))
        with self.assertRaisesRegex(ValueError, "joint_mapping_hash"):
            mismatched.load_state_dict(state)


class JointGapControllerTest(unittest.TestCase):
    def test_controller_computes_joint_gap_without_generic_gap(self) -> None:
        controller = _controller(lambda_joint=0.2)
        _seed_controller(controller)
        self.assertTrue(controller.on_iteration_end(0))
        self.assertIsNotNone(controller.joint_gap_result)
        self.assertIsNone(controller.gap_result)
        self.assertIsNone(controller.bin_calibration)
        metrics = controller.metrics()
        self.assertIn("joint_gap/gap_mean", metrics)
        self.assertIn("joint_gap/segment_top1_mass", metrics)

    def test_controller_lambda_zero_keeps_raw_sampler_score_identity(self) -> None:
        controller = _controller(lambda_joint=0.0)
        _seed_controller(controller)
        controller.on_iteration_end(0)
        self.assertTrue(torch.equal(controller.joint_gap_result.corrected_priority, controller.segment_error_result.error))

    def test_joint_gap_checkpoint_round_trip(self) -> None:
        controller = _controller(lambda_joint=0.2)
        _seed_controller(controller)
        controller.on_iteration_end(0)
        state = controller.state_dict()
        expected = controller.sample(32)
        restored = _controller(lambda_joint=0.2)
        restored.load_state_dict(state)
        actual = restored.sample(32)
        self.assertTrue(all(torch.equal(left, right) for left, right in zip(expected, actual)))
        self.assertTrue(torch.equal(restored.joint_gap_result.correction, controller.joint_gap_result.correction))

    def test_joint_gap_resume_requires_joint_gap_namespace(self) -> None:
        controller = _controller(lambda_joint=0.2)
        _seed_controller(controller)
        controller.on_iteration_end(0)
        state = controller.state_dict()
        state.pop("joint_gap")
        with self.assertRaisesRegex(ValueError, "Joint-gap resume requires"):
            _controller(lambda_joint=0.2).load_state_dict(state)

    def test_joint_gap_checkpoint_mapping_mismatch_fails(self) -> None:
        controller = _controller(lambda_joint=0.2)
        _seed_controller(controller)
        controller.on_iteration_end(0)
        state = controller.state_dict()
        state["joint_gap"]["joint_mapping_identity"]["joint_mapping_hash"] = "bad"
        with self.assertRaisesRegex(ValueError, "mapping hash"):
            _controller(lambda_joint=0.2).load_state_dict(state)

    def test_raw_error_joint_gap_requires_joint_gap_enabled(self) -> None:
        settings = _settings(lambda_joint=0.2)
        settings["joint_gap"] = {"enabled": False}
        with self.assertRaisesRegex(ValueError, "joint_gap.enabled"):
            OnlineLearningController(
                num_envs=1,
                motion_lengths=torch.tensor([10]),
                segment_motion_ids=torch.tensor([0, 0]),
                segment_start_frames=torch.tensor([0, 5]),
                segment_end_frames=torch.tensor([5, 10]),
                motion_mode="raw_error",
                segment_mode="raw_error_joint_gap",
                settings=settings,
                motion_eligible_mask=torch.ones(1, dtype=torch.bool),
                segment_eligible_mask=torch.ones(2, dtype=torch.bool),
                difficulty_bins=torch.tensor([0, 1]),
                device="cpu",
            )


if __name__ == "__main__":
    unittest.main()
