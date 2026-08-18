from __future__ import annotations

import copy
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

from whole_body_tracking.utils.online_learning import (  # noqa: E402
    OnlineLearningController,
    canonical_joint_mapping_hash,
)


JOINT_NAMES = ("hip", "knee", "ankle")
DIFFICULTY_IDENTITY = {
    "schema_version": "fixture",
    "metadata_sha256": "difficulty-sha",
    "profile_sha256": "profile-sha",
    "num_bins": 2,
}


def _settings(*, lambda_joint: float = 0.0, top_k: int = 2, raw_gate_mode: str = "within_motion_median"):
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
            "top_k": top_k,
            "raw_gate": {"enabled": True, "mode": raw_gate_mode},
            "transform": "tanh",
            "joint_names": list(JOINT_NAMES),
            "joint_mapping_hash": canonical_joint_mapping_hash(JOINT_NAMES),
            "difficulty_metadata_identity": dict(DIFFICULTY_IDENTITY),
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


def _raw_settings() -> dict[str, object]:
    settings = _settings(lambda_joint=0.0)
    settings.pop("joint_gap")
    return settings


def _controller(*, lambda_joint: float = 0.0, settings: dict[str, object] | None = None) -> OnlineLearningController:
    return OnlineLearningController(
        num_envs=2,
        motion_lengths=torch.tensor([10, 10]),
        segment_motion_ids=torch.tensor([0, 0, 1, 1]),
        segment_start_frames=torch.tensor([0, 5, 0, 5]),
        segment_end_frames=torch.tensor([5, 10, 5, 10]),
        motion_mode="raw_error",
        segment_mode="raw_error_joint_gap",
        settings=_settings(lambda_joint=lambda_joint) if settings is None else settings,
        motion_eligible_mask=torch.ones(2, dtype=torch.bool),
        segment_eligible_mask=torch.ones(4, dtype=torch.bool),
        difficulty_bins=torch.tensor([0, 0, 1, 1]),
        device="cpu",
    )


def _raw_controller() -> OnlineLearningController:
    return OnlineLearningController(
        num_envs=2,
        motion_lengths=torch.tensor([10, 10]),
        segment_motion_ids=torch.tensor([0, 0, 1, 1]),
        segment_start_frames=torch.tensor([0, 5, 0, 5]),
        segment_end_frames=torch.tensor([5, 10, 5, 10]),
        motion_mode="raw_error",
        segment_mode="raw_error",
        settings=_raw_settings(),
        motion_eligible_mask=torch.ones(2, dtype=torch.bool),
        segment_eligible_mask=torch.ones(4, dtype=torch.bool),
        difficulty_bins=None,
        device="cpu",
    )


def _seed(controller: OnlineLearningController) -> None:
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
    if stats.joint_gap is not None:
        stats.joint_gap.segment_joint_error_ema.copy_(
            torch.tensor(
                [
                    [0.1, 1.0, 1.0],
                    [0.9, 1.2, 1.1],
                    [0.2, 0.5, 3.0],
                    [1.0, 0.7, 3.8],
                ]
            )
        )
        stats.joint_gap.segment_joint_error_initialized.fill_(True)


def _stateful_controller(lambda_joint: float) -> OnlineLearningController:
    controller = _controller(lambda_joint=lambda_joint)
    _seed(controller)
    controller.on_iteration_end(0)
    return controller


class JointGapCheckpointTest(unittest.TestCase):
    def test_checkpoint_round_trip_preserves_joint_gap_state(self) -> None:
        controller = _stateful_controller(lambda_joint=0.1)
        state = controller.state_dict()
        restored = _controller(lambda_joint=0.1)
        restored.load_state_dict(state)
        self.assertTrue(torch.equal(restored.statistics.joint_gap.segment_joint_error_ema, controller.statistics.joint_gap.segment_joint_error_ema))
        self.assertTrue(torch.equal(restored.statistics.joint_gap.segment_joint_pending_sum, controller.statistics.joint_gap.segment_joint_pending_sum))
        self.assertTrue(torch.equal(restored.joint_bin_calibration.mean, controller.joint_bin_calibration.mean))
        self.assertTrue(torch.equal(restored.joint_bin_calibration.sigma, controller.joint_bin_calibration.sigma))
        self.assertTrue(torch.equal(restored.joint_bin_calibration.sigma_floor_mask, controller.joint_bin_calibration.sigma_floor_mask))
        self.assertTrue(torch.equal(restored.joint_gap_result.correction, controller.joint_gap_result.correction))
        self.assertEqual(restored.joint_gap_settings["difficulty_metadata_identity"], DIFFICULTY_IDENTITY)

    def test_resume_determinism_lambda_zero_and_nonzero(self) -> None:
        for lambda_joint in (0.0, 0.1):
            with self.subTest(lambda_joint=lambda_joint):
                direct = _stateful_controller(lambda_joint=lambda_joint)
                saved = direct.state_dict()
                expected = direct.sample(64)

                resumed = _controller(lambda_joint=lambda_joint)
                resumed.load_state_dict(saved)
                actual = resumed.sample(64)
                self.assertTrue(all(torch.equal(a, b) for a, b in zip(expected, actual)))
                self.assertTrue(torch.equal(direct.sampler.generator.get_state(), resumed.sampler.generator.get_state()))

    def test_old_m7_raw_online_state_loads_when_joint_gap_disabled(self) -> None:
        raw = _raw_controller()
        _seed(raw)
        raw.on_iteration_end(0)
        state = raw.state_dict()
        restored = _raw_controller()
        restored.load_state_dict(state)
        self.assertIsNone(restored.statistics.joint_gap)
        self.assertTrue(torch.equal(restored.sampler.motion_probability, raw.sampler.motion_probability))

    def test_old_m7_raw_online_state_fails_loudly_when_joint_gap_enabled(self) -> None:
        raw = _raw_controller()
        _seed(raw)
        raw.on_iteration_end(0)
        with self.assertRaisesRegex(ValueError, "Joint-gap resume requires"):
            _controller(lambda_joint=0.0).load_state_dict(raw.state_dict())

    def test_joint_mapping_mismatch_fails_loudly(self) -> None:
        controller = _stateful_controller(lambda_joint=0.1)
        state = copy.deepcopy(controller.state_dict())
        state["joint_gap"]["joint_mapping_identity"]["joint_mapping_hash"] = "bad"
        with self.assertRaisesRegex(ValueError, "joint mapping hash"):
            _controller(lambda_joint=0.1).load_state_dict(state)

    def test_difficulty_identity_mismatch_fails_loudly(self) -> None:
        controller = _stateful_controller(lambda_joint=0.1)
        state = copy.deepcopy(controller.state_dict())
        state["joint_gap"]["difficulty_metadata_identity"]["metadata_sha256"] = "wrong"
        with self.assertRaisesRegex(ValueError, "difficulty metadata identity"):
            _controller(lambda_joint=0.1).load_state_dict(state)

    def test_config_identity_mismatches_fail_loudly(self) -> None:
        controller = _stateful_controller(lambda_joint=0.1)
        state = controller.state_dict()
        for name, settings in (
            ("top_k", _settings(lambda_joint=0.1, top_k=1)),
            ("num_joints", _settings(lambda_joint=0.1)),
            ("raw_gate", _settings(lambda_joint=0.1, raw_gate_mode="top_half")),
            ("ema", _settings(lambda_joint=0.1)),
        ):
            with self.subTest(name=name):
                mutated = copy.deepcopy(settings)
                if name == "num_joints":
                    mutated["joint_gap"]["num_joints"] = 4
                    mutated["joint_gap"]["joint_names"] = ["hip", "knee", "ankle", "toe"]
                    mutated["joint_gap"]["joint_mapping_hash"] = canonical_joint_mapping_hash(mutated["joint_gap"]["joint_names"])
                if name == "ema":
                    mutated["ema_decay"] = 0.25
                    mutated["joint_gap"]["ema_decay"] = 0.25
                if name == "raw_gate":
                    with self.assertRaisesRegex(NotImplementedError, "raw_gate"):
                        _controller(lambda_joint=0.1, settings=mutated)
                    continue
                with self.assertRaises(ValueError):
                    _controller(lambda_joint=0.1, settings=mutated).load_state_dict(state)


if __name__ == "__main__":
    unittest.main()
