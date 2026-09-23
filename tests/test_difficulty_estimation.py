from __future__ import annotations

import copy
import importlib.util
import json
import math
import sys
import tempfile
import types
import unittest
from pathlib import Path
from typing import Any, Mapping

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
UTILS_DIR = (
    PROJECT_ROOT
    / "source"
    / "whole_body_tracking"
    / "whole_body_tracking"
    / "utils"
)
TEST_PACKAGE = "wbt_difficulty_utils_for_tests"


def _load_utils_module(name: str) -> types.ModuleType:
    full_name = f"{TEST_PACKAGE}.{name}"
    module_path = UTILS_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(full_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {name} utilities from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = module
    spec.loader.exec_module(module)
    return module


package = types.ModuleType(TEST_PACKAGE)
package.__path__ = [str(UTILS_DIR)]
sys.modules[TEST_PACKAGE] = package
quality = _load_utils_module("quality")
difficulty = _load_utils_module("difficulty")


def _feature_index(name: str) -> int:
    return difficulty.FEATURE_NAMES.index(name)


def _make_motion(
    config: Mapping[str, Any],
    *,
    fps: float = 50.0,
    duration_seconds: float = 1.0,
    intensity: float = 0.0,
    support: str = "double",
) -> Any:
    """Build a smooth, schema-shaped G1-like motion without Isaac imports."""

    frame_count = max(2, int(round(fps * duration_seconds)))
    time = np.arange(frame_count, dtype=np.float64) / fps
    joint_names = ("joint_0", "joint_1", "joint_2")
    joint_frequencies = np.asarray([0.55, 0.85, 1.15], dtype=np.float64)
    joint_amplitudes = np.asarray([0.35, 0.25, 0.18], dtype=np.float64)
    joint_phases = np.asarray([0.0, 0.3, -0.4], dtype=np.float64)
    joint_angles = (
        2.0 * np.pi * time[:, None] * joint_frequencies[None, :]
        + joint_phases[None, :]
    )
    joint_pos = intensity * joint_amplitudes[None, :] * np.sin(joint_angles)
    joint_vel = (
        intensity
        * joint_amplitudes[None, :]
        * (2.0 * np.pi * joint_frequencies)[None, :]
        * np.cos(joint_angles)
    )

    root_name = str(config["root_body_name"])
    hand_names = tuple(str(item) for item in config["hand_body_names"])
    foot_names = tuple(str(item) for item in config["foot_body_names"])
    body_names = (root_name, *hand_names, *foot_names)
    body_lookup = {name: index for index, name in enumerate(body_names)}
    body_count = len(body_names)
    body_pos = np.zeros((frame_count, body_count, 3), dtype=np.float64)
    body_quat = np.zeros((frame_count, body_count, 4), dtype=np.float64)
    body_quat[..., 0] = 1.0
    body_lin_vel = np.zeros_like(body_pos)
    body_ang_vel = np.zeros_like(body_pos)

    root_frequency = 0.7
    root_omega = 2.0 * np.pi * root_frequency
    root_position = np.column_stack(
        (
            intensity * (0.15 * time + 0.025 * np.sin(root_omega * time)),
            intensity * 0.015 * np.sin(0.6 * root_omega * time),
            0.8 + intensity * 0.03 * np.sin(0.45 * root_omega * time),
        )
    )
    root_velocity = np.column_stack(
        (
            intensity * (0.15 + 0.025 * root_omega * np.cos(root_omega * time)),
            intensity * 0.015 * 0.6 * root_omega * np.cos(0.6 * root_omega * time),
            intensity * 0.03 * 0.45 * root_omega * np.cos(0.45 * root_omega * time),
        )
    )
    root_index = body_lookup[root_name]
    body_pos[:, root_index] = root_position
    body_lin_vel[:, root_index] = root_velocity

    yaw_rate = 0.5 * intensity
    yaw = yaw_rate * time
    root_quaternion = np.column_stack(
        (np.cos(0.5 * yaw), np.zeros(frame_count), np.zeros(frame_count), np.sin(0.5 * yaw))
    )
    body_quat[:, root_index] = root_quaternion
    body_ang_vel[:, root_index, 2] = yaw_rate

    for hand_number, hand_name in enumerate(hand_names):
        hand_index = body_lookup[hand_name]
        side = -1.0 if hand_number == 0 else 1.0
        arm_phase = 2.0 * np.pi * 0.9 * time + hand_number * np.pi
        local_motion = np.column_stack(
            (
                intensity * 0.04 * np.sin(arm_phase),
                np.full(frame_count, side * 0.28),
                0.35 + intensity * 0.025 * np.cos(arm_phase),
            )
        )
        local_velocity = np.column_stack(
            (
                intensity * 0.04 * 2.0 * np.pi * 0.9 * np.cos(arm_phase),
                np.zeros(frame_count),
                -intensity * 0.025 * 2.0 * np.pi * 0.9 * np.sin(arm_phase),
            )
        )
        body_pos[:, hand_index] = root_position + local_motion
        body_lin_vel[:, hand_index] = root_velocity + local_velocity
        body_quat[:, hand_index] = root_quaternion
        body_ang_vel[:, hand_index, 2] = yaw_rate

    if support == "double":
        sole_heights = (0.0, 0.0)
    elif support == "single":
        sole_heights = (0.0, 0.16)
    elif support == "flight":
        sole_heights = (0.16, 0.16)
    else:
        raise ValueError(f"Unknown support mode: {support}")
    for foot_number, (foot_name, sole_height) in enumerate(zip(foot_names, sole_heights, strict=True)):
        foot_index = body_lookup[foot_name]
        sole_offset = np.asarray(config["sole_local_offsets_m"][foot_name], dtype=np.float64)
        body_pos[:, foot_index, 0] = -0.10 if foot_number == 0 else 0.10
        body_pos[:, foot_index, 1] = 0.09 if foot_number == 0 else -0.09
        body_pos[:, foot_index, 2] = sole_height - sole_offset[2]

    return quality.MotionData(
        fps=float(fps),
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        body_pos_w=body_pos,
        body_quat_w=body_quat,
        body_lin_vel_w=body_lin_vel,
        body_ang_vel_w=body_ang_vel,
        joint_names=joint_names,
        body_names=body_names,
        path=f"<synthetic intensity={intensity:g} fps={fps:g}>",
    )


def _set_sole_height_trace(
    motion: Any,
    config: Mapping[str, Any],
    left_height: np.ndarray,
    right_height: np.ndarray,
) -> None:
    lookup = {name: index for index, name in enumerate(motion.body_names)}
    for foot_name, heights in zip(
        config["foot_body_names"], (left_height, right_height), strict=True
    ):
        offset = np.asarray(config["sole_local_offsets_m"][foot_name], dtype=np.float64)
        motion.body_pos_w[:, lookup[foot_name], 2] = np.asarray(heights) - offset[2]


def _only_scoring_features(
    base_config: Mapping[str, Any], definitions: Mapping[str, tuple[float, int]]
) -> dict[str, Any]:
    config = copy.deepcopy(base_config)
    for item in config["features"].values():
        item["weight"] = 0.0
        item["direction"] = 1
    for name, (weight, direction) in definitions.items():
        config["features"][name]["weight"] = weight
        config["features"][name]["direction"] = direction
    return config


class MotionDifficultyFeatureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = difficulty.load_difficulty_config()

    def test_stationary_slow_fast_are_ordered_after_train_fit(self) -> None:
        rows: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        for intensity in (0.0, 0.25, 1.2):
            motion = _make_motion(self.config, intensity=intensity)
            extracted = difficulty.extract_motion_difficulty_features(
                motion, [(0, motion.num_frames)], self.config
            )
            self.assertTrue(extracted.available[0].all())
            rows.append(extracted.values[0])
            masks.append(extracted.available[0])

        values = np.stack(rows)
        available = np.stack(masks)
        profile = difficulty.fit_difficulty_profile(values, available, self.config)
        transformed = difficulty.transform_difficulty_features(values, available, profile)

        self.assertLess(transformed.difficulty_raw[0], transformed.difficulty_raw[1])
        self.assertLess(transformed.difficulty_raw[1], transformed.difficulty_raw[2])
        self.assertLess(transformed.difficulty_score[0], transformed.difficulty_score[1])
        self.assertLess(transformed.difficulty_score[1], transformed.difficulty_score[2])
        speed_column = _feature_index("root_linear_speed_p95")
        self.assertAlmostEqual(values[0, speed_column], 0.0, places=12)
        self.assertLess(values[1, speed_column], values[2, speed_column])

    def test_single_support_and_flight_ratios(self) -> None:
        single_motion = _make_motion(self.config, support="single")
        single = difficulty.extract_motion_difficulty_features(
            single_motion, [(0, single_motion.num_frames)], self.config
        )
        self.assertAlmostEqual(single.values[0, _feature_index("single_support_ratio")], 1.0)
        self.assertAlmostEqual(single.values[0, _feature_index("flight_ratio")], 0.0)

        flight_motion = _make_motion(self.config, support="flight")
        flight = difficulty.extract_motion_difficulty_features(
            flight_motion, [(0, flight_motion.num_frames)], self.config
        )
        self.assertAlmostEqual(flight.values[0, _feature_index("single_support_ratio")], 0.0)
        self.assertAlmostEqual(flight.values[0, _feature_index("flight_ratio")], 1.0)

        invalid_feet = _make_motion(self.config)
        body_lookup = {name: index for index, name in enumerate(invalid_feet.body_names)}
        foot_indexes = [body_lookup[name] for name in self.config["foot_body_names"]]
        invalid_feet.body_pos_w[:, foot_indexes] = np.nan
        invalid = difficulty.extract_motion_difficulty_features(
            invalid_feet, [(0, invalid_feet.num_frames)], self.config
        )
        for feature_name in (
            "foot_swing_speed_p95",
            "end_effector_speed_mean",
            "end_effector_speed_p95",
            "double_support_ratio",
            "single_support_ratio",
            "flight_ratio",
            "left_contact_ratio",
            "right_contact_ratio",
            "contact_switch_count",
            "contact_switch_rate_per_second",
        ):
            column = _feature_index(feature_name)
            self.assertFalse(invalid.available[0, column])
            self.assertTrue(np.isnan(invalid.values[0, column]))
        with self.assertRaisesRegex(ValueError, "Non-finite sole positions"):
            difficulty.infer_debounced_foot_contacts(
                np.full((10, 2, 3), np.nan), invalid_feet.fps, self.config["contact"]
            )

        invalid_joint = _make_motion(self.config)
        invalid_joint.joint_vel[10, 0] = np.inf
        invalid_required = difficulty.extract_motion_difficulty_features(
            invalid_joint, [(0, invalid_joint.num_frames)], self.config
        )
        self.assertFalse(invalid_required.available[0, _feature_index("joint_speed_mean")])
        with self.assertRaisesRegex(ValueError, "Required difficulty features"):
            difficulty.fit_difficulty_profile(
                invalid_required.values, invalid_required.available, self.config
            )

    def test_contact_switch_rate_and_threshold_jitter_debounce(self) -> None:
        fps = 60.0
        motion = _make_motion(self.config, fps=fps)
        left = np.concatenate((np.zeros(20), np.full(20, 0.16), np.zeros(20)))
        right = np.zeros(60)
        _set_sole_height_trace(motion, self.config, left, right)
        extracted = difficulty.extract_motion_difficulty_features(
            motion, [(0, motion.num_frames)], self.config
        )
        self.assertEqual(extracted.values[0, _feature_index("contact_switch_count")], 2.0)
        self.assertAlmostEqual(
            extracted.values[0, _feature_index("contact_switch_rate_per_second")], 2.0
        )

        sole_positions = np.zeros((60, 2, 3), dtype=np.float64)
        sole_positions[..., 2] = 0.0
        sole_positions[30, :, 2] = 0.16
        contacts, _ = difficulty.infer_debounced_foot_contacts(
            sole_positions, fps, self.config["contact"]
        )
        self.assertTrue(contacts.all())
        self.assertEqual(np.count_nonzero(contacts[1:] != contacts[:-1]), 0)

        alternating = np.asarray([True, False, True, False, True])
        np.testing.assert_array_equal(
            difficulty.debounce_boolean_runs(alternating, 2),
            np.ones(alternating.shape, dtype=bool),
        )
        np.testing.assert_array_equal(
            difficulty.debounce_boolean_runs(~alternating, 2),
            np.zeros(alternating.shape, dtype=bool),
        )

    def test_quaternion_sign_equivalence_and_constant_angular_velocity(self) -> None:
        fps = 100.0
        angular_speed = 1.25
        time = np.arange(101, dtype=np.float64) / fps
        angle = angular_speed * time
        quaternions = np.column_stack(
            (np.cos(angle / 2.0), np.zeros(time.size), np.zeros(time.size), np.sin(angle / 2.0))
        )
        quaternions[1::2] *= -1.0

        velocity = difficulty.quaternion_angular_velocity_wxyz(quaternions, fps)
        np.testing.assert_allclose(
            np.linalg.norm(velocity, axis=1), angular_speed, rtol=0.0, atol=2.0e-12
        )

    def test_kinematic_features_are_stable_across_fps(self) -> None:
        compared_features = (
            "root_linear_speed_mean",
            "root_linear_speed_p95",
            "root_linear_acceleration_mean",
            "root_linear_acceleration_p95",
            "root_angular_speed_mean",
            "root_angular_speed_p95",
            "joint_speed_mean",
            "joint_speed_p95",
            "joint_acceleration_mean",
            "joint_acceleration_p95",
        )
        by_fps: dict[int, np.ndarray] = {}
        for fps in (25, 50, 100):
            motion = _make_motion(
                self.config, fps=float(fps), duration_seconds=2.0, intensity=0.8
            )
            extracted = difficulty.extract_motion_difficulty_features(
                motion, [(0, motion.num_frames)], self.config
            )
            by_fps[fps] = extracted.values[0]

        reference = by_fps[100]
        for fps in (25, 50):
            for feature_name in compared_features:
                column = _feature_index(feature_name)
                with self.subTest(fps=fps, feature=feature_name):
                    self.assertTrue(
                        np.isclose(by_fps[fps][column], reference[column], rtol=0.055, atol=2.0e-3),
                        msg=f"{by_fps[fps][column]} != {reference[column]}",
                    )

    def test_one_frame_tail_segment_uses_actual_duration(self) -> None:
        motion = _make_motion(self.config, fps=50.0, duration_seconds=1.02, intensity=0.6)
        self.assertEqual(motion.num_frames, 51)
        extracted = difficulty.extract_motion_difficulty_features(
            motion, [(0, 50), (50, 51)], self.config
        )

        np.testing.assert_allclose(extracted.duration_seconds, [1.0, 0.02], rtol=0.0, atol=1.0e-12)
        self.assertTrue(extracted.available[1].all())
        self.assertTrue(np.isfinite(extracted.values[1]).all())

        boundary_signal = np.asarray([[0.0], [0.0], [1.0], [1.0]])
        destination_derivative = difficulty._gradient_vectors(boundary_signal, 1.0)
        np.testing.assert_array_equal(destination_derivative[:, 0], [0.0, 0.0, 1.0, 0.0])
        self.assertEqual(np.max(destination_derivative[:2]), 0.0)


class DifficultyProfileTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.base_config = difficulty.load_difficulty_config()

    def _manual_profile_fixture(
        self,
    ) -> tuple[dict[str, Any], np.ndarray, np.ndarray, Any]:
        positive = difficulty.FEATURE_NAMES[0]
        negative = difficulty.FEATURE_NAMES[1]
        near_constant = difficulty.FEATURE_NAMES[2]
        config = _only_scoring_features(
            self.base_config,
            {
                positive: (1.0, 1),
                negative: (3.0, -1),
                near_constant: (5.0, 1),
            },
        )
        values = np.zeros((5, len(difficulty.FEATURE_NAMES)), dtype=np.float64)
        values[:, 0] = [-2.0, -1.0, 0.0, 1.0, 2.0]
        values[:, 1] = [2.0, 1.0, 0.0, -1.0, -2.0]
        values[:, 2] = 7.0
        available = np.ones_like(values, dtype=bool)
        profile = difficulty.fit_difficulty_profile(
            values,
            available,
            config,
            training_manifest_sha256="a" * 64,
            training_pool_fingerprint="b" * 64,
            segment_schema_version=1,
            git_commit="test-commit",
        )
        return config, values, available, profile

    def test_median_mad_near_constant_clipping_direction_and_composite(self) -> None:
        config, values, available, profile = self._manual_profile_fixture()
        del config
        self.assertEqual(profile.feature_medians[0], 0.0)
        self.assertAlmostEqual(profile.feature_scales[0], 1.4826, places=12)
        self.assertIn(difficulty.FEATURE_NAMES[2], profile.near_constant_features)
        self.assertEqual(profile.effective_feature_weights[2], 0.0)

        clipped = difficulty.robust_standardize(
            np.asarray([[100.0, -100.0, np.nan]]),
            np.asarray([[True, True, False]]),
            medians=[0.0, 0.0, 0.0],
            scales=[1.0, 1.0, 1.0],
            robust_clip=2.0,
        )
        np.testing.assert_array_equal(clipped[0, :2], [2.0, -2.0])
        self.assertTrue(np.isnan(clipped[0, 2]))

        transformed = difficulty.transform_difficulty_features(values, available, profile)
        expected_raw = (transformed.feature_z[:, 0] - 3.0 * transformed.feature_z[:, 1]) / 4.0
        np.testing.assert_allclose(transformed.difficulty_raw, expected_raw, rtol=0.0, atol=1.0e-12)
        np.testing.assert_allclose(
            transformed.feature_contributions[:, 0]
            + transformed.feature_contributions[:, 1],
            expected_raw,
            rtol=0.0,
            atol=1.0e-12,
        )

    def test_empirical_percentile_and_tied_bins_are_deterministic(self) -> None:
        knots, percentiles = difficulty._empirical_cdf_knots(
            np.asarray([0.0, 0.0, 1.0, 2.0, 2.0])
        )
        np.testing.assert_array_equal(knots, [0.0, 1.0, 2.0])
        np.testing.assert_allclose(percentiles, [0.125, 0.5, 0.875], rtol=0.0, atol=1.0e-12)
        mapped = difficulty.empirical_percentile(
            np.asarray([0.0, 0.0, 1.0, 2.0, 2.0]), knots, percentiles
        )
        self.assertEqual(mapped[0], mapped[1])
        self.assertEqual(mapped[3], mapped[4])

        config = _only_scoring_features(
            self.base_config, {difficulty.FEATURE_NAMES[0]: (1.0, 1)}
        )
        repeated = np.repeat(np.arange(10, dtype=np.float64), 2)
        values = np.zeros((repeated.size, len(difficulty.FEATURE_NAMES)), dtype=np.float64)
        values[:, 0] = repeated
        available = np.ones_like(values, dtype=bool)
        profile = difficulty.fit_difficulty_profile(values, available, config)
        first = difficulty.transform_difficulty_features(values, available, profile)
        second = difficulty.transform_difficulty_features(values, available, profile)
        np.testing.assert_array_equal(first.difficulty_score, second.difficulty_score)
        np.testing.assert_array_equal(first.difficulty_bin, second.difficulty_bin)
        self.assertTrue(np.all((first.difficulty_bin >= 0) & (first.difficulty_bin < 10)))
        for value in np.unique(repeated):
            selected = repeated == value
            self.assertEqual(np.unique(first.difficulty_score[selected]).size, 1)
            self.assertEqual(np.unique(first.difficulty_bin[selected]).size, 1)

    def test_profile_json_round_trip_preserves_transform(self) -> None:
        _, values, available, profile = self._manual_profile_fixture()
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "difficulty_profile.json"
            path.write_text(
                json.dumps(profile.to_dict(), ensure_ascii=False, sort_keys=True), encoding="utf-8"
            )
            loaded = difficulty.load_difficulty_profile(path)

        self.assertEqual(loaded.to_dict(), profile.to_dict())
        before = difficulty.transform_difficulty_features(values, available, profile)
        after = difficulty.transform_difficulty_features(values, available, loaded)
        np.testing.assert_array_equal(after.feature_z, before.feature_z)
        np.testing.assert_array_equal(after.difficulty_raw, before.difficulty_raw)
        np.testing.assert_array_equal(after.difficulty_score, before.difficulty_score)
        np.testing.assert_array_equal(after.difficulty_bin, before.difficulty_bin)

        invalid_profiles = []
        invalid_schema = copy.deepcopy(profile.to_dict())
        invalid_schema["segment_schema_version"] = 0
        invalid_profiles.append(invalid_schema)
        reversed_edges = copy.deepcopy(profile.to_dict())
        reversed_edges["difficulty_bin_edges"] = list(
            reversed(reversed_edges["difficulty_bin_edges"])
        )
        invalid_profiles.append(reversed_edges)
        collapsed_cdf = copy.deepcopy(profile.to_dict())
        collapsed_cdf["difficulty_raw_percentile_knots"] = [
            0.5 for _ in collapsed_cdf["difficulty_raw_percentile_knots"]
        ]
        invalid_profiles.append(collapsed_cdf)
        out_of_range_edges = copy.deepcopy(profile.to_dict())
        raw_maximum = max(out_of_range_edges["difficulty_raw_distribution_knots"])
        out_of_range_edges["difficulty_bin_edges"] = [
            raw_maximum + index + 1.0 for index in range(profile.num_bins - 1)
        ]
        invalid_profiles.append(out_of_range_edges)
        fractional_direction = copy.deepcopy(profile.to_dict())
        fractional_direction["feature_directions"][0] = 1.9
        invalid_profiles.append(fractional_direction)
        overflowing_direction = copy.deepcopy(profile.to_dict())
        overflowing_direction["feature_directions"][0] = 257
        invalid_profiles.append(overflowing_direction)
        invalid_coverage = copy.deepcopy(profile.to_dict())
        invalid_coverage["feature_coverage"][0] = 1.1
        invalid_profiles.append(invalid_coverage)
        for invalid_profile in invalid_profiles:
            with self.subTest(field=invalid_profile):
                with self.assertRaises(ValueError):
                    difficulty.DifficultyProfile.from_dict(invalid_profile)

    def test_motion_aggregation_uses_duration_weighted_mean(self) -> None:
        result = difficulty.aggregate_motion_difficulty(
            [0.0, 1.0], [1.0, 0.1], mean_weight=0.5, p90_weight=0.5, num_bins=10
        )
        expected_mean = 0.1 / 1.1
        expected_p90 = 0.9
        expected_score = 0.5 * (expected_mean + expected_p90)
        self.assertAlmostEqual(result["difficulty_mean"], expected_mean)
        self.assertAlmostEqual(result["difficulty_p90"], expected_p90)
        self.assertAlmostEqual(result["difficulty_score"], expected_score)
        self.assertEqual(result["difficulty_bin"], math.floor(expected_score * 10))
        self.assertEqual(result["segment_count"], 2)
        self.assertAlmostEqual(result["duration_seconds"], 1.1)


if __name__ == "__main__":
    unittest.main()
