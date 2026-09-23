from __future__ import annotations

import copy
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUALITY_PATH = (
    PROJECT_ROOT / "source" / "whole_body_tracking" / "whole_body_tracking" / "utils" / "quality.py"
)

_QUALITY_SPEC = importlib.util.spec_from_file_location("wbt_quality_for_tests", QUALITY_PATH)
if _QUALITY_SPEC is None or _QUALITY_SPEC.loader is None:
    raise RuntimeError(f"Unable to load quality utilities from {QUALITY_PATH}")
quality = importlib.util.module_from_spec(_QUALITY_SPEC)
sys.modules[_QUALITY_SPEC.name] = quality
_QUALITY_SPEC.loader.exec_module(quality)


def make_motion(
    num_frames: int = 101,
    *,
    fps: float = 50.0,
    include_feet: bool = True,
) -> dict[str, np.ndarray]:
    time = np.arange(num_frames, dtype=np.float64) / fps
    joint_pos = np.stack(
        (
            0.10 * np.sin(2.0 * np.pi * time),
            0.08 * np.cos(1.5 * np.pi * time),
        ),
        axis=1,
    )
    joint_vel = np.gradient(joint_pos, 1.0 / fps, axis=0)

    body_names = ["pelvis"]
    if include_feet:
        body_names.extend(("left_ankle_roll_link", "right_ankle_roll_link"))
    else:
        body_names.append("torso_link")
    body_count = len(body_names)
    body_pos = np.zeros((num_frames, body_count, 3), dtype=np.float64)
    body_pos[:, 0, 0] = 0.10 * time
    body_pos[:, 0, 2] = 1.0
    if include_feet:
        body_pos[:, 1, :] = np.array((0.0, 0.10, 0.05))
        body_pos[:, 2, :] = np.array((0.0, -0.10, 0.05))
    else:
        body_pos[:, 1, 2] = 1.3

    body_quat = np.zeros((num_frames, body_count, 4), dtype=np.float64)
    body_quat[..., 0] = 1.0
    body_lin_vel = np.gradient(body_pos, 1.0 / fps, axis=0)
    body_ang_vel = np.zeros((num_frames, body_count, 3), dtype=np.float64)
    return {
        "fps": np.asarray([fps], dtype=np.float32),
        "joint_pos": joint_pos.astype(np.float32),
        "joint_vel": joint_vel.astype(np.float32),
        "body_pos_w": body_pos.astype(np.float32),
        "body_quat_w": body_quat.astype(np.float32),
        "body_lin_vel_w": body_lin_vel.astype(np.float32),
        "body_ang_vel_w": body_ang_vel.astype(np.float32),
        "joint_names": np.asarray(("joint_a", "joint_b")),
        "body_names": np.asarray(body_names),
        "source_file": np.asarray("synthetic/source.npy"),
        "source_format": np.asarray("synthetic"),
    }


def direct_limits(
    *,
    lower: float = -2.0,
    upper: float = 2.0,
    velocity: float = 10.0,
) -> quality.JointLimitTable:
    names = ("joint_a", "joint_b")
    return quality.JointLimitTable(
        joint_names=names,
        lower=np.full(2, lower, dtype=np.float64),
        upper=np.full(2, upper, dtype=np.float64),
        velocity=np.full(2, velocity, dtype=np.float64),
        continuous=np.zeros(2, dtype=bool),
        source_path="synthetic.urdf",
    )


def recompute_joint_velocity(motion: dict[str, np.ndarray]) -> None:
    fps = float(np.asarray(motion["fps"]).reshape(-1)[0])
    motion["joint_vel"] = np.gradient(
        motion["joint_pos"].astype(np.float64), 1.0 / fps, axis=0
    ).astype(np.float32)


def recompute_body_velocity(motion: dict[str, np.ndarray]) -> None:
    fps = float(np.asarray(motion["fps"]).reshape(-1)[0])
    motion["body_lin_vel_w"] = np.gradient(
        motion["body_pos_w"].astype(np.float64), 1.0 / fps, axis=0
    ).astype(np.float32)


class QualityAuditTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = quality.load_quality_config()

    def audit(
        self,
        motion: dict[str, np.ndarray],
        *,
        config: dict | None = None,
        limits: quality.JointLimitTable | None = None,
        urdf_path: str | Path | None = None,
    ) -> quality.MotionQualityAudit:
        return quality.audit_motion_segments(
            motion,
            self.config if config is None else config,
            joint_limits=limits,
            urdf_path=urdf_path,
        )

    def test_clean_motion_passes_and_real_npz_schema_loads(self) -> None:
        motion = make_motion()
        result = self.audit(motion, limits=direct_limits())
        self.assertEqual(
            [(item.start_frame, item.end_frame_exclusive) for item in result.segment_results],
            [(0, 50), (50, 100), (100, 101)],
        )
        self.assertTrue(all(item.quality_status == "pass" for item in result.segment_results))
        self.assertTrue(all(not item.hard_violation for item in result.segment_results))
        self.assertTrue(all(item.quality_score > 0.99 for item in result.segment_results))
        self.assertTrue(self.config["provisional"])

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "motion.npz"
            np.savez(path, **motion)
            loaded = quality.load_motion_npz(path)
        self.assertEqual(loaded.joint_pos.shape, (101, 2))
        self.assertEqual(loaded.body_pos_w.shape, (101, 3, 3))
        self.assertEqual(loaded.joint_names, ("joint_a", "joint_b"))
        self.assertEqual(loaded.source_format, "synthetic")

    def test_schema_errors_are_not_swallowed(self) -> None:
        motion = make_motion()
        del motion["body_quat_w"]
        with self.assertRaisesRegex(quality.MotionSchemaError, "missing required WBT fields"):
            quality.validate_motion_npz(motion, path="broken.npz")

        motion = make_motion()
        motion["body_names"] = np.asarray(("pelvis", "left_ankle_roll_link"))
        with self.assertRaisesRegex(quality.MotionSchemaError, "body_names must have shape"):
            quality.validate_motion_npz(motion)

    def test_nan_and_inf_are_hard_segment_violations(self) -> None:
        cases = (("joint_pos", np.nan), ("body_lin_vel_w", np.inf))
        for field, value in cases:
            with self.subTest(field=field):
                motion = make_motion()
                motion[field][10].reshape(-1)[0] = value
                result = self.audit(motion, limits=direct_limits()).segment_results[0]
                metric = result.metrics["nonfinite_values"]
                self.assertTrue(metric.available)
                self.assertTrue(metric.hard_violation)
                self.assertEqual(metric.severity, 1.0)
                self.assertEqual(result.quality_status, "reject")

    def test_urdf_limits_and_light_vs_heavy_position_excess(self) -> None:
        urdf = """<robot name="fixture">
          <joint name="joint_b" type="revolute"><limit lower="-2" upper="2" velocity="10"/></joint>
          <joint name="joint_a" type="revolute"><limit lower="-2" upper="2" velocity="10"/></joint>
        </robot>"""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "fixture.urdf"
            path.write_text(urdf, encoding="utf-8")
            limits = quality.parse_urdf_joint_limits(path, ("joint_a", "joint_b"))
        self.assertEqual(limits.lower.tolist(), [-2.0, -2.0])
        self.assertEqual(limits.upper.tolist(), [2.0, 2.0])
        self.assertEqual(limits.velocity.tolist(), [10.0, 10.0])
        self.assertEqual(limits.position_coverage, 1.0)

        light = make_motion()
        light["joint_pos"][:, 0] = 2.05
        recompute_joint_velocity(light)
        light_result = self.audit(light, limits=limits).segment_results[0]
        light_metric = light_result.metrics["joint_position_limits"]
        self.assertGreater(light_metric.severity, 0.0)
        self.assertLess(light_metric.severity, 1.0)
        self.assertFalse(light_metric.hard_violation)
        self.assertEqual(light_result.quality_status, "borderline")

        heavy = make_motion()
        heavy["joint_pos"][:, 0] = 2.50
        recompute_joint_velocity(heavy)
        heavy_result = self.audit(heavy, limits=limits).segment_results[0]
        self.assertTrue(heavy_result.metrics["joint_position_limits"].hard_violation)
        self.assertEqual(heavy_result.quality_status, "reject")

    def test_velocity_field_mismatch_is_detected(self) -> None:
        motion = make_motion(num_frames=51)
        motion["joint_pos"][:, 0] = 0.20 * np.arange(51)
        motion["joint_vel"][:, 0] = 0.0
        result = self.audit(
            motion,
            limits=direct_limits(lower=-100.0, upper=100.0, velocity=100.0),
        ).segment_results[0]
        metric = result.metrics["joint_velocity_consistency"]
        self.assertGreaterEqual(metric.raw_value, 8.0)
        self.assertTrue(metric.hard_violation)
        self.assertEqual(result.quality_status, "reject")

        sparse = make_motion(num_frames=51)
        sparse["joint_vel"][25, 0] += 50.0
        sparse_result = self.audit(
            sparse,
            limits=direct_limits(lower=-100.0, upper=100.0, velocity=100.0),
        ).segment_results[0]
        sparse_metric = sparse_result.metrics["joint_velocity_consistency"]
        self.assertLess(sparse_metric.raw_value, 1.0)
        self.assertGreater(sparse_metric.details["max_error"], 20.0)
        self.assertEqual(sparse_metric.details["max_severity"], 1.0)
        self.assertTrue(sparse_metric.hard_violation)
        self.assertEqual(sparse_result.quality_status, "reject")

    def test_smooth_high_speed_is_not_an_isolated_spike_but_jump_is(self) -> None:
        limits = direct_limits(lower=-100.0, upper=100.0, velocity=1000.0)
        smooth = make_motion()
        smooth["joint_pos"][:, 0] = 0.50 * np.arange(101)
        recompute_joint_velocity(smooth)
        smooth_result = self.audit(smooth, limits=limits)
        self.assertTrue(all(item.quality_status == "pass" for item in smooth_result.segment_results))
        self.assertTrue(
            all(
                item.metrics["joint_position_continuity"].severity == 0.0
                for item in smooth_result.segment_results
            )
        )
        self.assertTrue(
            all(
                item.metrics["joint_acceleration_spike"].severity == 0.0
                for item in smooth_result.segment_results
            )
        )

        jumped = make_motion()
        jumped["joint_pos"][:, 0] = 0.50 * np.arange(101)
        # A large one-frame conversion jump on top of the same smooth trend.
        # It remains inside the deliberately wide synthetic URDF range so the
        # continuity metric, rather than a joint-limit violation, detects it.
        jumped["joint_pos"][50, 0] += 20.0
        recompute_joint_velocity(jumped)
        jumped_result = self.audit(jumped, limits=limits)
        destination_segment = jumped_result.segment_results[1]
        self.assertTrue(destination_segment.metrics["joint_position_continuity"].hard_violation)
        self.assertEqual(destination_segment.quality_status, "reject")
        self.assertGreater(
            max(
                item.metrics["joint_acceleration_spike"].raw_value
                for item in jumped_result.segment_results
            ),
            1.0,
        )

    def test_segment_boundary_transition_is_attributed_to_destination_segment(self) -> None:
        motion = make_motion()
        motion["joint_pos"][50, 0] += 5.0
        recompute_joint_velocity(motion)
        result = self.audit(
            motion,
            limits=direct_limits(lower=-100.0, upper=100.0, velocity=1000.0),
        )
        source = result.segment_results[0].metrics["joint_position_continuity"]
        destination = result.segment_results[1].metrics["joint_position_continuity"]
        self.assertFalse(source.hard_violation)
        self.assertTrue(destination.hard_violation)
        self.assertGreater(destination.details["jump_max"], 4.0)

    def test_wxyz_quaternion_sign_flip_is_not_a_rotation_jump(self) -> None:
        motion = make_motion()
        motion["body_quat_w"][1::2] *= -1.0
        result = self.audit(motion, limits=direct_limits())
        for segment in result.segment_results:
            self.assertEqual(segment.metrics["root_orientation_continuity"].severity, 0.0)
            self.assertEqual(segment.metrics["body_orientation_continuity"].severity, 0.0)
            self.assertAlmostEqual(
                segment.metrics["body_orientation_continuity"].details["jump_max"], 0.0
            )

    def test_isolated_180_degree_orientation_jump_is_rejected(self) -> None:
        motion = make_motion()
        motion["body_quat_w"][50:, :, 0] = 0.0
        motion["body_quat_w"][50:, :, 1] = 1.0
        result = self.audit(motion, limits=direct_limits()).segment_results[1]
        root_metric = result.metrics["root_orientation_continuity"]
        body_metric = result.metrics["body_orientation_continuity"]
        self.assertAlmostEqual(root_metric.details["jump_max"], np.pi, places=5)
        self.assertTrue(root_metric.hard_violation)
        self.assertTrue(body_metric.hard_violation)
        self.assertEqual(result.quality_status, "reject")

    def test_short_tail_segment_uses_full_motion_derivatives(self) -> None:
        motion = make_motion(num_frames=52)
        result = self.audit(motion, limits=direct_limits())
        self.assertEqual(
            [(item.start_frame, item.end_frame_exclusive) for item in result.segment_results],
            [(0, 50), (50, 52)],
        )
        tail = result.segment_results[1]
        self.assertEqual(tail.quality_status, "pass")
        self.assertTrue(tail.metrics["joint_velocity_consistency"].available)
        self.assertTrue(tail.metrics["joint_position_continuity"].available)

    def test_configured_local_sole_offset_detects_penetration(self) -> None:
        motion = make_motion()
        config = copy.deepcopy(self.config)
        for foot_name in config["ground"]["foot_body_names"]:
            config["ground"]["sole_local_offsets_m"][foot_name] = [0.0, 0.0, -0.20]
        result = self.audit(motion, config=config, limits=direct_limits()).segment_results[0]
        metric = result.metrics["ground_penetration"]
        self.assertGreater(metric.raw_value, 0.10)
        self.assertEqual(
            set(metric.details["sole_min_height_by_body_m"]),
            {"left_ankle_roll_link", "right_ankle_roll_link"},
        )
        self.assertTrue(
            all(value < 0.0 for value in metric.details["sole_min_height_by_body_m"].values())
        )
        self.assertTrue(metric.hard_violation)
        self.assertEqual(result.quality_status, "reject")

    def test_foot_sliding_is_contact_only_and_persistent_severe_is_hard(self) -> None:
        contact_motion = make_motion()
        contact_motion["body_pos_w"][:, 1:, 0] = 0.02 * np.arange(101)[:, None]
        contact_motion["body_pos_w"][:, 1:, 2] = 0.04
        recompute_body_velocity(contact_motion)
        contact_result = self.audit(contact_motion, limits=direct_limits()).segment_results[0]
        contact_metric = contact_result.metrics["foot_sliding"]
        self.assertGreaterEqual(contact_metric.raw_value, 0.9)
        self.assertGreater(contact_metric.severity, 0.0)
        self.assertFalse(contact_metric.hard_violation)

        severe_motion = make_motion()
        severe_motion["body_pos_w"][:, 1:, 0] = 0.06 * np.arange(101)[:, None]
        severe_motion["body_pos_w"][:, 1:, 2] = 0.04
        recompute_body_velocity(severe_motion)
        severe_result = self.audit(severe_motion, limits=direct_limits()).segment_results[0]
        severe_metric = severe_result.metrics["foot_sliding"]
        self.assertGreater(severe_metric.raw_value, 2.5)
        self.assertTrue(severe_metric.details["persistent_reject"])
        self.assertTrue(severe_metric.hard_violation)
        self.assertEqual(severe_result.quality_status, "reject")

        airborne_motion = make_motion()
        airborne_motion["body_pos_w"][:, 1:, 0] = 0.06 * np.arange(101)[:, None]
        airborne_motion["body_pos_w"][:, 1:, 2] = 1.0
        recompute_body_velocity(airborne_motion)
        airborne_result = self.audit(airborne_motion, limits=direct_limits()).segment_results[0]
        airborne_metric = airborne_result.metrics["foot_sliding"]
        self.assertEqual(airborne_metric.raw_value, 0.0)
        self.assertEqual(airborne_metric.details["contact_frame_count"], 0)
        self.assertEqual(airborne_metric.severity, 0.0)

    def test_single_frame_landing_impact_is_not_persistent_foot_sliding(self) -> None:
        motion = make_motion()
        motion["body_pos_w"][:, 1:, 2] = 1.0
        motion["body_pos_w"][25, 1:, 0] = 10.0
        motion["body_pos_w"][25, 1:, 2] = 0.04
        recompute_body_velocity(motion)

        result = self.audit(motion, limits=direct_limits()).segment_results[0]
        metric = result.metrics["foot_sliding"]
        self.assertLess(metric.details["contact_frame_count"], 10)
        self.assertFalse(metric.details["persistent_reject"])
        self.assertFalse(metric.hard_violation)

    def test_missing_configured_foot_body_names_raise_clear_error(self) -> None:
        motion = make_motion(include_feet=False)
        with self.assertRaisesRegex(ValueError, "Configured foot body names are absent"):
            self.audit(motion, limits=direct_limits())

    def test_missing_required_metrics_are_unavailable_not_zero_score_but_reject(self) -> None:
        motion = make_motion(include_feet=False)
        config = copy.deepcopy(self.config)
        config["ground"]["require_configured_foot_bodies"] = False
        result = self.audit(motion, config=config, urdf_path="/definitely/missing/g1.urdf")
        self.assertIn("does not exist", result.joint_limit_error)
        segment = result.segment_results[0]
        for name in (
            "joint_position_limits",
            "joint_velocity_limits",
            "ground_penetration",
            "foot_sliding",
        ):
            self.assertFalse(segment.metrics[name].available)
            self.assertEqual(segment.metrics[name].severity, 0.0)
        self.assertLess(segment.metric_coverage, 1.0)
        self.assertTrue(segment.insufficient_metrics)
        self.assertEqual(segment.quality_score, 1.0)
        self.assertEqual(segment.quality_status, "reject")
        self.assertTrue(any(reason.startswith("missing_required_metrics:") for reason in segment.status_reasons))

    def test_optional_metric_coverage_replaces_fixed_available_count(self) -> None:
        metrics = {
            name: quality.MetricResult(0.0, 0.0, True, False)
            for name in quality.METRIC_NAMES
        }
        metric_configs = {
            name: {"weight": 1.0}
            for name in quality.METRIC_NAMES
        }
        status = {
            "reject_score_threshold": 0.55,
            "pass_score_threshold": 0.90,
            "required_metrics": ["nonfinite_values"],
            "optional_metric_coverage_profile": ["quaternion_norm", "joint_position_limits"],
            "minimum_optional_metric_coverage": 0.5,
            "reject_severity_count": 2,
            "borderline_on_warning": True,
        }

        metrics["joint_position_limits"] = quality.MetricResult(float("nan"), 0.0, False, False)
        score, status_name, _, insufficient, _, _, optional_coverage, reasons = quality._score_and_status(
            metrics, metric_configs, status
        )
        self.assertEqual(score, 1.0)
        self.assertEqual(optional_coverage, 0.5)
        self.assertFalse(insufficient)
        self.assertEqual(status_name, "pass")
        self.assertEqual(reasons, ())

        metrics["quaternion_norm"] = quality.MetricResult(float("nan"), 0.0, False, False)
        _, status_name, _, insufficient, _, _, optional_coverage, reasons = quality._score_and_status(
            metrics, metric_configs, status
        )
        self.assertEqual(optional_coverage, 0.0)
        self.assertTrue(insufficient)
        self.assertEqual(status_name, "reject")
        self.assertTrue(any(reason.startswith("optional_metric_coverage_below_threshold") for reason in reasons))

    def test_quality_score_status_rules_hand_examples(self) -> None:
        metric_configs = {
            name: {"weight": 1.0}
            for name in quality.METRIC_NAMES
        }
        required = ["nonfinite_values"]
        optional_profile = ["quaternion_norm", "joint_position_limits"]
        status = {
            "reject_score_threshold": 0.55,
            "pass_score_threshold": 0.90,
            "required_metrics": required,
            "optional_metric_coverage_profile": optional_profile,
            "minimum_optional_metric_coverage": 0.5,
            "reject_severity_count": 2,
            "borderline_on_warning": True,
        }

        def classify(overrides: dict[str, quality.MetricResult]):
            metrics = {
                name: quality.MetricResult(0.0, 0.0, True, False)
                for name in quality.METRIC_NAMES
            }
            metrics.update(overrides)
            return quality._score_and_status(metrics, metric_configs, status)

        self.assertEqual(classify({})[1], "pass")
        self.assertEqual(
            classify({"quaternion_norm": quality.MetricResult(0.5, 0.5, True, False)})[1],
            "borderline",
        )
        self.assertEqual(
            classify({"quaternion_norm": quality.MetricResult(1.0, 1.0, True, True)})[1],
            "reject",
        )
        self.assertEqual(
            classify(
                {
                    "quaternion_norm": quality.MetricResult(1.0, 1.0, True, False),
                    "joint_position_limits": quality.MetricResult(1.0, 1.0, True, False),
                }
            )[1],
            "reject",
        )
        self.assertEqual(
            classify({"nonfinite_values": quality.MetricResult(float("nan"), 0.0, False, False)})[1],
            "reject",
        )
        optional_missing = classify(
            {"joint_position_limits": quality.MetricResult(float("nan"), 0.0, False, False)}
        )
        self.assertEqual(optional_missing[1], "pass")
        self.assertEqual(optional_missing[6], 0.5)


if __name__ == "__main__":
    unittest.main()
