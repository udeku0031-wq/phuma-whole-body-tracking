"""Pure NumPy segment-level quality auditing for converted WBT motions.

The quality core deliberately has no Isaac Sim, Torch, Hydra, or PyYAML
dependency.  Offline tooling may pass the exact right-open segment bounds
produced by :class:`FixedLengthSegmentIndex`; the fallback bound builder uses
the same round-to-nearest-even convention for standalone tests and inspection.

All temporal derivatives and transition signals are computed over the complete
motion before they are aggregated into segments.  Consequently, a jump from
frame ``i - 1`` to frame ``i`` is attributed to the segment containing frame
``i`` and cannot disappear at a segment boundary.
"""

from __future__ import annotations

import copy
import json
import math
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


QUALITY_SCHEMA_VERSION = 1
PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_QUALITY_CONFIG_PATH = PROJECT_ROOT / "configs" / "quality" / "g1_segment_quality.yaml"

REQUIRED_MOTION_FIELDS = (
    "fps",
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
    "joint_names",
    "body_names",
)

METRIC_NAMES = (
    "nonfinite_values",
    "quaternion_norm",
    "joint_position_limits",
    "joint_velocity_limits",
    "joint_velocity_consistency",
    "body_velocity_consistency",
    "joint_acceleration_spike",
    "joint_jerk_spike",
    "root_linear_acceleration_spike",
    "root_angular_acceleration_spike",
    "root_position_continuity",
    "root_orientation_continuity",
    "body_position_continuity",
    "body_orientation_continuity",
    "joint_position_continuity",
    "ground_penetration",
    "foot_sliding",
)

METRIC_SEMANTICS = {
    "nonfinite_values": (
        "Fraction of NaN/Inf values over all numeric WBT trajectory arrays in the segment."
    ),
    "quaternion_norm": (
        "Maximum | ||q_wxyz||_2 - 1 | over body_quat_w in the segment."
    ),
    "joint_position_limits": (
        "Maximum signed excess outside URDF lower/upper position limits, aligned by joint_names."
    ),
    "joint_velocity_limits": (
        "Maximum excess of |joint_vel| over URDF velocity limits, aligned by joint_names."
    ),
    "joint_velocity_consistency": (
        "P95 finite-difference joint velocity error, with an additional max-error channel."
    ),
    "body_velocity_consistency": (
        "P95 world-frame finite-difference body linear-velocity vector error, with an additional max-error channel."
    ),
    "joint_acceleration_spike": (
        "Maximum local-isolation ratio of joint acceleration magnitude; sustained high dynamics are not spikes."
    ),
    "joint_jerk_spike": (
        "Maximum local-isolation ratio of joint jerk magnitude; sustained high dynamics are not spikes."
    ),
    "root_linear_acceleration_spike": (
        "Maximum local-isolation ratio of root linear acceleration from stored body_lin_vel_w."
    ),
    "root_angular_acceleration_spike": (
        "Maximum local-isolation ratio of root angular acceleration from stored body_ang_vel_w."
    ),
    "root_position_continuity": (
        "Maximum local-isolation ratio of the root world-position frame-to-frame transition."
    ),
    "root_orientation_continuity": (
        "Maximum local-isolation ratio of root quaternion geodesic frame-to-frame transition."
    ),
    "body_position_continuity": (
        "Maximum local-isolation ratio of all body world-position frame-to-frame transitions."
    ),
    "body_orientation_continuity": (
        "Maximum local-isolation ratio of all body quaternion geodesic frame-to-frame transitions."
    ),
    "joint_position_continuity": (
        "Maximum local-isolation ratio of unwrapped joint-position frame-to-frame transitions."
    ),
    "ground_penetration": (
        "Maximum depth of configured sole reference points below ground.z_m."
    ),
    "foot_sliding": (
        "P95 horizontal configured-sole speed on kinematically inferred contact frames; only persistent severe contact is hard."
    ),
}


class MotionSchemaError(ValueError):
    """Raised when a file cannot be interpreted as a WBT motion."""


@dataclass
class MotionData:
    """Validated in-memory representation of one converted WBT motion."""

    fps: float
    joint_pos: np.ndarray
    joint_vel: np.ndarray
    body_pos_w: np.ndarray
    body_quat_w: np.ndarray
    body_lin_vel_w: np.ndarray
    body_ang_vel_w: np.ndarray
    joint_names: tuple[str, ...]
    body_names: tuple[str, ...]
    path: str = "<memory>"
    source_file: str = ""
    source_format: str = ""

    @property
    def num_frames(self) -> int:
        return int(self.joint_pos.shape[0])


@dataclass
class JointLimitTable:
    """URDF limits aligned to the WBT ``joint_names`` order.

    Unavailable limits are represented by NaN.  This lets callers distinguish
    an unavailable metric from a valid trajectory with zero violations.
    """

    joint_names: tuple[str, ...]
    lower: np.ndarray
    upper: np.ndarray
    velocity: np.ndarray
    continuous: np.ndarray
    source_path: str | None = None
    error: str | None = None

    @property
    def position_available(self) -> np.ndarray:
        return np.isfinite(self.lower) & np.isfinite(self.upper)

    @property
    def velocity_available(self) -> np.ndarray:
        return np.isfinite(self.velocity) & (self.velocity > 0.0)

    @property
    def position_coverage(self) -> float:
        return float(np.mean(self.position_available)) if self.position_available.size else 0.0

    @property
    def velocity_coverage(self) -> float:
        return float(np.mean(self.velocity_available)) if self.velocity_available.size else 0.0


@dataclass
class MetricResult:
    """One transparent segment-level quality metric."""

    raw_value: float
    severity: float
    available: bool
    hard_violation: bool
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_value": _json_value(self.raw_value),
            "severity": float(self.severity),
            "available": bool(self.available),
            "hard_violation": bool(self.hard_violation),
            "details": {key: _json_value(value) for key, value in self.details.items()},
        }


@dataclass
class SegmentQualityResult:
    """Quality result for one right-open motion segment."""

    local_segment_id: int
    start_frame: int
    end_frame_exclusive: int
    quality_score: float
    quality_status: str
    hard_violation: bool
    insufficient_metrics: bool
    available_metric_count: int
    metric_coverage: float
    optional_metric_coverage: float
    metrics: dict[str, MetricResult]
    status_reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": QUALITY_SCHEMA_VERSION,
            "local_segment_id": self.local_segment_id,
            "start_frame": self.start_frame,
            "end_frame_exclusive": self.end_frame_exclusive,
            "quality_score": self.quality_score,
            "quality_status": self.quality_status,
            "hard_violation": self.hard_violation,
            "insufficient_metrics": self.insufficient_metrics,
            "available_metric_count": self.available_metric_count,
            "metric_coverage": self.metric_coverage,
            "optional_metric_coverage": self.optional_metric_coverage,
            "status_reasons": list(self.status_reasons),
            "metrics": {name: result.to_dict() for name, result in self.metrics.items()},
        }


@dataclass
class MotionQualityAudit:
    """All segment results and availability metadata for one motion."""

    motion_path: str
    fps: float
    num_frames: int
    segment_length_seconds: float
    segment_results: list[SegmentQualityResult]
    joint_position_limit_coverage: float
    joint_velocity_limit_coverage: float
    joint_limit_source: str | None
    joint_limit_error: str | None


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _validate_quality_config(config: Mapping[str, Any]) -> None:
    if int(config.get("schema_version", -1)) != QUALITY_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported quality config schema {config.get('schema_version')}; "
            f"expected {QUALITY_SCHEMA_VERSION}."
        )
    segment_length = float(config.get("segment_length_seconds", 0.0))
    if not math.isfinite(segment_length) or segment_length <= 0.0:
        raise ValueError("segment_length_seconds must be finite and greater than zero.")

    metric_configs = config.get("metrics")
    if not isinstance(metric_configs, Mapping):
        raise ValueError("Quality config must contain a metrics mapping.")
    missing_metrics = sorted(set(METRIC_NAMES).difference(metric_configs))
    if missing_metrics:
        raise ValueError(f"Quality config is missing metric definitions: {missing_metrics}")

    for name in METRIC_NAMES:
        metric = metric_configs[name]
        if not isinstance(metric, Mapping):
            raise ValueError(f"Metric config '{name}' must be a mapping.")
        warning = float(metric.get("warning_threshold", float("nan")))
        reject = float(metric.get("reject_threshold", float("nan")))
        weight = float(metric.get("weight", float("nan")))
        if not math.isfinite(warning) or not math.isfinite(reject) or warning >= reject:
            raise ValueError(f"Metric '{name}' must have finite warning_threshold < reject_threshold.")
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError(f"Metric '{name}' weight must be finite and non-negative.")
        if not isinstance(metric.get("unit"), str) or not str(metric["unit"]).strip():
            raise ValueError(f"Metric '{name}' must declare a non-empty unit.")
        if not isinstance(metric.get("description"), str) or not str(metric["description"]).strip():
            raise ValueError(f"Metric '{name}' must declare a non-empty description.")
        if not isinstance(metric.get("hard_at_reject"), bool):
            raise ValueError(f"Metric '{name}' hard_at_reject must be boolean.")
        has_max_warning = "max_warning_threshold" in metric
        has_max_reject = "max_reject_threshold" in metric
        if has_max_warning != has_max_reject:
            raise ValueError(
                f"Metric '{name}' must define max_warning_threshold and max_reject_threshold together."
            )
        if has_max_warning:
            max_warning = float(metric["max_warning_threshold"])
            max_reject = float(metric["max_reject_threshold"])
            if not math.isfinite(max_warning) or not math.isfinite(max_reject) or max_warning >= max_reject:
                raise ValueError(
                    f"Metric '{name}' must have finite max_warning_threshold < max_reject_threshold."
                )
        if "absolute_floor" in metric:
            floor = float(metric["absolute_floor"])
            multiplier = float(metric.get("relative_multiplier", 0.0))
            if not math.isfinite(floor) or floor <= 0.0:
                raise ValueError(f"Metric '{name}' absolute_floor must be positive and finite.")
            if not math.isfinite(multiplier) or multiplier <= 1.0:
                raise ValueError(f"Metric '{name}' relative_multiplier must be greater than one.")

    status = config.get("status")
    if not isinstance(status, Mapping):
        raise ValueError("Quality config must contain a status mapping.")
    reject_score = float(status.get("reject_score_threshold", float("nan")))
    pass_score = float(status.get("pass_score_threshold", float("nan")))
    if not (0.0 <= reject_score < pass_score <= 1.0):
        raise ValueError("Status thresholds must satisfy 0 <= reject < pass <= 1.")
    required_metrics = status.get("required_metrics", ())
    if (
        not isinstance(required_metrics, list)
        or not all(isinstance(name, str) for name in required_metrics)
        or len(set(required_metrics)) != len(required_metrics)
    ):
        raise ValueError("status.required_metrics must be a duplicate-free list.")
    unknown_required = set(required_metrics).difference(METRIC_NAMES)
    if unknown_required:
        raise ValueError(f"status.required_metrics contains unknown metrics: {sorted(unknown_required)}")
    optional_metric_names = tuple(name for name in METRIC_NAMES if name not in set(required_metrics))
    coverage_profile = status.get("optional_metric_coverage_profile", list(optional_metric_names))
    if (
        not isinstance(coverage_profile, list)
        or not all(isinstance(name, str) for name in coverage_profile)
        or len(set(coverage_profile)) != len(coverage_profile)
    ):
        raise ValueError("status.optional_metric_coverage_profile must be a duplicate-free list.")
    unknown_profile_metrics = set(coverage_profile).difference(METRIC_NAMES)
    if unknown_profile_metrics:
        raise ValueError(
            "status.optional_metric_coverage_profile contains unknown metrics: "
            f"{sorted(unknown_profile_metrics)}"
        )
    required_profile_metrics = set(coverage_profile).intersection(required_metrics)
    if required_profile_metrics:
        raise ValueError(
            "status.optional_metric_coverage_profile must not include required metrics: "
            f"{sorted(required_profile_metrics)}"
        )
    coverage = float(status.get("minimum_optional_metric_coverage", float("nan")))
    if not math.isfinite(coverage) or not 0.0 <= coverage <= 1.0:
        raise ValueError("status.minimum_optional_metric_coverage must be a finite value in [0, 1].")
    reject_severity_count = int(status.get("reject_severity_count", 0))
    if reject_severity_count < 1 or reject_severity_count > len(METRIC_NAMES):
        raise ValueError(f"reject_severity_count must be in [1, {len(METRIC_NAMES)}].")

    sliding = metric_configs["foot_sliding"]
    persistent_ratio = float(sliding.get("persistent_reject_frame_ratio", float("nan")))
    persistent_samples = int(sliding.get("persistent_reject_min_contact_samples", 0))
    if not math.isfinite(persistent_ratio) or not 0.0 < persistent_ratio <= 1.0:
        raise ValueError("foot_sliding.persistent_reject_frame_ratio must be in (0, 1].")
    if persistent_samples < 1:
        raise ValueError("foot_sliding.persistent_reject_min_contact_samples must be at least 1.")

    ground = config.get("ground")
    if not isinstance(ground, Mapping):
        raise ValueError("Quality config must contain a ground mapping.")
    ground_z = float(ground.get("z_m", float("nan")))
    contact_height = float(ground.get("contact_height_threshold_m", float("nan")))
    contact_vertical_speed = float(ground.get("contact_vertical_speed_threshold_mps", float("nan")))
    foot_names = ground.get("foot_body_names", ())
    if (
        not isinstance(foot_names, list)
        or not foot_names
        or not all(isinstance(name, str) and name for name in foot_names)
        or len(set(foot_names)) != len(foot_names)
    ):
        raise ValueError("ground.foot_body_names must be a non-empty duplicate-free list of strings.")
    offsets = ground.get("sole_local_offsets_m", {})
    if not isinstance(offsets, Mapping):
        raise ValueError("ground.sole_local_offsets_m must be a mapping from foot body name to xyz offset.")
    missing_offsets = [name for name in foot_names if name not in offsets]
    if missing_offsets:
        raise ValueError(f"ground.sole_local_offsets_m is missing offsets for: {missing_offsets}")
    for name in foot_names:
        offset = np.asarray(offsets[name], dtype=np.float64)
        if offset.shape != (3,) or not np.all(np.isfinite(offset)):
            raise ValueError(f"ground.sole_local_offsets_m['{name}'] must contain three finite values.")
    if not isinstance(ground.get("require_configured_foot_bodies", True), bool):
        raise ValueError("ground.require_configured_foot_bodies must be boolean.")
    if not math.isfinite(ground_z):
        raise ValueError("ground.z_m must be finite.")
    if not math.isfinite(contact_height) or contact_height < 0.0:
        raise ValueError("ground.contact_height_threshold_m must be finite and non-negative.")
    if not math.isfinite(contact_vertical_speed) or contact_vertical_speed < 0.0:
        raise ValueError("ground.contact_vertical_speed_threshold_mps must be finite and non-negative.")


def load_quality_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the JSON-compatible YAML quality configuration.

    The repository uses a ``.yaml`` suffix for discoverability, but the file is
    intentionally valid JSON so this core does not gain a PyYAML dependency.
    """

    config_path = DEFAULT_QUALITY_CONFIG_PATH if path is None else Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Quality config does not exist: {config_path}")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Quality config must use JSON-compatible YAML syntax: {config_path}: {exc}") from exc
    if not isinstance(config, dict):
        raise ValueError("Quality config root must be an object.")
    _validate_quality_config(config)
    return config


def parse_urdf_joint_limits(
    urdf_path: str | Path | None, joint_names: Sequence[str]
) -> JointLimitTable:
    """Parse URDF position and velocity limits aligned to ``joint_names``.

    A missing or unreadable URDF is represented as an unavailable table rather
    than guessed limits.  Callers can decide whether missing coverage is fatal.
    """

    names = tuple(str(name) for name in joint_names)
    count = len(names)
    lower = np.full(count, np.nan, dtype=np.float64)
    upper = np.full(count, np.nan, dtype=np.float64)
    velocity = np.full(count, np.nan, dtype=np.float64)
    continuous = np.zeros(count, dtype=bool)

    if urdf_path is None:
        return JointLimitTable(names, lower, upper, velocity, continuous, error="URDF path was not provided.")

    path = Path(urdf_path).expanduser()
    if not path.is_file():
        return JointLimitTable(
            names,
            lower,
            upper,
            velocity,
            continuous,
            source_path=str(path),
            error=f"URDF does not exist: {path}",
        )

    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError) as exc:
        return JointLimitTable(
            names,
            lower,
            upper,
            velocity,
            continuous,
            source_path=str(path.resolve()),
            error=f"Unable to parse URDF: {type(exc).__name__}: {exc}",
        )

    joints = {element.attrib.get("name", ""): element for element in root.findall(".//joint")}
    for index, name in enumerate(names):
        element = joints.get(name)
        if element is None:
            continue
        joint_type = element.attrib.get("type", "")
        continuous[index] = joint_type == "continuous"
        limit = element.find("limit")
        if limit is None:
            continue
        if not continuous[index]:
            try:
                lower[index] = float(limit.attrib["lower"])
                upper[index] = float(limit.attrib["upper"])
            except (KeyError, TypeError, ValueError):
                lower[index] = np.nan
                upper[index] = np.nan
        try:
            velocity[index] = float(limit.attrib["velocity"])
        except (KeyError, TypeError, ValueError):
            velocity[index] = np.nan

    return JointLimitTable(
        names,
        lower,
        upper,
        velocity,
        continuous,
        source_path=str(path.resolve()),
    )


def _keys(data: Any) -> set[str]:
    if hasattr(data, "files"):
        return set(data.files)
    if isinstance(data, Mapping):
        return set(data)
    raise MotionSchemaError("Motion data must be a mapping or NumPy NPZ archive.")


def _name_tuple(value: Any, field_name: str, expected_count: int) -> tuple[str, ...]:
    array = np.asarray(value)
    if array.ndim != 1 or array.shape[0] != expected_count:
        raise MotionSchemaError(
            f"{field_name} must have shape ({expected_count},), got {array.shape}."
        )
    names = tuple(str(item) for item in array.tolist())
    if any(not name for name in names):
        raise MotionSchemaError(f"{field_name} must not contain empty names.")
    if len(set(names)) != len(names):
        raise MotionSchemaError(f"{field_name} must contain unique names.")
    return names


def _optional_scalar_text(data: Any, key: str, keys: set[str]) -> str:
    if key not in keys:
        return ""
    array = np.asarray(data[key])
    return str(array.reshape(()).item()) if array.size == 1 else ""


def validate_motion_npz(data: Any, *, path: str = "<memory>") -> MotionData:
    """Validate WBT shapes and names without hiding non-finite trajectory data.

    Non-finite values inside trajectory tensors are intentionally retained so
    the affected segment receives a hard ``nonfinite_values`` violation.
    """

    keys = _keys(data)
    missing = sorted(set(REQUIRED_MOTION_FIELDS).difference(keys))
    if missing:
        raise MotionSchemaError(f"{path}: missing required WBT fields: {missing}")

    fps_array = np.asarray(data["fps"], dtype=np.float64)
    if fps_array.size != 1:
        raise MotionSchemaError(f"{path}: fps must contain exactly one value, got shape {fps_array.shape}.")
    fps = float(fps_array.reshape(-1)[0])
    if not math.isfinite(fps) or fps <= 0.0:
        raise MotionSchemaError(f"{path}: fps must be finite and greater than zero, got {fps}.")

    arrays = {
        name: np.asarray(data[name], dtype=np.float64)
        for name in (
            "joint_pos",
            "joint_vel",
            "body_pos_w",
            "body_quat_w",
            "body_lin_vel_w",
            "body_ang_vel_w",
        )
    }
    joint_pos = arrays["joint_pos"]
    if joint_pos.ndim != 2 or joint_pos.shape[0] < 2 or joint_pos.shape[1] < 1:
        raise MotionSchemaError(
            f"{path}: joint_pos must have shape (T >= 2, num_joints >= 1), got {joint_pos.shape}."
        )
    if arrays["joint_vel"].shape != joint_pos.shape:
        raise MotionSchemaError(
            f"{path}: joint_vel shape {arrays['joint_vel'].shape} does not match joint_pos {joint_pos.shape}."
        )

    frame_count = joint_pos.shape[0]
    body_count: int | None = None
    for name, width in (
        ("body_pos_w", 3),
        ("body_quat_w", 4),
        ("body_lin_vel_w", 3),
        ("body_ang_vel_w", 3),
    ):
        array = arrays[name]
        if array.ndim != 3 or array.shape[0] != frame_count or array.shape[2] != width:
            raise MotionSchemaError(
                f"{path}: {name} must have shape ({frame_count}, num_bodies, {width}), got {array.shape}."
            )
        if body_count is None:
            body_count = int(array.shape[1])
        elif array.shape[1] != body_count:
            raise MotionSchemaError(
                f"{path}: {name} contains {array.shape[1]} bodies; expected {body_count}."
            )
    assert body_count is not None

    joint_names = _name_tuple(data["joint_names"], "joint_names", joint_pos.shape[1])
    body_names = _name_tuple(data["body_names"], "body_names", body_count)
    return MotionData(
        fps=fps,
        joint_pos=joint_pos,
        joint_vel=arrays["joint_vel"],
        body_pos_w=arrays["body_pos_w"],
        body_quat_w=arrays["body_quat_w"],
        body_lin_vel_w=arrays["body_lin_vel_w"],
        body_ang_vel_w=arrays["body_ang_vel_w"],
        joint_names=joint_names,
        body_names=body_names,
        path=path,
        source_file=_optional_scalar_text(data, "source_file", keys),
        source_format=_optional_scalar_text(data, "source_format", keys),
    )


def load_motion_npz(path: str | Path) -> MotionData:
    """Load and validate one real WBT ``.npz`` without pickle support."""

    motion_path = Path(path)
    try:
        with np.load(motion_path, allow_pickle=False) as loaded:
            return validate_motion_npz(loaded, path=str(motion_path.resolve()))
    except MotionSchemaError:
        raise
    except Exception as exc:
        raise MotionSchemaError(
            f"{motion_path}: unable to load WBT NPZ: {type(exc).__name__}: {exc}"
        ) from exc


def build_stage0_compatible_segment_bounds(
    num_frames: int, fps: float, segment_length_seconds: float
) -> tuple[tuple[int, int], ...]:
    """Build right-open bounds using Stage 0's round-to-nearest-even rule.

    Production metadata tooling should preferably pass bounds obtained directly
    from ``FixedLengthSegmentIndex.metadata()``.
    """

    if num_frames < 1:
        raise ValueError("num_frames must be positive.")
    if not math.isfinite(fps) or fps <= 0.0:
        raise ValueError("fps must be finite and positive.")
    if not math.isfinite(segment_length_seconds) or segment_length_seconds <= 0.0:
        raise ValueError("segment_length_seconds must be finite and positive.")
    segment_frames = max(1, int(round(fps * segment_length_seconds)))
    return tuple(
        (start, min(start + segment_frames, num_frames))
        for start in range(0, num_frames, segment_frames)
    )


def _validate_segment_bounds(
    bounds: Sequence[Sequence[int]], num_frames: int
) -> tuple[tuple[int, int], ...]:
    normalized = tuple((int(item[0]), int(item[1])) for item in bounds)
    if not normalized:
        raise ValueError("segment_bounds must not be empty.")
    expected_start = 0
    for start, end in normalized:
        if start != expected_start or end <= start or end > num_frames:
            raise ValueError(
                "segment_bounds must be contiguous, positive, right-open intervals covering the motion."
            )
        expected_start = end
    if expected_start != num_frames:
        raise ValueError("segment_bounds must cover every motion frame exactly once.")
    return normalized


def _gradient(values: np.ndarray, dt: float) -> np.ndarray:
    return np.gradient(values.astype(np.float64, copy=False), dt, axis=0, edge_order=1)


def _finite(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)[np.isfinite(values)]


def _percentile(values: np.ndarray, quantile: float) -> float:
    finite = _finite(values)
    return float(np.percentile(finite, quantile)) if finite.size else float("nan")


def _maximum(values: np.ndarray) -> float:
    finite = _finite(values)
    return float(np.max(finite)) if finite.size else float("nan")


def _minimum(values: np.ndarray) -> float:
    finite = _finite(values)
    return float(np.min(finite)) if finite.size else float("nan")


def _mean(values: np.ndarray) -> float:
    finite = _finite(values)
    return float(np.mean(finite)) if finite.size else float("nan")


def _piecewise_severity(raw_value: float, warning: float, reject: float) -> float:
    if not math.isfinite(raw_value):
        return 0.0
    if raw_value <= warning:
        return 0.0
    if raw_value >= reject:
        return 1.0
    return float((raw_value - warning) / (reject - warning))


def _severity(raw_value: float, metric_cfg: Mapping[str, Any]) -> float:
    return _piecewise_severity(
        raw_value,
        float(metric_cfg["warning_threshold"]),
        float(metric_cfg["reject_threshold"]),
    )


def _metric_result(
    raw_value: float,
    metric_cfg: Mapping[str, Any],
    *,
    available: bool = True,
    details: Mapping[str, Any] | None = None,
    force_hard: bool = False,
) -> MetricResult:
    if not available or not math.isfinite(raw_value):
        return MetricResult(float("nan"), 0.0, False, False, dict(details or {}))
    severity = _severity(float(raw_value), metric_cfg)
    hard = force_hard or (
        bool(metric_cfg.get("hard_at_reject", False))
        and float(raw_value) >= float(metric_cfg["reject_threshold"])
    )
    return MetricResult(float(raw_value), severity, True, hard, dict(details or {}))


def normalize_quaternions_wxyz(quaternions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return normalized WXYZ quaternions and their original norms.

    Invalid or near-zero quaternions are represented by NaNs in the normalized
    result.  This helper is shared by the quality and intrinsic-difficulty
    pipelines so both interpret the converted G1 rotations identically.
    """

    norms = np.linalg.norm(quaternions, axis=-1)
    valid = np.isfinite(norms) & (norms > 1.0e-12) & np.all(np.isfinite(quaternions), axis=-1)
    normalized = np.full_like(quaternions, np.nan, dtype=np.float64)
    normalized[valid] = quaternions[valid] / norms[valid, None]
    return normalized, norms


def quaternion_geodesic_transitions_wxyz(quaternions: np.ndarray) -> np.ndarray:
    """Return frame-to-frame rotation angles with ``q`` and ``-q`` equivalent."""

    normalized, _ = normalize_quaternions_wxyz(quaternions)
    output = np.full(quaternions.shape[:-1], np.nan, dtype=np.float64)
    if quaternions.shape[0] < 2:
        return output
    dots = np.sum(normalized[:-1] * normalized[1:], axis=-1)
    # abs(dot) makes q and -q exactly equivalent rotations.
    angles = 2.0 * np.arccos(np.clip(np.abs(dots), 0.0, 1.0))
    output[1:] = angles
    return output


def rotate_local_vectors_wxyz(quaternions: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """Rotate local XYZ vectors by WXYZ quaternions."""

    normalized, _ = normalize_quaternions_wxyz(quaternions)
    vector = np.broadcast_to(vectors, normalized.shape[:-1] + (3,))
    xyz = normalized[..., 1:]
    twice_cross = 2.0 * np.cross(xyz, vector)
    return vector + normalized[..., :1] * twice_cross + np.cross(xyz, twice_cross)


# Backward-compatible private aliases for the existing module-one audit.  New
# consumers should use the public names above.
_normalize_quaternions = normalize_quaternions_wxyz
_quaternion_geodesic_transitions = quaternion_geodesic_transitions_wxyz
_rotate_local_vectors_wxyz = rotate_local_vectors_wxyz


def _isolated_spike_score(
    magnitudes: np.ndarray,
    *,
    absolute_floor: float,
    relative_multiplier: float,
    radius: int = 2,
) -> np.ndarray:
    """Return a dimensionless local-isolation score for each frame/item."""

    values = np.asarray(magnitudes, dtype=np.float64)
    was_vector = values.ndim == 1
    if was_vector:
        values = values[:, None]
    scores = np.full_like(values, np.nan)
    for frame in range(values.shape[0]):
        before = values[max(0, frame - radius) : frame]
        after = values[frame + 1 : min(values.shape[0], frame + radius + 1)]
        neighbors = np.concatenate((before, after), axis=0)
        with np.errstate(all="ignore"):
            baseline = np.nanmedian(neighbors, axis=0) if neighbors.size else np.zeros(values.shape[1])
        baseline = np.where(np.isfinite(baseline), baseline, 0.0)
        threshold = np.maximum(float(absolute_floor), float(relative_multiplier) * baseline)
        current = values[frame]
        valid = np.isfinite(current)
        scores[frame, valid] = current[valid] / threshold[valid]
    return scores[:, 0] if was_vector else scores


def _spike_metric(
    name: str,
    magnitudes: np.ndarray | None,
    scores: np.ndarray | None,
    start: int,
    end: int,
    metric_configs: Mapping[str, Any],
) -> MetricResult:
    cfg = metric_configs[name]
    if magnitudes is None or scores is None:
        return _metric_result(float("nan"), cfg, available=False, details={"reason": "required body unavailable"})
    segment_magnitude = magnitudes[start:end]
    segment_score = scores[start:end]
    raw = _maximum(segment_score)
    finite_scores = _finite(segment_score)
    return _metric_result(
        raw,
        cfg,
        available=finite_scores.size > 0,
        details={
            "physical_p95": _percentile(segment_magnitude, 95.0),
            "physical_max": _maximum(segment_magnitude),
            "spike_frame_ratio": (
                float(np.mean(finite_scores > 1.0)) if finite_scores.size else float("nan")
            ),
        },
    )


def _continuity_metric(
    name: str,
    jumps: np.ndarray | None,
    scores: np.ndarray | None,
    start: int,
    end: int,
    metric_configs: Mapping[str, Any],
) -> MetricResult:
    cfg = metric_configs[name]
    if jumps is None or scores is None:
        return _metric_result(float("nan"), cfg, available=False, details={"reason": "root body unavailable"})
    segment_jumps = jumps[start:end]
    segment_scores = scores[start:end]
    raw = _maximum(segment_scores)
    finite_scores = _finite(segment_scores)
    return _metric_result(
        raw,
        cfg,
        available=finite_scores.size > 0,
        details={
            "jump_p95": _percentile(segment_jumps, 95.0),
            "jump_max": _maximum(segment_jumps),
            "isolated_jump_ratio": (
                float(np.mean(finite_scores > 1.0)) if finite_scores.size else float("nan")
            ),
        },
    )


def _score_and_status(
    metrics: Mapping[str, MetricResult],
    metric_configs: Mapping[str, Any],
    status_cfg: Mapping[str, Any],
) -> tuple[float, str, bool, bool, int, float, float, tuple[str, ...]]:
    """Apply the provisional quality-score and status rules.

    Required metrics are schema-critical: if any are unavailable, the segment
    is rejected even though unavailable metrics contribute neither severity nor
    weight.  Optional coverage is computed over the explicit
    ``optional_metric_coverage_profile`` instead of the current total metric
    count, so adding a new optional diagnostic later does not silently change
    old pass/borderline/reject labels.
    """

    available_metric_count = sum(1 for result in metrics.values() if result.available)
    metric_coverage = float(available_metric_count / len(METRIC_NAMES))
    weight_sum = sum(
        float(metric_configs[name]["weight"])
        for name, result in metrics.items()
        if result.available
    )
    weighted_severity = sum(
        float(metric_configs[name]["weight"]) * result.severity
        for name, result in metrics.items()
        if result.available
    )
    required_metrics = tuple(status_cfg.get("required_metrics", ()))
    missing_required_metrics = [name for name in required_metrics if not metrics[name].available]

    optional_profile = tuple(status_cfg.get("optional_metric_coverage_profile", ()))
    if optional_profile:
        available_optional_metrics = sum(1 for name in optional_profile if metrics[name].available)
        optional_metric_coverage = float(available_optional_metrics / len(optional_profile))
    else:
        optional_metric_coverage = 1.0
    insufficient_optional_metrics = optional_metric_coverage < float(
        status_cfg["minimum_optional_metric_coverage"]
    )

    insufficient = (
        insufficient_optional_metrics
        or weight_sum <= 0.0
        or bool(missing_required_metrics)
    )
    quality_score = (
        float(np.clip(1.0 - weighted_severity / weight_sum, 0.0, 1.0))
        if weight_sum
        else 0.0
    )
    hard_violation = any(result.hard_violation for result in metrics.values())
    severe_metric_count = sum(
        1 for result in metrics.values() if result.available and result.severity >= 1.0
    )
    warning_present = any(
        result.available and result.severity > 0.0 for result in metrics.values()
    )

    reasons: list[str] = []
    if hard_violation:
        reasons.append("hard_violation")
    if insufficient:
        reasons.append("insufficient_metrics")
    if missing_required_metrics:
        reasons.append("missing_required_metrics:" + ",".join(missing_required_metrics))
    if insufficient_optional_metrics:
        reasons.append(
            "optional_metric_coverage_below_threshold:"
            f"{optional_metric_coverage:.6g}<"
            f"{float(status_cfg['minimum_optional_metric_coverage']):.6g}"
        )
    if quality_score < float(status_cfg["reject_score_threshold"]):
        reasons.append("score_below_reject_threshold")
    if severe_metric_count >= int(status_cfg.get("reject_severity_count", len(METRIC_NAMES) + 1)):
        reasons.append("severe_metric_count")

    if reasons:
        quality_status = "reject"
    elif quality_score < float(status_cfg["pass_score_threshold"]) or (
        bool(status_cfg.get("borderline_on_warning", True)) and warning_present
    ):
        quality_status = "borderline"
        if quality_score < float(status_cfg["pass_score_threshold"]):
            reasons.append("score_below_pass_threshold")
        if warning_present:
            reasons.append("metric_warning")
    else:
        quality_status = "pass"

    return (
        quality_score,
        quality_status,
        hard_violation,
        insufficient,
        available_metric_count,
        metric_coverage,
        optional_metric_coverage,
        tuple(reasons),
    )


def _limit_excess(
    values: np.ndarray, lower: np.ndarray | None, upper: np.ndarray
) -> np.ndarray:
    result = np.full_like(values, np.nan, dtype=np.float64)
    if lower is None:
        available = np.isfinite(upper)
        result[:, available] = np.maximum(np.abs(values[:, available]) - upper[available], 0.0)
    else:
        available = np.isfinite(lower) & np.isfinite(upper)
        result[:, available] = np.maximum(
            np.maximum(lower[available] - values[:, available], values[:, available] - upper[available]),
            0.0,
        )
    return result


def _limit_metric(
    name: str,
    excess: np.ndarray,
    coverage: float,
    start: int,
    end: int,
    metric_configs: Mapping[str, Any],
) -> MetricResult:
    cfg = metric_configs[name]
    segment = excess[start:end]
    finite = np.isfinite(segment)
    available = bool(np.any(finite))
    positive = segment[finite & (segment > 0.0)]
    frame_violation = np.any(np.where(finite, segment > 0.0, False), axis=1) if segment.ndim == 2 else np.array([])
    raw = _maximum(segment)
    return _metric_result(
        raw,
        cfg,
        available=available,
        details={
            "coverage": coverage,
            "violation_frame_ratio": float(np.mean(frame_violation)) if frame_violation.size else float("nan"),
            "max_excess": raw,
            "mean_excess": float(np.mean(positive)) if positive.size else 0.0,
        },
    )


def _consistency_metric(
    name: str,
    errors: np.ndarray,
    start: int,
    end: int,
    metric_configs: Mapping[str, Any],
) -> MetricResult:
    cfg = metric_configs[name]
    segment = errors[start:end]
    raw = _percentile(segment, 95.0)
    maximum = _maximum(segment)
    result = _metric_result(
        raw,
        cfg,
        available=_finite(segment).size > 0,
        details={
            "mean_error": _mean(segment),
            "p95_error": raw,
            "max_error": maximum,
        },
    )
    if result.available and "max_warning_threshold" in cfg:
        max_severity = _piecewise_severity(
            maximum,
            float(cfg["max_warning_threshold"]),
            float(cfg["max_reject_threshold"]),
        )
        result.details["p95_severity"] = result.severity
        result.details["max_severity"] = max_severity
        result.severity = max(result.severity, max_severity)
        if bool(cfg.get("hard_at_reject", False)) and maximum >= float(cfg["max_reject_threshold"]):
            result.hard_violation = True
    return result


def _resolve_urdf_path(config: Mapping[str, Any], explicit_path: str | Path | None) -> Path | None:
    if explicit_path is not None:
        return Path(explicit_path).expanduser()
    robot = config.get("robot", {})
    configured = robot.get("urdf_path") if isinstance(robot, Mapping) else None
    if not configured:
        return None
    path = Path(str(configured)).expanduser()
    if path.is_absolute() or path.is_file():
        return path
    return PROJECT_ROOT / path


def _coerce_motion(motion: MotionData | Mapping[str, Any] | str | Path) -> MotionData:
    if isinstance(motion, MotionData):
        return motion
    if isinstance(motion, (str, Path)):
        return load_motion_npz(motion)
    return validate_motion_npz(motion)


def _coerce_config(config: Mapping[str, Any] | str | Path | None) -> dict[str, Any]:
    if config is None or isinstance(config, (str, Path)):
        return load_quality_config(config)
    copied = copy.deepcopy(dict(config))
    _validate_quality_config(copied)
    return copied


def audit_motion_segments(
    motion: MotionData | Mapping[str, Any] | str | Path,
    config: Mapping[str, Any] | str | Path | None = None,
    *,
    segment_bounds: Sequence[Sequence[int]] | None = None,
    joint_limits: JointLimitTable | None = None,
    urdf_path: str | Path | None = None,
) -> MotionQualityAudit:
    """Audit every right-open segment in one converted WBT motion."""

    motion_data = _coerce_motion(motion)
    quality_config = _coerce_config(config)
    metric_configs = quality_config["metrics"]
    segment_length = float(quality_config["segment_length_seconds"])
    if segment_bounds is None:
        bounds = build_stage0_compatible_segment_bounds(
            motion_data.num_frames, motion_data.fps, segment_length
        )
    else:
        bounds = _validate_segment_bounds(segment_bounds, motion_data.num_frames)

    if joint_limits is None:
        joint_limits = parse_urdf_joint_limits(
            _resolve_urdf_path(quality_config, urdf_path), motion_data.joint_names
        )
    if joint_limits.joint_names != motion_data.joint_names:
        raise ValueError("joint_limits must be aligned to motion joint_names.")

    dt = 1.0 / motion_data.fps
    frame_count = motion_data.num_frames

    joint_pos_for_diff = motion_data.joint_pos.copy()
    for joint_index in np.flatnonzero(joint_limits.continuous):
        joint_pos_for_diff[:, joint_index] = np.unwrap(joint_pos_for_diff[:, joint_index])

    joint_velocity_from_position = _gradient(joint_pos_for_diff, dt)
    body_velocity_from_position = _gradient(motion_data.body_pos_w, dt)
    joint_velocity_error = np.abs(joint_velocity_from_position - motion_data.joint_vel)
    body_velocity_error = np.linalg.norm(
        body_velocity_from_position - motion_data.body_lin_vel_w, axis=-1
    )

    joint_acceleration = _gradient(motion_data.joint_vel, dt)
    joint_jerk = _gradient(joint_acceleration, dt)
    joint_acceleration_magnitude = np.max(np.abs(joint_acceleration), axis=1)
    joint_jerk_magnitude = np.max(np.abs(joint_jerk), axis=1)

    body_position_jump = np.full(motion_data.body_pos_w.shape[:2], np.nan, dtype=np.float64)
    body_position_jump[1:] = np.linalg.norm(np.diff(motion_data.body_pos_w, axis=0), axis=-1)
    body_orientation_jump = _quaternion_geodesic_transitions(motion_data.body_quat_w)
    joint_position_jump = np.full(motion_data.joint_pos.shape, np.nan, dtype=np.float64)
    joint_position_jump[1:] = np.abs(np.diff(joint_pos_for_diff, axis=0))

    def spike_scores(name: str, magnitude: np.ndarray) -> np.ndarray:
        cfg = metric_configs[name]
        return _isolated_spike_score(
            magnitude,
            absolute_floor=float(cfg["absolute_floor"]),
            relative_multiplier=float(cfg["relative_multiplier"]),
        )

    joint_acceleration_score = spike_scores("joint_acceleration_spike", joint_acceleration_magnitude)
    joint_jerk_score = spike_scores("joint_jerk_spike", joint_jerk_magnitude)

    root_cfg = quality_config.get("root", {})
    root_name = str(root_cfg.get("body_name", "pelvis")) if isinstance(root_cfg, Mapping) else "pelvis"
    root_index = motion_data.body_names.index(root_name) if root_name in motion_data.body_names else None
    root_position_jump = None
    root_orientation_jump = None
    root_position_score = None
    root_orientation_score = None
    root_linear_acceleration_magnitude = None
    root_angular_acceleration_magnitude = None
    root_linear_acceleration_score = None
    root_angular_acceleration_score = None
    if root_index is not None:
        root_position_jump = body_position_jump[:, root_index]
        root_orientation_jump = body_orientation_jump[:, root_index]
        root_position_score = spike_scores("root_position_continuity", root_position_jump)
        root_orientation_score = spike_scores("root_orientation_continuity", root_orientation_jump)
        root_linear_acceleration_magnitude = np.linalg.norm(
            _gradient(motion_data.body_lin_vel_w[:, root_index], dt), axis=-1
        )
        root_angular_acceleration_magnitude = np.linalg.norm(
            _gradient(motion_data.body_ang_vel_w[:, root_index], dt), axis=-1
        )
        root_linear_acceleration_score = spike_scores(
            "root_linear_acceleration_spike", root_linear_acceleration_magnitude
        )
        root_angular_acceleration_score = spike_scores(
            "root_angular_acceleration_spike", root_angular_acceleration_magnitude
        )

    body_position_score = spike_scores("body_position_continuity", body_position_jump)
    body_orientation_score = spike_scores("body_orientation_continuity", body_orientation_jump)
    joint_position_score = spike_scores("joint_position_continuity", joint_position_jump)

    position_excess = _limit_excess(
        motion_data.joint_pos, joint_limits.lower, joint_limits.upper
    )
    velocity_excess = _limit_excess(
        motion_data.joint_vel, None, joint_limits.velocity
    )

    numeric_arrays = (
        motion_data.joint_pos,
        motion_data.joint_vel,
        motion_data.body_pos_w,
        motion_data.body_quat_w,
        motion_data.body_lin_vel_w,
        motion_data.body_ang_vel_w,
    )
    nonfinite_per_frame = np.zeros(frame_count, dtype=np.int64)
    values_per_frame = 0
    for array in numeric_arrays:
        flattened = array.reshape(frame_count, -1)
        nonfinite_per_frame += np.count_nonzero(~np.isfinite(flattened), axis=1)
        values_per_frame += flattened.shape[1]

    _, quaternion_norm = _normalize_quaternions(motion_data.body_quat_w)
    quaternion_norm_error = np.abs(quaternion_norm - 1.0)

    ground_cfg = quality_config["ground"]
    configured_foot_names = tuple(str(name) for name in ground_cfg.get("foot_body_names", ()))
    offsets = ground_cfg.get("sole_local_offsets_m", {})
    foot_indexes: list[int] = []
    foot_offsets: list[np.ndarray] = []
    missing_foot_names: list[str] = []
    for name in configured_foot_names:
        if name not in motion_data.body_names:
            missing_foot_names.append(name)
            continue
        offset = np.asarray(offsets.get(name, (0.0, 0.0, 0.0)), dtype=np.float64)
        if offset.shape != (3,) or not np.all(np.isfinite(offset)):
            raise ValueError(f"Sole offset for '{name}' must contain three finite values.")
        foot_indexes.append(motion_data.body_names.index(name))
        foot_offsets.append(offset)
    if missing_foot_names and bool(ground_cfg.get("require_configured_foot_bodies", True)):
        raise ValueError(
            "Configured foot body names are absent from motion body_names: "
            f"{missing_foot_names}. Available body_names include: {list(motion_data.body_names[:20])}."
        )

    penetration_depth = None
    sole_height = None
    foot_contact = None
    foot_horizontal_speed = None
    matched_foot_names: tuple[str, ...] = ()
    if foot_indexes:
        matched_foot_names = tuple(motion_data.body_names[index] for index in foot_indexes)
        foot_position = motion_data.body_pos_w[:, foot_indexes, :]
        foot_quaternion = motion_data.body_quat_w[:, foot_indexes, :]
        local_offsets = np.asarray(foot_offsets, dtype=np.float64)[None, :, :]
        sole_position = foot_position + _rotate_local_vectors_wxyz(foot_quaternion, local_offsets)
        sole_velocity = _gradient(sole_position, dt)
        ground_z = float(ground_cfg["z_m"])
        sole_height = sole_position[..., 2] - ground_z
        penetration_depth = np.maximum(-sole_height, 0.0)
        foot_contact = (
            (sole_height <= float(ground_cfg["contact_height_threshold_m"]))
            & (
                np.abs(sole_velocity[..., 2])
                <= float(ground_cfg["contact_vertical_speed_threshold_mps"])
            )
        )
        foot_horizontal_speed = np.linalg.norm(sole_velocity[..., :2], axis=-1)

    segment_results: list[SegmentQualityResult] = []
    status_cfg = quality_config["status"]
    for local_id, (start, end) in enumerate(bounds):
        metrics: dict[str, MetricResult] = {}

        nonfinite_raw = float(np.sum(nonfinite_per_frame[start:end])) / float(
            (end - start) * values_per_frame
        )
        metrics["nonfinite_values"] = _metric_result(
            nonfinite_raw,
            metric_configs["nonfinite_values"],
            details={
                "nonfinite_count": int(np.sum(nonfinite_per_frame[start:end])),
                "value_count": int((end - start) * values_per_frame),
            },
        )

        quaternion_segment = quaternion_norm_error[start:end]
        metrics["quaternion_norm"] = _metric_result(
            _maximum(quaternion_segment),
            metric_configs["quaternion_norm"],
            available=_finite(quaternion_segment).size > 0,
            details={
                "mean_norm_error": _mean(quaternion_segment),
                "max_norm_error": _maximum(quaternion_segment),
            },
        )
        metrics["joint_position_limits"] = _limit_metric(
            "joint_position_limits",
            position_excess,
            joint_limits.position_coverage,
            start,
            end,
            metric_configs,
        )
        metrics["joint_velocity_limits"] = _limit_metric(
            "joint_velocity_limits",
            velocity_excess,
            joint_limits.velocity_coverage,
            start,
            end,
            metric_configs,
        )
        metrics["joint_velocity_consistency"] = _consistency_metric(
            "joint_velocity_consistency", joint_velocity_error, start, end, metric_configs
        )
        metrics["body_velocity_consistency"] = _consistency_metric(
            "body_velocity_consistency", body_velocity_error, start, end, metric_configs
        )
        metrics["joint_acceleration_spike"] = _spike_metric(
            "joint_acceleration_spike",
            joint_acceleration_magnitude,
            joint_acceleration_score,
            start,
            end,
            metric_configs,
        )
        metrics["joint_jerk_spike"] = _spike_metric(
            "joint_jerk_spike",
            joint_jerk_magnitude,
            joint_jerk_score,
            start,
            end,
            metric_configs,
        )
        metrics["root_linear_acceleration_spike"] = _spike_metric(
            "root_linear_acceleration_spike",
            root_linear_acceleration_magnitude,
            root_linear_acceleration_score,
            start,
            end,
            metric_configs,
        )
        metrics["root_angular_acceleration_spike"] = _spike_metric(
            "root_angular_acceleration_spike",
            root_angular_acceleration_magnitude,
            root_angular_acceleration_score,
            start,
            end,
            metric_configs,
        )
        metrics["root_position_continuity"] = _continuity_metric(
            "root_position_continuity",
            root_position_jump,
            root_position_score,
            start,
            end,
            metric_configs,
        )
        metrics["root_orientation_continuity"] = _continuity_metric(
            "root_orientation_continuity",
            root_orientation_jump,
            root_orientation_score,
            start,
            end,
            metric_configs,
        )
        metrics["body_position_continuity"] = _continuity_metric(
            "body_position_continuity",
            body_position_jump,
            body_position_score,
            start,
            end,
            metric_configs,
        )
        metrics["body_orientation_continuity"] = _continuity_metric(
            "body_orientation_continuity",
            body_orientation_jump,
            body_orientation_score,
            start,
            end,
            metric_configs,
        )
        metrics["joint_position_continuity"] = _continuity_metric(
            "joint_position_continuity",
            joint_position_jump,
            joint_position_score,
            start,
            end,
            metric_configs,
        )

        penetration_cfg = metric_configs["ground_penetration"]
        if penetration_depth is None:
            metrics["ground_penetration"] = _metric_result(
                float("nan"),
                penetration_cfg,
                available=False,
                details={"reason": "configured foot body names are absent"},
            )
        else:
            segment_penetration = penetration_depth[start:end]
            penetration_raw = _maximum(segment_penetration)
            metrics["ground_penetration"] = _metric_result(
                penetration_raw,
                penetration_cfg,
                available=_finite(segment_penetration).size > 0,
                details={
                    "matched_foot_count": len(foot_indexes),
                    "matched_foot_names": matched_foot_names,
                    "coordinate_frame": "world",
                    "ground_z_m": float(ground_cfg["z_m"]),
                    "height_reference": "configured sole reference point",
                    "sole_local_offsets_m": {
                        name: [float(value) for value in foot_offsets[index].tolist()]
                        for index, name in enumerate(matched_foot_names)
                    },
                    "sole_offset_source_note": str(ground_cfg.get("sole_offset_note", "")),
                    "sole_min_height_by_body_m": {
                        name: _minimum(sole_height[start:end, index])
                        for index, name in enumerate(matched_foot_names)
                    },
                    "max_depth_m": penetration_raw,
                    "penetration_frame_ratio": float(
                        np.mean(np.any(segment_penetration > 0.0, axis=1))
                    ),
                },
            )

        sliding_cfg = metric_configs["foot_sliding"]
        if foot_contact is None or foot_horizontal_speed is None:
            metrics["foot_sliding"] = _metric_result(
                float("nan"),
                sliding_cfg,
                available=False,
                details={"reason": "configured foot body names are absent"},
            )
        else:
            contact = foot_contact[start:end]
            speed = foot_horizontal_speed[start:end]
            contact_speeds = speed[contact & np.isfinite(speed)]
            sliding_raw = float(np.percentile(contact_speeds, 95.0)) if contact_speeds.size else 0.0
            severe_contact_count = int(
                np.count_nonzero(contact_speeds >= float(sliding_cfg["reject_threshold"]))
            )
            severe_contact_ratio = (
                float(severe_contact_count / contact_speeds.size) if contact_speeds.size else 0.0
            )
            persistent_reject = (
                contact_speeds.size >= int(sliding_cfg["persistent_reject_min_contact_samples"])
                and severe_contact_ratio >= float(sliding_cfg["persistent_reject_frame_ratio"])
            )
            sliding_result = _metric_result(
                sliding_raw,
                sliding_cfg,
                details={
                    "matched_foot_count": len(foot_indexes),
                    "matched_foot_names": matched_foot_names,
                    "coordinate_frame": "world",
                    "height_reference": "configured sole reference point",
                    "contact_height_threshold_m": float(ground_cfg["contact_height_threshold_m"]),
                    "contact_vertical_speed_threshold_mps": float(
                        ground_cfg["contact_vertical_speed_threshold_mps"]
                    ),
                    "contact_frame_count": int(contact_speeds.size),
                    "contact_speed_mean_mps": (
                        float(np.mean(contact_speeds)) if contact_speeds.size else 0.0
                    ),
                    "contact_speed_p95_mps": sliding_raw,
                    "contact_speed_max_mps": (
                        float(np.max(contact_speeds)) if contact_speeds.size else 0.0
                    ),
                    "sliding_frame_ratio": (
                        float(
                            np.mean(
                                contact_speeds
                                > float(sliding_cfg["warning_threshold"])
                            )
                        )
                        if contact_speeds.size
                        else 0.0
                    ),
                    "severe_contact_count": severe_contact_count,
                    "severe_contact_ratio": severe_contact_ratio,
                    "persistent_reject": persistent_reject,
                },
            )
            if persistent_reject:
                sliding_result.hard_violation = True
            metrics["foot_sliding"] = sliding_result

        (
            quality_score,
            quality_status,
            hard_violation,
            insufficient,
            available_metric_count,
            metric_coverage,
            optional_metric_coverage,
            reasons,
        ) = _score_and_status(metrics, metric_configs, status_cfg)

        segment_results.append(
            SegmentQualityResult(
                local_segment_id=local_id,
                start_frame=start,
                end_frame_exclusive=end,
                quality_score=quality_score,
                quality_status=quality_status,
                hard_violation=hard_violation,
                insufficient_metrics=insufficient,
                available_metric_count=available_metric_count,
                metric_coverage=metric_coverage,
                optional_metric_coverage=optional_metric_coverage,
                metrics=metrics,
                status_reasons=reasons,
            )
        )

    return MotionQualityAudit(
        motion_path=motion_data.path,
        fps=motion_data.fps,
        num_frames=motion_data.num_frames,
        segment_length_seconds=segment_length,
        segment_results=segment_results,
        joint_position_limit_coverage=joint_limits.position_coverage,
        joint_velocity_limit_coverage=joint_limits.velocity_coverage,
        joint_limit_source=joint_limits.source_path,
        joint_limit_error=joint_limits.error,
    )


__all__ = [
    "DEFAULT_QUALITY_CONFIG_PATH",
    "METRIC_NAMES",
    "QUALITY_SCHEMA_VERSION",
    "JointLimitTable",
    "MetricResult",
    "METRIC_SEMANTICS",
    "MotionData",
    "MotionQualityAudit",
    "MotionSchemaError",
    "SegmentQualityResult",
    "audit_motion_segments",
    "build_stage0_compatible_segment_bounds",
    "load_motion_npz",
    "load_quality_config",
    "parse_urdf_joint_limits",
    "validate_motion_npz",
]
