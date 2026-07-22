"""Policy-independent intrinsic difficulty for converted WBT/G1 trajectories.

The module is deliberately NumPy-only (apart from reusing module one's WBT
schema and sole-frame helpers).  Every derivative, contact state, and switch
signal is computed over a complete motion before the shared Stage-0 segment
bounds are applied.  No policy output, reward, success signal, or quality
label is accepted by this API.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

import numpy as np

try:  # Package import during training/installed use.
    from .quality import MotionData, normalize_quaternions_wxyz, rotate_local_vectors_wxyz
except ImportError:  # Direct pure-file import used by offline builders/tests.
    from quality import MotionData, normalize_quaternions_wxyz, rotate_local_vectors_wxyz


DIFFICULTY_CONFIG_SCHEMA_VERSION: Final[int] = 1
DIFFICULTY_PROFILE_SCHEMA_VERSION: Final[str] = "wbt.difficulty_profile.v1"
DEFAULT_ALGORITHM_SCHEMA_VERSION: Final[str] = "wbt.intrinsic_difficulty.kinematic_contact.v2"
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[4]
DEFAULT_DIFFICULTY_CONFIG_PATH: Final[Path] = (
    PROJECT_ROOT / "configs" / "difficulty" / "g1_segment_difficulty.yaml"
)

FEATURE_NAMES: Final[tuple[str, ...]] = (
    "root_linear_speed_mean",
    "root_linear_speed_p95",
    "root_linear_acceleration_mean",
    "root_linear_acceleration_p95",
    "root_angular_speed_mean",
    "root_angular_speed_p95",
    "root_angular_acceleration_mean",
    "root_angular_acceleration_p95",
    "joint_speed_mean",
    "joint_speed_p95",
    "joint_acceleration_mean",
    "joint_acceleration_p95",
    "joint_range_mean",
    "joint_range_p90",
    "body_height_mean",
    "body_height_range",
    "body_height_std",
    "hand_speed_p95",
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
)


def canonical_json_sha256(value: Mapping[str, Any]) -> str:
    """Hash a JSON-compatible mapping with deterministic serialization."""

    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _finite_scalar(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Expected a finite numeric value, got {value!r}.")
    return result


def _integer_value(value: Any, name: str) -> int:
    """Return a JSON integer without silently truncating floats or booleans."""

    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer, got {value!r}.")
    return int(value)


def _validate_sha256(value: str, name: str) -> None:
    """Validate a required Profile SHA256 identity."""

    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value.lower()
    ):
        raise ValueError(f"Difficulty profile field '{name}' is not a SHA256 hex digest.")


def validate_difficulty_config(config: Mapping[str, Any]) -> None:
    """Validate the frozen feature definition and all unit-bearing parameters."""

    schema_version = _integer_value(config.get("schema_version", -1), "schema_version")
    if schema_version != DIFFICULTY_CONFIG_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported difficulty config schema {config.get('schema_version')}; "
            f"expected {DIFFICULTY_CONFIG_SCHEMA_VERSION}."
        )
    algorithm = config.get("algorithm_schema_version")
    if not isinstance(algorithm, str) or not algorithm.strip():
        raise ValueError("algorithm_schema_version must be a non-empty string.")
    if algorithm != DEFAULT_ALGORITHM_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported difficulty algorithm schema {algorithm!r}; "
            f"expected {DEFAULT_ALGORITHM_SCHEMA_VERSION!r}."
        )
    segment_length = _finite_scalar(config.get("segment_length_seconds", float("nan")))
    if segment_length <= 0.0:
        raise ValueError("segment_length_seconds must be greater than zero.")
    num_bins = _integer_value(config.get("num_bins", 0), "num_bins")
    if num_bins < 2:
        raise ValueError("num_bins must be at least 2.")
    scale_epsilon = _finite_scalar(config.get("robust_scale_epsilon", float("nan")))
    near_constant = _finite_scalar(config.get("near_constant_scale_threshold", float("nan")))
    fallback_quantiles = np.asarray(config.get("zero_mad_fallback_quantiles"), dtype=np.float64)
    fallback_divisor = _finite_scalar(
        config.get("zero_mad_fallback_scale_divisor", float("nan"))
    )
    robust_clip = _finite_scalar(config.get("robust_clip", float("nan")))
    if (
        scale_epsilon <= 0.0
        or near_constant < scale_epsilon
        or fallback_quantiles.shape != (2,)
        or not np.isfinite(fallback_quantiles).all()
        or fallback_quantiles[0] < 0.0
        or fallback_quantiles[1] > 1.0
        or fallback_quantiles[0] >= fallback_quantiles[1]
        or fallback_divisor <= 0.0
        or robust_clip <= 0.0
    ):
        raise ValueError(
            "robust_scale_epsilon must be positive, near_constant_scale_threshold must be at least epsilon, "
            "fallback quantiles/divisor must be valid, and robust_clip must be positive."
        )
    optional_coverage = _finite_scalar(config.get("minimum_optional_feature_coverage", float("nan")))
    if not 0.0 <= optional_coverage <= 1.0:
        raise ValueError("minimum_optional_feature_coverage must be in [0, 1].")

    features = config.get("features")
    if not isinstance(features, Mapping):
        raise ValueError("Difficulty config must contain a features mapping.")
    if tuple(features) != FEATURE_NAMES:
        missing = sorted(set(FEATURE_NAMES).difference(features))
        extra = sorted(set(features).difference(FEATURE_NAMES))
        raise ValueError(
            "Difficulty feature order/schema does not match the implementation; "
            f"missing={missing}, extra={extra}."
        )
    positive_weight = False
    for name in FEATURE_NAMES:
        item = features[name]
        if not isinstance(item, Mapping):
            raise ValueError(f"Feature config '{name}' must be a mapping.")
        for text_field in ("unit", "aggregation", "family", "description"):
            if not isinstance(item.get(text_field), str) or not str(item[text_field]).strip():
                raise ValueError(f"Feature '{name}' must declare a non-empty {text_field}.")
        if not isinstance(item.get("required"), bool):
            raise ValueError(f"Feature '{name}' required must be boolean.")
        direction = _integer_value(item.get("direction", 0), f"features.{name}.direction")
        if direction not in {-1, 1}:
            raise ValueError(f"Feature '{name}' direction must be -1 or +1.")
        weight = _finite_scalar(item.get("weight", float("nan")))
        if weight < 0.0:
            raise ValueError(f"Feature '{name}' weight must be non-negative.")
        positive_weight |= weight > 0.0
    if not positive_weight:
        raise ValueError("At least one difficulty feature must have positive weight.")

    root_name = config.get("root_body_name")
    if not isinstance(root_name, str) or not root_name:
        raise ValueError("root_body_name must be a non-empty string.")
    for field, expected in (("hand_body_names", 2), ("foot_body_names", 2)):
        names = config.get(field)
        if (
            not isinstance(names, list)
            or len(names) != expected
            or not all(isinstance(name, str) and name for name in names)
            or len(set(names)) != len(names)
        ):
            raise ValueError(f"{field} must contain exactly {expected} unique non-empty names.")
    offsets = config.get("sole_local_offsets_m")
    if not isinstance(offsets, Mapping):
        raise ValueError("sole_local_offsets_m must be a mapping.")
    for name in config["foot_body_names"]:
        offset = np.asarray(offsets.get(name), dtype=np.float64)
        if offset.shape != (3,) or not np.isfinite(offset).all():
            raise ValueError(f"sole_local_offsets_m['{name}'] must contain three finite metre values.")

    contact = config.get("contact")
    if not isinstance(contact, Mapping):
        raise ValueError("Difficulty config must contain a contact mapping.")
    ground_z = _finite_scalar(contact.get("ground_z_m", float("nan")))
    enter_height = _finite_scalar(contact.get("contact_height_threshold_m", float("nan")))
    release_height = _finite_scalar(contact.get("release_height_threshold_m", float("nan")))
    enter_speed = _finite_scalar(contact.get("contact_vertical_speed_threshold_mps", float("nan")))
    release_speed = _finite_scalar(contact.get("release_vertical_speed_threshold_mps", float("nan")))
    minimum_duration = _finite_scalar(contact.get("minimum_state_duration_seconds", float("nan")))
    del ground_z
    if (
        enter_height < 0.0
        or release_height < enter_height
        or enter_speed < 0.0
        or release_speed < enter_speed
        or minimum_duration < 0.0
    ):
        raise ValueError(
            "Contact thresholds must be non-negative; release thresholds must be no smaller than entry "
            "thresholds; minimum_state_duration_seconds must be non-negative."
        )

    aggregation = config.get("motion_aggregation")
    if not isinstance(aggregation, Mapping):
        raise ValueError("Difficulty config must contain motion_aggregation.")
    mean_weight = _finite_scalar(aggregation.get("duration_weighted_mean_weight", float("nan")))
    p90_weight = _finite_scalar(aggregation.get("segment_p90_weight", float("nan")))
    if mean_weight < 0.0 or p90_weight < 0.0 or mean_weight + p90_weight <= 0.0:
        raise ValueError("Motion aggregation weights must be non-negative with positive total weight.")


def load_difficulty_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the JSON-compatible YAML difficulty definition."""

    config_path = DEFAULT_DIFFICULTY_CONFIG_PATH if path is None else Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Difficulty config does not exist: {config_path}")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Difficulty config must use JSON-compatible YAML syntax: {config_path}: {exc}") from exc
    if not isinstance(config, dict):
        raise ValueError("Difficulty config root must be an object.")
    validate_difficulty_config(config)
    return config


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
                "segment_bounds must be contiguous positive right-open intervals covering the complete motion."
            )
        expected_start = end
    if expected_start != num_frames:
        raise ValueError("segment_bounds must cover every motion frame exactly once.")
    return normalized


def quaternion_angular_velocity_wxyz(quaternions: np.ndarray, fps: float) -> np.ndarray:
    """Return shortest-path rotation-vector velocities for a WXYZ sequence.

    The transition from frame ``i-1`` to ``i`` is assigned to frame ``i``.
    Frame zero copies the first valid transition so a segment does not gain an
    artificial zero solely because the sequence has no predecessor.
    """

    q = np.asarray(quaternions, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != 4:
        raise ValueError(f"quaternions must have shape (T, 4), got {q.shape}.")
    if q.shape[0] < 2:
        raise ValueError("At least two quaternion frames are required.")
    fps_value = _finite_scalar(fps)
    if fps_value <= 0.0:
        raise ValueError("fps must be positive.")
    normalized, _ = normalize_quaternions_wxyz(q)
    previous = normalized[:-1]
    current = normalized[1:].copy()
    dots = np.sum(previous * current, axis=1)
    current[dots < 0.0] *= -1.0

    # conjugate(previous) * current in WXYZ convention.
    w1 = previous[:, 0]
    v1 = -previous[:, 1:]
    w2 = current[:, 0]
    v2 = current[:, 1:]
    relative_w = w1 * w2 - np.sum(v1 * v2, axis=1)
    relative_v = w1[:, None] * v2 + w2[:, None] * v1 + np.cross(v1, v2)
    negative = relative_w < 0.0
    relative_w[negative] *= -1.0
    relative_v[negative] *= -1.0
    vector_norm = np.linalg.norm(relative_v, axis=1)
    angle = 2.0 * np.arctan2(vector_norm, np.clip(relative_w, 0.0, None))
    rotation_vector = np.zeros_like(relative_v)
    valid = np.isfinite(angle) & np.isfinite(vector_norm)
    nonzero = valid & (vector_norm > 1.0e-12)
    rotation_vector[nonzero] = relative_v[nonzero] * (angle[nonzero] / vector_norm[nonzero])[:, None]
    rotation_vector[~valid] = np.nan

    velocity = np.full((q.shape[0], 3), np.nan, dtype=np.float64)
    velocity[1:] = rotation_vector * fps_value
    velocity[0] = velocity[1]
    return velocity


def _gradient_vectors(values: np.ndarray, fps: float) -> np.ndarray:
    """Differentiate a full motion and attribute each transition to its destination frame."""

    array = np.asarray(values, dtype=np.float64)
    if array.shape[0] < 2:
        return np.full_like(array, np.nan, dtype=np.float64)
    derivatives = np.full_like(array, np.nan, dtype=np.float64)
    derivatives[1:] = (array[1:] - array[:-1]) * float(fps)
    derivatives[0] = derivatives[1]
    return derivatives


def _hysteretic_contact_state(
    height: np.ndarray,
    vertical_speed: np.ndarray,
    *,
    enter_height: float,
    release_height: float,
    enter_vertical_speed: float,
    release_vertical_speed: float,
) -> np.ndarray:
    heights = np.asarray(height, dtype=np.float64)
    speeds = np.abs(np.asarray(vertical_speed, dtype=np.float64))
    if heights.ndim != 1 or speeds.shape != heights.shape:
        raise ValueError("height and vertical_speed must be same-shape one-dimensional arrays.")
    finite = np.isfinite(heights) & np.isfinite(speeds)
    state = bool(
        finite[0] and heights[0] <= enter_height and speeds[0] <= enter_vertical_speed
    )
    output = np.zeros(heights.size, dtype=bool)
    for index in range(heights.size):
        if not finite[index]:
            state = False
        elif state:
            state = bool(
                heights[index] <= release_height and speeds[index] <= release_vertical_speed
            )
        else:
            state = bool(
                heights[index] <= enter_height and speeds[index] <= enter_vertical_speed
            )
        output[index] = state
    return output


def debounce_boolean_runs(values: Sequence[bool] | np.ndarray, minimum_frames: int) -> np.ndarray:
    """Require a candidate state to persist before accepting its transition.

    Confirmed transitions are attributed to the candidate run's first frame.
    Alternating threshold jitter therefore remains in the initial stable state
    instead of being resolved according to an in-place run traversal order.
    """

    raw = np.asarray(values, dtype=bool)
    if raw.ndim != 1:
        raise ValueError("values must be one-dimensional.")
    if minimum_frames < 1:
        raise ValueError("minimum_frames must be at least one.")
    if raw.size <= 1 or minimum_frames == 1:
        return raw.copy()

    output = np.full(raw.shape, raw[0], dtype=bool)
    stable_state = bool(raw[0])
    candidate_state = stable_state
    candidate_start = 0
    candidate_count = 0
    confirmed_transition = False
    for index in range(1, raw.size):
        observed = bool(raw[index])
        if observed == stable_state:
            candidate_state = stable_state
            candidate_count = 0
            output[index] = stable_state
            continue
        if observed != candidate_state:
            candidate_state = observed
            candidate_start = index
            candidate_count = 1
        else:
            candidate_count += 1
        output[index] = stable_state
        if candidate_count < minimum_frames:
            continue

        stable_state = candidate_state
        output[candidate_start : index + 1] = stable_state
        if not confirmed_transition and candidate_start < minimum_frames:
            output[:candidate_start] = stable_state
        confirmed_transition = True
        candidate_count = 0
    return output


def infer_debounced_foot_contacts(
    sole_positions_w: np.ndarray, fps: float, contact_config: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    """Infer full-motion foot contacts using hysteresis and time-based debounce.

    Returns ``(contacts, sole_velocities_w)`` with shapes ``(T, 2)`` and
    ``(T, 2, 3)``.  The input positions are the same configured sole reference
    points used by module one's geometric audit.
    """

    positions = np.asarray(sole_positions_w, dtype=np.float64)
    if positions.ndim != 3 or positions.shape[1:] != (2, 3):
        raise ValueError(f"sole_positions_w must have shape (T, 2, 3), got {positions.shape}.")
    if positions.shape[0] < 2:
        raise ValueError("At least two frames are required for contact inference.")
    if not np.isfinite(positions).all():
        raise ValueError("Non-finite sole positions cannot be interpreted as contact or flight.")
    fps_value = _finite_scalar(fps)
    if fps_value <= 0.0:
        raise ValueError("fps must be positive for contact inference.")
    velocities = _gradient_vectors(positions, fps_value)
    ground_z = float(contact_config["ground_z_m"])
    minimum_frames = max(
        1, int(round(float(contact_config["minimum_state_duration_seconds"]) * fps_value))
    )
    contacts = np.zeros((positions.shape[0], 2), dtype=bool)
    for foot_index in range(2):
        state = _hysteretic_contact_state(
            positions[:, foot_index, 2] - ground_z,
            velocities[:, foot_index, 2],
            enter_height=float(contact_config["contact_height_threshold_m"]),
            release_height=float(contact_config["release_height_threshold_m"]),
            enter_vertical_speed=float(contact_config["contact_vertical_speed_threshold_mps"]),
            release_vertical_speed=float(contact_config["release_vertical_speed_threshold_mps"]),
        )
        contacts[:, foot_index] = debounce_boolean_runs(state, minimum_frames)
    return contacts, velocities


@dataclass(frozen=True)
class MotionDifficultyFeatures:
    """All configured raw features for the Stage-0 segments of one motion."""

    feature_names: tuple[str, ...]
    values: np.ndarray
    available: np.ndarray
    duration_seconds: np.ndarray

    @property
    def num_segments(self) -> int:
        return int(self.values.shape[0])


def _finite_stat(values: np.ndarray, statistic: str) -> tuple[float, bool]:
    finite = np.asarray(values, dtype=np.float64)
    if finite.size == 0 or not np.isfinite(finite).all():
        return float("nan"), False
    if statistic == "mean":
        return float(np.mean(finite)), True
    if statistic == "std":
        return float(np.std(finite)), True
    if statistic == "range":
        return float(np.max(finite) - np.min(finite)), True
    if statistic == "p90":
        return float(np.percentile(finite, 90.0)), True
    if statistic == "p95":
        return float(np.percentile(finite, 95.0)), True
    raise ValueError(f"Unknown statistic: {statistic}")


def _joint_ranges(joint_pos: np.ndarray) -> np.ndarray:
    ranges = np.full(joint_pos.shape[1], np.nan, dtype=np.float64)
    for joint_index in range(joint_pos.shape[1]):
        values = joint_pos[:, joint_index]
        if values.size and np.isfinite(values).all():
            ranges[joint_index] = np.max(values) - np.min(values)
    return ranges


def extract_motion_difficulty_features(
    motion: MotionData,
    segment_bounds: Sequence[Sequence[int]],
    config: Mapping[str, Any],
) -> MotionDifficultyFeatures:
    """Extract policy-independent features after full-motion preprocessing."""

    validate_difficulty_config(config)
    bounds = _validate_segment_bounds(segment_bounds, motion.num_frames)
    body_lookup = {name: index for index, name in enumerate(motion.body_names)}
    root_name = str(config["root_body_name"])
    if root_name not in body_lookup:
        raise ValueError(f"{motion.path}: configured root body '{root_name}' is unavailable.")
    root_index = body_lookup[root_name]
    fps = float(motion.fps)

    root_velocity = motion.body_lin_vel_w[:, root_index]
    root_linear_speed = np.linalg.norm(root_velocity, axis=1)
    root_linear_acceleration = np.linalg.norm(_gradient_vectors(root_velocity, fps), axis=1)
    root_angular_velocity = quaternion_angular_velocity_wxyz(
        motion.body_quat_w[:, root_index], fps
    )
    root_angular_speed = np.linalg.norm(root_angular_velocity, axis=1)
    root_angular_acceleration = np.linalg.norm(
        _gradient_vectors(root_angular_velocity, fps), axis=1
    )
    joint_speed = np.abs(motion.joint_vel)
    joint_acceleration = np.abs(_gradient_vectors(motion.joint_vel, fps))
    root_height = motion.body_pos_w[:, root_index, 2]

    hand_names = tuple(str(name) for name in config["hand_body_names"])
    hand_available = all(name in body_lookup for name in hand_names)
    hand_speeds: np.ndarray | None = None
    if hand_available:
        hand_indexes = [body_lookup[name] for name in hand_names]
        hand_speeds = np.linalg.norm(motion.body_lin_vel_w[:, hand_indexes], axis=2)

    foot_names = tuple(str(name) for name in config["foot_body_names"])
    foot_available = all(name in body_lookup for name in foot_names)
    sole_speeds: np.ndarray | None = None
    contacts: np.ndarray | None = None
    switch_signal: np.ndarray | None = None
    if foot_available:
        foot_indexes = [body_lookup[name] for name in foot_names]
        foot_quaternions = motion.body_quat_w[:, foot_indexes]
        offsets = np.asarray(
            [config["sole_local_offsets_m"][name] for name in foot_names], dtype=np.float64
        )
        rotated_offsets = rotate_local_vectors_wxyz(
            foot_quaternions, offsets[None, :, :]
        )
        sole_positions = motion.body_pos_w[:, foot_indexes] + rotated_offsets
        # Contact debounce is a full-motion state machine.  Any non-finite sole
        # input can affect later state, so conservatively disable every
        # foot/contact feature for this motion instead of interpreting the bad
        # frame as flight or silently dropping it from an aggregate.
        if np.isfinite(sole_positions).all():
            contacts, sole_velocities = infer_debounced_foot_contacts(
                sole_positions, fps, config["contact"]
            )
            if np.isfinite(sole_velocities).all():
                sole_speeds = np.linalg.norm(sole_velocities, axis=2)
                switch_signal = np.zeros(motion.num_frames, dtype=np.int64)
                switch_signal[1:] = np.count_nonzero(contacts[1:] != contacts[:-1], axis=1)
            else:
                contacts = None

    feature_indexes = {name: index for index, name in enumerate(FEATURE_NAMES)}
    values = np.full((len(bounds), len(FEATURE_NAMES)), np.nan, dtype=np.float64)
    available = np.zeros_like(values, dtype=bool)
    durations = np.empty(len(bounds), dtype=np.float64)

    def assign(segment_index: int, name: str, result: tuple[float, bool]) -> None:
        value, is_available = result
        column = feature_indexes[name]
        values[segment_index, column] = value
        available[segment_index, column] = bool(is_available and math.isfinite(value))

    for segment_index, (start, end) in enumerate(bounds):
        duration = (end - start) / fps
        durations[segment_index] = duration
        assign(segment_index, "root_linear_speed_mean", _finite_stat(root_linear_speed[start:end], "mean"))
        assign(segment_index, "root_linear_speed_p95", _finite_stat(root_linear_speed[start:end], "p95"))
        assign(
            segment_index,
            "root_linear_acceleration_mean",
            _finite_stat(root_linear_acceleration[start:end], "mean"),
        )
        assign(
            segment_index,
            "root_linear_acceleration_p95",
            _finite_stat(root_linear_acceleration[start:end], "p95"),
        )
        assign(segment_index, "root_angular_speed_mean", _finite_stat(root_angular_speed[start:end], "mean"))
        assign(segment_index, "root_angular_speed_p95", _finite_stat(root_angular_speed[start:end], "p95"))
        assign(
            segment_index,
            "root_angular_acceleration_mean",
            _finite_stat(root_angular_acceleration[start:end], "mean"),
        )
        assign(
            segment_index,
            "root_angular_acceleration_p95",
            _finite_stat(root_angular_acceleration[start:end], "p95"),
        )
        assign(segment_index, "joint_speed_mean", _finite_stat(joint_speed[start:end], "mean"))
        assign(segment_index, "joint_speed_p95", _finite_stat(joint_speed[start:end], "p95"))
        assign(
            segment_index,
            "joint_acceleration_mean",
            _finite_stat(joint_acceleration[start:end], "mean"),
        )
        assign(
            segment_index,
            "joint_acceleration_p95",
            _finite_stat(joint_acceleration[start:end], "p95"),
        )
        ranges = _joint_ranges(motion.joint_pos[start:end])
        assign(segment_index, "joint_range_mean", _finite_stat(ranges, "mean"))
        assign(segment_index, "joint_range_p90", _finite_stat(ranges, "p90"))
        assign(segment_index, "body_height_mean", _finite_stat(root_height[start:end], "mean"))
        assign(segment_index, "body_height_range", _finite_stat(root_height[start:end], "range"))
        assign(segment_index, "body_height_std", _finite_stat(root_height[start:end], "std"))

        if hand_speeds is not None:
            assign(segment_index, "hand_speed_p95", _finite_stat(hand_speeds[start:end], "p95"))
        if sole_speeds is not None and contacts is not None:
            swing_mask = ~contacts[start:end]
            swing_values = sole_speeds[start:end][swing_mask]
            if swing_values.size == 0:
                assign(segment_index, "foot_swing_speed_p95", (0.0, True))
            else:
                assign(segment_index, "foot_swing_speed_p95", _finite_stat(swing_values, "p95"))
        if hand_speeds is not None and sole_speeds is not None:
            end_effector_speeds = np.concatenate(
                (hand_speeds[start:end], sole_speeds[start:end]), axis=1
            )
            assign(segment_index, "end_effector_speed_mean", _finite_stat(end_effector_speeds, "mean"))
            assign(segment_index, "end_effector_speed_p95", _finite_stat(end_effector_speeds, "p95"))

        if contacts is not None and switch_signal is not None:
            segment_contacts = contacts[start:end]
            left = segment_contacts[:, 0]
            right = segment_contacts[:, 1]
            assign(segment_index, "double_support_ratio", (float(np.mean(left & right)), True))
            assign(segment_index, "single_support_ratio", (float(np.mean(left ^ right)), True))
            assign(segment_index, "flight_ratio", (float(np.mean(~left & ~right)), True))
            assign(segment_index, "left_contact_ratio", (float(np.mean(left)), True))
            assign(segment_index, "right_contact_ratio", (float(np.mean(right)), True))
            switch_count = int(np.sum(switch_signal[start:end]))
            assign(segment_index, "contact_switch_count", (float(switch_count), True))
            assign(
                segment_index,
                "contact_switch_rate_per_second",
                (float(switch_count) / duration, True),
            )

    return MotionDifficultyFeatures(FEATURE_NAMES, values, available, durations)


@dataclass(frozen=True)
class DifficultyTransform:
    """Profile-transformed segment values."""

    feature_z: np.ndarray
    feature_contributions: np.ndarray
    difficulty_raw: np.ndarray
    difficulty_score: np.ndarray
    difficulty_bin: np.ndarray


@dataclass(frozen=True)
class DifficultyProfile:
    """Frozen Train-fitted robust scaling, empirical CDF, and bin mapping."""

    schema_version: str
    algorithm_schema_version: str
    training_manifest_sha256: str
    training_pool_fingerprint: str
    segment_schema_version: int
    segment_length_seconds: float
    feature_names: tuple[str, ...]
    feature_units: tuple[str, ...]
    feature_directions: np.ndarray
    feature_weights: np.ndarray
    effective_feature_weights: np.ndarray
    feature_medians: np.ndarray
    feature_scales: np.ndarray
    feature_coverage: np.ndarray
    required_features: tuple[str, ...]
    optional_feature_profile: tuple[str, ...]
    minimum_optional_feature_coverage: float
    near_constant_features: tuple[str, ...]
    robust_clip: float
    difficulty_raw_values: np.ndarray
    difficulty_raw_percentiles: np.ndarray
    difficulty_bin_edges: np.ndarray
    num_bins: int
    motion_mean_weight: float
    motion_p90_weight: float
    config_sha256: str
    git_commit: str
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a canonical JSON-compatible profile payload."""

        return {
            "schema_version": self.schema_version,
            "algorithm_schema_version": self.algorithm_schema_version,
            "training_manifest_sha256": self.training_manifest_sha256,
            "training_pool_fingerprint": self.training_pool_fingerprint,
            "segment_schema_version": self.segment_schema_version,
            "segment_length_seconds": self.segment_length_seconds,
            "feature_names": list(self.feature_names),
            "feature_units": list(self.feature_units),
            "feature_directions": self.feature_directions.astype(int).tolist(),
            "feature_weights": self.feature_weights.tolist(),
            "effective_feature_weights": self.effective_feature_weights.tolist(),
            "feature_medians": self.feature_medians.tolist(),
            "feature_scales": self.feature_scales.tolist(),
            "feature_coverage": self.feature_coverage.tolist(),
            "required_features": list(self.required_features),
            "optional_feature_profile": list(self.optional_feature_profile),
            "minimum_optional_feature_coverage": self.minimum_optional_feature_coverage,
            "near_constant_features": list(self.near_constant_features),
            "robust_clip": self.robust_clip,
            "difficulty_raw_distribution_knots": self.difficulty_raw_values.tolist(),
            "difficulty_raw_percentile_knots": self.difficulty_raw_percentiles.tolist(),
            "difficulty_bin_edges": self.difficulty_bin_edges.tolist(),
            "num_bins": self.num_bins,
            "motion_aggregation_weights": {
                "duration_weighted_mean": self.motion_mean_weight,
                "segment_p90": self.motion_p90_weight,
            },
            "config_sha256": self.config_sha256,
            "git_commit": self.git_commit,
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DifficultyProfile":
        """Build and validate a profile loaded from JSON."""

        if value.get("schema_version") != DIFFICULTY_PROFILE_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported difficulty profile schema {value.get('schema_version')!r}; "
                f"expected {DIFFICULTY_PROFILE_SCHEMA_VERSION!r}."
            )
        algorithm_schema_version = str(value.get("algorithm_schema_version", ""))
        if algorithm_schema_version != DEFAULT_ALGORITHM_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported difficulty Profile algorithm schema {algorithm_schema_version!r}; "
                f"expected {DEFAULT_ALGORITHM_SCHEMA_VERSION!r}."
            )
        training_manifest_sha256 = str(value.get("training_manifest_sha256", ""))
        training_pool_fingerprint = str(value.get("training_pool_fingerprint", ""))
        config_sha256 = str(value.get("config_sha256", ""))
        _validate_sha256(training_manifest_sha256, "training_manifest_sha256")
        _validate_sha256(training_pool_fingerprint, "training_pool_fingerprint")
        _validate_sha256(config_sha256, "config_sha256")
        segment_schema_version = _integer_value(
            value.get("segment_schema_version", -1), "segment_schema_version"
        )
        if segment_schema_version < 1:
            raise ValueError("Difficulty profile segment_schema_version must be positive.")
        segment_length_seconds = _finite_scalar(
            value.get("segment_length_seconds", float("nan"))
        )
        if segment_length_seconds <= 0.0:
            raise ValueError("Difficulty profile segment_length_seconds must be positive.")
        names = tuple(str(name) for name in value.get("feature_names", ()))
        if names != FEATURE_NAMES:
            raise ValueError("Difficulty profile feature names/order do not match this implementation.")
        count = len(names)

        def vector(field: str, dtype: Any = np.float64) -> np.ndarray:
            result = np.asarray(value.get(field), dtype=dtype)
            if result.shape != (count,) or not np.isfinite(result).all():
                raise ValueError(f"Difficulty profile field '{field}' must be a finite ({count},) vector.")
            return result

        directions_raw = np.asarray(value.get("feature_directions"))
        if directions_raw.shape != (count,) or directions_raw.dtype.kind not in "iu":
            raise ValueError(
                f"Difficulty profile field 'feature_directions' must be an integer ({count},) vector."
            )
        if not np.isin(directions_raw, (-1, 1)).all():
            raise ValueError("Difficulty profile feature directions must be -1 or +1.")
        directions = directions_raw.astype(np.int8, copy=False)

        raw_values = np.asarray(value.get("difficulty_raw_distribution_knots"), dtype=np.float64)
        raw_percentiles = np.asarray(value.get("difficulty_raw_percentile_knots"), dtype=np.float64)
        if (
            raw_values.ndim != 1
            or raw_values.size == 0
            or raw_percentiles.shape != raw_values.shape
            or not np.isfinite(raw_values).all()
            or not np.isfinite(raw_percentiles).all()
            or np.any(np.diff(raw_values) <= 0.0)
            or (raw_percentiles.size > 1 and np.any(np.diff(raw_percentiles) <= 0.0))
            or np.any(raw_percentiles < 0.0)
            or np.any(raw_percentiles > 1.0)
        ):
            raise ValueError("Difficulty profile empirical-CDF knots are invalid.")
        num_bins = _integer_value(value.get("num_bins", 0), "num_bins")
        edges = np.asarray(value.get("difficulty_bin_edges"), dtype=np.float64)
        if (
            num_bins < 2
            or edges.shape != (num_bins - 1,)
            or not np.isfinite(edges).all()
            or np.any(np.diff(edges) < 0.0)
            or np.any(edges < raw_values[0])
            or np.any(edges > raw_values[-1])
        ):
            raise ValueError("Difficulty profile bin edges are invalid.")
        aggregation = value.get("motion_aggregation_weights")
        if not isinstance(aggregation, Mapping):
            raise ValueError("Difficulty profile motion_aggregation_weights is invalid.")
        units = tuple(str(unit) for unit in value.get("feature_units", ()))
        if len(units) != count or any(not unit for unit in units):
            raise ValueError("Difficulty profile feature_units are invalid.")
        required = tuple(str(name) for name in value.get("required_features", ()))
        optional = tuple(str(name) for name in value.get("optional_feature_profile", ()))
        if (
            len(required) + len(optional) != count
            or len(set(required)) != len(required)
            or len(set(optional)) != len(optional)
            or set(required).intersection(optional)
            or set(required).union(optional) != set(names)
        ):
            raise ValueError("Difficulty profile required/optional feature partition is invalid.")

        near_constant_features = tuple(
            str(name) for name in value.get("near_constant_features", ())
        )
        if (
            len(set(near_constant_features)) != len(near_constant_features)
            or not set(near_constant_features).issubset(names)
        ):
            raise ValueError("Difficulty profile near_constant_features are invalid.")
        minimum_optional_coverage = _finite_scalar(
            value.get("minimum_optional_feature_coverage", float("nan"))
        )
        if not 0.0 <= minimum_optional_coverage <= 1.0:
            raise ValueError(
                "Difficulty profile minimum_optional_feature_coverage must be in [0, 1]."
            )
        robust_clip = _finite_scalar(value.get("robust_clip", float("nan")))
        if robust_clip <= 0.0:
            raise ValueError("Difficulty profile robust_clip must be positive.")
        motion_mean_weight = _finite_scalar(
            aggregation.get("duration_weighted_mean", float("nan"))
        )
        motion_p90_weight = _finite_scalar(aggregation.get("segment_p90", float("nan")))
        if (
            motion_mean_weight < 0.0
            or motion_p90_weight < 0.0
            or motion_mean_weight + motion_p90_weight <= 0.0
        ):
            raise ValueError("Difficulty profile motion aggregation weights are invalid.")

        profile = cls(
            schema_version=str(value["schema_version"]),
            algorithm_schema_version=algorithm_schema_version,
            training_manifest_sha256=training_manifest_sha256,
            training_pool_fingerprint=training_pool_fingerprint,
            segment_schema_version=segment_schema_version,
            segment_length_seconds=segment_length_seconds,
            feature_names=names,
            feature_units=units,
            feature_directions=directions,
            feature_weights=vector("feature_weights"),
            effective_feature_weights=vector("effective_feature_weights"),
            feature_medians=vector("feature_medians"),
            feature_scales=vector("feature_scales"),
            feature_coverage=vector("feature_coverage"),
            required_features=required,
            optional_feature_profile=optional,
            minimum_optional_feature_coverage=minimum_optional_coverage,
            near_constant_features=near_constant_features,
            robust_clip=robust_clip,
            difficulty_raw_values=raw_values,
            difficulty_raw_percentiles=raw_percentiles,
            difficulty_bin_edges=edges,
            num_bins=num_bins,
            motion_mean_weight=motion_mean_weight,
            motion_p90_weight=motion_p90_weight,
            config_sha256=config_sha256,
            git_commit=str(value.get("git_commit", "")),
            warnings=tuple(str(item) for item in value.get("warnings", ())),
        )
        if np.any(profile.feature_weights < 0.0) or np.any(profile.effective_feature_weights < 0.0):
            raise ValueError("Difficulty profile feature weights must be non-negative.")
        if np.any(profile.effective_feature_weights > profile.feature_weights):
            raise ValueError("Difficulty profile effective weights cannot exceed configured weights.")
        if np.sum(profile.effective_feature_weights) <= 0.0:
            raise ValueError("Difficulty profile has no active feature weight.")
        if np.any(profile.feature_scales <= 0.0):
            raise ValueError("Difficulty profile feature scales must be positive.")
        if np.any(profile.feature_coverage < 0.0) or np.any(profile.feature_coverage > 1.0):
            raise ValueError("Difficulty profile feature coverage must be in [0, 1].")
        near_constant_indexes = np.asarray(
            [profile.feature_names.index(name) for name in profile.near_constant_features],
            dtype=np.int64,
        )
        if near_constant_indexes.size and np.any(
            profile.effective_feature_weights[near_constant_indexes] != 0.0
        ):
            raise ValueError("Difficulty profile near-constant features must have zero effective weight.")
        return profile


def load_difficulty_profile(path: str | Path) -> DifficultyProfile:
    """Load a frozen Train-fitted profile without refitting any parameter."""

    profile_path = Path(path)
    if not profile_path.is_file():
        raise FileNotFoundError(f"Difficulty profile does not exist: {profile_path}")
    try:
        payload = json.loads(profile_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid difficulty profile JSON '{profile_path}': {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("Difficulty profile root must be an object.")
    return DifficultyProfile.from_dict(payload)


def _feature_definition_vectors(
    config: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    features = config["features"]
    directions = np.asarray([int(features[name]["direction"]) for name in FEATURE_NAMES], dtype=np.int8)
    weights = np.asarray([float(features[name]["weight"]) for name in FEATURE_NAMES], dtype=np.float64)
    units = tuple(str(features[name]["unit"]) for name in FEATURE_NAMES)
    required = tuple(name for name in FEATURE_NAMES if bool(features[name]["required"]))
    optional = tuple(name for name in FEATURE_NAMES if not bool(features[name]["required"]))
    return directions, weights, units, required, optional


def _validate_feature_matrix(
    values: np.ndarray,
    available: np.ndarray,
    feature_names: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(values, dtype=np.float64)
    mask = np.asarray(available, dtype=bool)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] != len(FEATURE_NAMES):
        raise ValueError(f"feature values must have shape (N > 0, {len(FEATURE_NAMES)}).")
    if mask.shape != matrix.shape:
        raise ValueError("feature availability mask must match feature values.")
    if tuple(str(name) for name in feature_names) != FEATURE_NAMES:
        raise ValueError("feature_names/order do not match the difficulty feature schema.")
    mask = mask & np.isfinite(matrix)
    return matrix, mask


def robust_standardize(
    values: np.ndarray,
    available: np.ndarray,
    medians: Sequence[float],
    scales: Sequence[float],
    robust_clip: float,
) -> np.ndarray:
    """Apply frozen median/MAD scaling and clipping, preserving missingness."""

    matrix = np.asarray(values, dtype=np.float64)
    mask = np.asarray(available, dtype=bool) & np.isfinite(matrix)
    centers = np.asarray(medians, dtype=np.float64)
    scale_values = np.asarray(scales, dtype=np.float64)
    if centers.shape != (matrix.shape[1],) or scale_values.shape != centers.shape:
        raise ValueError("medians/scales must match the feature dimension.")
    if not np.isfinite(centers).all() or not np.isfinite(scale_values).all() or np.any(scale_values <= 0.0):
        raise ValueError("medians and positive scales must be finite.")
    clip = _finite_scalar(robust_clip)
    if clip <= 0.0:
        raise ValueError("robust_clip must be positive.")
    z = np.full_like(matrix, np.nan, dtype=np.float64)
    standardized = (matrix - centers[None, :]) / scale_values[None, :]
    z[mask] = np.clip(standardized[mask], -clip, clip)
    return z


def _empirical_cdf_knots(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sorted_values = np.sort(np.asarray(values, dtype=np.float64))
    if sorted_values.ndim != 1 or sorted_values.size == 0 or not np.isfinite(sorted_values).all():
        raise ValueError("Empirical CDF requires a non-empty finite vector.")
    unique, first, counts = np.unique(sorted_values, return_index=True, return_counts=True)
    if sorted_values.size == 1:
        percentiles = np.asarray([0.5], dtype=np.float64)
    else:
        midrank = first.astype(np.float64) + 0.5 * (counts.astype(np.float64) - 1.0)
        percentiles = midrank / float(sorted_values.size - 1)
    return unique, percentiles


def empirical_percentile(
    raw_values: np.ndarray, cdf_values: np.ndarray, cdf_percentiles: np.ndarray
) -> np.ndarray:
    """Map raw scores through the frozen empirical CDF with linear interpolation."""

    raw = np.asarray(raw_values, dtype=np.float64)
    knots = np.asarray(cdf_values, dtype=np.float64)
    percentiles = np.asarray(cdf_percentiles, dtype=np.float64)
    if knots.size == 1:
        result = np.full(raw.shape, 0.5, dtype=np.float64)
        result[raw < knots[0]] = 0.0
        result[raw > knots[0]] = 1.0
        return result
    return np.interp(raw, knots, percentiles, left=0.0, right=1.0)


def _quantile(values: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    try:
        return np.quantile(values, probabilities, method="linear")
    except TypeError:  # NumPy < 1.22 compatibility.
        return np.quantile(values, probabilities, interpolation="linear")


def fit_difficulty_profile(
    values: np.ndarray,
    available: np.ndarray,
    config: Mapping[str, Any],
    *,
    feature_names: Sequence[str] = FEATURE_NAMES,
    training_manifest_sha256: str = "",
    training_pool_fingerprint: str = "",
    segment_schema_version: int = 1,
    config_sha256: str | None = None,
    git_commit: str = "",
) -> DifficultyProfile:
    """Fit Train-only robust statistics, empirical CDF, and quantile bins."""

    validate_difficulty_config(config)
    matrix, mask = _validate_feature_matrix(values, available, feature_names)
    directions, weights, units, required, optional = _feature_definition_vectors(config)
    name_to_index = {name: index for index, name in enumerate(FEATURE_NAMES)}
    missing_required = [
        name for name in required if not np.all(mask[:, name_to_index[name]])
    ]
    if missing_required:
        raise ValueError(
            "Required difficulty features are unavailable for one or more Train segments: "
            f"{missing_required}. Missing values are never interpreted as easy or difficult."
        )
    optional_indexes = np.asarray([name_to_index[name] for name in optional], dtype=np.int64)
    optional_coverage_by_segment = (
        np.mean(mask[:, optional_indexes], axis=1) if optional_indexes.size else np.ones(matrix.shape[0])
    )
    minimum_optional = float(config["minimum_optional_feature_coverage"])
    if np.any(optional_coverage_by_segment < minimum_optional):
        raise ValueError(
            "Optional difficulty feature coverage is below the configured minimum for "
            f"{int(np.count_nonzero(optional_coverage_by_segment < minimum_optional))} Train segment(s)."
        )

    medians = np.empty(matrix.shape[1], dtype=np.float64)
    scales = np.empty(matrix.shape[1], dtype=np.float64)
    coverage = np.mean(mask, axis=0)
    near_constant_mask = np.zeros(matrix.shape[1], dtype=bool)
    epsilon = float(config["robust_scale_epsilon"])
    near_threshold = float(config["near_constant_scale_threshold"])
    fallback_quantiles = np.asarray(config["zero_mad_fallback_quantiles"], dtype=np.float64)
    fallback_divisor = float(config["zero_mad_fallback_scale_divisor"])
    warnings: list[str] = []
    for column, name in enumerate(FEATURE_NAMES):
        observed = matrix[mask[:, column], column]
        if observed.size == 0:
            medians[column] = 0.0
            scales[column] = epsilon
            near_constant_mask[column] = True
            warnings.append(f"feature '{name}' has zero coverage and is disabled")
            continue
        median = float(np.median(observed))
        mad_scale = 1.4826 * float(np.median(np.abs(observed - median)))
        fallback_bounds = _quantile(observed, fallback_quantiles)
        fallback_scale = float(fallback_bounds[1] - fallback_bounds[0]) / fallback_divisor
        fitted_scale = mad_scale if mad_scale >= near_threshold else fallback_scale
        medians[column] = median
        scales[column] = max(fitted_scale, epsilon)
        near_constant_mask[column] = fitted_scale < near_threshold
        if mad_scale < near_threshold <= fallback_scale:
            warnings.append(
                f"feature '{name}' used the configured zero-MAD quantile scale fallback "
                f"({fallback_scale:.9g})"
            )
        if near_constant_mask[column]:
            warnings.append(
                f"feature '{name}' is near-constant (MAD scale={mad_scale:.9g}, "
                f"fallback scale={fallback_scale:.9g}) and is disabled"
            )

    effective_weights = weights.copy()
    effective_weights[near_constant_mask] = 0.0
    # A Profile uses a single feature set and denominator.  Optional features
    # that were not available for every Train segment remain diagnostic only.
    incomplete_optional = np.zeros(matrix.shape[1], dtype=bool)
    incomplete_optional[optional_indexes] = coverage[optional_indexes] < 1.0
    for column in np.flatnonzero(incomplete_optional & (effective_weights > 0.0)):
        warnings.append(
            f"optional feature '{FEATURE_NAMES[column]}' has coverage {coverage[column]:.6f} and is disabled"
        )
    effective_weights[incomplete_optional] = 0.0
    if np.sum(effective_weights) <= 0.0:
        raise ValueError("All configured scoring features are near-constant or unavailable.")

    z = robust_standardize(matrix, mask, medians, scales, float(config["robust_clip"]))
    active = effective_weights > 0.0
    if not np.all(mask[:, active]):
        raise RuntimeError("Profile active features unexpectedly contain missing values.")
    directed = z[:, active] * directions[active][None, :]
    raw = np.sum(directed * effective_weights[active][None, :], axis=1) / float(
        np.sum(effective_weights[active])
    )
    if not np.isfinite(raw).all():
        raise ValueError("Difficulty raw score contains NaN/Inf after robust standardization.")
    cdf_values, cdf_percentiles = _empirical_cdf_knots(raw)
    num_bins = int(config["num_bins"])
    edges = _quantile(raw, np.arange(1, num_bins, dtype=np.float64) / num_bins)
    if np.any(np.diff(edges) <= 0.0):
        warnings.append(
            "difficulty quantile edges contain ties; deterministic mapping is retained but one or more "
            "bins may be empty"
        )
    train_bins = np.searchsorted(edges, raw, side="right")
    represented_bins = int(np.unique(train_bins).size)
    if represented_bins != num_bins:
        warnings.append(
            f"training raw scores represent {represented_bins} of {num_bins} requested difficulty bins"
        )

    aggregation = config["motion_aggregation"]
    return DifficultyProfile(
        schema_version=DIFFICULTY_PROFILE_SCHEMA_VERSION,
        algorithm_schema_version=str(config["algorithm_schema_version"]),
        training_manifest_sha256=str(training_manifest_sha256),
        training_pool_fingerprint=str(training_pool_fingerprint),
        segment_schema_version=int(segment_schema_version),
        segment_length_seconds=float(config["segment_length_seconds"]),
        feature_names=FEATURE_NAMES,
        feature_units=units,
        feature_directions=directions,
        feature_weights=weights,
        effective_feature_weights=effective_weights,
        feature_medians=medians,
        feature_scales=scales,
        feature_coverage=coverage,
        required_features=required,
        optional_feature_profile=optional,
        minimum_optional_feature_coverage=minimum_optional,
        near_constant_features=tuple(
            name for name, is_constant in zip(FEATURE_NAMES, near_constant_mask, strict=True) if is_constant
        ),
        robust_clip=float(config["robust_clip"]),
        difficulty_raw_values=cdf_values,
        difficulty_raw_percentiles=cdf_percentiles,
        difficulty_bin_edges=np.asarray(edges, dtype=np.float64),
        num_bins=num_bins,
        motion_mean_weight=float(aggregation["duration_weighted_mean_weight"]),
        motion_p90_weight=float(aggregation["segment_p90_weight"]),
        config_sha256=config_sha256 or canonical_json_sha256(config),
        git_commit=str(git_commit),
        warnings=tuple(warnings),
    )


def transform_difficulty_features(
    values: np.ndarray,
    available: np.ndarray,
    profile: DifficultyProfile,
    *,
    feature_names: Sequence[str] = FEATURE_NAMES,
) -> DifficultyTransform:
    """Apply a frozen profile without fitting or changing any parameter."""

    matrix, mask = _validate_feature_matrix(values, available, feature_names)
    active = profile.effective_feature_weights > 0.0
    if not np.all(mask[:, active]):
        missing_columns = np.flatnonzero(np.any(~mask[:, active], axis=0))
        active_names = np.asarray(profile.feature_names, dtype=object)[active]
        missing_names = active_names[missing_columns].tolist()
        raise ValueError(
            "Frozen Profile scoring features are unavailable in transform data: "
            f"{missing_names}. The scoring denominator is never changed per segment."
        )
    name_to_index = {name: index for index, name in enumerate(profile.feature_names)}
    required_indexes = np.asarray([name_to_index[name] for name in profile.required_features], dtype=np.int64)
    if required_indexes.size and not np.all(mask[:, required_indexes]):
        raise ValueError("One or more frozen required features are unavailable in transform data.")
    optional_indexes = np.asarray(
        [name_to_index[name] for name in profile.optional_feature_profile], dtype=np.int64
    )
    optional_coverage = (
        np.mean(mask[:, optional_indexes], axis=1) if optional_indexes.size else np.ones(matrix.shape[0])
    )
    if np.any(optional_coverage < profile.minimum_optional_feature_coverage):
        raise ValueError("Transform optional feature coverage is below the frozen Profile minimum.")

    z = robust_standardize(
        matrix, mask, profile.feature_medians, profile.feature_scales, profile.robust_clip
    )
    weight_sum = float(np.sum(profile.effective_feature_weights[active]))
    contributions = np.zeros_like(matrix, dtype=np.float64)
    contributions[:, active] = (
        z[:, active]
        * profile.feature_directions[active][None, :]
        * profile.effective_feature_weights[active][None, :]
        / weight_sum
    )
    raw = np.sum(contributions[:, active], axis=1)
    score = empirical_percentile(
        raw, profile.difficulty_raw_values, profile.difficulty_raw_percentiles
    )
    bins = np.searchsorted(profile.difficulty_bin_edges, raw, side="right").astype(np.int16)
    if (
        not np.isfinite(raw).all()
        or not np.isfinite(score).all()
        or np.any(score < 0.0)
        or np.any(score > 1.0)
        or np.any(bins < 0)
        or np.any(bins >= profile.num_bins)
    ):
        raise RuntimeError("Frozen difficulty transform produced an invalid raw/score/bin value.")
    return DifficultyTransform(z, contributions, raw, score, bins)


def aggregate_motion_difficulty(
    segment_scores: Sequence[float],
    duration_seconds: Sequence[float],
    *,
    mean_weight: float = 0.5,
    p90_weight: float = 0.5,
    num_bins: int = 10,
) -> dict[str, float | int]:
    """Aggregate segment percentiles without allowing a short tail to dominate."""

    scores = np.asarray(segment_scores, dtype=np.float64)
    durations = np.asarray(duration_seconds, dtype=np.float64)
    if scores.ndim != 1 or scores.size == 0 or durations.shape != scores.shape:
        raise ValueError("segment_scores and duration_seconds must be same-shape non-empty vectors.")
    if (
        not np.isfinite(scores).all()
        or np.any(scores < 0.0)
        or np.any(scores > 1.0)
        or not np.isfinite(durations).all()
        or np.any(durations <= 0.0)
    ):
        raise ValueError("Motion aggregation requires finite [0,1] scores and positive durations.")
    first_weight = _finite_scalar(mean_weight)
    second_weight = _finite_scalar(p90_weight)
    if first_weight < 0.0 or second_weight < 0.0 or first_weight + second_weight <= 0.0:
        raise ValueError("Motion aggregation weights must be non-negative with positive total.")
    if num_bins < 2:
        raise ValueError("num_bins must be at least 2.")
    weighted_mean = float(np.average(scores, weights=durations))
    p90 = float(np.percentile(scores, 90.0))
    score = (first_weight * weighted_mean + second_weight * p90) / (first_weight + second_weight)
    motion_bin = min(int(math.floor(score * num_bins)), num_bins - 1)
    return {
        "difficulty_mean": weighted_mean,
        "difficulty_p90": p90,
        "difficulty_score": score,
        "difficulty_bin": motion_bin,
        "segment_count": int(scores.size),
        "duration_seconds": float(np.sum(durations)),
    }
