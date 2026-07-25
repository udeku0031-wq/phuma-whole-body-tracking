"""Deterministic, policy-independent motion clustering utilities.

The clustering input is deliberately limited to the raw segment feature
matrix produced by module two.  Difficulty scores/bins, quality labels,
source categories, rewards, and policy errors are not accepted by this API.

The implementation is NumPy-only so the offline builder and its CPU tests do
not require Isaac Sim or scikit-learn.  A Train-fitted
:class:`MotionClusteringProfile` freezes aggregation definitions, robust
scaling, optional PCA, and canonicalized KMeans centroids for later
``transform`` calls.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

import numpy as np


MOTION_CLUSTERING_CONFIG_SCHEMA_VERSION: Final[int] = 1
MOTION_CLUSTERING_PROFILE_SCHEMA_VERSION: Final[str] = (
    "wbt.motion_clustering_profile.v1"
)
DEFAULT_ALGORITHM_SCHEMA_VERSION: Final[str] = (
    "wbt.motion_clustering.numpy_kmeans.v1"
)
KMEANS_IMPLEMENTATION_IDENTITY: Final[str] = (
    "numpy.kmeans_plusplus.lloyd.canonical.v1"
)

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[4]
DEFAULT_MOTION_CLUSTERING_CONFIG_PATH: Final[Path] = (
    PROJECT_ROOT / "configs" / "diversity" / "g1_motion_clustering.yaml"
)

SUPPORTED_AGGREGATIONS: Final[tuple[str, ...]] = (
    "duration_weighted_mean",
    "p90",
    "duration_weighted_std",
)


def canonical_json_sha256(value: Mapping[str, Any]) -> str:
    """Hash a JSON-compatible mapping with deterministic serialization."""

    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _finite_scalar(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number.") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number.")
    return result


def _integer_value(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer.")
    return int(value)


def _string_tuple(
    value: Any,
    name: str,
    *,
    allow_empty: bool = False,
    unique: bool = True,
) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a list.")
    result = tuple(str(item) for item in value)
    if (not allow_empty and not result) or any(not item for item in result):
        raise ValueError(f"{name} must contain non-empty strings.")
    if unique and len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicates.")
    return result


def _quantile(values: np.ndarray, probabilities: float | np.ndarray) -> np.ndarray:
    try:
        return np.quantile(values, probabilities, method="linear")
    except TypeError:  # NumPy < 1.22 compatibility.
        return np.quantile(values, probabilities, interpolation="linear")


def validate_motion_clustering_config(config: Mapping[str, Any]) -> None:
    """Validate the complete clustering definition before reading data."""

    if not isinstance(config, Mapping):
        raise ValueError("Motion clustering config root must be a mapping.")
    schema_version = _integer_value(
        config.get("schema_version", -1), "schema_version"
    )
    if schema_version != MOTION_CLUSTERING_CONFIG_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported motion clustering config schema {schema_version}; "
            f"expected {MOTION_CLUSTERING_CONFIG_SCHEMA_VERSION}."
        )
    algorithm = config.get("algorithm_schema_version")
    if algorithm != DEFAULT_ALGORITHM_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported motion clustering algorithm schema {algorithm!r}; "
            f"expected {DEFAULT_ALGORITHM_SCHEMA_VERSION!r}."
        )

    num_clusters = _integer_value(config.get("num_clusters"), "num_clusters")
    random_seed = _integer_value(config.get("random_seed"), "random_seed")
    if num_clusters < 2:
        raise ValueError("num_clusters must be at least 2.")
    if random_seed < 0 or random_seed > np.iinfo(np.uint32).max:
        raise ValueError("random_seed must be in [0, 2**32 - 1].")

    feature_list = _string_tuple(config.get("feature_list"), "feature_list")
    required = _string_tuple(
        config.get("required_features"), "required_features", allow_empty=True
    )
    optional = _string_tuple(
        config.get("optional_features"), "optional_features", allow_empty=True
    )
    if (
        set(required).intersection(optional)
        or set(required).union(optional) != set(feature_list)
    ):
        raise ValueError(
            "required_features and optional_features must be a disjoint "
            "partition of feature_list."
        )

    aggregations = config.get("feature_aggregations")
    if not isinstance(aggregations, Mapping) or set(aggregations) != set(feature_list):
        raise ValueError(
            "feature_aggregations must contain exactly one entry per feature_list item."
        )
    for feature_name in feature_list:
        items = _string_tuple(
            aggregations[feature_name],
            f"feature_aggregations.{feature_name}",
        )
        unsupported = sorted(set(items).difference(SUPPORTED_AGGREGATIONS))
        if unsupported:
            raise ValueError(
                f"Feature '{feature_name}' uses unsupported aggregations: {unsupported}."
            )

    units = config.get("feature_units")
    if not isinstance(units, Mapping) or set(units) != set(feature_list):
        raise ValueError(
            "feature_units must contain exactly one entry per feature_list item."
        )
    if any(not isinstance(units[name], str) or not units[name] for name in feature_list):
        raise ValueError("Every clustering source feature must declare a non-empty unit.")

    minimum_optional = _finite_scalar(
        config.get("minimum_optional_feature_coverage"),
        "minimum_optional_feature_coverage",
    )
    epsilon = _finite_scalar(
        config.get("robust_scale_epsilon"), "robust_scale_epsilon"
    )
    near_threshold = _finite_scalar(
        config.get("near_constant_scale_threshold"),
        "near_constant_scale_threshold",
    )
    fallback_quantiles = np.asarray(
        config.get("zero_mad_fallback_quantiles"), dtype=np.float64
    )
    fallback_divisor = _finite_scalar(
        config.get("zero_mad_fallback_scale_divisor"),
        "zero_mad_fallback_scale_divisor",
    )
    robust_clip = _finite_scalar(config.get("robust_clip"), "robust_clip")
    if not 0.0 <= minimum_optional <= 1.0:
        raise ValueError("minimum_optional_feature_coverage must be in [0, 1].")
    if (
        epsilon <= 0.0
        or near_threshold < epsilon
        or fallback_quantiles.shape != (2,)
        or not np.isfinite(fallback_quantiles).all()
        or fallback_quantiles[0] < 0.0
        or fallback_quantiles[1] > 1.0
        or fallback_quantiles[0] >= fallback_quantiles[1]
        or fallback_divisor <= 0.0
        or robust_clip <= 0.0
    ):
        raise ValueError(
            "Robust-scaling epsilon/near-constant threshold, fallback "
            "quantiles/divisor, and clipping limit are invalid."
        )

    use_pca = config.get("use_pca")
    if not isinstance(use_pca, bool):
        raise ValueError("use_pca must be boolean.")
    pca_components = config.get("pca_components")
    explained_target = config.get("explained_variance_target")
    if use_pca:
        selected = int(pca_components is not None) + int(explained_target is not None)
        if selected != 1:
            raise ValueError(
                "When use_pca is true, exactly one of pca_components or "
                "explained_variance_target must be set."
            )
        if pca_components is not None and _integer_value(
            pca_components, "pca_components"
        ) < 1:
            raise ValueError("pca_components must be positive.")
        if explained_target is not None:
            target = _finite_scalar(
                explained_target, "explained_variance_target"
            )
            if not 0.0 < target <= 1.0:
                raise ValueError("explained_variance_target must be in (0, 1].")
    elif pca_components is not None or explained_target is not None:
        raise ValueError(
            "pca_components and explained_variance_target must be null when "
            "use_pca is false."
        )

    n_init = _integer_value(config.get("kmeans_n_init"), "kmeans_n_init")
    max_iterations = _integer_value(
        config.get("kmeans_max_iterations"), "kmeans_max_iterations"
    )
    tolerance = _finite_scalar(config.get("kmeans_tolerance"), "kmeans_tolerance")
    if n_init < 1 or max_iterations < 1 or tolerance < 0.0:
        raise ValueError(
            "kmeans_n_init/max_iterations must be positive and tolerance non-negative."
        )
    if config.get("canonicalize_labels") is not True:
        raise ValueError(
            "canonicalize_labels must be true; frozen cluster IDs require "
            "deterministic centroid ordering."
        )

    minimum_size = _integer_value(
        config.get("minimum_cluster_size_warning"),
        "minimum_cluster_size_warning",
    )
    if minimum_size < 1:
        raise ValueError("minimum_cluster_size_warning must be positive.")
    budget_mode = config.get("cluster_budget_mode")
    if budget_mode not in {"sqrt_size_with_floor", "size_power_with_floor"}:
        raise ValueError("Unsupported cluster_budget_mode.")
    minimum_fraction = _finite_scalar(
        config.get("minimum_budget_fraction_of_uniform"),
        "minimum_budget_fraction_of_uniform",
    )
    size_exponent = _finite_scalar(
        config.get("cluster_size_exponent"), "cluster_size_exponent"
    )
    if not 0.0 <= minimum_fraction <= 1.0:
        raise ValueError("minimum_budget_fraction_of_uniform must be in [0, 1].")
    if not 0.0 <= size_exponent <= 1.0:
        raise ValueError("cluster_size_exponent must be in [0, 1].")
    if not isinstance(config.get("provisional"), bool):
        raise ValueError("provisional must be boolean.")


def load_motion_clustering_config(
    path: str | Path | None = None,
) -> dict[str, Any]:
    """Load the JSON-compatible YAML clustering configuration."""

    config_path = (
        DEFAULT_MOTION_CLUSTERING_CONFIG_PATH if path is None else Path(path)
    )
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Motion clustering config does not exist: {config_path}"
        )
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON-compatible motion clustering config '{config_path}': {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError("Motion clustering config root must be an object.")
    validate_motion_clustering_config(payload)
    return payload


def _expanded_feature_definition(
    config: Mapping[str, Any],
) -> tuple[
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
]:
    """Return expanded names, sources, aggregations, units, required, optional."""

    feature_list = tuple(str(name) for name in config["feature_list"])
    required_sources = set(str(name) for name in config["required_features"])
    names: list[str] = []
    sources: list[str] = []
    aggregations: list[str] = []
    units: list[str] = []
    required: list[str] = []
    optional: list[str] = []
    for source_name in feature_list:
        for aggregation in config["feature_aggregations"][source_name]:
            expanded_name = f"{source_name}__{aggregation}"
            names.append(expanded_name)
            sources.append(source_name)
            aggregations.append(str(aggregation))
            units.append(str(config["feature_units"][source_name]))
            (required if source_name in required_sources else optional).append(
                expanded_name
            )
    return (
        tuple(names),
        tuple(sources),
        tuple(aggregations),
        tuple(units),
        tuple(required),
        tuple(optional),
    )


@dataclass(frozen=True)
class MotionFeatureMatrix:
    """Raw segment features aggregated into one row per motion."""

    feature_names: tuple[str, ...]
    source_feature_names: tuple[str, ...]
    aggregations: tuple[str, ...]
    feature_units: tuple[str, ...]
    required_features: tuple[str, ...]
    optional_features: tuple[str, ...]
    values: np.ndarray
    available_mask: np.ndarray
    motion_duration_seconds: np.ndarray
    segment_counts: np.ndarray

    @property
    def num_motions(self) -> int:
        return int(self.values.shape[0])

    @property
    def num_features(self) -> int:
        return int(self.values.shape[1])

    @property
    def feature_available_mask(self) -> np.ndarray:
        """Compatibility alias matching module-two metadata terminology."""

        return self.available_mask

    @property
    def feature_coverage(self) -> np.ndarray:
        return np.mean(self.available_mask, axis=0)


def _validate_aggregate_inputs(
    metadata: Any,
) -> tuple[
    tuple[str, ...],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    required_attributes = (
        "feature_names",
        "feature_values",
        "feature_available_mask",
        "duration_seconds",
        "motion_segment_offsets",
    )
    missing = [
        name for name in required_attributes if not hasattr(metadata, name)
    ]
    if missing:
        raise ValueError(
            "Segment difficulty metadata is missing clustering inputs: "
            f"{missing}."
        )
    feature_names = tuple(str(name) for name in np.asarray(metadata.feature_names))
    if not feature_names or any(not name for name in feature_names):
        raise ValueError("Segment feature names must be non-empty.")
    if len(set(feature_names)) != len(feature_names):
        raise ValueError("Segment feature names must be unique.")

    values = np.asarray(metadata.feature_values, dtype=np.float64)
    available_raw = np.asarray(metadata.feature_available_mask)
    if values.ndim != 2 or values.shape[1] != len(feature_names):
        raise ValueError(
            "Segment feature_values shape does not match feature_names."
        )
    if available_raw.shape != values.shape:
        raise ValueError(
            "Segment feature_available_mask must match feature_values."
        )
    available = available_raw.astype(bool, copy=False)
    if np.any(available & ~np.isfinite(values)):
        raise ValueError(
            "Available segment clustering features contain NaN or Inf."
        )

    durations = np.asarray(metadata.duration_seconds, dtype=np.float64)
    if (
        durations.shape != (values.shape[0],)
        or not np.isfinite(durations).all()
        or np.any(durations <= 0.0)
    ):
        raise ValueError(
            "Segment duration_seconds must be a positive finite vector."
        )

    offsets_raw = np.asarray(metadata.motion_segment_offsets)
    if offsets_raw.ndim != 1 or offsets_raw.dtype.kind not in "iu":
        raise ValueError(
            "motion_segment_offsets must be a one-dimensional integer vector."
        )
    offsets = offsets_raw.astype(np.int64, copy=False)
    if (
        offsets.size < 2
        or offsets[0] != 0
        or offsets[-1] != values.shape[0]
        or np.any(np.diff(offsets) <= 0)
    ):
        raise ValueError(
            "motion_segment_offsets must cover all segments with at least "
            "one segment per motion."
        )
    if hasattr(metadata, "motion_id"):
        motion_id = np.asarray(metadata.motion_id)
        expected = np.repeat(
            np.arange(offsets.size - 1, dtype=np.int64), np.diff(offsets)
        )
        if motion_id.shape != expected.shape or not np.array_equal(
            motion_id, expected
        ):
            raise ValueError(
                "Segment motion_id/order does not match motion_segment_offsets."
            )
    if hasattr(metadata, "motion_keys"):
        motion_keys = np.asarray(metadata.motion_keys)
        if motion_keys.shape != (offsets.size - 1,):
            raise ValueError(
                "motion_keys/order does not match motion_segment_offsets."
            )
    return feature_names, values, available, durations, offsets


def _aggregate_column(
    values: np.ndarray,
    durations: np.ndarray,
    aggregation: str,
) -> float:
    if aggregation == "duration_weighted_mean":
        return float(np.average(values, weights=durations))
    if aggregation == "p90":
        return float(_quantile(values, 0.90))
    if aggregation == "duration_weighted_std":
        mean = float(np.average(values, weights=durations))
        variance = float(
            np.average(np.square(values - mean), weights=durations)
        )
        return math.sqrt(max(variance, 0.0))
    raise ValueError(f"Unsupported motion aggregation {aggregation!r}.")


def _aggregate_motion_features_with_definition(
    metadata: Any,
    *,
    feature_names: Sequence[str],
    source_feature_names: Sequence[str],
    aggregations: Sequence[str],
    feature_units: Sequence[str],
    required_features: Sequence[str],
    optional_features: Sequence[str],
) -> MotionFeatureMatrix:
    (
        segment_feature_names,
        segment_values,
        segment_available,
        durations,
        offsets,
    ) = _validate_aggregate_inputs(metadata)
    names = tuple(str(name) for name in feature_names)
    sources = tuple(str(name) for name in source_feature_names)
    aggregate_names = tuple(str(name) for name in aggregations)
    units = tuple(str(name) for name in feature_units)
    count = len(names)
    if (
        count == 0
        or len(sources) != count
        or len(aggregate_names) != count
        or len(units) != count
        or len(set(names)) != count
    ):
        raise ValueError("Expanded motion feature definition is invalid.")
    if any(name not in SUPPORTED_AGGREGATIONS for name in aggregate_names):
        raise ValueError("Expanded motion feature definition has an unsupported aggregation.")
    required = tuple(str(name) for name in required_features)
    optional = tuple(str(name) for name in optional_features)
    if (
        set(required).intersection(optional)
        or set(required).union(optional) != set(names)
    ):
        raise ValueError("Expanded required/optional feature partition is invalid.")

    source_indexes = {
        name: index for index, name in enumerate(segment_feature_names)
    }
    required_sources = {
        source
        for name, source in zip(names, sources, strict=True)
        if name in set(required)
    }
    missing_required_sources = sorted(
        required_sources.difference(source_indexes)
    )
    if missing_required_sources:
        raise ValueError(
            "Required clustering source features are absent from difficulty "
            f"metadata: {missing_required_sources}."
        )

    num_motions = offsets.size - 1
    result = np.full((num_motions, count), np.nan, dtype=np.float64)
    available_result = np.zeros((num_motions, count), dtype=bool)
    for motion_index, (start, end) in enumerate(
        zip(offsets[:-1], offsets[1:], strict=True)
    ):
        motion_durations = durations[start:end]
        for column, (source_name, aggregation) in enumerate(
            zip(sources, aggregate_names, strict=True)
        ):
            source_index = source_indexes.get(source_name)
            if source_index is None:
                continue
            observed_mask = segment_available[start:end, source_index]
            # A motion aggregate represents the complete motion.  Partial
            # segment coverage is kept unavailable instead of silently
            # changing the represented time interval.
            if not np.all(observed_mask):
                continue
            observed = segment_values[start:end, source_index]
            aggregate = _aggregate_column(
                observed, motion_durations, aggregation
            )
            if not math.isfinite(aggregate):
                raise ValueError(
                    f"Motion aggregation '{names[column]}' produced NaN or Inf."
                )
            result[motion_index, column] = aggregate
            available_result[motion_index, column] = True

    motion_durations = np.add.reduceat(durations, offsets[:-1])
    segment_counts = np.diff(offsets)
    return MotionFeatureMatrix(
        feature_names=names,
        source_feature_names=sources,
        aggregations=aggregate_names,
        feature_units=units,
        required_features=required,
        optional_features=optional,
        values=result,
        available_mask=available_result,
        motion_duration_seconds=motion_durations.astype(
            np.float64, copy=False
        ),
        segment_counts=segment_counts.astype(np.int64, copy=False),
    )


def aggregate_motion_features(
    metadata: Any, config: Mapping[str, Any]
) -> MotionFeatureMatrix:
    """Aggregate module-two raw segment features into motion features.

    Duration-weighted aggregations use each segment's actual duration, so a
    short tail segment cannot receive a full segment's weight.
    """

    validate_motion_clustering_config(config)
    (
        names,
        sources,
        aggregations,
        units,
        required,
        optional,
    ) = _expanded_feature_definition(config)
    return _aggregate_motion_features_with_definition(
        metadata,
        feature_names=names,
        source_feature_names=sources,
        aggregations=aggregations,
        feature_units=units,
        required_features=required,
        optional_features=optional,
    )


@dataclass(frozen=True)
class RobustScalerState:
    centers: np.ndarray
    scales: np.ndarray
    coverage: np.ndarray
    near_constant_mask: np.ndarray
    active_mask: np.ndarray
    warnings: tuple[str, ...]


def fit_robust_scaler(
    values: np.ndarray,
    available_mask: np.ndarray,
    *,
    required_mask: Sequence[bool],
    optional_mask: Sequence[bool],
    minimum_optional_feature_coverage: float,
    robust_scale_epsilon: float,
    near_constant_scale_threshold: float,
    zero_mad_fallback_quantiles: Sequence[float],
    zero_mad_fallback_scale_divisor: float,
) -> RobustScalerState:
    """Fit median/MAD scaling with a quantile fallback on Train motions."""

    matrix = np.asarray(values, dtype=np.float64)
    available = np.asarray(available_mask, dtype=bool)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or available.shape != matrix.shape:
        raise ValueError(
            "Robust scaler values/availability must be same-shape non-empty matrices."
        )
    if np.any(available & ~np.isfinite(matrix)):
        raise ValueError("Available scaler input contains NaN or Inf.")
    required = np.asarray(required_mask, dtype=bool)
    optional = np.asarray(optional_mask, dtype=bool)
    if (
        required.shape != (matrix.shape[1],)
        or optional.shape != required.shape
        or np.any(required & optional)
        or not np.all(required | optional)
    ):
        raise ValueError("required_mask/optional_mask must partition feature columns.")
    if np.any(~available[:, required]):
        bad = np.flatnonzero(np.any(~available[:, required], axis=0))
        required_columns = np.flatnonzero(required)
        raise ValueError(
            "Required motion clustering features are unavailable for one or "
            f"more Train motions (column indexes {required_columns[bad].tolist()})."
        )

    minimum_optional = _finite_scalar(
        minimum_optional_feature_coverage,
        "minimum_optional_feature_coverage",
    )
    if not 0.0 <= minimum_optional <= 1.0:
        raise ValueError("minimum_optional_feature_coverage must be in [0, 1].")
    coverage = np.mean(available, axis=0)
    low_optional = optional & (coverage < minimum_optional)
    if np.any(low_optional):
        raise ValueError(
            "Optional motion clustering feature coverage is below the "
            f"configured minimum for columns {np.flatnonzero(low_optional).tolist()}."
        )

    epsilon = _finite_scalar(
        robust_scale_epsilon, "robust_scale_epsilon"
    )
    near_threshold = _finite_scalar(
        near_constant_scale_threshold, "near_constant_scale_threshold"
    )
    fallback_quantiles = np.asarray(
        zero_mad_fallback_quantiles, dtype=np.float64
    )
    fallback_divisor = _finite_scalar(
        zero_mad_fallback_scale_divisor,
        "zero_mad_fallback_scale_divisor",
    )
    if (
        epsilon <= 0.0
        or near_threshold < epsilon
        or fallback_quantiles.shape != (2,)
        or not np.isfinite(fallback_quantiles).all()
        or fallback_quantiles[0] < 0.0
        or fallback_quantiles[1] > 1.0
        or fallback_quantiles[0] >= fallback_quantiles[1]
        or fallback_divisor <= 0.0
    ):
        raise ValueError("Robust scaler fallback settings are invalid.")

    num_features = matrix.shape[1]
    centers = np.empty(num_features, dtype=np.float64)
    scales = np.empty(num_features, dtype=np.float64)
    near_constant = np.zeros(num_features, dtype=bool)
    warnings: list[str] = []
    for column in range(num_features):
        observed = matrix[available[:, column], column]
        if observed.size == 0:
            centers[column] = 0.0
            scales[column] = epsilon
            near_constant[column] = True
            warnings.append(
                f"feature column {column} has zero coverage and is disabled"
            )
            continue
        center = float(np.median(observed))
        mad_scale = 1.4826 * float(
            np.median(np.abs(observed - center))
        )
        bounds = _quantile(observed, fallback_quantiles)
        fallback_scale = float(bounds[1] - bounds[0]) / fallback_divisor
        fitted_scale = (
            mad_scale if mad_scale >= near_threshold else fallback_scale
        )
        centers[column] = center
        scales[column] = max(fitted_scale, epsilon)
        near_constant[column] = fitted_scale < near_threshold
        if mad_scale < near_threshold <= fallback_scale:
            warnings.append(
                f"feature column {column} used zero-MAD quantile fallback "
                f"scale {fallback_scale:.9g}"
            )
        if near_constant[column]:
            warnings.append(
                f"feature column {column} is near-constant and is disabled"
            )

    # A frozen Euclidean space needs the same dimensions for every row.
    # Optional columns with incomplete Train coverage remain diagnostics,
    # rather than receiving an implicit zero/median physical observation.
    incomplete_optional = optional & (coverage < 1.0)
    for column in np.flatnonzero(incomplete_optional):
        warnings.append(
            f"optional feature column {column} has coverage "
            f"{coverage[column]:.6f} and is disabled"
        )
    active = ~near_constant & ~incomplete_optional
    if not np.any(active):
        raise ValueError(
            "All motion clustering features are near-constant or unavailable."
        )
    return RobustScalerState(
        centers=centers,
        scales=scales,
        coverage=coverage,
        near_constant_mask=near_constant,
        active_mask=active,
        warnings=tuple(warnings),
    )


def transform_robust_scaler(
    values: np.ndarray,
    available_mask: np.ndarray,
    centers: Sequence[float],
    scales: Sequence[float],
    robust_clip: float,
) -> np.ndarray:
    """Apply frozen robust scaling while preserving missingness as NaN."""

    matrix = np.asarray(values, dtype=np.float64)
    available = np.asarray(available_mask, dtype=bool)
    center_values = np.asarray(centers, dtype=np.float64)
    scale_values = np.asarray(scales, dtype=np.float64)
    if matrix.ndim != 2 or available.shape != matrix.shape:
        raise ValueError("Scaler values and availability must be same-shape matrices.")
    if (
        center_values.shape != (matrix.shape[1],)
        or scale_values.shape != center_values.shape
        or not np.isfinite(center_values).all()
        or not np.isfinite(scale_values).all()
        or np.any(scale_values <= 0.0)
    ):
        raise ValueError("Frozen scaler centers/scales are invalid.")
    if np.any(available & ~np.isfinite(matrix)):
        raise ValueError("Available scaler transform input contains NaN or Inf.")
    clip = _finite_scalar(robust_clip, "robust_clip")
    if clip <= 0.0:
        raise ValueError("robust_clip must be positive.")
    result = np.full_like(matrix, np.nan, dtype=np.float64)
    scaled = (matrix - center_values[None, :]) / scale_values[None, :]
    result[available] = np.clip(scaled[available], -clip, clip)
    return result


@dataclass(frozen=True)
class PCAState:
    mean: np.ndarray
    components: np.ndarray
    explained_variance: np.ndarray
    explained_variance_ratio: np.ndarray


def _canonicalize_pca_component_signs(
    components: np.ndarray,
) -> np.ndarray:
    result = np.asarray(components, dtype=np.float64).copy()
    for row in range(result.shape[0]):
        pivot = int(np.argmax(np.abs(result[row])))
        if result[row, pivot] < 0.0:
            result[row] *= -1.0
    return result


def fit_pca(
    values: np.ndarray,
    *,
    num_components: int | None = None,
    explained_variance_target: float | None = None,
) -> PCAState:
    """Fit deterministic NumPy PCA and canonicalize every component's sign."""

    matrix = np.asarray(values, dtype=np.float64)
    if (
        matrix.ndim != 2
        or matrix.shape[0] < 2
        or matrix.shape[1] < 1
        or not np.isfinite(matrix).all()
    ):
        raise ValueError("PCA requires a finite matrix with at least two rows.")
    if (num_components is None) == (explained_variance_target is None):
        raise ValueError(
            "Exactly one of num_components or explained_variance_target is required."
        )
    maximum = min(matrix.shape)
    if num_components is not None:
        components_count = _integer_value(num_components, "num_components")
        if not 1 <= components_count <= maximum:
            raise ValueError(
                f"num_components must be in [1, {maximum}] for this matrix."
            )
    else:
        target = _finite_scalar(
            explained_variance_target, "explained_variance_target"
        )
        if not 0.0 < target <= 1.0:
            raise ValueError("explained_variance_target must be in (0, 1].")

    mean = np.mean(matrix, axis=0)
    centered = matrix - mean[None, :]
    _, singular_values, vh = np.linalg.svd(
        centered, full_matrices=False
    )
    all_variance = np.square(singular_values) / float(matrix.shape[0] - 1)
    total_variance = float(np.sum(all_variance))
    if not math.isfinite(total_variance) or total_variance <= 0.0:
        raise ValueError("PCA input has no non-zero finite variance.")
    all_ratio = all_variance / total_variance
    if num_components is None:
        components_count = int(
            np.searchsorted(
                np.cumsum(all_ratio),
                float(explained_variance_target),
                side="left",
            )
            + 1
        )
        components_count = min(components_count, maximum)
    components = _canonicalize_pca_component_signs(
        vh[:components_count]
    )
    return PCAState(
        mean=mean.astype(np.float64, copy=False),
        components=components,
        explained_variance=all_variance[:components_count].astype(
            np.float64, copy=False
        ),
        explained_variance_ratio=all_ratio[:components_count].astype(
            np.float64, copy=False
        ),
    )


def transform_pca(values: np.ndarray, state: PCAState) -> np.ndarray:
    """Apply a frozen PCA transform without refitting."""

    matrix = np.asarray(values, dtype=np.float64)
    mean = np.asarray(state.mean, dtype=np.float64)
    components = np.asarray(state.components, dtype=np.float64)
    if (
        matrix.ndim != 2
        or mean.shape != (matrix.shape[1],)
        or components.ndim != 2
        or components.shape[1] != matrix.shape[1]
        or not np.isfinite(matrix).all()
        or not np.isfinite(mean).all()
        or not np.isfinite(components).all()
    ):
        raise ValueError("Frozen PCA state/input dimensions or values are invalid.")
    return (matrix - mean[None, :]) @ components.T


@dataclass(frozen=True)
class KMeansResult:
    labels: np.ndarray
    centroids: np.ndarray
    inertia: float
    n_iterations: int
    converged: bool
    canonical_label_remap: np.ndarray


def _squared_distances(
    values: np.ndarray, centroids: np.ndarray
) -> np.ndarray:
    differences = values[:, None, :] - centroids[None, :, :]
    distances = np.einsum(
        "nkd,nkd->nk", differences, differences, optimize=True
    )
    return np.maximum(distances, 0.0)


def _kmeans_plus_plus(
    values: np.ndarray, num_clusters: int, rng: np.random.Generator
) -> np.ndarray:
    num_samples = values.shape[0]
    selected: list[int] = [int(rng.integers(num_samples))]
    closest_squared = np.sum(
        np.square(values - values[selected[0]][None, :]), axis=1
    )
    while len(selected) < num_clusters:
        total = float(np.sum(closest_squared))
        if not math.isfinite(total):
            raise ValueError("KMeans++ distances contain NaN or Inf.")
        if total <= 0.0:
            unselected = np.setdiff1d(
                np.arange(num_samples, dtype=np.int64),
                np.asarray(selected, dtype=np.int64),
                assume_unique=False,
            )
            if unselected.size == 0:
                raise ValueError(
                    "KMeans++ cannot choose distinct centers from the input."
                )
            next_index = int(unselected[0])
        else:
            threshold = float(rng.random()) * total
            next_index = int(
                np.searchsorted(
                    np.cumsum(closest_squared), threshold, side="right"
                )
            )
            next_index = min(next_index, num_samples - 1)
            if next_index in selected or closest_squared[next_index] <= 0.0:
                candidates = np.flatnonzero(closest_squared > 0.0)
                if candidates.size == 0:
                    raise ValueError(
                        "KMeans++ input contains fewer distinct points than clusters."
                    )
                # Largest-distance repair is deterministic; index order breaks ties.
                next_index = int(
                    candidates[
                        np.argmax(closest_squared[candidates])
                    ]
                )
        selected.append(next_index)
        new_squared = np.sum(
            np.square(values - values[next_index][None, :]), axis=1
        )
        closest_squared = np.minimum(closest_squared, new_squared)
    return values[np.asarray(selected, dtype=np.int64)].copy()


def _repair_empty_clusters(
    labels: np.ndarray,
    squared_distances: np.ndarray,
    num_clusters: int,
) -> np.ndarray:
    repaired = labels.copy()
    counts = np.bincount(repaired, minlength=num_clusters).astype(np.int64)
    assigned_distance = squared_distances[
        np.arange(repaired.size, dtype=np.int64), repaired
    ]
    for empty_cluster in np.flatnonzero(counts == 0):
        candidates = np.flatnonzero(counts[repaired] > 1)
        if candidates.size == 0:
            raise RuntimeError("Unable to repair an empty KMeans cluster.")
        candidate_distances = assigned_distance[candidates]
        maximum = float(np.max(candidate_distances))
        # np.flatnonzero/order makes the smallest row index the stable tie-break.
        selected = int(candidates[np.flatnonzero(candidate_distances == maximum)[0]])
        donor = int(repaired[selected])
        repaired[selected] = int(empty_cluster)
        counts[donor] -= 1
        counts[empty_cluster] += 1
        assigned_distance[selected] = 0.0
    return repaired


def _canonicalize_clusters(
    labels: np.ndarray, centroids: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.asarray(
        sorted(
            range(centroids.shape[0]),
            key=lambda index: (
                tuple(float(value) for value in centroids[index]),
                index,
            ),
        ),
        dtype=np.int64,
    )
    remap = np.empty(centroids.shape[0], dtype=np.int64)
    remap[order] = np.arange(centroids.shape[0], dtype=np.int64)
    return remap[labels], centroids[order].copy(), remap


def _single_kmeans(
    values: np.ndarray,
    *,
    num_clusters: int,
    rng: np.random.Generator,
    max_iterations: int,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray, float, int, bool]:
    centroids = _kmeans_plus_plus(values, num_clusters, rng)
    previous_labels: np.ndarray | None = None
    converged = False
    labels = np.zeros(values.shape[0], dtype=np.int64)
    n_iterations = 0
    for iteration in range(1, max_iterations + 1):
        squared = _squared_distances(values, centroids)
        labels = np.argmin(squared, axis=1).astype(np.int64)
        labels = _repair_empty_clusters(labels, squared, num_clusters)
        updated = np.vstack(
            [np.mean(values[labels == cluster], axis=0) for cluster in range(num_clusters)]
        )
        if not np.isfinite(updated).all():
            raise RuntimeError("KMeans produced a non-finite centroid.")
        shift = float(
            np.max(np.linalg.norm(updated - centroids, axis=1))
        )
        labels_unchanged = (
            previous_labels is not None
            and np.array_equal(labels, previous_labels)
        )
        centroids = updated
        n_iterations = iteration
        if labels_unchanged or shift <= tolerance:
            converged = True
            break
        previous_labels = labels.copy()

    # Make returned assignments correspond to the returned centroids.  Exact
    # label convergence normally makes this a no-op; the bounded refinement
    # handles a tolerance stop near a Voronoi boundary.
    for _ in range(max_iterations):
        squared = _squared_distances(values, centroids)
        nearest = np.argmin(squared, axis=1).astype(np.int64)
        nearest = _repair_empty_clusters(nearest, squared, num_clusters)
        if np.array_equal(nearest, labels):
            labels = nearest
            break
        labels = nearest
        centroids = np.vstack(
            [np.mean(values[labels == cluster], axis=0) for cluster in range(num_clusters)]
        )
    final_squared = _squared_distances(values, centroids)
    inertia = float(
        np.sum(
            final_squared[
                np.arange(values.shape[0], dtype=np.int64), labels
            ]
        )
    )
    return labels, centroids, inertia, n_iterations, converged


def deterministic_kmeans(
    values: np.ndarray,
    *,
    num_clusters: int,
    random_seed: int = 42,
    n_init: int = 20,
    max_iterations: int = 300,
    tolerance: float = 1.0e-6,
    canonicalize_labels: bool = True,
) -> KMeansResult:
    """Run deterministic KMeans++/Lloyd with multiple independent starts."""

    matrix = np.asarray(values, dtype=np.float64)
    if (
        matrix.ndim != 2
        or matrix.shape[0] == 0
        or matrix.shape[1] == 0
        or not np.isfinite(matrix).all()
    ):
        raise ValueError("KMeans requires a non-empty finite 2-D matrix.")
    clusters = _integer_value(num_clusters, "num_clusters")
    seed = _integer_value(random_seed, "random_seed")
    initializations = _integer_value(n_init, "n_init")
    iterations = _integer_value(max_iterations, "max_iterations")
    threshold = _finite_scalar(tolerance, "tolerance")
    if (
        clusters < 1
        or clusters > matrix.shape[0]
        or seed < 0
        or seed > np.iinfo(np.uint32).max
        or initializations < 1
        or iterations < 1
        or threshold < 0.0
    ):
        raise ValueError("KMeans cluster count/seed/iteration settings are invalid.")
    if np.unique(matrix, axis=0).shape[0] < clusters:
        raise ValueError(
            "KMeans input has fewer distinct points than num_clusters."
        )

    best: KMeansResult | None = None
    best_key: tuple[float, tuple[float, ...], tuple[int, ...]] | None = None
    for initialization in range(initializations):
        rng = np.random.default_rng(
            np.random.SeedSequence([seed, initialization])
        )
        labels, centroids, inertia, n_iterations, converged = _single_kmeans(
            matrix,
            num_clusters=clusters,
            rng=rng,
            max_iterations=iterations,
            tolerance=threshold,
        )
        if canonicalize_labels:
            labels, centroids, remap = _canonicalize_clusters(
                labels, centroids
            )
        else:
            remap = np.arange(clusters, dtype=np.int64)
        result = KMeansResult(
            labels=labels,
            centroids=centroids,
            inertia=inertia,
            n_iterations=n_iterations,
            converged=converged,
            canonical_label_remap=remap,
        )
        key = (
            inertia,
            tuple(float(item) for item in centroids.ravel()),
            tuple(int(item) for item in labels),
        )
        if best is None:
            best, best_key = result, key
            continue
        assert best_key is not None
        inertia_tolerance = 1.0e-12 * max(1.0, abs(best_key[0]))
        if inertia < best_key[0] - inertia_tolerance or (
            abs(inertia - best_key[0]) <= inertia_tolerance
            and key[1:] < best_key[1:]
        ):
            best, best_key = result, key
    assert best is not None
    return best


@dataclass(frozen=True)
class MotionClusteringProfile:
    """Frozen aggregation, scaler, PCA, and canonical centroid state."""

    schema_version: str
    algorithm_schema_version: str
    feature_names: tuple[str, ...]
    source_feature_names: tuple[str, ...]
    feature_aggregations: tuple[str, ...]
    feature_units: tuple[str, ...]
    required_features: tuple[str, ...]
    optional_features: tuple[str, ...]
    minimum_optional_feature_coverage: float
    feature_centers: np.ndarray
    feature_scales: np.ndarray
    feature_coverage: np.ndarray
    near_constant_features: tuple[str, ...]
    active_feature_mask: np.ndarray
    robust_clip: float
    use_pca: bool
    pca_mean: np.ndarray
    pca_components: np.ndarray
    pca_explained_variance: np.ndarray
    pca_explained_variance_ratio: np.ndarray
    num_clusters: int
    random_seed: int
    kmeans_n_init: int
    kmeans_max_iterations: int
    kmeans_tolerance: float
    kmeans_inertia: float
    kmeans_n_iterations: int
    kmeans_converged: bool
    centroids: np.ndarray
    canonical_label_remap: np.ndarray
    training_manifest_sha256: str
    training_pool_fingerprint: str
    difficulty_metadata_sha256: str
    difficulty_profile_sha256: str
    config_sha256: str
    git_commit: str
    warnings: tuple[str, ...]

    @property
    def active_feature_names(self) -> tuple[str, ...]:
        return tuple(
            name
            for name, active in zip(
                self.feature_names, self.active_feature_mask, strict=True
            )
            if active
        )

    @property
    def sha256(self) -> str:
        return canonical_json_sha256(self.to_dict())

    @property
    def profile_sha256(self) -> str:
        return self.sha256

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "algorithm_schema_version": self.algorithm_schema_version,
            "feature_names": list(self.feature_names),
            "source_feature_names": list(self.source_feature_names),
            "feature_aggregations": list(self.feature_aggregations),
            "feature_units": list(self.feature_units),
            "required_features": list(self.required_features),
            "optional_features": list(self.optional_features),
            "minimum_optional_feature_coverage": (
                self.minimum_optional_feature_coverage
            ),
            "feature_centers": self.feature_centers.tolist(),
            "feature_scales": self.feature_scales.tolist(),
            "feature_coverage": self.feature_coverage.tolist(),
            "near_constant_features": list(self.near_constant_features),
            "active_feature_mask": self.active_feature_mask.tolist(),
            "robust_clip": self.robust_clip,
            "pca": {
                "enabled": self.use_pca,
                "mean": self.pca_mean.tolist(),
                "components": self.pca_components.tolist(),
                "explained_variance": self.pca_explained_variance.tolist(),
                "explained_variance_ratio": (
                    self.pca_explained_variance_ratio.tolist()
                ),
            },
            "algorithm": {
                "name": "KMeans",
                "implementation_identity": KMEANS_IMPLEMENTATION_IDENTITY,
                "random_seed": self.random_seed,
                "n_init": self.kmeans_n_init,
                "max_iterations": self.kmeans_max_iterations,
                "tolerance": self.kmeans_tolerance,
                "inertia": self.kmeans_inertia,
                "n_iterations": self.kmeans_n_iterations,
                "converged": self.kmeans_converged,
            },
            "num_clusters": self.num_clusters,
            "centroids": self.centroids.tolist(),
            "canonical_label_remap": self.canonical_label_remap.tolist(),
            "training_manifest_sha256": self.training_manifest_sha256,
            "training_pool_fingerprint": self.training_pool_fingerprint,
            "difficulty_metadata_sha256": self.difficulty_metadata_sha256,
            "difficulty_profile_sha256": self.difficulty_profile_sha256,
            "config_sha256": self.config_sha256,
            "git_commit": self.git_commit,
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MotionClusteringProfile":
        """Strictly validate and load a frozen profile mapping."""

        if not isinstance(value, Mapping):
            raise ValueError("Motion clustering profile root must be an object.")
        if value.get("schema_version") != MOTION_CLUSTERING_PROFILE_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported motion clustering profile schema "
                f"{value.get('schema_version')!r}."
            )
        if value.get("algorithm_schema_version") != DEFAULT_ALGORITHM_SCHEMA_VERSION:
            raise ValueError(
                "Unsupported motion clustering profile algorithm schema."
            )
        feature_names = _string_tuple(
            value.get("feature_names"), "feature_names"
        )
        sources = _string_tuple(
            value.get("source_feature_names"),
            "source_feature_names",
            unique=False,
        )
        aggregations = _string_tuple(
            value.get("feature_aggregations"),
            "feature_aggregations",
            unique=False,
        )
        units = tuple(str(item) for item in value.get("feature_units", ()))
        count = len(feature_names)
        if (
            len(sources) != count
            or len(aggregations) != count
            or len(units) != count
            or any(not unit for unit in units)
            or any(item not in SUPPORTED_AGGREGATIONS for item in aggregations)
        ):
            raise ValueError("Profile expanded feature definition is invalid.")
        required = _string_tuple(
            value.get("required_features"),
            "required_features",
            allow_empty=True,
        )
        optional = _string_tuple(
            value.get("optional_features"),
            "optional_features",
            allow_empty=True,
        )
        if (
            set(required).intersection(optional)
            or set(required).union(optional) != set(feature_names)
        ):
            raise ValueError("Profile required/optional features are invalid.")

        def finite_vector(field: str, length: int) -> np.ndarray:
            result = np.asarray(value.get(field), dtype=np.float64)
            if result.shape != (length,) or not np.isfinite(result).all():
                raise ValueError(
                    f"Profile field '{field}' must be a finite ({length},) vector."
                )
            return result

        centers = finite_vector("feature_centers", count)
        scales = finite_vector("feature_scales", count)
        coverage = finite_vector("feature_coverage", count)
        if np.any(scales <= 0.0):
            raise ValueError("Profile feature_scales must be positive.")
        if np.any(coverage < 0.0) or np.any(coverage > 1.0):
            raise ValueError("Profile feature_coverage must be in [0, 1].")
        active_raw = np.asarray(value.get("active_feature_mask"))
        if active_raw.shape != (count,) or active_raw.dtype.kind != "b":
            raise ValueError(
                "Profile active_feature_mask must be a boolean feature vector."
            )
        active = active_raw.astype(bool, copy=False)
        if not np.any(active):
            raise ValueError("Profile must contain at least one active feature.")
        near_constant = _string_tuple(
            value.get("near_constant_features"),
            "near_constant_features",
            allow_empty=True,
        )
        if not set(near_constant).issubset(feature_names):
            raise ValueError("Profile near_constant_features are invalid.")
        near_indexes = np.asarray(
            [feature_names.index(name) for name in near_constant],
            dtype=np.int64,
        )
        if near_indexes.size and np.any(active[near_indexes]):
            raise ValueError(
                "Profile near-constant features cannot remain active."
            )
        minimum_optional = _finite_scalar(
            value.get("minimum_optional_feature_coverage"),
            "minimum_optional_feature_coverage",
        )
        robust_clip = _finite_scalar(
            value.get("robust_clip"), "robust_clip"
        )
        if not 0.0 <= minimum_optional <= 1.0 or robust_clip <= 0.0:
            raise ValueError(
                "Profile optional coverage/clipping settings are invalid."
            )

        pca = value.get("pca")
        if not isinstance(pca, Mapping) or not isinstance(
            pca.get("enabled"), bool
        ):
            raise ValueError("Profile pca state is invalid.")
        use_pca = bool(pca["enabled"])
        pca_mean = np.asarray(pca.get("mean"), dtype=np.float64)
        pca_components = np.asarray(pca.get("components"), dtype=np.float64)
        pca_variance = np.asarray(
            pca.get("explained_variance"), dtype=np.float64
        )
        pca_ratio = np.asarray(
            pca.get("explained_variance_ratio"), dtype=np.float64
        )
        active_count = int(np.count_nonzero(active))
        if use_pca:
            if (
                pca_mean.shape != (active_count,)
                or pca_components.ndim != 2
                or pca_components.shape[0] < 1
                or pca_components.shape[1] != active_count
                or pca_variance.shape != (pca_components.shape[0],)
                or pca_ratio.shape != pca_variance.shape
                or not np.isfinite(pca_mean).all()
                or not np.isfinite(pca_components).all()
                or not np.isfinite(pca_variance).all()
                or not np.isfinite(pca_ratio).all()
                or np.any(pca_variance < 0.0)
                or np.any(pca_ratio < 0.0)
            ):
                raise ValueError("Profile frozen PCA arrays are invalid.")
            cluster_dimension = pca_components.shape[0]
        else:
            if (
                pca_mean.size
                or pca_components.size
                or pca_variance.size
                or pca_ratio.size
            ):
                raise ValueError(
                    "Disabled Profile PCA state must contain empty arrays."
                )
            pca_mean = np.empty(0, dtype=np.float64)
            pca_components = np.empty((0, active_count), dtype=np.float64)
            pca_variance = np.empty(0, dtype=np.float64)
            pca_ratio = np.empty(0, dtype=np.float64)
            cluster_dimension = active_count

        algorithm = value.get("algorithm")
        if (
            not isinstance(algorithm, Mapping)
            or algorithm.get("name") != "KMeans"
            or algorithm.get("implementation_identity")
            != KMEANS_IMPLEMENTATION_IDENTITY
        ):
            raise ValueError("Profile KMeans implementation identity is invalid.")
        num_clusters = _integer_value(
            value.get("num_clusters"), "num_clusters"
        )
        random_seed = _integer_value(
            algorithm.get("random_seed"), "algorithm.random_seed"
        )
        n_init = _integer_value(
            algorithm.get("n_init"), "algorithm.n_init"
        )
        max_iterations = _integer_value(
            algorithm.get("max_iterations"),
            "algorithm.max_iterations",
        )
        tolerance = _finite_scalar(
            algorithm.get("tolerance"), "algorithm.tolerance"
        )
        inertia = _finite_scalar(
            algorithm.get("inertia"), "algorithm.inertia"
        )
        n_iterations = _integer_value(
            algorithm.get("n_iterations"),
            "algorithm.n_iterations",
        )
        converged = algorithm.get("converged")
        if (
            num_clusters < 1
            or random_seed < 0
            or n_init < 1
            or max_iterations < 1
            or tolerance < 0.0
            or inertia < 0.0
            or n_iterations < 1
            or not isinstance(converged, bool)
        ):
            raise ValueError("Profile KMeans settings are invalid.")
        centroids = np.asarray(value.get("centroids"), dtype=np.float64)
        if (
            centroids.shape != (num_clusters, cluster_dimension)
            or not np.isfinite(centroids).all()
        ):
            raise ValueError("Profile centroids shape/values are invalid.")
        centroid_order = sorted(
            range(num_clusters),
            key=lambda index: (
                tuple(float(item) for item in centroids[index]),
                index,
            ),
        )
        if centroid_order != list(range(num_clusters)):
            raise ValueError(
                "Profile centroids are not in canonical lexicographic label order."
            )
        remap_raw = np.asarray(value.get("canonical_label_remap"))
        if (
            remap_raw.shape != (num_clusters,)
            or remap_raw.dtype.kind not in "iu"
        ):
            raise ValueError("Profile canonical_label_remap is invalid.")
        remap = remap_raw.astype(np.int64, copy=False)
        if not np.array_equal(
            np.sort(remap), np.arange(num_clusters, dtype=np.int64)
        ):
            raise ValueError(
                "Profile canonical_label_remap must be a permutation."
            )

        warnings_raw = value.get("warnings", ())
        if not isinstance(warnings_raw, (list, tuple)):
            raise ValueError("Profile warnings must be a list.")
        warnings = tuple(str(item) for item in warnings_raw)
        identity_names = (
            "training_manifest_sha256",
            "training_pool_fingerprint",
            "difficulty_metadata_sha256",
            "difficulty_profile_sha256",
            "config_sha256",
            "git_commit",
        )
        identities = {
            name: str(value.get(name, "")) for name in identity_names
        }
        for name in (
            "training_manifest_sha256",
            "training_pool_fingerprint",
            "difficulty_metadata_sha256",
            "difficulty_profile_sha256",
            "config_sha256",
        ):
            identity = identities[name]
            if identity and not _is_sha256(identity):
                raise ValueError(
                    f"Profile identity '{name}' must be a lowercase SHA256 digest."
                )
        return cls(
            schema_version=MOTION_CLUSTERING_PROFILE_SCHEMA_VERSION,
            algorithm_schema_version=DEFAULT_ALGORITHM_SCHEMA_VERSION,
            feature_names=feature_names,
            source_feature_names=sources,
            feature_aggregations=aggregations,
            feature_units=units,
            required_features=required,
            optional_features=optional,
            minimum_optional_feature_coverage=minimum_optional,
            feature_centers=centers,
            feature_scales=scales,
            feature_coverage=coverage,
            near_constant_features=near_constant,
            active_feature_mask=active,
            robust_clip=robust_clip,
            use_pca=use_pca,
            pca_mean=pca_mean,
            pca_components=pca_components,
            pca_explained_variance=pca_variance,
            pca_explained_variance_ratio=pca_ratio,
            num_clusters=num_clusters,
            random_seed=random_seed,
            kmeans_n_init=n_init,
            kmeans_max_iterations=max_iterations,
            kmeans_tolerance=tolerance,
            kmeans_inertia=inertia,
            kmeans_n_iterations=n_iterations,
            kmeans_converged=bool(converged),
            centroids=centroids,
            canonical_label_remap=remap,
            training_manifest_sha256=identities[
                "training_manifest_sha256"
            ],
            training_pool_fingerprint=identities[
                "training_pool_fingerprint"
            ],
            difficulty_metadata_sha256=identities[
                "difficulty_metadata_sha256"
            ],
            difficulty_profile_sha256=identities[
                "difficulty_profile_sha256"
            ],
            config_sha256=identities["config_sha256"],
            git_commit=identities["git_commit"],
            warnings=warnings,
        )

    def save_json(self, path: str | Path) -> None:
        """Write a deterministic, human-readable frozen profile."""

        profile_path = Path(path)
        profile_path.write_text(
            json.dumps(
                self.to_dict(),
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load_json(
        cls,
        path: str | Path,
        *,
        require_frozen_identity: bool = False,
    ) -> "MotionClusteringProfile":
        profile_path = Path(path)
        if not profile_path.is_file():
            raise FileNotFoundError(
                f"Motion clustering profile does not exist: {profile_path}"
            )
        try:
            payload = json.loads(profile_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid motion clustering profile JSON '{profile_path}': {exc}"
            ) from exc
        profile = cls.from_dict(payload)
        if require_frozen_identity:
            for name in (
                "training_manifest_sha256",
                "training_pool_fingerprint",
                "difficulty_metadata_sha256",
                "difficulty_profile_sha256",
                "config_sha256",
            ):
                if not _is_sha256(getattr(profile, name)):
                    raise ValueError(
                        f"Frozen Profile identity '{name}' must be a lowercase "
                        "SHA256 digest."
                    )
        return profile


@dataclass(frozen=True)
class MotionClusteringResult:
    """Motion assignments and every frozen/intermediate feature space."""

    profile: MotionClusteringProfile
    motion_features: MotionFeatureMatrix
    standardized_features: np.ndarray
    clustering_features: np.ndarray
    cluster_id: np.ndarray
    distance_to_centroid: np.ndarray
    normalized_distance_to_centroid: np.ndarray

    @property
    def labels(self) -> np.ndarray:
        return self.cluster_id

    @property
    def cluster_ids(self) -> np.ndarray:
        return self.cluster_id


def _validate_motion_feature_matrix(features: MotionFeatureMatrix) -> None:
    if not isinstance(features, MotionFeatureMatrix):
        raise TypeError("Expected a MotionFeatureMatrix.")
    count = len(features.feature_names)
    if (
        count == 0
        or len(features.source_feature_names) != count
        or len(features.aggregations) != count
        or len(features.feature_units) != count
        or len(set(features.feature_names)) != count
        or features.values.ndim != 2
        or features.values.shape[1] != count
        or features.values.shape[0] == 0
        or features.available_mask.shape != features.values.shape
        or features.motion_duration_seconds.shape
        != (features.values.shape[0],)
        or features.segment_counts.shape != (features.values.shape[0],)
    ):
        raise ValueError("MotionFeatureMatrix structure is invalid.")
    if np.any(features.available_mask & ~np.isfinite(features.values)):
        raise ValueError(
            "Available MotionFeatureMatrix values contain NaN or Inf."
        )
    if (
        not np.isfinite(features.motion_duration_seconds).all()
        or np.any(features.motion_duration_seconds <= 0.0)
        or np.any(features.segment_counts <= 0)
    ):
        raise ValueError("MotionFeatureMatrix duration/count values are invalid.")
    if (
        set(features.required_features).intersection(
            features.optional_features
        )
        or set(features.required_features).union(
            features.optional_features
        )
        != set(features.feature_names)
    ):
        raise ValueError(
            "MotionFeatureMatrix required/optional partition is invalid."
        )


def _normalise_centroid_distances(
    labels: np.ndarray, distances: np.ndarray, num_clusters: int
) -> np.ndarray:
    result = np.zeros_like(distances, dtype=np.float64)
    for cluster in range(num_clusters):
        mask = labels == cluster
        cluster_distances = distances[mask]
        if cluster_distances.size == 0:
            continue
        median = float(np.median(cluster_distances))
        if median > 0.0:
            result[mask] = cluster_distances / median
        else:
            maximum = float(np.max(cluster_distances))
            result[mask] = (
                cluster_distances / maximum if maximum > 0.0 else 0.0
            )
    return result


def fit_motion_clustering(
    motion_features: MotionFeatureMatrix,
    config: Mapping[str, Any],
    *,
    training_manifest_sha256: str = "",
    training_pool_fingerprint: str = "",
    difficulty_metadata_sha256: str = "",
    difficulty_profile_sha256: str = "",
    config_sha256: str | None = None,
    git_commit: str = "",
) -> MotionClusteringResult:
    """Fit Train-only scaling/PCA/KMeans from already aggregated motions."""

    validate_motion_clustering_config(config)
    _validate_motion_feature_matrix(motion_features)
    expected = _expanded_feature_definition(config)
    if (
        motion_features.feature_names != expected[0]
        or motion_features.source_feature_names != expected[1]
        or motion_features.aggregations != expected[2]
        or motion_features.feature_units != expected[3]
        or motion_features.required_features != expected[4]
        or motion_features.optional_features != expected[5]
    ):
        raise ValueError(
            "MotionFeatureMatrix definition/order does not match clustering config."
        )
    if motion_features.num_motions < int(config["num_clusters"]):
        raise ValueError(
            "num_clusters cannot exceed the number of Train motions."
        )

    required_set = set(motion_features.required_features)
    required_mask = np.asarray(
        [name in required_set for name in motion_features.feature_names],
        dtype=bool,
    )
    scaler = fit_robust_scaler(
        motion_features.values,
        motion_features.available_mask,
        required_mask=required_mask,
        optional_mask=~required_mask,
        minimum_optional_feature_coverage=float(
            config["minimum_optional_feature_coverage"]
        ),
        robust_scale_epsilon=float(config["robust_scale_epsilon"]),
        near_constant_scale_threshold=float(
            config["near_constant_scale_threshold"]
        ),
        zero_mad_fallback_quantiles=config[
            "zero_mad_fallback_quantiles"
        ],
        zero_mad_fallback_scale_divisor=float(
            config["zero_mad_fallback_scale_divisor"]
        ),
    )
    standardized = transform_robust_scaler(
        motion_features.values,
        motion_features.available_mask,
        scaler.centers,
        scaler.scales,
        float(config["robust_clip"]),
    )
    active_standardized = standardized[:, scaler.active_mask]
    if not np.isfinite(active_standardized).all():
        raise RuntimeError(
            "Active Train clustering features unexpectedly contain missing values."
        )

    if bool(config["use_pca"]):
        pca_state = fit_pca(
            active_standardized,
            num_components=(
                int(config["pca_components"])
                if config["pca_components"] is not None
                else None
            ),
            explained_variance_target=(
                float(config["explained_variance_target"])
                if config["explained_variance_target"] is not None
                else None
            ),
        )
        clustering_values = transform_pca(
            active_standardized, pca_state
        )
    else:
        pca_state = PCAState(
            mean=np.empty(0, dtype=np.float64),
            components=np.empty(
                (0, active_standardized.shape[1]), dtype=np.float64
            ),
            explained_variance=np.empty(0, dtype=np.float64),
            explained_variance_ratio=np.empty(0, dtype=np.float64),
        )
        clustering_values = active_standardized.copy()

    kmeans = deterministic_kmeans(
        clustering_values,
        num_clusters=int(config["num_clusters"]),
        random_seed=int(config["random_seed"]),
        n_init=int(config["kmeans_n_init"]),
        max_iterations=int(config["kmeans_max_iterations"]),
        tolerance=float(config["kmeans_tolerance"]),
        canonicalize_labels=True,
    )
    centroid_squared = _squared_distances(
        clustering_values, kmeans.centroids
    )
    labels = np.argmin(centroid_squared, axis=1).astype(np.int64)
    # In a well-defined fitted KMeans result, the frozen nearest-centroid
    # transform must reproduce Train labels exactly.
    if not np.array_equal(labels, kmeans.labels):
        raise RuntimeError(
            "Final KMeans labels do not match frozen nearest-centroid assignment."
        )
    distances = np.sqrt(
        centroid_squared[
            np.arange(labels.size, dtype=np.int64), labels
        ]
    )
    normalized = _normalise_centroid_distances(
        labels, distances, int(config["num_clusters"])
    )
    near_names = tuple(
        name
        for name, near in zip(
            motion_features.feature_names,
            scaler.near_constant_mask,
            strict=True,
        )
        if near
    )
    warnings = list(scaler.warnings)
    cluster_sizes = np.bincount(
        labels, minlength=int(config["num_clusters"])
    )
    warning_size = int(config["minimum_cluster_size_warning"])
    for cluster, size in enumerate(cluster_sizes):
        if int(size) < warning_size:
            warnings.append(
                f"cluster {cluster} has {int(size)} motions, below warning "
                f"threshold {warning_size}"
            )

    profile = MotionClusteringProfile(
        schema_version=MOTION_CLUSTERING_PROFILE_SCHEMA_VERSION,
        algorithm_schema_version=DEFAULT_ALGORITHM_SCHEMA_VERSION,
        feature_names=motion_features.feature_names,
        source_feature_names=motion_features.source_feature_names,
        feature_aggregations=motion_features.aggregations,
        feature_units=motion_features.feature_units,
        required_features=motion_features.required_features,
        optional_features=motion_features.optional_features,
        minimum_optional_feature_coverage=float(
            config["minimum_optional_feature_coverage"]
        ),
        feature_centers=scaler.centers,
        feature_scales=scaler.scales,
        feature_coverage=scaler.coverage,
        near_constant_features=near_names,
        active_feature_mask=scaler.active_mask,
        robust_clip=float(config["robust_clip"]),
        use_pca=bool(config["use_pca"]),
        pca_mean=pca_state.mean,
        pca_components=pca_state.components,
        pca_explained_variance=pca_state.explained_variance,
        pca_explained_variance_ratio=pca_state.explained_variance_ratio,
        num_clusters=int(config["num_clusters"]),
        random_seed=int(config["random_seed"]),
        kmeans_n_init=int(config["kmeans_n_init"]),
        kmeans_max_iterations=int(config["kmeans_max_iterations"]),
        kmeans_tolerance=float(config["kmeans_tolerance"]),
        kmeans_inertia=kmeans.inertia,
        kmeans_n_iterations=kmeans.n_iterations,
        kmeans_converged=kmeans.converged,
        centroids=kmeans.centroids,
        canonical_label_remap=kmeans.canonical_label_remap,
        training_manifest_sha256=str(training_manifest_sha256),
        training_pool_fingerprint=str(training_pool_fingerprint),
        difficulty_metadata_sha256=str(difficulty_metadata_sha256),
        difficulty_profile_sha256=str(difficulty_profile_sha256),
        config_sha256=(
            str(config_sha256)
            if config_sha256 is not None
            else canonical_json_sha256(config)
        ),
        git_commit=str(git_commit),
        warnings=tuple(warnings),
    )
    # Exercise the same strict validation used by JSON loading before the
    # fitted profile is allowed to escape.
    profile = MotionClusteringProfile.from_dict(profile.to_dict())
    return MotionClusteringResult(
        profile=profile,
        motion_features=motion_features,
        standardized_features=standardized,
        clustering_features=clustering_values,
        cluster_id=labels,
        distance_to_centroid=distances,
        normalized_distance_to_centroid=normalized,
    )


def fit_transform_motion_clustering(
    metadata: Any,
    config: Mapping[str, Any],
    **identity: Any,
) -> MotionClusteringResult:
    """Aggregate raw segment metadata and fit/assign Train motions."""

    features = aggregate_motion_features(metadata, config)
    return fit_motion_clustering(features, config, **identity)


def _aggregate_for_profile(
    metadata: Any, profile: MotionClusteringProfile
) -> MotionFeatureMatrix:
    return _aggregate_motion_features_with_definition(
        metadata,
        feature_names=profile.feature_names,
        source_feature_names=profile.source_feature_names,
        aggregations=profile.feature_aggregations,
        feature_units=profile.feature_units,
        required_features=profile.required_features,
        optional_features=profile.optional_features,
    )


def transform_motion_features(
    motion_features: MotionFeatureMatrix,
    profile: MotionClusteringProfile,
) -> MotionClusteringResult:
    """Assign motions using only a frozen Profile (never refit)."""

    _validate_motion_feature_matrix(motion_features)
    # Round-trip validation also protects callers that manually constructed a
    # dataclass instead of using ``from_dict``.
    profile = MotionClusteringProfile.from_dict(profile.to_dict())
    if (
        motion_features.feature_names != profile.feature_names
        or motion_features.source_feature_names
        != profile.source_feature_names
        or motion_features.aggregations != profile.feature_aggregations
        or motion_features.feature_units != profile.feature_units
        or motion_features.required_features != profile.required_features
        or motion_features.optional_features != profile.optional_features
    ):
        raise ValueError(
            "Transform motion feature definition/order does not match the frozen Profile."
        )
    required_set = set(profile.required_features)
    required_mask = np.asarray(
        [name in required_set for name in profile.feature_names], dtype=bool
    )
    if np.any(~motion_features.available_mask[:, required_mask]):
        raise ValueError(
            "One or more frozen required motion features are unavailable in transform data."
        )
    optional_mask = ~required_mask
    optional_coverage = np.mean(
        motion_features.available_mask[:, optional_mask], axis=0
    )
    if optional_coverage.size and np.any(
        optional_coverage < profile.minimum_optional_feature_coverage
    ):
        raise ValueError(
            "Transform optional motion feature coverage is below the frozen "
            "Profile minimum."
        )
    if np.any(
        ~motion_features.available_mask[:, profile.active_feature_mask]
    ):
        missing = np.flatnonzero(
            np.any(
                ~motion_features.available_mask[
                    :, profile.active_feature_mask
                ],
                axis=0,
            )
        )
        active_names = np.asarray(
            profile.active_feature_names, dtype=object
        )
        raise ValueError(
            "Frozen active clustering features are unavailable in transform "
            f"data: {active_names[missing].tolist()}."
        )

    standardized = transform_robust_scaler(
        motion_features.values,
        motion_features.available_mask,
        profile.feature_centers,
        profile.feature_scales,
        profile.robust_clip,
    )
    active = standardized[:, profile.active_feature_mask]
    if profile.use_pca:
        clustering_values = transform_pca(
            active,
            PCAState(
                mean=profile.pca_mean,
                components=profile.pca_components,
                explained_variance=profile.pca_explained_variance,
                explained_variance_ratio=(
                    profile.pca_explained_variance_ratio
                ),
            ),
        )
    else:
        clustering_values = active.copy()
    squared = _squared_distances(
        clustering_values, profile.centroids
    )
    labels = np.argmin(squared, axis=1).astype(np.int64)
    distances = np.sqrt(
        squared[np.arange(labels.size, dtype=np.int64), labels]
    )
    normalized = _normalise_centroid_distances(
        labels, distances, profile.num_clusters
    )
    return MotionClusteringResult(
        profile=profile,
        motion_features=motion_features,
        standardized_features=standardized,
        clustering_features=clustering_values,
        cluster_id=labels,
        distance_to_centroid=distances,
        normalized_distance_to_centroid=normalized,
    )


def transform_motion_clustering(
    metadata: Any, profile: MotionClusteringProfile
) -> MotionClusteringResult:
    """Aggregate segment metadata and apply a frozen clustering Profile."""

    return transform_motion_features(
        _aggregate_for_profile(metadata, profile), profile
    )


def save_motion_clustering_profile(
    profile: MotionClusteringProfile, path: str | Path
) -> None:
    profile.save_json(path)


def load_motion_clustering_profile(
    path: str | Path,
) -> MotionClusteringProfile:
    return MotionClusteringProfile.load_json(
        path, require_frozen_identity=True
    )


# Concise aliases used by offline builders.
fit_transform = fit_transform_motion_clustering
transform = transform_motion_clustering
