"""Frozen, policy-independent segment-difficulty metadata utilities.

The loader in this module validates a compact ``.npz`` produced by the
offline difficulty pipeline.  It intentionally depends only on NumPy and the
Python standard library and has no dependency on quality labels, policy
outputs, rewards, or training state.
"""

from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Mapping, Sequence

import numpy as np


DIFFICULTY_METADATA_SCHEMA_VERSION: Final[str] = "wbt.segment_difficulty.v1"

_REQUIRED_ARRAYS: Final[tuple[str, ...]] = (
    "schema_version",
    "algorithm_schema_version",
    "segment_schema_version",
    "segment_length_seconds",
    "manifest_sha256",
    "manifest_motion_count",
    "profile_sha256",
    "difficulty_config_sha256",
    "pool_fingerprint",
    "num_bins",
    "motion_keys",
    "motion_lengths",
    "motion_fps",
    "motion_segment_offsets",
    "global_segment_id",
    "motion_id",
    "local_segment_id",
    "start_frame",
    "end_frame_exclusive",
    "duration_seconds",
    "difficulty_raw",
    "difficulty_score",
    "difficulty_bin",
    "feature_names",
    "feature_values",
    "feature_z",
    "feature_available_mask",
    "available_feature_count",
    "optional_feature_coverage",
    "near_constant_features",
)


def sha256_file(path: str | os.PathLike[str]) -> str:
    """Return the SHA256 digest of a file without loading it all at once."""

    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_manifest_entries(path: str | os.PathLike[str]) -> list[str]:
    """Read ordered non-comment manifest entries using canonical POSIX keys."""

    entries: list[str] = []
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            item = line.strip()
            if item and not item.startswith("#"):
                entries.append(Path(item).as_posix())
    if not entries:
        raise ValueError(f"Motion manifest is empty: {path}")
    return entries


def resolve_manifest_entries(
    path: str | os.PathLike[str], *, working_directory: str | os.PathLike[str] | None = None
) -> tuple[list[str], list[str]]:
    """Return ordered manifest keys and paths using MotionLoader semantics."""

    manifest_path = Path(path).resolve()
    cwd = Path.cwd() if working_directory is None else Path(working_directory).resolve()
    keys = canonical_manifest_entries(manifest_path)
    resolved: list[str] = []
    for key in keys:
        item = Path(key)
        if item.is_absolute():
            candidate = item
        else:
            manifest_candidate = manifest_path.parent / item
            cwd_candidate = cwd / item
            candidate = manifest_candidate if manifest_candidate.exists() else cwd_candidate
        resolved.append(str(candidate.resolve()))
    return keys, resolved


def _scalar_text(value: np.ndarray, name: str) -> str:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"Difficulty metadata field '{name}' must be scalar.")
    if array.dtype.kind not in "SU":
        raise ValueError(f"Difficulty metadata field '{name}' must be text.")
    item = array.reshape(()).item()
    result = item.decode("utf-8") if isinstance(item, bytes) else str(item)
    if not result:
        raise ValueError(f"Difficulty metadata field '{name}' must not be empty.")
    return result


def _scalar_int(value: np.ndarray, name: str) -> int:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"Difficulty metadata field '{name}' must be scalar.")
    if array.dtype.kind not in "iu":
        raise ValueError(f"Difficulty metadata field '{name}' must be an integer.")
    return int(array.reshape(()).item())


def _scalar_float(value: np.ndarray, name: str) -> float:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"Difficulty metadata field '{name}' must be scalar.")
    if array.dtype.kind not in "fiu":
        raise ValueError(f"Difficulty metadata field '{name}' must be numeric.")
    result = float(array.reshape(()).item())
    if not math.isfinite(result):
        raise ValueError(f"Difficulty metadata field '{name}' must be finite.")
    return result


def _integer_array(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in "iu":
        raise ValueError(f"Difficulty metadata field '{name}' must contain integers.")
    return array.astype(np.int64, copy=False)


def _numeric_array(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in "fiu":
        raise ValueError(f"Difficulty metadata field '{name}' must be numeric.")
    return array.astype(np.float64, copy=False)


def _boolean_array(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind != "b":
        raise ValueError(f"Difficulty metadata field '{name}' must contain booleans.")
    return array.astype(bool, copy=False)


def _text_array(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in "SU":
        raise ValueError(f"Difficulty metadata field '{name}' must contain text.")
    return array.astype(str, copy=False)


def _validate_sha256(name: str, digest: str) -> None:
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest.lower()):
        raise ValueError(f"Difficulty metadata field '{name}' is not a SHA256 hex digest.")


@dataclass(frozen=True)
class SegmentDifficultyMetadata:
    """Structurally validated segment difficulty arrays from one frozen NPZ."""

    path: str
    metadata_sha256: str
    schema_version: str
    algorithm_schema_version: str
    segment_schema_version: int
    segment_length_seconds: float
    manifest_sha256: str
    manifest_motion_count: int
    profile_sha256: str
    difficulty_config_sha256: str
    pool_fingerprint: str
    num_bins: int
    motion_keys: np.ndarray
    motion_lengths: np.ndarray
    motion_fps: np.ndarray
    motion_segment_offsets: np.ndarray
    global_segment_id: np.ndarray
    motion_id: np.ndarray
    local_segment_id: np.ndarray
    start_frame: np.ndarray
    end_frame_exclusive: np.ndarray
    duration_seconds: np.ndarray
    difficulty_raw: np.ndarray
    difficulty_score: np.ndarray
    difficulty_bin: np.ndarray
    feature_names: np.ndarray
    feature_values: np.ndarray
    feature_z: np.ndarray
    feature_available_mask: np.ndarray
    available_feature_count: np.ndarray
    optional_feature_coverage: np.ndarray
    near_constant_features: np.ndarray

    @property
    def num_motions(self) -> int:
        return int(self.motion_lengths.size)

    @property
    def num_segments(self) -> int:
        return int(self.global_segment_id.size)

    @property
    def num_features(self) -> int:
        return int(self.feature_names.size)

    @property
    def difficulty_profile_sha256(self) -> str:
        """Alias used by checkpoint code that spells out the identity type."""

        return self.profile_sha256

    @property
    def near_constant_mask(self) -> np.ndarray:
        """Return one boolean per feature in profile order."""

        return np.isin(self.feature_names, self.near_constant_features)

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> "SegmentDifficultyMetadata":
        """Load and strictly validate frozen difficulty metadata."""

        resolved_path = str(Path(path).resolve())
        if not os.path.isfile(resolved_path):
            raise FileNotFoundError(f"Difficulty metadata file does not exist: {resolved_path}")
        metadata_digest = sha256_file(resolved_path)
        try:
            archive = np.load(resolved_path, allow_pickle=False)
        except Exception as exc:
            raise ValueError(f"Unable to load difficulty metadata '{resolved_path}': {exc}") from exc
        try:
            missing = sorted(set(_REQUIRED_ARRAYS).difference(archive.files))
            if missing:
                raise ValueError(f"Difficulty metadata is missing fields: {missing}")
            values = {name: np.asarray(archive[name]) for name in _REQUIRED_ARRAYS}
        finally:
            archive.close()

        schema_version = _scalar_text(values["schema_version"], "schema_version")
        if schema_version != DIFFICULTY_METADATA_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported difficulty metadata schema '{schema_version}'; "
                f"expected '{DIFFICULTY_METADATA_SCHEMA_VERSION}'."
            )
        algorithm_schema_version = _scalar_text(
            values["algorithm_schema_version"], "algorithm_schema_version"
        )
        segment_schema_version = _scalar_int(values["segment_schema_version"], "segment_schema_version")
        if segment_schema_version < 1:
            raise ValueError("Difficulty metadata segment_schema_version must be positive.")
        segment_length_seconds = _scalar_float(values["segment_length_seconds"], "segment_length_seconds")
        if segment_length_seconds <= 0.0:
            raise ValueError("Difficulty metadata segment_length_seconds must be greater than zero.")
        manifest_sha256 = _scalar_text(values["manifest_sha256"], "manifest_sha256")
        profile_sha256 = _scalar_text(values["profile_sha256"], "profile_sha256")
        difficulty_config_sha256 = _scalar_text(
            values["difficulty_config_sha256"], "difficulty_config_sha256"
        )
        pool_fingerprint = _scalar_text(values["pool_fingerprint"], "pool_fingerprint")
        for name, digest in (
            ("manifest_sha256", manifest_sha256),
            ("profile_sha256", profile_sha256),
            ("difficulty_config_sha256", difficulty_config_sha256),
            ("pool_fingerprint", pool_fingerprint),
        ):
            _validate_sha256(name, digest)
        num_bins = _scalar_int(values["num_bins"], "num_bins")
        if num_bins < 2:
            raise ValueError("Difficulty metadata num_bins must be at least two.")

        motion_keys = _text_array(values["motion_keys"], "motion_keys")
        motion_lengths = _integer_array(values["motion_lengths"], "motion_lengths")
        motion_fps = _numeric_array(values["motion_fps"], "motion_fps")
        offsets = _integer_array(values["motion_segment_offsets"], "motion_segment_offsets")
        if motion_keys.ndim != 1 or motion_lengths.ndim != 1 or motion_fps.ndim != 1:
            raise ValueError("Difficulty metadata motion arrays must be one-dimensional.")
        num_motions = int(motion_lengths.size)
        if num_motions == 0 or motion_keys.size != num_motions or motion_fps.size != num_motions:
            raise ValueError("Difficulty metadata motion arrays are empty or have inconsistent lengths.")
        if np.any(np.char.str_len(motion_keys) == 0):
            raise ValueError("Difficulty metadata motion_keys must not contain empty strings.")
        manifest_motion_count = _scalar_int(values["manifest_motion_count"], "manifest_motion_count")
        if manifest_motion_count != num_motions:
            raise ValueError("Difficulty metadata manifest_motion_count does not match motion arrays.")
        if np.any(motion_lengths < 1) or not np.isfinite(motion_fps).all() or np.any(motion_fps <= 0.0):
            raise ValueError("Difficulty metadata contains invalid motion lengths or FPS values.")
        if offsets.shape != (num_motions + 1,) or offsets[0] != 0 or np.any(np.diff(offsets) <= 0):
            raise ValueError("Difficulty metadata motion_segment_offsets is invalid.")

        segment_arrays: dict[str, np.ndarray] = {
            "global_segment_id": _integer_array(values["global_segment_id"], "global_segment_id"),
            "motion_id": _integer_array(values["motion_id"], "motion_id"),
            "local_segment_id": _integer_array(values["local_segment_id"], "local_segment_id"),
            "start_frame": _integer_array(values["start_frame"], "start_frame"),
            "end_frame_exclusive": _integer_array(
                values["end_frame_exclusive"], "end_frame_exclusive"
            ),
            "duration_seconds": _numeric_array(values["duration_seconds"], "duration_seconds"),
            "difficulty_raw": _numeric_array(values["difficulty_raw"], "difficulty_raw"),
            "difficulty_score": _numeric_array(values["difficulty_score"], "difficulty_score"),
            "difficulty_bin": _integer_array(values["difficulty_bin"], "difficulty_bin"),
            "available_feature_count": _integer_array(
                values["available_feature_count"], "available_feature_count"
            ),
            "optional_feature_coverage": _numeric_array(
                values["optional_feature_coverage"], "optional_feature_coverage"
            ),
        }
        num_segments = int(offsets[-1])
        for name, array in segment_arrays.items():
            if array.shape != (num_segments,):
                raise ValueError(f"Difficulty metadata field '{name}' must have shape ({num_segments},).")
        if not np.array_equal(segment_arrays["global_segment_id"], np.arange(num_segments, dtype=np.int64)):
            raise ValueError("Difficulty metadata global_segment_id must be contiguous and ordered.")

        expected_motion_ids = np.repeat(np.arange(num_motions, dtype=np.int64), np.diff(offsets))
        expected_local_ids = np.arange(num_segments, dtype=np.int64) - offsets[expected_motion_ids]
        if not np.array_equal(segment_arrays["motion_id"], expected_motion_ids):
            raise ValueError("Difficulty metadata motion_id does not match motion_segment_offsets.")
        if not np.array_equal(segment_arrays["local_segment_id"], expected_local_ids):
            raise ValueError("Difficulty metadata local_segment_id does not match motion_segment_offsets.")

        # Reconstruct Stage 0's round-to-nearest-even segment layout.  This
        # catches a self-consistent but incompatible reimplementation of the
        # global segment mapping before training can consume it.
        segment_frames = np.maximum(1, np.rint(motion_fps * segment_length_seconds).astype(np.int64))
        expected_counts = (motion_lengths + segment_frames - 1) // segment_frames
        if not np.array_equal(np.diff(offsets), expected_counts):
            raise ValueError("Difficulty metadata segment offsets do not match the Stage 0 segment layout.")
        expected_starts = expected_local_ids * segment_frames[expected_motion_ids]
        expected_ends = np.minimum(
            expected_starts + segment_frames[expected_motion_ids], motion_lengths[expected_motion_ids]
        )
        if not np.array_equal(segment_arrays["start_frame"], expected_starts):
            raise ValueError("Difficulty metadata start_frame does not match the Stage 0 segment layout.")
        if not np.array_equal(segment_arrays["end_frame_exclusive"], expected_ends):
            raise ValueError("Difficulty metadata end_frame_exclusive does not match the Stage 0 segment layout.")

        durations = segment_arrays["duration_seconds"]
        expected_durations = (expected_ends - expected_starts) / motion_fps[expected_motion_ids]
        if not np.isfinite(durations).all() or np.any(durations <= 0.0) or not np.allclose(
            durations, expected_durations, rtol=1.0e-7, atol=1.0e-9
        ):
            raise ValueError("Difficulty metadata duration_seconds does not match segment bounds and FPS.")
        raw = segment_arrays["difficulty_raw"]
        scores = segment_arrays["difficulty_score"]
        bins = segment_arrays["difficulty_bin"]
        if not np.isfinite(raw).all():
            raise ValueError("Difficulty metadata difficulty_raw must contain only finite values.")
        if not np.isfinite(scores).all() or np.any(scores < 0.0) or np.any(scores > 1.0):
            raise ValueError("Difficulty metadata difficulty_score must contain finite values in [0, 1].")
        if np.any(bins < 0) or np.any(bins >= num_bins):
            raise ValueError(f"Difficulty metadata difficulty_bin must be in [0, {num_bins - 1}].")

        feature_names = _text_array(values["feature_names"], "feature_names")
        near_constant_features = _text_array(values["near_constant_features"], "near_constant_features")
        if feature_names.ndim != 1 or feature_names.size == 0:
            raise ValueError("Difficulty metadata feature_names must be a non-empty one-dimensional array.")
        if np.any(np.char.str_len(feature_names) == 0) or np.unique(feature_names).size != feature_names.size:
            raise ValueError("Difficulty metadata feature_names must be non-empty and unique.")
        if near_constant_features.ndim != 1:
            raise ValueError("Difficulty metadata near_constant_features must be one-dimensional.")
        if (
            np.any(np.char.str_len(near_constant_features) == 0)
            or np.unique(near_constant_features).size != near_constant_features.size
            or not np.isin(near_constant_features, feature_names).all()
        ):
            raise ValueError(
                "Difficulty metadata near_constant_features must be unique members of feature_names."
            )

        feature_values = _numeric_array(values["feature_values"], "feature_values")
        feature_z = _numeric_array(values["feature_z"], "feature_z")
        available_mask = _boolean_array(values["feature_available_mask"], "feature_available_mask")
        feature_shape = (num_segments, int(feature_names.size))
        if feature_values.shape != feature_shape or feature_z.shape != feature_shape:
            raise ValueError(
                "Difficulty metadata feature_values and feature_z must have shape "
                f"{feature_shape}."
            )
        if available_mask.shape != feature_shape:
            raise ValueError(f"Difficulty metadata feature_available_mask must have shape {feature_shape}.")
        if not np.isfinite(feature_values[available_mask]).all() or not np.isfinite(
            feature_z[available_mask]
        ).all():
            raise ValueError("Difficulty metadata available feature values and z-scores must be finite.")
        unavailable_mask = ~available_mask
        if not np.isnan(feature_values[unavailable_mask]).all() or not np.isnan(
            feature_z[unavailable_mask]
        ).all():
            raise ValueError("Difficulty metadata unavailable feature entries must be represented by NaN.")
        available_count = segment_arrays["available_feature_count"]
        if not np.array_equal(available_count, np.count_nonzero(available_mask, axis=1)):
            raise ValueError("Difficulty metadata available_feature_count does not match its availability mask.")
        if np.any(available_count == 0):
            raise ValueError("Every difficulty segment must have at least one available feature.")
        coverage = segment_arrays["optional_feature_coverage"]
        if not np.isfinite(coverage).all() or np.any(coverage < 0.0) or np.any(coverage > 1.0):
            raise ValueError("Difficulty metadata optional_feature_coverage must be finite and in [0, 1].")

        return cls(
            path=resolved_path,
            metadata_sha256=metadata_digest,
            schema_version=schema_version,
            algorithm_schema_version=algorithm_schema_version,
            segment_schema_version=segment_schema_version,
            segment_length_seconds=segment_length_seconds,
            manifest_sha256=manifest_sha256,
            manifest_motion_count=manifest_motion_count,
            profile_sha256=profile_sha256,
            difficulty_config_sha256=difficulty_config_sha256,
            pool_fingerprint=pool_fingerprint,
            num_bins=num_bins,
            motion_keys=motion_keys.copy(),
            motion_lengths=motion_lengths.copy(),
            motion_fps=motion_fps.copy(),
            motion_segment_offsets=offsets.copy(),
            feature_names=feature_names.copy(),
            feature_values=feature_values.copy(),
            feature_z=feature_z.copy(),
            feature_available_mask=available_mask.copy(),
            near_constant_features=near_constant_features.copy(),
            **{name: array.copy() for name, array in segment_arrays.items()},
        )

    def validate_against(
        self,
        *,
        manifest_path: str | os.PathLike[str],
        motion_keys: Sequence[str],
        motion_lengths: Sequence[int],
        motion_fps: Sequence[float],
        motion_segment_offsets: Sequence[int],
        segment_start_frames: Sequence[int],
        segment_end_frames: Sequence[int],
        segment_length_seconds: float,
        segment_schema_version: int,
        pool_fingerprint: str,
        segment_global_ids: Sequence[int] | None = None,
        segment_motion_ids: Sequence[int] | None = None,
        segment_local_ids: Sequence[int] | None = None,
        segment_duration_seconds: Sequence[float] | None = None,
        expected_profile_sha256: str | None = None,
        expected_difficulty_config_sha256: str | None = None,
        expected_num_bins: int | None = None,
        strict: bool = True,
    ) -> bool:
        """Validate that metadata describes the exact ordered motion pool and segments."""

        mismatches: list[str] = []
        if sha256_file(manifest_path) != self.manifest_sha256:
            mismatches.append("manifest SHA256")
        normalized_keys = np.asarray([Path(key).as_posix() for key in motion_keys], dtype=str)
        if not np.array_equal(normalized_keys, self.motion_keys):
            mismatches.append("manifest motion order")
        current_lengths = np.asarray(motion_lengths, dtype=np.int64)
        current_fps = np.asarray(motion_fps, dtype=np.float64)
        current_offsets = np.asarray(motion_segment_offsets, dtype=np.int64)
        current_starts = np.asarray(segment_start_frames, dtype=np.int64)
        current_ends = np.asarray(segment_end_frames, dtype=np.int64)
        if not np.array_equal(current_lengths, self.motion_lengths):
            mismatches.append("motion frame counts")
        if current_fps.shape != self.motion_fps.shape or not np.allclose(
            current_fps, self.motion_fps, rtol=0.0, atol=1.0e-12
        ):
            mismatches.append("motion FPS")
        if not np.array_equal(current_offsets, self.motion_segment_offsets):
            mismatches.append("segment offsets/global count")
        if not np.array_equal(current_starts, self.start_frame):
            mismatches.append("segment start frames")
        if not np.array_equal(current_ends, self.end_frame_exclusive):
            mismatches.append("segment end frames")
        if not math.isclose(
            float(segment_length_seconds), self.segment_length_seconds, rel_tol=0.0, abs_tol=1.0e-12
        ):
            mismatches.append("segment length")
        if int(segment_schema_version) != self.segment_schema_version:
            mismatches.append("segment schema version")
        if pool_fingerprint != self.pool_fingerprint:
            mismatches.append("ordered motion pool fingerprint")

        optional_arrays: tuple[tuple[str, Sequence[int] | Sequence[float] | None, np.ndarray], ...] = (
            ("global segment IDs", segment_global_ids, self.global_segment_id),
            ("segment motion IDs", segment_motion_ids, self.motion_id),
            ("local segment IDs", segment_local_ids, self.local_segment_id),
            ("segment durations", segment_duration_seconds, self.duration_seconds),
        )
        for label, supplied, expected in optional_arrays:
            if supplied is None:
                continue
            current = np.asarray(supplied, dtype=expected.dtype)
            if np.issubdtype(expected.dtype, np.floating):
                matches = current.shape == expected.shape and np.allclose(
                    current, expected, rtol=1.0e-7, atol=1.0e-9
                )
            else:
                matches = np.array_equal(current, expected)
            if not matches:
                mismatches.append(label)

        if expected_profile_sha256 is not None and expected_profile_sha256 != self.profile_sha256:
            mismatches.append("difficulty profile SHA256")
        if (
            expected_difficulty_config_sha256 is not None
            and expected_difficulty_config_sha256 != self.difficulty_config_sha256
        ):
            mismatches.append("difficulty config SHA256")
        if expected_num_bins is not None and int(expected_num_bins) != self.num_bins:
            mismatches.append("difficulty bin count")
        if mismatches and strict:
            raise ValueError(
                "Difficulty metadata does not match training data: " + ", ".join(mismatches) + "."
            )
        return not mismatches

    def difficulty_metrics(self) -> dict[str, float | int]:
        """Return low-cardinality static metrics suitable for W&B logging."""

        metrics: dict[str, float | int] = {
            "num_segments": self.num_segments,
            "num_motions": self.num_motions,
            "num_bins": self.num_bins,
            "score_mean": float(np.mean(self.difficulty_score)),
            "score_std": float(np.std(self.difficulty_score)),
            "score_p10": float(np.quantile(self.difficulty_score, 0.10)),
            "score_p50": float(np.quantile(self.difficulty_score, 0.50)),
            "score_p90": float(np.quantile(self.difficulty_score, 0.90)),
            "raw_mean": float(np.mean(self.difficulty_raw)),
            "raw_std": float(np.std(self.difficulty_raw)),
            "optional_feature_coverage_mean": float(np.mean(self.optional_feature_coverage)),
            "available_feature_count_mean": float(np.mean(self.available_feature_count)),
            "near_constant_feature_count": int(self.near_constant_features.size),
            "metadata_match_ok": 1,
        }
        for bin_id in range(self.num_bins):
            count = int(np.count_nonzero(self.difficulty_bin == bin_id))
            metrics[f"bin_{bin_id}_count"] = count
            metrics[f"bin_{bin_id}_ratio"] = count / self.num_segments
        return metrics

    def identity_state(self) -> dict[str, object]:
        """Return immutable identities needed for checkpoint compatibility."""

        return {
            "schema_version": self.schema_version,
            "algorithm_schema_version": self.algorithm_schema_version,
            "segment_schema_version": self.segment_schema_version,
            "segment_length_seconds": self.segment_length_seconds,
            "metadata_path": self.path,
            "metadata_sha256": self.metadata_sha256,
            "profile_sha256": self.profile_sha256,
            "difficulty_config_sha256": self.difficulty_config_sha256,
            "manifest_sha256": self.manifest_sha256,
            "pool_fingerprint": self.pool_fingerprint,
            "manifest_motion_count": self.manifest_motion_count,
            "num_bins": self.num_bins,
        }


def metadata_npz_payload(
    *,
    algorithm_schema_version: str,
    segment_schema_version: int,
    segment_length_seconds: float,
    manifest_sha256: str,
    profile_sha256: str,
    difficulty_config_sha256: str,
    pool_fingerprint: str,
    num_bins: int,
    motion_keys: Sequence[str],
    motion_lengths: Sequence[int],
    motion_fps: Sequence[float],
    motion_segment_offsets: Sequence[int],
    global_segment_id: Sequence[int],
    motion_id: Sequence[int],
    local_segment_id: Sequence[int],
    start_frame: Sequence[int],
    end_frame_exclusive: Sequence[int],
    duration_seconds: Sequence[float],
    difficulty_raw: Sequence[float],
    difficulty_score: Sequence[float],
    difficulty_bin: Sequence[int],
    feature_names: Sequence[str],
    feature_values: Sequence[Sequence[float]] | np.ndarray,
    feature_z: Sequence[Sequence[float]] | np.ndarray,
    feature_available_mask: Sequence[Sequence[bool]] | np.ndarray,
    optional_feature_coverage: Sequence[float],
    near_constant_features: Sequence[str],
    available_feature_count: Sequence[int] | None = None,
) -> Mapping[str, np.ndarray]:
    """Build the compact, trajectory-free payload written by the offline CLI."""

    names = np.asarray(feature_names, dtype=str)
    values = np.asarray(feature_values, dtype=np.float32)
    z_values = np.asarray(feature_z, dtype=np.float32)
    availability = np.asarray(feature_available_mask, dtype=bool)
    if names.ndim != 1 or names.size == 0:
        raise ValueError("feature_names must be a non-empty one-dimensional sequence.")
    if values.ndim != 2 or values.shape[1] != names.size:
        raise ValueError("feature_values must have shape (num_segments, num_features).")
    if z_values.shape != values.shape or availability.shape != values.shape:
        raise ValueError("feature_z and feature_available_mask must match feature_values shape.")
    derived_available_count = np.count_nonzero(availability, axis=1).astype(np.int64)
    if available_feature_count is None:
        stored_available_count = derived_available_count
    else:
        stored_available_count = np.asarray(available_feature_count, dtype=np.int64)
        if not np.array_equal(stored_available_count, derived_available_count):
            raise ValueError("available_feature_count does not match feature_available_mask.")

    return {
        "schema_version": np.asarray(DIFFICULTY_METADATA_SCHEMA_VERSION),
        "algorithm_schema_version": np.asarray(str(algorithm_schema_version)),
        "segment_schema_version": np.asarray(segment_schema_version, dtype=np.int64),
        "segment_length_seconds": np.asarray(segment_length_seconds, dtype=np.float64),
        "manifest_sha256": np.asarray(manifest_sha256),
        "manifest_motion_count": np.asarray(len(motion_keys), dtype=np.int64),
        "profile_sha256": np.asarray(profile_sha256),
        "difficulty_config_sha256": np.asarray(difficulty_config_sha256),
        "pool_fingerprint": np.asarray(pool_fingerprint),
        "num_bins": np.asarray(num_bins, dtype=np.int64),
        "motion_keys": np.asarray(motion_keys, dtype=str),
        "motion_lengths": np.asarray(motion_lengths, dtype=np.int64),
        "motion_fps": np.asarray(motion_fps, dtype=np.float64),
        "motion_segment_offsets": np.asarray(motion_segment_offsets, dtype=np.int64),
        "global_segment_id": np.asarray(global_segment_id, dtype=np.int64),
        "motion_id": np.asarray(motion_id, dtype=np.int64),
        "local_segment_id": np.asarray(local_segment_id, dtype=np.int64),
        "start_frame": np.asarray(start_frame, dtype=np.int64),
        "end_frame_exclusive": np.asarray(end_frame_exclusive, dtype=np.int64),
        "duration_seconds": np.asarray(duration_seconds, dtype=np.float64),
        "difficulty_raw": np.asarray(difficulty_raw, dtype=np.float32),
        "difficulty_score": np.asarray(difficulty_score, dtype=np.float32),
        "difficulty_bin": np.asarray(difficulty_bin, dtype=np.int16),
        "feature_names": names,
        "feature_values": values,
        "feature_z": z_values,
        "feature_available_mask": availability,
        "available_feature_count": stored_available_count,
        "optional_feature_coverage": np.asarray(optional_feature_coverage, dtype=np.float32),
        "near_constant_features": np.asarray(near_constant_features, dtype=str),
    }
