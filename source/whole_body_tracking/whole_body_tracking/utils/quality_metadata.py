"""Static segment-quality metadata shared by offline audit and training.

This module intentionally depends only on NumPy and the Python standard
library.  Training loads a frozen ``.npz`` produced by the offline audit; it
never recomputes quality from policy-dependent signals.
"""

from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Mapping, Sequence

import numpy as np


QUALITY_METADATA_SCHEMA_VERSION: Final[str] = "wbt.segment_quality.v1"
QUALITY_STATUS_TO_CODE: Final[dict[str, int]] = {"pass": 0, "borderline": 1, "reject": 2}
QUALITY_CODE_TO_STATUS: Final[dict[int, str]] = {
    code: status for status, code in QUALITY_STATUS_TO_CODE.items()
}

_REQUIRED_ARRAYS: Final[tuple[str, ...]] = (
    "schema_version",
    "segment_schema_version",
    "segment_length_seconds",
    "manifest_sha256",
    "manifest_motion_count",
    "quality_config_sha256",
    "pool_fingerprint",
    "motion_keys",
    "motion_lengths",
    "motion_fps",
    "motion_segment_offsets",
    "global_segment_id",
    "motion_id",
    "local_segment_id",
    "start_frame",
    "end_frame_exclusive",
    "quality_score",
    "quality_status",
    "pass_mask",
    "borderline_mask",
    "reject_mask",
)


def sha256_file(path: str | os.PathLike[str]) -> str:
    """Return the SHA256 digest of a file without loading it all at once."""

    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_manifest_entries(path: str | os.PathLike[str]) -> list[str]:
    """Read the ordered, non-comment manifest entries used as motion keys."""

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
    """Return ordered manifest keys and resolved paths using MotionLoader semantics."""

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
        raise ValueError(f"Quality metadata field '{name}' must be scalar.")
    return str(array.reshape(()).item())


def _scalar_int(value: np.ndarray, name: str) -> int:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"Quality metadata field '{name}' must be scalar.")
    return int(array.reshape(()).item())


def _scalar_float(value: np.ndarray, name: str) -> float:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"Quality metadata field '{name}' must be scalar.")
    result = float(array.reshape(()).item())
    if not math.isfinite(result):
        raise ValueError(f"Quality metadata field '{name}' must be finite.")
    return result


@dataclass(frozen=True)
class SegmentQualityMetadata:
    """Validated, immutable arrays loaded from one quality metadata NPZ."""

    path: str
    metadata_sha256: str
    schema_version: str
    segment_schema_version: int
    segment_length_seconds: float
    manifest_sha256: str
    manifest_motion_count: int
    quality_config_sha256: str
    pool_fingerprint: str
    motion_keys: np.ndarray
    motion_lengths: np.ndarray
    motion_fps: np.ndarray
    motion_segment_offsets: np.ndarray
    global_segment_id: np.ndarray
    motion_id: np.ndarray
    local_segment_id: np.ndarray
    start_frame: np.ndarray
    end_frame_exclusive: np.ndarray
    quality_score: np.ndarray
    quality_status: np.ndarray
    pass_mask: np.ndarray
    borderline_mask: np.ndarray
    reject_mask: np.ndarray

    @property
    def num_motions(self) -> int:
        return int(self.motion_lengths.size)

    @property
    def num_segments(self) -> int:
        return int(self.global_segment_id.size)

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> "SegmentQualityMetadata":
        """Load and structurally validate a frozen metadata file."""

        resolved_path = str(Path(path).resolve())
        if not os.path.isfile(resolved_path):
            raise FileNotFoundError(f"Quality metadata file does not exist: {resolved_path}")
        metadata_digest = sha256_file(resolved_path)
        try:
            archive = np.load(resolved_path, allow_pickle=False)
        except Exception as exc:
            raise ValueError(f"Unable to load quality metadata '{resolved_path}': {exc}") from exc
        try:
            missing = sorted(set(_REQUIRED_ARRAYS).difference(archive.files))
            if missing:
                raise ValueError(f"Quality metadata is missing fields: {missing}")
            values = {name: np.asarray(archive[name]) for name in _REQUIRED_ARRAYS}
        finally:
            archive.close()

        schema_version = _scalar_text(values["schema_version"], "schema_version")
        if schema_version != QUALITY_METADATA_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported quality metadata schema '{schema_version}'; "
                f"expected '{QUALITY_METADATA_SCHEMA_VERSION}'."
            )
        segment_schema_version = _scalar_int(values["segment_schema_version"], "segment_schema_version")
        segment_length_seconds = _scalar_float(values["segment_length_seconds"], "segment_length_seconds")
        if segment_length_seconds <= 0.0:
            raise ValueError("Quality metadata segment_length_seconds must be greater than zero.")
        manifest_sha256 = _scalar_text(values["manifest_sha256"], "manifest_sha256")
        quality_config_sha256 = _scalar_text(values["quality_config_sha256"], "quality_config_sha256")
        pool_fingerprint = _scalar_text(values["pool_fingerprint"], "pool_fingerprint")
        for name, digest in (
            ("manifest_sha256", manifest_sha256),
            ("quality_config_sha256", quality_config_sha256),
            ("pool_fingerprint", pool_fingerprint),
        ):
            if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest.lower()):
                raise ValueError(f"Quality metadata field '{name}' is not a SHA256 hex digest.")

        motion_keys = values["motion_keys"].astype(str, copy=False)
        motion_lengths = values["motion_lengths"].astype(np.int64, copy=False)
        motion_fps = values["motion_fps"].astype(np.float64, copy=False)
        offsets = values["motion_segment_offsets"].astype(np.int64, copy=False)
        if motion_keys.ndim != 1 or motion_lengths.ndim != 1 or motion_fps.ndim != 1:
            raise ValueError("Quality metadata motion arrays must be one-dimensional.")
        num_motions = int(motion_lengths.size)
        if num_motions == 0 or motion_keys.size != num_motions or motion_fps.size != num_motions:
            raise ValueError("Quality metadata motion arrays are empty or have inconsistent lengths.")
        manifest_motion_count = _scalar_int(values["manifest_motion_count"], "manifest_motion_count")
        if manifest_motion_count != num_motions:
            raise ValueError("Quality metadata manifest_motion_count does not match motion arrays.")
        if np.any(motion_lengths < 1) or not np.isfinite(motion_fps).all() or np.any(motion_fps <= 0.0):
            raise ValueError("Quality metadata contains invalid motion lengths or FPS values.")
        if offsets.shape != (num_motions + 1,) or offsets[0] != 0 or np.any(np.diff(offsets) <= 0):
            raise ValueError("Quality metadata motion_segment_offsets is invalid.")

        segment_arrays: dict[str, np.ndarray] = {
            "global_segment_id": values["global_segment_id"].astype(np.int64, copy=False),
            "motion_id": values["motion_id"].astype(np.int64, copy=False),
            "local_segment_id": values["local_segment_id"].astype(np.int64, copy=False),
            "start_frame": values["start_frame"].astype(np.int64, copy=False),
            "end_frame_exclusive": values["end_frame_exclusive"].astype(np.int64, copy=False),
            "quality_score": values["quality_score"].astype(np.float64, copy=False),
            "quality_status": values["quality_status"].astype(np.int8, copy=False),
            "pass_mask": values["pass_mask"].astype(bool, copy=False),
            "borderline_mask": values["borderline_mask"].astype(bool, copy=False),
            "reject_mask": values["reject_mask"].astype(bool, copy=False),
        }
        num_segments = int(offsets[-1])
        for name, array in segment_arrays.items():
            if array.shape != (num_segments,):
                raise ValueError(f"Quality metadata field '{name}' must have shape ({num_segments},).")
        if not np.array_equal(segment_arrays["global_segment_id"], np.arange(num_segments, dtype=np.int64)):
            raise ValueError("Quality metadata global_segment_id must be contiguous and ordered.")
        expected_motion_ids = np.repeat(np.arange(num_motions, dtype=np.int64), np.diff(offsets))
        expected_local_ids = np.arange(num_segments, dtype=np.int64) - offsets[expected_motion_ids]
        if not np.array_equal(segment_arrays["motion_id"], expected_motion_ids):
            raise ValueError("Quality metadata motion_id does not match motion_segment_offsets.")
        if not np.array_equal(segment_arrays["local_segment_id"], expected_local_ids):
            raise ValueError("Quality metadata local_segment_id does not match motion_segment_offsets.")
        if np.any(segment_arrays["start_frame"] < 0) or np.any(
            segment_arrays["start_frame"] >= segment_arrays["end_frame_exclusive"]
        ):
            raise ValueError("Quality metadata contains invalid segment frame bounds.")
        if np.any(segment_arrays["end_frame_exclusive"] > motion_lengths[expected_motion_ids]):
            raise ValueError("Quality metadata segment end exceeds its motion frame count.")

        scores = segment_arrays["quality_score"]
        statuses = segment_arrays["quality_status"]
        if not np.isfinite(scores).all() or np.any(scores < 0.0) or np.any(scores > 1.0):
            raise ValueError("Quality metadata scores must be finite values in [0, 1].")
        if not np.isin(statuses, list(QUALITY_CODE_TO_STATUS)).all():
            raise ValueError("Quality metadata contains an unknown quality_status code.")
        pass_mask = segment_arrays["pass_mask"]
        borderline_mask = segment_arrays["borderline_mask"]
        reject_mask = segment_arrays["reject_mask"]
        mask_sum = pass_mask.astype(np.int8) + borderline_mask.astype(np.int8) + reject_mask.astype(np.int8)
        if not np.all(mask_sum == 1):
            raise ValueError("Quality status masks must be mutually exclusive and exhaustive.")
        if not (
            np.array_equal(pass_mask, statuses == QUALITY_STATUS_TO_CODE["pass"])
            and np.array_equal(borderline_mask, statuses == QUALITY_STATUS_TO_CODE["borderline"])
            and np.array_equal(reject_mask, statuses == QUALITY_STATUS_TO_CODE["reject"])
        ):
            raise ValueError("Quality status masks do not match quality_status codes.")

        return cls(
            path=resolved_path,
            metadata_sha256=metadata_digest,
            schema_version=schema_version,
            segment_schema_version=segment_schema_version,
            segment_length_seconds=segment_length_seconds,
            manifest_sha256=manifest_sha256,
            manifest_motion_count=manifest_motion_count,
            quality_config_sha256=quality_config_sha256,
            pool_fingerprint=pool_fingerprint,
            motion_keys=motion_keys.copy(),
            motion_lengths=motion_lengths.copy(),
            motion_fps=motion_fps.copy(),
            motion_segment_offsets=offsets.copy(),
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
        strict: bool = True,
    ) -> bool:
        """Validate that metadata describes the exact ordered training pool."""

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
        if mismatches and strict:
            raise ValueError("Quality metadata does not match training data: " + ", ".join(mismatches) + ".")
        return not mismatches

    def accepted_mask(self, *, include_borderline: bool = True) -> np.ndarray:
        """Return the M1 mask: pass plus optionally borderline, never reject."""

        return self.pass_mask | (self.borderline_mask if include_borderline else False)

    def quality_metrics(self) -> dict[str, float | int]:
        """Return low-cardinality static metrics suitable for W&B."""

        count = self.num_segments
        num_pass = int(np.count_nonzero(self.pass_mask))
        num_borderline = int(np.count_nonzero(self.borderline_mask))
        num_reject = int(np.count_nonzero(self.reject_mask))
        return {
            "num_pass_segments": num_pass,
            "num_borderline_segments": num_borderline,
            "num_reject_segments": num_reject,
            "pass_ratio": num_pass / count,
            "borderline_ratio": num_borderline / count,
            "reject_ratio": num_reject / count,
            "mean_quality_score": float(np.mean(self.quality_score)),
            "min_quality_score": float(np.min(self.quality_score)),
            "metadata_match_ok": 1,
        }

    def identity_state(self) -> dict[str, object]:
        """Return static identity fields for checkpoint compatibility checks."""

        return {
            "schema_version": self.schema_version,
            "segment_schema_version": self.segment_schema_version,
            "metadata_path": self.path,
            "metadata_sha256": self.metadata_sha256,
            "quality_config_sha256": self.quality_config_sha256,
            "manifest_sha256": self.manifest_sha256,
            "pool_fingerprint": self.pool_fingerprint,
        }


def metadata_npz_payload(
    *,
    segment_schema_version: int,
    segment_length_seconds: float,
    manifest_sha256: str,
    quality_config_sha256: str,
    pool_fingerprint: str,
    motion_keys: Sequence[str],
    motion_lengths: Sequence[int],
    motion_fps: Sequence[float],
    motion_segment_offsets: Sequence[int],
    global_segment_id: Sequence[int],
    motion_id: Sequence[int],
    local_segment_id: Sequence[int],
    start_frame: Sequence[int],
    end_frame_exclusive: Sequence[int],
    quality_score: Sequence[float],
    quality_status: Sequence[int],
) -> Mapping[str, np.ndarray]:
    """Build the compact, trajectory-free payload written by the audit CLI."""

    statuses = np.asarray(quality_status, dtype=np.int8)
    return {
        "schema_version": np.asarray(QUALITY_METADATA_SCHEMA_VERSION),
        "segment_schema_version": np.asarray(segment_schema_version, dtype=np.int64),
        "segment_length_seconds": np.asarray(segment_length_seconds, dtype=np.float64),
        "manifest_sha256": np.asarray(manifest_sha256),
        "manifest_motion_count": np.asarray(len(motion_keys), dtype=np.int64),
        "quality_config_sha256": np.asarray(quality_config_sha256),
        "pool_fingerprint": np.asarray(pool_fingerprint),
        "motion_keys": np.asarray(motion_keys, dtype=str),
        "motion_lengths": np.asarray(motion_lengths, dtype=np.int64),
        "motion_fps": np.asarray(motion_fps, dtype=np.float64),
        "motion_segment_offsets": np.asarray(motion_segment_offsets, dtype=np.int64),
        "global_segment_id": np.asarray(global_segment_id, dtype=np.int64),
        "motion_id": np.asarray(motion_id, dtype=np.int64),
        "local_segment_id": np.asarray(local_segment_id, dtype=np.int64),
        "start_frame": np.asarray(start_frame, dtype=np.int64),
        "end_frame_exclusive": np.asarray(end_frame_exclusive, dtype=np.int64),
        "quality_score": np.asarray(quality_score, dtype=np.float32),
        "quality_status": statuses,
        "pass_mask": statuses == QUALITY_STATUS_TO_CODE["pass"],
        "borderline_mask": statuses == QUALITY_STATUS_TO_CODE["borderline"],
        "reject_mask": statuses == QUALITY_STATUS_TO_CODE["reject"],
    }
