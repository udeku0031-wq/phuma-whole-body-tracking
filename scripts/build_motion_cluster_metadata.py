#!/usr/bin/env python3
"""Build frozen motion-cluster metadata from module-two segment features.

The clustering input is intentionally limited to the policy-independent raw
feature matrix already stored by module two.  This script never reloads a
trajectory to recompute kinematics, and source/category, quality, difficulty
score/bin, and policy statistics are diagnostics only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shlex
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
UTILS_DIR = (
    PROJECT_ROOT
    / "source"
    / "whole_body_tracking"
    / "whole_body_tracking"
    / "utils"
)
# Avoid importing the package root: it eagerly imports simulator tasks while
# this builder is deliberately a deterministic CPU/NumPy pipeline.
sys.path.insert(0, str(UTILS_DIR))

import cluster_metadata  # noqa: E402
import difficulty  # noqa: E402
import difficulty_metadata  # noqa: E402
import motion_clustering  # noqa: E402
import quality_metadata  # noqa: E402


OUTPUT_FILENAMES = (
    "motion_cluster_metadata.csv",
    "motion_cluster_metadata.npz",
    "cluster_profile.json",
    "cluster_summary.json",
    "cluster_feature_statistics.csv",
    "cluster_centroids.csv",
    "cluster_review_motions.csv",
    "cluster_config_resolved.json",
    "normalized_manifest.txt",
)

_NOT_USED_FOR_FITTING = {
    "source_category_was_not_used_for_fitting": True,
    "quality_was_not_used_for_fitting": True,
    "difficulty_score_bin_was_not_used_for_fitting": True,
    "policy_statistics_were_not_used_for_fitting": True,
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate frozen module-two trajectory features by motion, then "
            "fit or apply a deterministic Train-frozen clustering profile."
        )
    )
    parser.add_argument(
        "--manifest", type=Path, required=True, help="Ordered local motion manifest."
    )
    parser.add_argument(
        "--difficulty-metadata",
        type=Path,
        required=True,
        help="Frozen module-two segment_difficulty_metadata.npz.",
    )
    parser.add_argument(
        "--difficulty-profile",
        type=Path,
        required=True,
        help="Frozen module-two difficulty_profile.json.",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Metadata output directory."
    )
    parser.add_argument(
        "--cluster-config",
        type=Path,
        default=motion_clustering.DEFAULT_MOTION_CLUSTERING_CONFIG_PATH,
        help="JSON-compatible YAML clustering definition.",
    )
    parser.add_argument(
        "--mode", choices=("fit_transform", "transform"), default="fit_transform"
    )
    parser.add_argument(
        "--profile",
        type=Path,
        default=None,
        help="Train-fitted cluster_profile.json required by transform mode.",
    )
    parser.add_argument(
        "--quality-metadata",
        type=Path,
        default=None,
        help="Optional module-one NPZ used only for mapping/quality diagnostics.",
    )
    parser.add_argument(
        "--max-motions",
        type=int,
        default=None,
        help="Use a stable prefix of the supplied manifest and metadata.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic review and sampled-silhouette seed (fit seed is in config).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Reject optional diagnostic metadata provenance mismatches.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of known output files.",
    )
    parser.add_argument(
        "--diagnose-k",
        nargs="+",
        default=(),
        metavar="K",
        help=(
            "Optional Train-only K diagnostics, e.g. '--diagnose-k 6 8 10 12' "
            "or '--diagnose-k 6,8,10,12'. Never auto-selects K."
        ),
    )
    return parser.parse_args(argv)


def _parse_diagnose_k(raw_values: Sequence[str]) -> tuple[int, ...]:
    values: list[int] = []
    for raw in raw_values:
        for token in str(raw).split(","):
            token = token.strip()
            if not token:
                continue
            try:
                value = int(token)
            except ValueError as exc:
                raise ValueError(f"--diagnose-k contains a non-integer value: {token!r}") from exc
            if value < 2:
                raise ValueError("--diagnose-k values must be at least two.")
            if value not in values:
                values.append(value)
    return tuple(values)


def _validate_args(args: argparse.Namespace) -> tuple[int, ...]:
    if args.max_motions is not None and args.max_motions < 1:
        raise ValueError("--max-motions must be at least one.")
    if args.seed < 0 or args.seed > np.iinfo(np.uint32).max:
        raise ValueError("--seed must be in [0, 2**32 - 1].")
    if args.mode == "transform" and args.profile is None:
        raise ValueError("--profile is required in transform mode.")
    if args.mode == "fit_transform" and args.profile is not None:
        raise ValueError("--profile must not be supplied in fit_transform mode.")
    diagnose_k = _parse_diagnose_k(args.diagnose_k)
    if args.mode == "transform" and diagnose_k:
        raise ValueError("--diagnose-k is Train-fit-only and cannot be used in transform mode.")
    return diagnose_k


def _prepare_output_dir(path: Path, overwrite: bool) -> None:
    path.mkdir(parents=True, exist_ok=True)
    collisions = [name for name in OUTPUT_FILENAMES if (path / name).exists()]
    if collisions and not overwrite:
        raise FileExistsError(
            f"Output files already exist in {path}: {collisions}. "
            "Pass --overwrite to replace them."
        )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_csv(
    path: Path, rows: Iterable[Mapping[str, Any]], columns: Sequence[str]
) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=columns, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _fit_manifest_guard(path: Path) -> None:
    tokens = {
        token
        for token in path.stem.lower().replace("-", "_").split("_")
        if token
    }
    forbidden = {"validation", "valid", "val", "test"}
    if tokens.intersection(forbidden):
        raise ValueError(
            "fit_transform is Train-only; refusing a manifest whose name "
            "indicates Validation/Test."
        )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _motion_pool_fingerprint(
    motion_files: Sequence[str],
    motion_lengths: Sequence[int],
    motion_fps: Sequence[float],
) -> str:
    """Match Stage-0's relocation-stable ordered motion-pool identity."""

    if not (
        len(motion_files) == len(motion_lengths) == len(motion_fps)
    ) or not motion_files:
        raise ValueError("Motion pool identity inputs are empty or inconsistent.")
    normalized = [
        os.path.abspath(os.path.normpath(path)) for path in motion_files
    ]
    common_root = os.path.commonpath(
        [os.path.dirname(path) for path in normalized]
    )
    digest = hashlib.sha256()
    for path, length, fps in zip(
        normalized, motion_lengths, motion_fps, strict=True
    ):
        digest.update(os.path.relpath(path, common_root).encode("utf-8"))
        digest.update(b"\0")
        file_size = os.path.getsize(path) if os.path.isfile(path) else -1
        digest.update(str(file_size).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(int(length)).encode("ascii"))
        digest.update(b"\0")
        digest.update(format(float(fps), ".17g").encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _validate_upstream_identities(
    *,
    manifest: Path,
    manifest_keys: Sequence[str],
    metadata: difficulty_metadata.SegmentDifficultyMetadata,
    difficulty_profile_path: Path,
    mode: str,
) -> tuple[difficulty.DifficultyProfile, str]:
    """Strictly bind the manifest, metadata, and frozen difficulty Profile."""

    metadata.validate_against(
        manifest_path=manifest,
        motion_keys=manifest_keys,
        motion_lengths=metadata.motion_lengths,
        motion_fps=metadata.motion_fps,
        motion_segment_offsets=metadata.motion_segment_offsets,
        segment_start_frames=metadata.start_frame,
        segment_end_frames=metadata.end_frame_exclusive,
        segment_length_seconds=metadata.segment_length_seconds,
        segment_schema_version=metadata.segment_schema_version,
        pool_fingerprint=metadata.pool_fingerprint,
        segment_global_ids=metadata.global_segment_id,
        segment_motion_ids=metadata.motion_id,
        segment_local_ids=metadata.local_segment_id,
        segment_duration_seconds=metadata.duration_seconds,
        expected_profile_sha256=metadata.profile_sha256,
        expected_difficulty_config_sha256=metadata.difficulty_config_sha256,
        expected_num_bins=metadata.num_bins,
        strict=True,
    )
    profile_sha256 = difficulty_metadata.sha256_file(difficulty_profile_path)
    if profile_sha256 != metadata.profile_sha256:
        raise ValueError(
            "Difficulty Profile file SHA256 does not match difficulty metadata."
        )
    profile = difficulty.load_difficulty_profile(difficulty_profile_path)
    mismatches: list[str] = []
    if profile.algorithm_schema_version != metadata.algorithm_schema_version:
        mismatches.append("algorithm schema")
    if profile.segment_schema_version != metadata.segment_schema_version:
        mismatches.append("segment schema")
    if not math.isclose(
        profile.segment_length_seconds,
        metadata.segment_length_seconds,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        mismatches.append("segment length")
    if profile.config_sha256 != metadata.difficulty_config_sha256:
        mismatches.append("difficulty config SHA256")
    if tuple(profile.feature_names) != tuple(str(item) for item in metadata.feature_names):
        mismatches.append("feature names/order")
    # A fit must use difficulty metadata built from its own Train-fitted
    # Profile.  A transform metadata set legitimately has target identities
    # while its frozen difficulty Profile continues to identify Train.
    if mode == "fit_transform":
        if profile.training_manifest_sha256 != metadata.manifest_sha256:
            mismatches.append("Profile training manifest SHA256")
        if profile.training_pool_fingerprint != metadata.pool_fingerprint:
            mismatches.append("Profile training pool fingerprint")
    if mismatches:
        raise ValueError(
            "Difficulty Profile identity does not match difficulty metadata: "
            + ", ".join(mismatches)
            + "."
        )
    return profile, profile_sha256


def _prefix_metadata_view(
    metadata: difficulty_metadata.SegmentDifficultyMetadata, motion_count: int
) -> SimpleNamespace:
    segment_count = int(metadata.motion_segment_offsets[motion_count])
    return SimpleNamespace(
        feature_names=metadata.feature_names.copy(),
        feature_values=metadata.feature_values[:segment_count].copy(),
        feature_available_mask=metadata.feature_available_mask[:segment_count].copy(),
        duration_seconds=metadata.duration_seconds[:segment_count].copy(),
        motion_segment_offsets=metadata.motion_segment_offsets[: motion_count + 1].copy(),
        motion_id=metadata.motion_id[:segment_count].copy(),
        motion_keys=metadata.motion_keys[:motion_count].copy(),
    )


def _describe(values: Sequence[float] | np.ndarray) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return {
            "count": 0,
            "min": None,
            "mean": None,
            "std": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    return {
        "count": int(finite.size),
        "min": float(np.min(finite)),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "p50": float(np.percentile(finite, 50.0)),
        "p90": float(np.percentile(finite, 90.0)),
        "p95": float(np.percentile(finite, 95.0)),
        "p99": float(np.percentile(finite, 99.0)),
        "max": float(np.max(finite)),
    }


def _finite_mean(values: np.ndarray) -> float | None:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return None if finite.size == 0 else float(np.mean(finite))


def _gini(values: Sequence[int] | np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    total = float(np.sum(array))
    if array.size == 0 or total <= 0.0:
        return 0.0
    differences = np.abs(array[:, None] - array[None, :])
    return float(np.sum(differences) / (2.0 * array.size * total))


def _silhouette_score(
    values: np.ndarray,
    labels: np.ndarray,
    *,
    seed: int,
    max_samples: int = 2000,
) -> tuple[float | None, int]:
    """Return a deterministic exact-or-sampled Euclidean silhouette score."""

    matrix = np.asarray(values, dtype=np.float64)
    cluster_id = np.asarray(labels, dtype=np.int64)
    represented = np.unique(cluster_id)
    if matrix.shape[0] < 2 or represented.size < 2:
        return None, 0
    if matrix.shape[0] <= max_samples:
        sample = np.arange(matrix.shape[0], dtype=np.int64)
    else:
        rng = np.random.default_rng(seed)
        sample = np.sort(
            rng.choice(matrix.shape[0], size=max_samples, replace=False)
        ).astype(np.int64, copy=False)
    cluster_members = {
        int(cluster): np.flatnonzero(cluster_id == cluster)
        for cluster in represented
    }
    scores = np.zeros(sample.size, dtype=np.float64)
    matrix_squared_norm = np.sum(np.square(matrix), axis=1)
    for chunk_start in range(0, sample.size, 128):
        chunk_indexes = sample[chunk_start : chunk_start + 128]
        chunk = matrix[chunk_indexes]
        squared_distances = (
            np.sum(np.square(chunk), axis=1)[:, None]
            + matrix_squared_norm[None, :]
            - 2.0 * (chunk @ matrix.T)
        )
        # Roundoff can make an exact self-distance a tiny negative value.
        distances = np.sqrt(np.maximum(squared_distances, 0.0))
        for row_offset, global_index in enumerate(chunk_indexes):
            own = int(cluster_id[global_index])
            own_members = cluster_members[own]
            if own_members.size <= 1:
                scores[chunk_start + row_offset] = 0.0
                continue
            within = float(
                np.sum(distances[row_offset, own_members])
                / (own_members.size - 1)
            )
            nearest_other = min(
                float(np.mean(distances[row_offset, members]))
                for cluster, members in cluster_members.items()
                if cluster != own
            )
            denominator = max(within, nearest_other)
            scores[chunk_start + row_offset] = (
                (nearest_other - within) / denominator
                if denominator > 0.0
                else 0.0
            )
    return float(np.mean(scores)), int(sample.size)


def _inferred_source_category(key: str) -> str:
    parts = Path(key).as_posix().split("/")
    if "g1_all" in parts:
        index = parts.index("g1_all")
        if index + 1 < len(parts):
            return parts[index + 1]
    if len(parts) > 1:
        return parts[-2]
    return "unknown"


def _motion_difficulty_diagnostics(
    metadata: difficulty_metadata.SegmentDifficultyMetadata,
    motion_count: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for motion_id in range(motion_count):
        first = int(metadata.motion_segment_offsets[motion_id])
        last = int(metadata.motion_segment_offsets[motion_id + 1])
        scores = metadata.difficulty_score[first:last]
        durations = metadata.duration_seconds[first:last]
        bins = metadata.difficulty_bin[first:last]
        rows.append(
            {
                "score_mean": float(np.average(scores, weights=durations)),
                "score_p90": float(np.percentile(scores, 90.0)),
                "bin_counts": np.bincount(
                    bins, minlength=metadata.num_bins
                ).astype(np.int64),
            }
        )
    return rows


def _load_quality_diagnostics(
    path: Path | None,
    *,
    manifest: Path,
    manifest_keys: Sequence[str],
    metadata: difficulty_metadata.SegmentDifficultyMetadata,
    motion_count: int,
    strict: bool,
) -> tuple[list[dict[str, Any]] | None, dict[str, Any] | None, list[str]]:
    if path is None:
        return None, None, []
    qmeta = quality_metadata.SegmentQualityMetadata.load(path)
    layout_mismatches: list[str] = []
    comparisons = (
        ("motion order", qmeta.motion_keys, metadata.motion_keys),
        ("motion lengths", qmeta.motion_lengths, metadata.motion_lengths),
        ("segment offsets", qmeta.motion_segment_offsets, metadata.motion_segment_offsets),
        ("segment starts", qmeta.start_frame, metadata.start_frame),
        ("segment ends", qmeta.end_frame_exclusive, metadata.end_frame_exclusive),
        ("segment motion IDs", qmeta.motion_id, metadata.motion_id),
        ("local segment IDs", qmeta.local_segment_id, metadata.local_segment_id),
    )
    for label, current, expected in comparisons:
        if not np.array_equal(current, expected):
            layout_mismatches.append(label)
    if (
        qmeta.motion_fps.shape != metadata.motion_fps.shape
        or not np.allclose(
            qmeta.motion_fps, metadata.motion_fps, rtol=0.0, atol=1.0e-12
        )
    ):
        layout_mismatches.append("motion FPS")
    if qmeta.segment_schema_version != metadata.segment_schema_version:
        layout_mismatches.append("segment schema")
    if not math.isclose(
        qmeta.segment_length_seconds,
        metadata.segment_length_seconds,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        layout_mismatches.append("segment length")
    provenance_mismatches: list[str] = []
    if qmeta.manifest_sha256 != metadata.manifest_sha256:
        provenance_mismatches.append("manifest SHA256")
    if qmeta.pool_fingerprint != metadata.pool_fingerprint:
        provenance_mismatches.append("pool fingerprint")
    if difficulty_metadata.sha256_file(manifest) != qmeta.manifest_sha256:
        provenance_mismatches.append("supplied manifest SHA256")
    if tuple(str(item) for item in qmeta.motion_keys) != tuple(manifest_keys):
        layout_mismatches.append("supplied manifest order")
    mismatches = layout_mismatches + provenance_mismatches
    if mismatches and strict:
        raise ValueError(
            "Quality metadata does not match difficulty metadata: "
            + ", ".join(mismatches)
            + "."
        )
    if layout_mismatches:
        warning = (
            "quality diagnostics skipped because layout does not match: "
            + ", ".join(sorted(set(layout_mismatches)))
        )
        return None, {"mapping_match_ok": False, "used_for_fitting": False}, [warning]

    motion_rows: list[dict[str, Any]] = []
    accepted_codes = {
        quality_metadata.QUALITY_STATUS_TO_CODE["pass"],
        quality_metadata.QUALITY_STATUS_TO_CODE["borderline"],
    }
    for motion_id in range(motion_count):
        first = int(metadata.motion_segment_offsets[motion_id])
        last = int(metadata.motion_segment_offsets[motion_id + 1])
        statuses = qmeta.quality_status[first:last]
        status_counts = np.bincount(statuses, minlength=3).astype(np.int64)
        motion_rows.append(
            {
                "score_mean": float(np.mean(qmeta.quality_score[first:last])),
                "status_counts": status_counts,
                "eligible_segment_count": int(
                    np.count_nonzero(np.isin(statuses, tuple(accepted_codes)))
                ),
            }
        )
    mapping_ok = not provenance_mismatches
    diagnostic = {
        "mapping_match_ok": mapping_ok,
        "used_for_fitting": False,
        "metadata_sha256": qmeta.metadata_sha256,
        "provenance_warnings": sorted(set(provenance_mismatches)),
    }
    warnings = (
        [
            "quality diagnostics used with non-strict provenance mismatch: "
            + ", ".join(sorted(set(provenance_mismatches)))
        ]
        if provenance_mismatches
        else []
    )
    return motion_rows, diagnostic, warnings


def _centroids_in_original_z_space(
    profile: motion_clustering.MotionClusteringProfile,
) -> np.ndarray:
    active_count = int(np.count_nonzero(profile.active_feature_mask))
    if profile.use_pca:
        active = (
            profile.centroids @ profile.pca_components + profile.pca_mean
        )
    else:
        active = profile.centroids
    if active.shape != (profile.num_clusters, active_count):
        raise RuntimeError("Frozen centroid dimension does not match active feature state.")
    expanded = np.full(
        (profile.num_clusters, len(profile.feature_names)),
        np.nan,
        dtype=np.float64,
    )
    expanded[:, profile.active_feature_mask] = active
    return expanded


def _top_centroid_features(
    centroid_z: np.ndarray,
    feature_names: Sequence[str],
    *,
    count: int = 5,
) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    values = np.asarray(centroid_z, dtype=np.float64)
    finite = np.flatnonzero(np.isfinite(values))
    positive_order = sorted(
        finite.tolist(), key=lambda index: (-values[index], str(feature_names[index]))
    )
    negative_order = sorted(
        finite.tolist(), key=lambda index: (values[index], str(feature_names[index]))
    )
    positive = [
        {"feature": str(feature_names[index]), "z": float(values[index])}
        for index in positive_order
    ][:count]
    negative = [
        {"feature": str(feature_names[index]), "z": float(values[index])}
        for index in negative_order
    ][:count]
    return positive, negative


def _format_top_features(items: Sequence[Mapping[str, float]]) -> str:
    return ";".join(
        f"{item['feature']}:{float(item['z']):.6g}" for item in items
    )


def _cluster_distance_details(
    result: motion_clustering.MotionClusteringResult,
) -> tuple[np.ndarray, np.ndarray]:
    difference = (
        result.clustering_features[:, None, :] - result.profile.centroids[None, :, :]
    )
    distances = np.sqrt(np.sum(np.square(difference), axis=2))
    order = np.argsort(distances, axis=1, kind="mergesort")
    boundary_margin = (
        distances[np.arange(distances.shape[0]), order[:, 1]]
        - distances[np.arange(distances.shape[0]), order[:, 0]]
    )
    return distances, boundary_margin


def _representative_records(
    indexes: Sequence[int] | np.ndarray,
    *,
    motion_keys: Sequence[str],
    motion_paths: Sequence[str],
    distances: np.ndarray,
    boundary_margin: np.ndarray,
) -> list[dict[str, Any]]:
    return [
        {
            "motion_id": int(index),
            "motion_key": str(motion_keys[index]),
            "motion_path": str(motion_paths[index]),
            "distance_to_centroid": float(distances[index]),
            "boundary_margin": float(boundary_margin[index]),
        }
        for index in indexes
    ]


def _review_rows(
    *,
    result: motion_clustering.MotionClusteringResult,
    motion_keys: Sequence[str],
    motion_paths: Sequence[str],
    source_categories: Sequence[str],
    difficulty_rows: Sequence[Mapping[str, Any]],
    quality_rows: Sequence[Mapping[str, Any]] | None,
    centroid_z: np.ndarray,
    boundary_margin: np.ndarray,
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    labels = result.cluster_id
    distances = result.distance_to_centroid
    names = result.profile.feature_names
    for cluster_id in range(result.profile.num_clusters):
        members = np.flatnonzero(labels == cluster_id)
        by_distance = members[
            np.argsort(distances[members], kind="mergesort")
        ]
        by_boundary = members[
            np.argsort(boundary_margin[members], kind="mergesort")
        ]
        rng = random.Random((int(seed) << 16) + cluster_id)
        random_members = members.tolist()
        rng.shuffle(random_members)
        selections = (
            ("nearest", by_distance[:10]),
            ("boundary", by_boundary[:5]),
            ("farthest", by_distance[::-1][:5]),
            ("random", random_members[:10]),
        )
        top_positive, top_negative = _top_centroid_features(
            centroid_z[cluster_id], names
        )
        for bucket, selected in selections:
            for index_value in selected:
                index = int(index_value)
                difficulty_row = difficulty_rows[index]
                quality_row = quality_rows[index] if quality_rows is not None else None
                row = {
                    "review_bucket": bucket,
                    "cluster_id": cluster_id,
                    "motion_id": index,
                    "motion_key": str(motion_keys[index]),
                    "motion_path": str(motion_paths[index]),
                    "distance_to_centroid": float(distances[index]),
                    "normalized_distance_to_centroid": float(
                        result.normalized_distance_to_centroid[index]
                    ),
                    "boundary_margin": float(boundary_margin[index]),
                    "top_positive_centroid_features": _format_top_features(
                        top_positive
                    ),
                    "top_negative_centroid_features": _format_top_features(
                        top_negative
                    ),
                    "difficulty_score_mean": float(difficulty_row["score_mean"]),
                    "difficulty_score_p90": float(difficulty_row["score_p90"]),
                    "quality_score_mean": (
                        "" if quality_row is None else float(quality_row["score_mean"])
                    ),
                    "quality_pass_segments": (
                        ""
                        if quality_row is None
                        else int(quality_row["status_counts"][0])
                    ),
                    "quality_borderline_segments": (
                        ""
                        if quality_row is None
                        else int(quality_row["status_counts"][1])
                    ),
                    "quality_reject_segments": (
                        ""
                        if quality_row is None
                        else int(quality_row["status_counts"][2])
                    ),
                    "source_category_diagnostic_only": source_categories[index],
                    "replay_command": (
                        "python scripts/replay_npz.py --motion_file "
                        + shlex.quote(str(motion_paths[index]))
                    ),
                }
                rows.append(row)
    return rows


def _diagnose_alternative_k(
    *,
    features: motion_clustering.MotionFeatureMatrix,
    base_config: Mapping[str, Any],
    values: Sequence[int],
    seed: int,
) -> list[dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []
    for num_clusters in values:
        if num_clusters > features.num_motions:
            raise ValueError(
                f"--diagnose-k value {num_clusters} exceeds motion count "
                f"{features.num_motions}."
            )
        config = dict(base_config)
        config["num_clusters"] = int(num_clusters)
        fitted = motion_clustering.fit_motion_clustering(features, config)
        sizes = np.bincount(
            fitted.cluster_id, minlength=num_clusters
        ).astype(np.int64)
        silhouette, sample_count = _silhouette_score(
            fitted.clustering_features,
            fitted.cluster_id,
            seed=seed + num_clusters,
        )
        diagnostics.append(
            {
                "num_clusters": int(num_clusters),
                "inertia": float(fitted.profile.kmeans_inertia),
                "silhouette_score": silhouette,
                "silhouette_sample_count": sample_count,
                "cluster_sizes": sizes.tolist(),
                "minimum_cluster_size": int(np.min(sizes)),
                "maximum_cluster_size": int(np.max(sizes)),
                "cluster_size_gini": _gini(sizes),
                "minimum_to_maximum_size_ratio": (
                    float(np.min(sizes) / np.max(sizes))
                    if np.max(sizes) > 0
                    else 0.0
                ),
            }
        )
    return diagnostics


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    diagnose_k = _validate_args(args)

    manifest = args.manifest.expanduser().resolve()
    difficulty_metadata_path = args.difficulty_metadata.expanduser().resolve()
    difficulty_profile_path = args.difficulty_profile.expanduser().resolve()
    cluster_config_path = args.cluster_config.expanduser().resolve()
    for label, path in (
        ("Manifest", manifest),
        ("Difficulty metadata", difficulty_metadata_path),
        ("Difficulty Profile", difficulty_profile_path),
        ("Cluster config", cluster_config_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    if manifest.suffix.lower() != ".txt":
        raise ValueError("Motion cluster metadata requires a local .txt manifest.")
    if args.mode == "fit_transform":
        _fit_manifest_guard(manifest)

    output_dir = args.output_dir.expanduser().resolve()
    _prepare_output_dir(output_dir, args.overwrite)

    manifest_keys, manifest_paths = difficulty_metadata.resolve_manifest_entries(
        manifest, working_directory=PROJECT_ROOT
    )
    metadata = difficulty_metadata.SegmentDifficultyMetadata.load(
        difficulty_metadata_path
    )
    if len(manifest_keys) != metadata.num_motions:
        raise ValueError(
            "Manifest motion count does not match difficulty metadata."
        )
    if tuple(manifest_keys) != tuple(str(item) for item in metadata.motion_keys):
        raise ValueError(
            "Manifest motion order does not match difficulty metadata."
        )
    upstream_difficulty_profile, difficulty_profile_sha256 = (
        _validate_upstream_identities(
            manifest=manifest,
            manifest_keys=manifest_keys,
            metadata=metadata,
            difficulty_profile_path=difficulty_profile_path,
            mode=args.mode,
        )
    )

    selected_count = (
        metadata.num_motions
        if args.max_motions is None
        else min(args.max_motions, metadata.num_motions)
    )
    selected_keys = manifest_keys[:selected_count]
    selected_paths = manifest_paths[:selected_count]
    selected_lengths = metadata.motion_lengths[:selected_count]
    selected_fps = metadata.motion_fps[:selected_count]
    selected_offsets = metadata.motion_segment_offsets[: selected_count + 1]
    normalized_manifest_text = "".join(f"{key}\n" for key in selected_keys)
    if args.max_motions is None:
        effective_manifest_sha256 = metadata.manifest_sha256
        effective_pool_fingerprint = metadata.pool_fingerprint
    else:
        effective_manifest_sha256 = _sha256_text(normalized_manifest_text)
        effective_pool_fingerprint = _motion_pool_fingerprint(
            selected_paths, selected_lengths.tolist(), selected_fps.tolist()
        )

    config = motion_clustering.load_motion_clustering_config(cluster_config_path)
    config_sha256 = motion_clustering.canonical_json_sha256(config)
    upstream_units = dict(
        zip(
            upstream_difficulty_profile.feature_names,
            upstream_difficulty_profile.feature_units,
            strict=True,
        )
    )
    unit_mismatches = [
        (
            feature_name,
            config["feature_units"][feature_name],
            upstream_units.get(feature_name),
        )
        for feature_name in config["feature_list"]
        if config["feature_units"][feature_name]
        != upstream_units.get(feature_name)
    ]
    if unit_mismatches:
        details = ", ".join(
            f"{name}: cluster={configured!r}, module2={upstream!r}"
            for name, configured, upstream in unit_mismatches
        )
        raise ValueError(
            "Cluster feature units do not match the frozen module-two Profile: "
            + details
            + "."
        )
    if int(config["num_clusters"]) > selected_count:
        raise ValueError(
            "Configured num_clusters cannot exceed the selected motion count."
        )
    metadata_view = _prefix_metadata_view(metadata, selected_count)
    git_commit = _git_commit()
    cluster_profile_output = output_dir / "cluster_profile.json"
    if args.mode == "fit_transform":
        result = motion_clustering.fit_transform_motion_clustering(
            metadata_view,
            config,
            training_manifest_sha256=effective_manifest_sha256,
            training_pool_fingerprint=effective_pool_fingerprint,
            difficulty_metadata_sha256=metadata.metadata_sha256,
            difficulty_profile_sha256=difficulty_profile_sha256,
            config_sha256=config_sha256,
            git_commit=git_commit,
        )
    else:
        source_profile_path = args.profile.expanduser().resolve()
        profile = motion_clustering.load_motion_clustering_profile(
            source_profile_path
        )
        profile_mismatches: list[str] = []
        if profile.config_sha256 != config_sha256:
            profile_mismatches.append("cluster config SHA256")
        if profile.algorithm_schema_version != config["algorithm_schema_version"]:
            profile_mismatches.append("algorithm schema")
        if profile.num_clusters != int(config["num_clusters"]):
            profile_mismatches.append("cluster count")
        if profile.difficulty_profile_sha256 != difficulty_profile_sha256:
            profile_mismatches.append("difficulty Profile SHA256")
        if profile_mismatches:
            raise ValueError(
                "Frozen cluster Profile does not match transform inputs: "
                + ", ".join(profile_mismatches)
                + "."
            )
        result = motion_clustering.transform_motion_clustering(
            metadata_view, profile
        )

    profile = result.profile
    cluster_profile_sha256 = profile.sha256
    cluster_sizes = np.bincount(
        result.cluster_id, minlength=profile.num_clusters
    ).astype(np.int64)
    source_categories = [
        _inferred_source_category(key) for key in selected_keys
    ]
    difficulty_rows = _motion_difficulty_diagnostics(metadata, selected_count)
    quality_path = (
        None
        if args.quality_metadata is None
        else args.quality_metadata.expanduser().resolve()
    )
    quality_rows, quality_diagnostic, diagnostic_warnings = (
        _load_quality_diagnostics(
            quality_path,
            manifest=manifest,
            manifest_keys=manifest_keys,
            metadata=metadata,
            motion_count=selected_count,
            strict=args.strict,
        )
    )

    all_centroid_distances, boundary_margin = _cluster_distance_details(result)
    assigned_distances = all_centroid_distances[
        np.arange(selected_count, dtype=np.int64), result.cluster_id
    ]
    if not np.allclose(
        assigned_distances,
        result.distance_to_centroid,
        rtol=1.0e-12,
        atol=1.0e-12,
    ):
        raise RuntimeError("Assigned centroid distance diagnostics are inconsistent.")
    centroid_z = _centroids_in_original_z_space(profile)

    motion_rows: list[dict[str, Any]] = []
    for motion_id in range(selected_count):
        row: dict[str, Any] = {
            "schema_version": cluster_metadata.CLUSTER_METADATA_SCHEMA_VERSION,
            "algorithm_schema_version": profile.algorithm_schema_version,
            "manifest_index": motion_id,
            "motion_id": motion_id,
            "motion_key": selected_keys[motion_id],
            "motion_path": selected_paths[motion_id],
            "duration_seconds": float(
                result.motion_features.motion_duration_seconds[motion_id]
            ),
            "segment_count": int(result.motion_features.segment_counts[motion_id]),
            "cluster_id": int(result.cluster_id[motion_id]),
            "distance_to_centroid": float(result.distance_to_centroid[motion_id]),
            "normalized_distance_to_centroid": float(
                result.normalized_distance_to_centroid[motion_id]
            ),
            "boundary_margin": float(boundary_margin[motion_id]),
            "cluster_size": int(cluster_sizes[result.cluster_id[motion_id]]),
            "source_category_diagnostic_only": source_categories[motion_id],
        }
        for feature_index, name in enumerate(result.motion_features.feature_names):
            available = bool(
                result.motion_features.available_mask[motion_id, feature_index]
            )
            row[name] = (
                float(result.motion_features.values[motion_id, feature_index])
                if available
                else ""
            )
            standardized = result.standardized_features[
                motion_id, feature_index
            ]
            row[f"z_{name}"] = (
                float(standardized) if np.isfinite(standardized) else ""
            )
            row[f"available_{name}"] = available
        motion_rows.append(row)

    feature_statistics_rows: list[dict[str, Any]] = []
    required = set(profile.required_features)
    near_constant = set(profile.near_constant_features)
    for feature_index, name in enumerate(profile.feature_names):
        available = result.motion_features.available_mask[:, feature_index]
        raw_values = result.motion_features.values[available, feature_index]
        standardized = result.standardized_features[:, feature_index]
        row = {
            "feature": name,
            "source_segment_feature": profile.source_feature_names[feature_index],
            "aggregation": profile.feature_aggregations[feature_index],
            "unit": profile.feature_units[feature_index],
            "required": name in required,
            "coverage": float(np.mean(available)),
            "center": float(profile.feature_centers[feature_index]),
            "scale": float(profile.feature_scales[feature_index]),
            "near_constant": name in near_constant,
            "active_for_distance": bool(
                profile.active_feature_mask[feature_index]
            ),
            **{f"raw_{key}": value for key, value in _describe(raw_values).items()},
            **{
                f"z_{key}": value
                for key, value in _describe(standardized).items()
            },
        }
        for cluster_id in range(profile.num_clusters):
            members = result.cluster_id == cluster_id
            row[f"cluster_{cluster_id}_raw_mean"] = _finite_mean(
                result.motion_features.values[members, feature_index]
            )
            row[f"cluster_{cluster_id}_z_mean"] = _finite_mean(
                standardized[members]
            )
        feature_statistics_rows.append(row)

    centroid_rows: list[dict[str, Any]] = []
    for cluster_id in range(profile.num_clusters):
        row = {
            "cluster_id": cluster_id,
            "cluster_size": int(cluster_sizes[cluster_id]),
        }
        for dimension, value in enumerate(profile.centroids[cluster_id]):
            row[f"distance_space_{dimension}"] = float(value)
        for feature_index, name in enumerate(profile.feature_names):
            value = centroid_z[cluster_id, feature_index]
            row[f"z_{name}"] = float(value) if np.isfinite(value) else ""
        centroid_rows.append(row)

    review_rows = _review_rows(
        result=result,
        motion_keys=selected_keys,
        motion_paths=selected_paths,
        source_categories=source_categories,
        difficulty_rows=difficulty_rows,
        quality_rows=quality_rows,
        centroid_z=centroid_z,
        boundary_margin=boundary_margin,
        seed=args.seed,
    )

    silhouette, silhouette_sample_count = _silhouette_score(
        result.clustering_features, result.cluster_id, seed=args.seed
    )
    centroid_pairwise = np.sqrt(
        np.sum(
            np.square(
                profile.centroids[:, None, :] - profile.centroids[None, :, :]
            ),
            axis=2,
        )
    )
    off_diagonal = centroid_pairwise[
        ~np.eye(profile.num_clusters, dtype=bool)
    ]
    per_cluster: dict[str, Any] = {}
    for cluster_id in range(profile.num_clusters):
        members = np.flatnonzero(result.cluster_id == cluster_id)
        by_distance = members[
            np.argsort(result.distance_to_centroid[members], kind="mergesort")
        ]
        by_boundary = members[
            np.argsort(boundary_margin[members], kind="mergesort")
        ]
        raw_means = {
            name: _finite_mean(result.motion_features.values[members, index])
            for index, name in enumerate(profile.feature_names)
        }
        z_means = {
            name: _finite_mean(result.standardized_features[members, index])
            for index, name in enumerate(profile.feature_names)
        }
        difficulty_bin_counts = (
            np.sum(
                np.stack(
                    [difficulty_rows[index]["bin_counts"] for index in members],
                    axis=0,
                ),
                axis=0,
            ).astype(np.int64)
            if members.size
            else np.zeros(metadata.num_bins, dtype=np.int64)
        )
        category_counts = Counter(source_categories[index] for index in members)
        top_positive, top_negative = _top_centroid_features(
            centroid_z[cluster_id], profile.feature_names
        )
        quality_summary: dict[str, Any] | None = None
        if quality_rows is not None:
            status_counts = (
                np.sum(
                    np.stack(
                        [
                            quality_rows[index]["status_counts"]
                            for index in members
                        ],
                        axis=0,
                    ),
                    axis=0,
                ).astype(np.int64)
                if members.size
                else np.zeros(3, dtype=np.int64)
            )
            status_total = int(np.sum(status_counts))
            quality_summary = {
                "status_counts": {
                    "pass": int(status_counts[0]),
                    "borderline": int(status_counts[1]),
                    "reject": int(status_counts[2]),
                },
                "status_ratios": {
                    "pass": (
                        float(status_counts[0] / status_total)
                        if status_total
                        else 0.0
                    ),
                    "borderline": (
                        float(status_counts[1] / status_total)
                        if status_total
                        else 0.0
                    ),
                    "reject": (
                        float(status_counts[2] / status_total)
                        if status_total
                        else 0.0
                    ),
                },
                "eligible_motion_count": int(
                    sum(
                        quality_rows[index]["eligible_segment_count"] > 0
                        for index in members
                    )
                ),
                "quality_score_mean": _finite_mean(
                    np.asarray(
                        [quality_rows[index]["score_mean"] for index in members]
                    )
                ),
                "used_for_fitting": False,
            }
        per_cluster[str(cluster_id)] = {
            "size": int(members.size),
            "ratio": float(members.size / selected_count),
            "within_cluster_distance": _describe(
                result.distance_to_centroid[members]
            ),
            "duration_seconds": _describe(
                result.motion_features.motion_duration_seconds[members]
            ),
            "feature_means": raw_means,
            "feature_z_scores": z_means,
            "centroid_z_scores": {
                name: (
                    float(centroid_z[cluster_id, index])
                    if np.isfinite(centroid_z[cluster_id, index])
                    else None
                )
                for index, name in enumerate(profile.feature_names)
            },
            "top_positive_centroid_features": top_positive,
            "top_negative_centroid_features": top_negative,
            "difficulty_diagnostics": {
                "bin_counts": difficulty_bin_counts.tolist(),
                "bin_ratios": (
                    difficulty_bin_counts / np.sum(difficulty_bin_counts)
                ).tolist()
                if np.sum(difficulty_bin_counts)
                else np.zeros(metadata.num_bins, dtype=np.float64).tolist(),
                "motion_score_mean": _describe(
                    np.asarray(
                        [difficulty_rows[index]["score_mean"] for index in members]
                    )
                ),
                "used_for_fitting": False,
            },
            "quality_diagnostics": quality_summary,
            "source_category_composition": {
                category: {
                    "count": int(count),
                    "ratio": float(count / members.size) if members.size else 0.0,
                }
                for category, count in sorted(category_counts.items())
            },
            "source_category_used_for_fitting": False,
            "representative_motions": {
                "nearest": _representative_records(
                    by_distance[:5],
                    motion_keys=selected_keys,
                    motion_paths=selected_paths,
                    distances=result.distance_to_centroid,
                    boundary_margin=boundary_margin,
                ),
                "boundary": _representative_records(
                    by_boundary[:5],
                    motion_keys=selected_keys,
                    motion_paths=selected_paths,
                    distances=result.distance_to_centroid,
                    boundary_margin=boundary_margin,
                ),
                "farthest": _representative_records(
                    by_distance[::-1][:5],
                    motion_keys=selected_keys,
                    motion_paths=selected_paths,
                    distances=result.distance_to_centroid,
                    boundary_margin=boundary_margin,
                ),
            },
        }

    alternative_k = _diagnose_alternative_k(
        features=result.motion_features,
        base_config=config,
        values=diagnose_k,
        seed=args.seed,
    )
    warnings = list(profile.warnings) + diagnostic_warnings
    minimum_warning = int(config["minimum_cluster_size_warning"])
    for cluster_id, size in enumerate(cluster_sizes):
        if int(size) < minimum_warning:
            warnings.append(
                f"cluster {cluster_id} has {int(size)} motions, below "
                f"warning threshold {minimum_warning}"
            )
    summary = {
        "schema_version": "wbt.motion_cluster_summary.v1",
        "mode": args.mode,
        "manifest": str(
            output_dir / "normalized_manifest.txt"
            if args.max_motions is not None
            else manifest
        ),
        "original_manifest": str(manifest),
        "original_manifest_motion_count": metadata.num_motions,
        "motion_count": selected_count,
        "segment_count": int(selected_offsets[-1]),
        "num_clusters": profile.num_clusters,
        "cluster_sizes": cluster_sizes.tolist(),
        "cluster_size_ratios": (cluster_sizes / selected_count).tolist(),
        "minimum_cluster_size": int(np.min(cluster_sizes)),
        "maximum_cluster_size": int(np.max(cluster_sizes)),
        "cluster_size_gini": _gini(cluster_sizes),
        "inertia": float(np.sum(np.square(result.distance_to_centroid))),
        "profile_training_inertia": float(profile.kmeans_inertia),
        "silhouette_score": silhouette,
        "silhouette_sample_count": silhouette_sample_count,
        "centroid_pairwise_distances": {
            "matrix": centroid_pairwise.tolist(),
            "minimum_nonzero": (
                float(np.min(off_diagonal)) if off_diagonal.size else None
            ),
            "mean_nonzero": (
                float(np.mean(off_diagonal)) if off_diagonal.size else None
            ),
            "maximum": (
                float(np.max(off_diagonal)) if off_diagonal.size else None
            ),
        },
        "within_cluster_distance": {
            "global": _describe(result.distance_to_centroid),
            "per_cluster": {
                str(cluster_id): _describe(
                    result.distance_to_centroid[
                        result.cluster_id == cluster_id
                    ]
                )
                for cluster_id in range(profile.num_clusters)
            },
        },
        "per_cluster": per_cluster,
        "feature_availability": {
            name: float(np.mean(result.motion_features.available_mask[:, index]))
            for index, name in enumerate(profile.feature_names)
        },
        "near_constant_features": list(profile.near_constant_features),
        "cluster_profile_sha256": cluster_profile_sha256,
        "cluster_config_sha256": config_sha256,
        "difficulty_metadata_sha256": metadata.metadata_sha256,
        "difficulty_profile_sha256": difficulty_profile_sha256,
        "manifest_sha256": effective_manifest_sha256,
        "pool_fingerprint": effective_pool_fingerprint,
        "quality_diagnostics": quality_diagnostic,
        "diagnose_k": {
            "requested": list(diagnose_k),
            "auto_selected": False,
            "results": alternative_k,
            "note": (
                "Train-only diagnostics; no K was selected from Validation or Test."
            ),
        },
        "fitting_inputs": {
            "raw_module_two_segment_feature_matrix_used": True,
            "source_category_used": False,
            "quality_used": False,
            "difficulty_score_or_bin_used": False,
            "policy_statistics_used": False,
        },
        **_NOT_USED_FOR_FITTING,
        "review_seed": args.seed,
        "config_random_seed": int(config["random_seed"]),
        "git_commit": git_commit,
        "provisional": bool(config.get("provisional", True)),
        "warnings": sorted(set(warnings)),
    }

    # All expensive work and identity checks have succeeded.  Only now write
    # the artifact set so a failed fit cannot look like a completed build.
    normalized_manifest_path = output_dir / "normalized_manifest.txt"
    normalized_manifest_path.write_text(
        normalized_manifest_text, encoding="utf-8"
    )
    _write_json(output_dir / "cluster_config_resolved.json", config)
    if args.mode == "fit_transform":
        profile.save_json(cluster_profile_output)
    else:
        source_profile_path = args.profile.expanduser().resolve()
        if source_profile_path != cluster_profile_output:
            shutil.copyfile(source_profile_path, cluster_profile_output)
    # A copied transform Profile must deserialize to exactly the frozen state.
    written_profile = motion_clustering.load_motion_clustering_profile(
        cluster_profile_output
    )
    if written_profile.sha256 != cluster_profile_sha256:
        raise RuntimeError("Written cluster Profile identity changed unexpectedly.")

    _write_csv(
        output_dir / "motion_cluster_metadata.csv",
        motion_rows,
        tuple(motion_rows[0]),
    )
    _write_csv(
        output_dir / "cluster_feature_statistics.csv",
        feature_statistics_rows,
        tuple(feature_statistics_rows[0]),
    )
    _write_csv(
        output_dir / "cluster_centroids.csv",
        centroid_rows,
        tuple(centroid_rows[0]),
    )
    _write_csv(
        output_dir / "cluster_review_motions.csv",
        review_rows,
        tuple(review_rows[0]) if review_rows else (),
    )

    payload = cluster_metadata.metadata_npz_payload(
        algorithm_schema_version=profile.algorithm_schema_version,
        manifest_sha256=effective_manifest_sha256,
        pool_fingerprint=effective_pool_fingerprint,
        difficulty_metadata_sha256=metadata.metadata_sha256,
        difficulty_profile_sha256=difficulty_profile_sha256,
        cluster_profile_sha256=cluster_profile_sha256,
        cluster_config_sha256=config_sha256,
        motion_keys=selected_keys,
        motion_lengths=selected_lengths,
        motion_fps=selected_fps,
        motion_segment_offsets=selected_offsets,
        motion_id=np.arange(selected_count, dtype=np.int64),
        cluster_id=result.cluster_id,
        cluster_sizes=cluster_sizes,
        centroids=profile.centroids,
        feature_names=result.motion_features.feature_names,
        motion_feature_matrix=result.motion_features.values,
        feature_available_mask=result.motion_features.available_mask,
        standardized_feature_matrix=result.standardized_features,
        num_clusters=profile.num_clusters,
    )
    frozen = cluster_metadata.save_metadata(
        output_dir / "motion_cluster_metadata.npz",
        payload,
        overwrite=args.overwrite,
    )
    effective_manifest_path = (
        normalized_manifest_path if args.max_motions is not None else manifest
    )
    frozen.validate_against(
        manifest_path=effective_manifest_path,
        motion_keys=selected_keys,
        motion_lengths=selected_lengths,
        motion_fps=selected_fps,
        motion_segment_offsets=selected_offsets,
        pool_fingerprint=effective_pool_fingerprint,
        difficulty_metadata_sha256=metadata.metadata_sha256,
        difficulty_profile_sha256=difficulty_profile_sha256,
        expected_num_clusters=profile.num_clusters,
        strict=True,
    )
    _write_json(output_dir / "cluster_summary.json", summary)
    print(
        "Built frozen motion clusters: "
        f"motions={selected_count}, clusters={cluster_sizes.tolist()}, "
        f"silhouette={silhouette}, output={output_dir}"
    )


if __name__ == "__main__":
    main()
