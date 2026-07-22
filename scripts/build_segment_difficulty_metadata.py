#!/usr/bin/env python3
"""Build deterministic policy-independent difficulty metadata for WBT motions."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shlex
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
UTILS_DIR = PROJECT_ROOT / "source" / "whole_body_tracking" / "whole_body_tracking" / "utils"
# Avoid importing the package root: it eagerly imports Isaac tasks, which are
# unrelated to this deterministic CPU metadata pipeline.
sys.path.insert(0, str(UTILS_DIR))

import difficulty  # noqa: E402
import difficulty_metadata  # noqa: E402
import quality  # noqa: E402
import quality_metadata  # noqa: E402
import sampling  # noqa: E402


OUTPUT_FILENAMES = (
    "segment_difficulty_metadata.csv",
    "segment_difficulty_metadata.npz",
    "motion_difficulty_metadata.csv",
    "motion_difficulty_metadata.npz",
    "difficulty_profile.json",
    "difficulty_summary.json",
    "difficulty_feature_statistics.csv",
    "difficulty_review_segments.csv",
    "difficulty_config_resolved.json",
    "normalized_manifest.txt",
)


@dataclass(frozen=True)
class MotionFeatureResult:
    motion_id: int
    motion_key: str
    motion_path: str
    fps: float
    num_frames: int
    source_file: str
    source_format: str
    feature_values: np.ndarray
    feature_available: np.ndarray
    duration_seconds: np.ndarray
    bounds: tuple[tuple[int, int], ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate intrinsic motion difficulty from final WBT/G1 reference trajectories only. "
            "No policy, reward, success, or quality label enters the score."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True, help="Ordered local motion manifest.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Metadata output directory.")
    parser.add_argument(
        "--difficulty-config",
        type=Path,
        default=difficulty.DEFAULT_DIFFICULTY_CONFIG_PATH,
        help="JSON-compatible YAML feature definition.",
    )
    parser.add_argument(
        "--segment-length-seconds", type=float, default=1.0, help="Stage-0 segment duration override."
    )
    parser.add_argument(
        "--mode", choices=("fit_transform", "transform"), default="fit_transform"
    )
    parser.add_argument(
        "--profile", type=Path, default=None, help="Train-fitted profile required by transform mode."
    )
    parser.add_argument(
        "--quality-metadata",
        type=Path,
        default=None,
        help="Optional module-one NPZ used only for mapping checks and cross statistics.",
    )
    parser.add_argument(
        "--dataset-metadata",
        type=Path,
        default=None,
        help="Optional split/category/source CSV used only for diagnostics.",
    )
    parser.add_argument("--max-motions", type=int, default=None, help="Stable manifest prefix for a pilot.")
    parser.add_argument("--workers", type=int, default=1, help="Number of ordered CPU extraction workers.")
    parser.add_argument("--device", default="cpu", help="Reserved execution device; v1 supports cpu only.")
    parser.add_argument("--seed", type=int, default=42, help="Fixed review-sampling seed.")
    parser.add_argument("--strict", action="store_true", help="Reject metadata/split warnings when applicable.")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacement of known output files.")
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if not math.isfinite(args.segment_length_seconds) or args.segment_length_seconds <= 0.0:
        raise ValueError("--segment-length-seconds must be finite and positive.")
    if args.max_motions is not None and args.max_motions < 1:
        raise ValueError("--max-motions must be at least one.")
    if args.workers < 1:
        raise ValueError("--workers must be at least one.")
    if str(args.device).lower() != "cpu":
        raise ValueError("Module-two v1 is a deterministic NumPy pipeline and supports --device cpu only.")
    if args.mode == "transform" and args.profile is None:
        raise ValueError("--profile is required in transform mode.")
    if args.mode == "fit_transform" and args.profile is not None:
        raise ValueError("--profile must not be supplied in fit_transform mode.")


def _prepare_output_dir(path: Path, overwrite: bool) -> None:
    path.mkdir(parents=True, exist_ok=True)
    collisions = [name for name in OUTPUT_FILENAMES if (path / name).exists()]
    if collisions and not overwrite:
        raise FileExistsError(
            f"Output files already exist in {path}: {collisions}. Pass --overwrite to replace them."
        )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]], columns: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, check=False, capture_output=True, text=True
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _fit_manifest_guard(path: Path) -> None:
    tokens = {token for token in path.stem.lower().replace("-", "_").split("_") if token}
    forbidden = {"validation", "valid", "val", "test"}
    if tokens.intersection(forbidden):
        raise ValueError(
            "fit_transform is Train-only; refusing a manifest whose name indicates Validation/Test."
        )


def _extract_one(task: tuple[int, str, str, Mapping[str, Any]]) -> MotionFeatureResult:
    motion_id, motion_key, motion_path, config = task
    motion = quality.load_motion_npz(motion_path)
    local_index = sampling.FixedLengthSegmentIndex(
        [motion.num_frames], [motion.fps], float(config["segment_length_seconds"]), device="cpu"
    )
    bounds = tuple(
        (int(start), int(end))
        for start, end in zip(
            local_index.segment_start_frames.tolist(),
            local_index.segment_end_frames.tolist(),
            strict=True,
        )
    )
    features = difficulty.extract_motion_difficulty_features(motion, bounds, config)
    return MotionFeatureResult(
        motion_id=motion_id,
        motion_key=motion_key,
        motion_path=motion_path,
        fps=float(motion.fps),
        num_frames=motion.num_frames,
        source_file=motion.source_file,
        source_format=motion.source_format,
        feature_values=features.values,
        feature_available=features.available,
        duration_seconds=features.duration_seconds,
        bounds=bounds,
    )


def _load_dataset_metadata(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None:
        return {}
    if not path.is_file():
        raise FileNotFoundError(f"Dataset metadata does not exist: {path}")
    result: dict[str, dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or "relative_path" not in reader.fieldnames:
            raise ValueError("--dataset-metadata must contain a relative_path column.")
        for row in reader:
            key = Path(str(row.get("relative_path", ""))).as_posix()
            if key:
                result[key] = {str(name): str(value or "") for name, value in row.items()}
    return result


def _inferred_category(key: str) -> str:
    parts = Path(key).as_posix().split("/")
    if "g1_all" in parts:
        index = parts.index("g1_all")
        if index + 1 < len(parts):
            return parts[index + 1]
    return parts[0] if len(parts) > 1 else "unknown"


def _provenance(
    result: MotionFeatureResult, dataset_rows: Mapping[str, Mapping[str, str]]
) -> dict[str, str]:
    metadata = dataset_rows.get(Path(result.motion_key).as_posix(), {})
    category = str(metadata.get("category", "")) or _inferred_category(result.motion_key)
    source_group = str(metadata.get("source_group", "")) or category
    split = str(metadata.get("split", "")) or "unknown"
    return {"category": category, "source_group": source_group, "split": split}


def _rank_average(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    sorted_values = array[order]
    ranks = np.empty(array.size, dtype=np.float64)
    start = 0
    while start < array.size:
        end = start + 1
        while end < array.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def _correlation(first: np.ndarray, second: np.ndarray, *, spearman: bool = False) -> float | None:
    x = np.asarray(first, dtype=np.float64)
    y = np.asarray(second, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(valid) < 2:
        return None
    x = x[valid]
    y = y[valid]
    if spearman:
        x = _rank_average(x)
        y = _rank_average(y)
    if np.std(x) <= 0.0 or np.std(y) <= 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _describe(values: Sequence[float] | np.ndarray) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return {"count": 0, "min": None, "mean": None, "std": None, "p50": None, "p90": None,
                "p95": None, "p99": None, "max": None}
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


def _group_summary(
    labels: Sequence[str], scores: np.ndarray, bins: np.ndarray, num_bins: int
) -> dict[str, Any]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        grouped[str(label)].append(index)
    return {
        label: {
            "segment_count": len(indexes),
            "score": _describe(scores[indexes]),
            "bin_counts": [
                int(np.count_nonzero(bins[indexes] == bin_id)) for bin_id in range(num_bins)
            ],
        }
        for label, indexes in sorted(grouped.items())
    }


def _top_contributions(contributions: np.ndarray, count: int = 3) -> list[str]:
    positive = np.where(contributions > 0.0, contributions, -np.inf)
    order = np.argsort(-positive, kind="mergesort")
    output: list[str] = []
    for index in order:
        if len(output) >= count or not np.isfinite(positive[index]):
            break
        output.append(f"{difficulty.FEATURE_NAMES[int(index)]}:{contributions[int(index)]:.6g}")
    while len(output) < count:
        output.append("")
    return output


def _review_rows(
    segment_rows: Sequence[Mapping[str, Any]],
    transform: difficulty.DifficultyTransform,
    profile: difficulty.DifficultyProfile,
    seed: int,
    quality_statuses: np.ndarray | None,
) -> list[dict[str, Any]]:
    scores = transform.difficulty_score
    raw = transform.difficulty_raw
    bins = transform.difficulty_bin
    selections: list[tuple[str, int]] = []
    order = np.argsort(scores, kind="mergesort")
    selections.extend(("lowest", int(index)) for index in order[:20])
    selections.extend(("highest", int(index)) for index in order[-20:][::-1])
    rng = random.Random(seed)
    for bin_id in range(profile.num_bins):
        candidates = np.flatnonzero(bins == bin_id).tolist()
        rng.shuffle(candidates)
        selections.extend((f"bin_{bin_id}_random", int(index)) for index in candidates[:10])
    for boundary_index, edge in enumerate(profile.difficulty_bin_edges, start=1):
        nearby = np.argsort(np.abs(raw - edge), kind="mergesort")[:4]
        selections.extend((f"boundary_{boundary_index}", int(index)) for index in nearby)
    middle = np.argsort(np.abs(scores - 0.5), kind="mergesort")[:20]
    selections.extend(("middle", int(index)) for index in middle)
    if quality_statuses is not None:
        high = scores >= np.percentile(scores, 90.0)
        low = scores <= np.percentile(scores, 10.0)
        for bucket, mask in (
            ("quality_cross_high_pass", high & (quality_statuses == 0)),
            ("quality_cross_high_reject", high & (quality_statuses == 2)),
            ("quality_cross_low_reject", low & (quality_statuses == 2)),
        ):
            selections.extend((bucket, int(index)) for index in np.flatnonzero(mask)[:20])

    rows: list[dict[str, Any]] = []
    for bucket, index in selections:
        source = dict(segment_rows[index])
        top = _top_contributions(transform.feature_contributions[index])
        motion_path = str(source["motion_path"])
        start = int(source["start_frame"])
        end = int(source["end_frame_exclusive"])
        rows.append(
            {
                "review_bucket": bucket,
                "motion_path": motion_path,
                "motion_id": source["motion_id"],
                "local_segment_id": source["local_segment_id"],
                "global_segment_id": source["global_segment_id"],
                "start_frame": start,
                "end_frame_exclusive": end,
                "difficulty_raw": source["difficulty_raw"],
                "difficulty_score": source["difficulty_score"],
                "difficulty_bin": source["difficulty_bin"],
                "top_contribution_1": top[0],
                "top_contribution_2": top[1],
                "top_contribution_3": top[2],
                "preview_command": (
                    f"python scripts/render_npz_preview.py --motion_file {shlex.quote(motion_path)} "
                    f"--start_frame {start} --end_frame_exclusive {end}"
                ),
                "replay_command": (
                    f"python scripts/replay_npz.py --motion_file {shlex.quote(motion_path)} "
                    f"--start_frame {start} --end_frame_exclusive {end}"
                ),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    _validate_args(args)
    manifest = args.manifest.expanduser().resolve()
    if not manifest.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {manifest}")
    if not manifest.name.endswith(".txt"):
        raise ValueError("Module-two metadata requires a local .txt manifest.")
    if args.mode == "fit_transform":
        _fit_manifest_guard(manifest)
    output_dir = args.output_dir.expanduser().resolve()
    _prepare_output_dir(output_dir, args.overwrite)

    all_keys, all_paths = difficulty_metadata.resolve_manifest_entries(
        manifest, working_directory=PROJECT_ROOT
    )
    selected_count = len(all_keys) if args.max_motions is None else min(args.max_motions, len(all_keys))
    motion_keys = all_keys[:selected_count]
    motion_paths = all_paths[:selected_count]
    normalized_manifest = output_dir / "normalized_manifest.txt"
    normalized_manifest.write_text("".join(f"{key}\n" for key in motion_keys), encoding="utf-8")
    effective_manifest = normalized_manifest if args.max_motions is not None else manifest
    manifest_sha256 = difficulty_metadata.sha256_file(effective_manifest)

    config = difficulty.load_difficulty_config(args.difficulty_config.expanduser().resolve())
    config["segment_length_seconds"] = float(args.segment_length_seconds)
    difficulty.validate_difficulty_config(config)
    config_sha256 = difficulty.canonical_json_sha256(config)
    _write_json(output_dir / "difficulty_config_resolved.json", config)

    tasks = [
        (index, key, path, config)
        for index, (key, path) in enumerate(zip(motion_keys, motion_paths, strict=True))
    ]
    results: list[MotionFeatureResult] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for task, result_or_error in zip(tasks, executor.map(_safe_extract, tasks), strict=True):
            result, error = result_or_error
            if error is not None:
                raise RuntimeError(f"Difficulty feature extraction failed for '{task[2]}': {error}")
            assert result is not None
            results.append(result)

    motion_lengths = [result.num_frames for result in results]
    motion_fps = [result.fps for result in results]
    segment_index = sampling.FixedLengthSegmentIndex(
        motion_lengths, motion_fps, args.segment_length_seconds, device="cpu"
    )
    pool_fingerprint = sampling.motion_pool_fingerprint(motion_paths, motion_lengths, motion_fps)
    for result in results:
        first = int(segment_index.motion_segment_offsets[result.motion_id].item())
        last = int(segment_index.motion_segment_offsets[result.motion_id + 1].item())
        expected = tuple(
            (int(start), int(end))
            for start, end in zip(
                segment_index.segment_start_frames[first:last].tolist(),
                segment_index.segment_end_frames[first:last].tolist(),
                strict=True,
            )
        )
        if result.bounds != expected:
            raise RuntimeError("Per-motion extraction bounds do not match the shared Stage-0 segment index.")

    feature_values = np.concatenate([result.feature_values for result in results], axis=0)
    feature_available = np.concatenate([result.feature_available for result in results], axis=0)
    durations = np.concatenate([result.duration_seconds for result in results], axis=0)
    if feature_values.shape[0] != segment_index.num_segments:
        raise RuntimeError("Feature row count does not match the Stage-0 global segment count.")

    dataset_rows = _load_dataset_metadata(
        None if args.dataset_metadata is None else args.dataset_metadata.expanduser().resolve()
    )
    provenance = [_provenance(result, dataset_rows) for result in results]
    if args.mode == "fit_transform":
        non_train = [
            result.motion_key
            for result, item in zip(results, provenance, strict=True)
            if item["split"].lower() not in {"", "unknown", "train"}
        ]
        if non_train:
            raise ValueError(
                "fit_transform dataset metadata identifies non-Train motions: "
                + ", ".join(non_train[:10])
            )

    profile_path = output_dir / "difficulty_profile.json"
    if args.mode == "fit_transform":
        profile = difficulty.fit_difficulty_profile(
            feature_values,
            feature_available,
            config,
            training_manifest_sha256=manifest_sha256,
            training_pool_fingerprint=pool_fingerprint,
            segment_schema_version=sampling.SAMPLING_STATE_VERSION,
            config_sha256=config_sha256,
            git_commit=_git_commit(),
        )
        _write_json(profile_path, profile.to_dict())
    else:
        source_profile = args.profile.expanduser().resolve()
        profile = difficulty.load_difficulty_profile(source_profile)
        if profile.config_sha256 != config_sha256:
            raise ValueError("Transform config SHA256 does not match the frozen Train Profile.")
        if profile.algorithm_schema_version != config["algorithm_schema_version"]:
            raise ValueError("Transform algorithm schema does not match the frozen Train Profile.")
        if profile.segment_schema_version != sampling.SAMPLING_STATE_VERSION:
            raise ValueError("Transform segment schema does not match the frozen Train Profile.")
        if not math.isclose(
            profile.segment_length_seconds, args.segment_length_seconds, rel_tol=0.0, abs_tol=1.0e-12
        ):
            raise ValueError("Transform segment length does not match the frozen Train Profile.")
        if source_profile != profile_path:
            shutil.copyfile(source_profile, profile_path)
    profile_sha256 = difficulty_metadata.sha256_file(profile_path)
    transform = difficulty.transform_difficulty_features(
        feature_values, feature_available, profile
    )

    global_ids = segment_index.metadata()
    global_motion_ids = global_ids["motion_id"].cpu().numpy().astype(np.int64)
    local_ids = global_ids["local_segment_id"].cpu().numpy().astype(np.int64)
    starts = global_ids["start_frame"].cpu().numpy().astype(np.int64)
    ends = global_ids["end_frame_exclusive"].cpu().numpy().astype(np.int64)
    computed_durations = (ends - starts) / np.asarray(motion_fps)[global_motion_ids]
    if not np.allclose(durations, computed_durations, rtol=0.0, atol=1.0e-12):
        raise RuntimeError("Extracted segment durations do not match Stage-0 bounds and FPS.")

    features_cfg = config["features"]
    optional_indexes = np.asarray(
        [index for index, name in enumerate(difficulty.FEATURE_NAMES) if not features_cfg[name]["required"]],
        dtype=np.int64,
    )
    optional_coverage = (
        np.mean(feature_available[:, optional_indexes], axis=1)
        if optional_indexes.size
        else np.ones(segment_index.num_segments)
    )
    categories_by_segment = [provenance[int(motion_id)]["category"] for motion_id in global_motion_ids]
    sources_by_segment = [provenance[int(motion_id)]["source_group"] for motion_id in global_motion_ids]

    segment_rows: list[dict[str, Any]] = []
    for global_id in range(segment_index.num_segments):
        motion_id = int(global_motion_ids[global_id])
        top = _top_contributions(transform.feature_contributions[global_id])
        row: dict[str, Any] = {
            "schema_version": difficulty_metadata.DIFFICULTY_METADATA_SCHEMA_VERSION,
            "algorithm_schema_version": profile.algorithm_schema_version,
            "manifest_index": motion_id,
            "motion_id": motion_id,
            "motion_key": results[motion_id].motion_key,
            "motion_path": results[motion_id].motion_path,
            "category": provenance[motion_id]["category"],
            "source_group": provenance[motion_id]["source_group"],
            "local_segment_id": int(local_ids[global_id]),
            "global_segment_id": global_id,
            "start_frame": int(starts[global_id]),
            "end_frame_exclusive": int(ends[global_id]),
            "duration_seconds": float(durations[global_id]),
            "fps": results[motion_id].fps,
            "difficulty_raw": float(transform.difficulty_raw[global_id]),
            "difficulty_score": float(transform.difficulty_score[global_id]),
            "difficulty_bin": int(transform.difficulty_bin[global_id]),
            "available_feature_count": int(np.count_nonzero(feature_available[global_id])),
            "optional_feature_coverage": float(optional_coverage[global_id]),
            "top_contribution_1": top[0],
            "top_contribution_2": top[1],
            "top_contribution_3": top[2],
        }
        for feature_index, name in enumerate(difficulty.FEATURE_NAMES):
            row[name] = (
                float(feature_values[global_id, feature_index])
                if feature_available[global_id, feature_index]
                else ""
            )
            row[f"z_{name}"] = (
                float(transform.feature_z[global_id, feature_index])
                if feature_available[global_id, feature_index]
                else ""
            )
            row[f"available_{name}"] = bool(feature_available[global_id, feature_index])
        segment_rows.append(row)

    segment_columns = tuple(segment_rows[0])
    _write_csv(output_dir / "segment_difficulty_metadata.csv", segment_rows, segment_columns)

    payload = difficulty_metadata.metadata_npz_payload(
        algorithm_schema_version=profile.algorithm_schema_version,
        segment_schema_version=sampling.SAMPLING_STATE_VERSION,
        segment_length_seconds=args.segment_length_seconds,
        manifest_sha256=manifest_sha256,
        profile_sha256=profile_sha256,
        difficulty_config_sha256=config_sha256,
        pool_fingerprint=pool_fingerprint,
        num_bins=profile.num_bins,
        motion_keys=motion_keys,
        motion_lengths=motion_lengths,
        motion_fps=motion_fps,
        motion_segment_offsets=segment_index.motion_segment_offsets.tolist(),
        global_segment_id=np.arange(segment_index.num_segments),
        motion_id=global_motion_ids,
        local_segment_id=local_ids,
        start_frame=starts,
        end_frame_exclusive=ends,
        duration_seconds=durations,
        difficulty_raw=transform.difficulty_raw,
        difficulty_score=transform.difficulty_score,
        difficulty_bin=transform.difficulty_bin,
        feature_names=difficulty.FEATURE_NAMES,
        feature_values=feature_values,
        feature_z=transform.feature_z,
        feature_available_mask=feature_available,
        optional_feature_coverage=optional_coverage,
        near_constant_features=profile.near_constant_features,
    )
    segment_npz = output_dir / "segment_difficulty_metadata.npz"
    np.savez_compressed(segment_npz, **payload)
    # Load immediately so no malformed artifact can be reported as successful.
    frozen_metadata = difficulty_metadata.SegmentDifficultyMetadata.load(segment_npz)
    frozen_metadata.validate_against(
        manifest_path=effective_manifest,
        motion_keys=motion_keys,
        motion_lengths=motion_lengths,
        motion_fps=motion_fps,
        motion_segment_offsets=segment_index.motion_segment_offsets.tolist(),
        segment_start_frames=starts,
        segment_end_frames=ends,
        segment_length_seconds=args.segment_length_seconds,
        segment_schema_version=sampling.SAMPLING_STATE_VERSION,
        pool_fingerprint=pool_fingerprint,
        expected_profile_sha256=profile_sha256,
        expected_difficulty_config_sha256=config_sha256,
        expected_num_bins=profile.num_bins,
    )

    motion_rows: list[dict[str, Any]] = []
    for motion_id, result in enumerate(results):
        first = int(segment_index.motion_segment_offsets[motion_id].item())
        last = int(segment_index.motion_segment_offsets[motion_id + 1].item())
        aggregate = difficulty.aggregate_motion_difficulty(
            transform.difficulty_score[first:last],
            durations[first:last],
            mean_weight=profile.motion_mean_weight,
            p90_weight=profile.motion_p90_weight,
            num_bins=profile.num_bins,
        )
        motion_rows.append(
            {
                "motion_id": motion_id,
                "motion_key": result.motion_key,
                "motion_path": result.motion_path,
                "category": provenance[motion_id]["category"],
                "source_group": provenance[motion_id]["source_group"],
                "segment_count": aggregate["segment_count"],
                "duration_seconds": aggregate["duration_seconds"],
                "difficulty_mean": aggregate["difficulty_mean"],
                "difficulty_p90": aggregate["difficulty_p90"],
                "difficulty_score": aggregate["difficulty_score"],
                "difficulty_bin": aggregate["difficulty_bin"],
            }
        )
    _write_csv(output_dir / "motion_difficulty_metadata.csv", motion_rows, tuple(motion_rows[0]))
    np.savez_compressed(
        output_dir / "motion_difficulty_metadata.npz",
        schema_version=np.asarray("wbt.motion_difficulty.v1"),
        profile_sha256=np.asarray(profile_sha256),
        difficulty_config_sha256=np.asarray(config_sha256),
        motion_id=np.asarray([row["motion_id"] for row in motion_rows], dtype=np.int64),
        motion_keys=np.asarray(motion_keys, dtype=str),
        segment_count=np.asarray([row["segment_count"] for row in motion_rows], dtype=np.int64),
        duration_seconds=np.asarray([row["duration_seconds"] for row in motion_rows], dtype=np.float64),
        difficulty_mean=np.asarray([row["difficulty_mean"] for row in motion_rows], dtype=np.float32),
        difficulty_p90=np.asarray([row["difficulty_p90"] for row in motion_rows], dtype=np.float32),
        difficulty_score=np.asarray([row["difficulty_score"] for row in motion_rows], dtype=np.float32),
        difficulty_bin=np.asarray([row["difficulty_bin"] for row in motion_rows], dtype=np.int16),
    )

    quality_statuses: np.ndarray | None = None
    quality_cross: dict[str, Any] | None = None
    if args.quality_metadata is not None:
        qmeta = quality_metadata.SegmentQualityMetadata.load(args.quality_metadata.expanduser().resolve())
        match = qmeta.validate_against(
            manifest_path=effective_manifest,
            motion_keys=motion_keys,
            motion_lengths=motion_lengths,
            motion_fps=motion_fps,
            motion_segment_offsets=segment_index.motion_segment_offsets.tolist(),
            segment_start_frames=starts,
            segment_end_frames=ends,
            segment_length_seconds=args.segment_length_seconds,
            segment_schema_version=sampling.SAMPLING_STATE_VERSION,
            pool_fingerprint=pool_fingerprint,
            strict=args.strict,
        )
        if match:
            quality_statuses = qmeta.quality_status.copy()
            cross_counts: dict[str, dict[str, int]] = {}
            for status_name, status_code in quality_metadata.QUALITY_STATUS_TO_CODE.items():
                mask = quality_statuses == status_code
                cross_counts[status_name] = {
                    f"bin_{bin_id}": int(np.count_nonzero(mask & (transform.difficulty_bin == bin_id)))
                    for bin_id in range(profile.num_bins)
                }
            quality_cross = {
                "mapping_match_ok": True,
                "segment_count_before": segment_index.num_segments,
                "segment_count_after": int(transform.difficulty_score.size),
                "status_by_difficulty_bin": cross_counts,
                "note": "Quality was loaded after Profile fit/transform and did not change any score, bin, or row.",
            }

    feature_statistics_rows: list[dict[str, Any]] = []
    feature_correlations: dict[str, dict[str, float | None]] = {}
    spearman_to_score: dict[str, float | None] = {}
    for index, name in enumerate(difficulty.FEATURE_NAMES):
        stats = _describe(feature_values[feature_available[:, index], index])
        feature_statistics_rows.append(
            {
                "feature": name,
                "unit": profile.feature_units[index],
                "required": name in profile.required_features,
                "direction": int(profile.feature_directions[index]),
                "configured_weight": float(profile.feature_weights[index]),
                "effective_weight": float(profile.effective_feature_weights[index]),
                "coverage": float(np.mean(feature_available[:, index])),
                "median": float(profile.feature_medians[index]),
                "scale": float(profile.feature_scales[index]),
                "near_constant": name in profile.near_constant_features,
                **stats,
                "spearman_to_difficulty_score": _correlation(
                    feature_values[:, index], transform.difficulty_score, spearman=True
                ),
                "mean_absolute_score_contribution": float(
                    np.mean(np.abs(transform.feature_contributions[:, index]))
                ),
            }
        )
        feature_correlations[name] = {
            other: _correlation(feature_values[:, index], feature_values[:, other_index])
            for other_index, other in enumerate(difficulty.FEATURE_NAMES)
        }
        spearman_to_score[name] = feature_statistics_rows[-1]["spearman_to_difficulty_score"]
    _write_csv(
        output_dir / "difficulty_feature_statistics.csv",
        feature_statistics_rows,
        tuple(feature_statistics_rows[0]),
    )

    review_rows = _review_rows(segment_rows, transform, profile, args.seed, quality_statuses)
    _write_csv(
        output_dir / "difficulty_review_segments.csv",
        review_rows,
        tuple(review_rows[0]) if review_rows else (),
    )

    bin_counts = np.bincount(transform.difficulty_bin, minlength=profile.num_bins)
    warnings = list(profile.warnings)
    if np.count_nonzero(bin_counts) != profile.num_bins:
        warnings.append(
            f"only {int(np.count_nonzero(bin_counts))} of {profile.num_bins} difficulty bins are represented"
        )
    minimum_bin_size = max(1, int(math.floor(0.005 * segment_index.num_segments)))
    if np.any(bin_counts < minimum_bin_size):
        warnings.append(f"one or more difficulty bins contain fewer than {minimum_bin_size} segments")
    tied_fraction = 1.0 - np.unique(transform.difficulty_score).size / transform.difficulty_score.size
    if tied_fraction > 0.25:
        warnings.append(f"{tied_fraction:.3%} of segment scores are tied beyond unique-score count")
    for name, value in spearman_to_score.items():
        if value is not None and abs(value) > 0.98 and profile.effective_feature_weights[
            difficulty.FEATURE_NAMES.index(name)
        ] > 0.0:
            warnings.append(f"feature '{name}' has |Spearman correlation| > 0.98 with difficulty score")
    for index, name in enumerate(difficulty.FEATURE_NAMES):
        coverage = float(np.mean(feature_available[:, index]))
        if coverage < float(config["minimum_optional_feature_coverage"]):
            warnings.append(
                f"feature '{name}' coverage {coverage:.6f} is below the configured optional minimum"
            )
    category_summary = _group_summary(
        categories_by_segment, transform.difficulty_score, transform.difficulty_bin, profile.num_bins
    )
    for category, item in category_summary.items():
        counts = np.asarray(item["bin_counts"], dtype=np.int64)
        if counts.sum() >= 20 and counts.max() / counts.sum() >= 0.90:
            warnings.append(f"category '{category}' has at least 90% of segments in one difficulty bin")

    motion_scores = np.asarray([row["difficulty_score"] for row in motion_rows], dtype=np.float64)
    summary = {
        "schema_version": "wbt.difficulty_summary.v1",
        "mode": args.mode,
        "policy_independent": True,
        "quality_affects_score": False,
        "manifest": str(effective_manifest),
        "manifest_sha256": manifest_sha256,
        "original_manifest": str(manifest),
        "original_manifest_motion_count": len(all_keys),
        "motion_count": len(results),
        "segment_count": segment_index.num_segments,
        "segment_length_seconds": args.segment_length_seconds,
        "profile_sha256": profile_sha256,
        "difficulty_config_sha256": config_sha256,
        "pool_fingerprint": pool_fingerprint,
        "num_bins": profile.num_bins,
        "difficulty_raw": _describe(transform.difficulty_raw),
        "difficulty_score": _describe(transform.difficulty_score),
        "bin_counts": bin_counts.astype(int).tolist(),
        "bin_ratios": (bin_counts / segment_index.num_segments).tolist(),
        "motion_difficulty_score": _describe(motion_scores),
        "feature_availability": {
            name: float(np.mean(feature_available[:, index]))
            for index, name in enumerate(difficulty.FEATURE_NAMES)
        },
        "near_constant_features": list(profile.near_constant_features),
        "feature_correlation_matrix": feature_correlations,
        "feature_to_score_spearman": spearman_to_score,
        "per_category": category_summary,
        "per_source_group": _group_summary(
            sources_by_segment, transform.difficulty_score, transform.difficulty_bin, profile.num_bins
        ),
        "quality_cross_statistics": quality_cross,
        "warnings": sorted(set(warnings)),
        "git_commit": _git_commit(),
        "provisional": bool(config.get("provisional", True)),
    }
    _write_json(output_dir / "difficulty_summary.json", summary)
    print(
        f"Built intrinsic difficulty metadata: motions={len(results)}, segments={segment_index.num_segments}, "
        f"bins={bin_counts.astype(int).tolist()}, output={output_dir}"
    )


def _safe_extract(
    task: tuple[int, str, str, Mapping[str, Any]]
) -> tuple[MotionFeatureResult | None, str | None]:
    try:
        return _extract_one(task), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


if __name__ == "__main__":
    main()
