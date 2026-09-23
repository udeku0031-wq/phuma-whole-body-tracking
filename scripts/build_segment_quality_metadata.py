#!/usr/bin/env python3
"""Build deterministic segment-level quality metadata for an ordered Train manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import shlex
import subprocess
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
UTILS_DIR = PROJECT_ROOT / "source" / "whole_body_tracking" / "whole_body_tracking" / "utils"
# Import the pure utility files directly.  Importing the top-level
# whole_body_tracking package eagerly imports Isaac tasks, which is unnecessary
# for this offline CPU audit.
sys.path.insert(0, str(UTILS_DIR))

import quality  # noqa: E402
import quality_metadata  # noqa: E402
import sampling  # noqa: E402


OUTPUT_FILENAMES = (
    "segment_quality_metadata.csv",
    "segment_quality_metadata.npz",
    "quality_summary.json",
    "quality_config_resolved.json",
    "quality_review_segments.csv",
    "empty_eligible_motions.csv",
    "normalized_manifest.txt",
)

BASE_CSV_COLUMNS = (
    "schema_version",
    "manifest_index",
    "motion_id",
    "motion_key",
    "motion_path",
    "category",
    "category_source",
    "source_group",
    "local_segment_id",
    "global_segment_id",
    "start_frame",
    "end_frame_exclusive",
    "fps",
    "num_frames",
    "motion_num_frames",
    "quality_score",
    "quality_status",
    "hard_violation",
    "insufficient_metrics",
    "available_metric_count",
    "metric_coverage",
    "optional_metric_coverage",
    "status_reasons",
    "primary_trigger_metrics",
    "joint_position_limit_coverage",
    "joint_velocity_limit_coverage",
    "joint_limit_source",
    "joint_limit_error",
    "joint_limit_violation_ratio",
    "joint_limit_max_excess",
    "joint_limit_mean_excess",
    "joint_limit_severity",
    "velocity_consistency_error",
    "velocity_consistency_p95",
    "velocity_consistency_max",
    "velocity_consistency_severity",
    "acceleration_spike_ratio",
    "acceleration_p95",
    "acceleration_max",
    "acceleration_severity",
    "jerk_spike_ratio",
    "jerk_p95",
    "jerk_max",
    "jerk_severity",
    "ground_penetration_depth",
    "ground_penetration_frame_ratio",
    "left_foot_min_height",
    "right_foot_min_height",
    "ground_penetration_severity",
    "foot_sliding_ratio",
    "foot_sliding_speed_mean",
    "foot_sliding_speed_p95",
    "foot_sliding_speed_max",
    "foot_sliding_severity",
    "continuity_violation_ratio",
    "continuity_severity",
    "root_position_jump_max",
    "root_orientation_jump_max",
    "body_position_jump_max",
    "body_orientation_jump_max",
    "joint_position_jump_max",
)

METRIC_CSV_COLUMNS = tuple(
    f"{metric}_{suffix}"
    for metric in quality.METRIC_NAMES
    for suffix in ("raw_value", "severity", "available", "hard_violation")
)
CSV_COLUMNS = BASE_CSV_COLUMNS + METRIC_CSV_COLUMNS

REVIEW_COLUMNS = (
    "review_bucket",
    "motion_path",
    "motion_id",
    "local_segment_id",
    "global_segment_id",
    "start_frame",
    "end_frame_exclusive",
    "quality_score",
    "quality_status",
    "primary_trigger_metrics",
    "recommended_play_command",
    "recommended_video_command",
)

EMPTY_MOTION_COLUMNS = (
    "motion_id",
    "motion_key",
    "motion_path",
    "segment_count",
    "pass_count",
    "borderline_count",
    "reject_count",
    "reject_ratio",
    "eligible_start_count",
    "primary_reject_metrics",
    "status_reasons",
)


@dataclass(frozen=True)
class MotionLayout:
    manifest_index: int
    motion_key: str
    motion_path: str
    num_frames: int
    fps: float
    joint_names: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit final converted WBT/G1 trajectories and build frozen segment quality metadata."
    )
    parser.add_argument("--manifest", type=Path, required=True, help="Ordered Train-only WBT .npz manifest.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--segment-length-seconds", type=float, default=1.0)
    parser.add_argument(
        "--quality-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "quality" / "g1_segment_quality.yaml",
    )
    parser.add_argument("--max-motions", type=int, default=None, help="Audit only the stable manifest prefix.")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--device", default="cpu", help="The NumPy audit currently supports only cpu.")
    parser.add_argument("--seed", type=int, default=42, help="Stable tie-breaking seed for review rows.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--strict", action="store_true", help="Stop on the first corrupt or invalid motion.")
    parser.add_argument("--urdf-path", type=Path, default=None, help="Optional explicit G1 URDF override.")
    parser.add_argument(
        "--dataset-metadata",
        type=Path,
        default=PROJECT_ROOT / "PHUMA_wbt_motions" / "manifests" / "splits_v1" / "metadata.csv",
        help="Existing split metadata used to prove Train membership and report category/source group.",
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.device != "cpu":
        raise ValueError("Segment quality auditing is NumPy-based; --device currently must be 'cpu'.")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1.")
    if args.max_motions is not None and args.max_motions < 1:
        raise ValueError("--max-motions must be at least 1 when provided.")
    if not math.isfinite(args.segment_length_seconds) or args.segment_length_seconds <= 0.0:
        raise ValueError("--segment-length-seconds must be finite and greater than zero.")
    manifest_name = args.manifest.stem.lower()
    if re.search(r"(^|[_-])(validation|test)($|[_-])", manifest_name):
        raise ValueError("Quality metadata must be built from Train only; Validation/Test manifests are refused.")


def _prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = [output_dir / name for name in OUTPUT_FILENAMES]
    stale_ambiguous_manifest = output_dir / "audited_manifest.txt"
    if overwrite and stale_ambiguous_manifest.exists():
        stale_ambiguous_manifest.unlink()
    elif stale_ambiguous_manifest.exists():
        targets.append(stale_ambiguous_manifest)
    existing = [str(path) for path in targets if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing quality outputs; pass --overwrite after reviewing them: "
            + ", ".join(existing)
        )


def _canonical_json(data: Mapping[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _format_csv_value(value: Any) -> Any:
    if isinstance(value, (np.bool_, bool)):
        return str(bool(value)).lower()
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return "" if not math.isfinite(float(value)) else format(float(value), ".10g")
    return value


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]], columns: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: _format_csv_value(row.get(name, "")) for name in columns})


def _inspect_motion(index: int, key: str, path: str) -> MotionLayout:
    try:
        with np.load(path, allow_pickle=False) as archive:
            required = {"fps", "joint_pos", "joint_names"}
            missing = sorted(required.difference(archive.files))
            if missing:
                raise quality.MotionSchemaError(f"missing preflight fields: {missing}")
            fps_array = np.asarray(archive["fps"], dtype=np.float64)
            joint_pos = np.asarray(archive["joint_pos"])
            joint_names_array = np.asarray(archive["joint_names"])
    except Exception as exc:
        raise quality.MotionSchemaError(f"{path}: preflight failed: {type(exc).__name__}: {exc}") from exc
    fps_value = float(fps_array.reshape(-1)[0]) if fps_array.size == 1 else float("nan")
    if not math.isfinite(fps_value) or fps_value <= 0:
        raise quality.MotionSchemaError(f"{path}: invalid fps during preflight.")
    if joint_pos.ndim != 2 or joint_pos.shape[0] < 2 or joint_pos.shape[1] < 1:
        raise quality.MotionSchemaError(f"{path}: invalid joint_pos shape {joint_pos.shape} during preflight.")
    if joint_names_array.shape != (joint_pos.shape[1],):
        raise quality.MotionSchemaError(
            f"{path}: joint_names shape {joint_names_array.shape} does not match {joint_pos.shape[1]} joints."
        )
    joint_names = tuple(str(name) for name in joint_names_array.tolist())
    return MotionLayout(index, key, path, int(joint_pos.shape[0]), fps_value, joint_names)


def _project_relative(path: str) -> str:
    try:
        return Path(path).resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return Path(path).resolve().as_posix()


def _load_split_metadata(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    mapping: dict[str, dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            for key in (row.get("relative_path", ""), row.get("path", "")):
                if key:
                    mapping[Path(key).as_posix()] = row
                    mapping[str(Path(key).resolve())] = row
    return mapping


def _diagnostic_category(path: str) -> str:
    parts = Path(path).parts
    if "g1_all" in parts:
        index = parts.index("g1_all")
        if index + 1 < len(parts):
            return parts[index + 1]
    return "__unknown__"


def _provenance_for_motion(layout: MotionLayout, split_rows: Mapping[str, Mapping[str, str]]) -> dict[str, str]:
    relative = _project_relative(layout.motion_path)
    row = split_rows.get(layout.motion_key) or split_rows.get(relative) or split_rows.get(layout.motion_path)
    if row is None:
        return {
            "category": _diagnostic_category(layout.motion_path),
            "category_source": "diagnostic_path_category",
            "source_group": "",
            "split": "unknown",
        }
    return {
        "category": row.get("category", ""),
        "category_source": "splits_v1_metadata",
        "source_group": row.get("source_group", ""),
        "split": row.get("split", ""),
    }


def _resolve_urdf_path(config: Mapping[str, Any], explicit_path: Path | None) -> Path | None:
    if explicit_path is not None:
        return explicit_path.expanduser().resolve()
    configured = config.get("robot", {}).get("urdf_path")
    if not configured:
        return None
    path = Path(str(configured)).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _metric_details(segment: quality.SegmentQualityResult, metric_name: str) -> Mapping[str, Any]:
    return segment.metrics[metric_name].details


def _continuity_aliases(segment: quality.SegmentQualityResult) -> dict[str, float]:
    names = (
        "root_position_continuity",
        "root_orientation_continuity",
        "body_position_continuity",
        "body_orientation_continuity",
        "joint_position_continuity",
    )
    available = [segment.metrics[name] for name in names if segment.metrics[name].available]
    return {
        "continuity_violation_ratio": max(
            (float(result.details.get("isolated_jump_ratio", 0.0)) for result in available), default=0.0
        ),
        "continuity_severity": max((result.severity for result in available), default=0.0),
        "root_position_jump_max": segment.metrics["root_position_continuity"].details.get("jump_max", float("nan")),
        "root_orientation_jump_max": segment.metrics["root_orientation_continuity"].details.get(
            "jump_max", float("nan")
        ),
        "body_position_jump_max": segment.metrics["body_position_continuity"].details.get(
            "jump_max", float("nan")
        ),
        "body_orientation_jump_max": segment.metrics["body_orientation_continuity"].details.get(
            "jump_max", float("nan")
        ),
        "joint_position_jump_max": segment.metrics["joint_position_continuity"].details.get(
            "jump_max", float("nan")
        ),
    }


def _foot_min_height(segment: quality.SegmentQualityResult, side: str) -> float:
    values = segment.metrics["ground_penetration"].details.get("sole_min_height_by_body_m", {})
    if not isinstance(values, Mapping):
        return float("nan")
    matched = [
        float(value)
        for name, value in values.items()
        if side in str(name).lower() and math.isfinite(float(value))
    ]
    return min(matched, default=float("nan"))


def _segment_row(
    layout: MotionLayout,
    provenance: Mapping[str, str],
    audit: quality.MotionQualityAudit,
    segment: quality.SegmentQualityResult,
    global_segment_id: int,
) -> dict[str, Any]:
    triggers = sorted(
        (name for name, result in segment.metrics.items() if result.available and result.severity > 0.0),
        key=lambda name: (-segment.metrics[name].severity, name),
    )
    row: dict[str, Any] = {
        "schema_version": quality_metadata.QUALITY_METADATA_SCHEMA_VERSION,
        "manifest_index": layout.manifest_index,
        "motion_id": layout.manifest_index,
        "motion_key": layout.motion_key,
        "motion_path": _project_relative(layout.motion_path),
        "category": provenance["category"],
        "category_source": provenance["category_source"],
        "source_group": provenance["source_group"],
        "local_segment_id": segment.local_segment_id,
        "global_segment_id": global_segment_id,
        "start_frame": segment.start_frame,
        "end_frame_exclusive": segment.end_frame_exclusive,
        "fps": audit.fps,
        "num_frames": segment.end_frame_exclusive - segment.start_frame,
        "motion_num_frames": audit.num_frames,
        "quality_score": segment.quality_score,
        "quality_status": segment.quality_status,
        "hard_violation": segment.hard_violation,
        "insufficient_metrics": segment.insufficient_metrics,
        "available_metric_count": segment.available_metric_count,
        "metric_coverage": segment.metric_coverage,
        "optional_metric_coverage": segment.optional_metric_coverage,
        "status_reasons": ";".join(segment.status_reasons),
        "primary_trigger_metrics": ";".join(triggers[:5]),
        "joint_position_limit_coverage": audit.joint_position_limit_coverage,
        "joint_velocity_limit_coverage": audit.joint_velocity_limit_coverage,
        "joint_limit_source": audit.joint_limit_source or "",
        "joint_limit_error": audit.joint_limit_error or "",
    }
    for name, result in segment.metrics.items():
        row[f"{name}_raw_value"] = result.raw_value
        row[f"{name}_severity"] = result.severity
        row[f"{name}_available"] = result.available
        row[f"{name}_hard_violation"] = result.hard_violation

    joint_limit = segment.metrics["joint_position_limits"]
    velocity_consistency = segment.metrics["joint_velocity_consistency"]
    acceleration = segment.metrics["joint_acceleration_spike"]
    jerk = segment.metrics["joint_jerk_spike"]
    penetration = segment.metrics["ground_penetration"]
    sliding = segment.metrics["foot_sliding"]
    row.update(
        {
            "joint_limit_violation_ratio": joint_limit.details.get("violation_frame_ratio", float("nan")),
            "joint_limit_max_excess": joint_limit.details.get("max_excess", float("nan")),
            "joint_limit_mean_excess": joint_limit.details.get("mean_excess", float("nan")),
            "joint_limit_severity": joint_limit.severity,
            "velocity_consistency_error": velocity_consistency.raw_value,
            "velocity_consistency_p95": velocity_consistency.details.get("p95_error", float("nan")),
            "velocity_consistency_max": velocity_consistency.details.get("max_error", float("nan")),
            "velocity_consistency_severity": velocity_consistency.severity,
            "acceleration_spike_ratio": acceleration.details.get("spike_frame_ratio", float("nan")),
            "acceleration_p95": acceleration.details.get("physical_p95", float("nan")),
            "acceleration_max": acceleration.details.get("physical_max", float("nan")),
            "acceleration_severity": acceleration.severity,
            "jerk_spike_ratio": jerk.details.get("spike_frame_ratio", float("nan")),
            "jerk_p95": jerk.details.get("physical_p95", float("nan")),
            "jerk_max": jerk.details.get("physical_max", float("nan")),
            "jerk_severity": jerk.severity,
            "ground_penetration_depth": penetration.details.get("max_depth_m", float("nan")),
            "ground_penetration_frame_ratio": penetration.details.get(
                "penetration_frame_ratio", float("nan")
            ),
            "left_foot_min_height": _foot_min_height(segment, "left"),
            "right_foot_min_height": _foot_min_height(segment, "right"),
            "ground_penetration_severity": penetration.severity,
            "foot_sliding_ratio": sliding.details.get("sliding_frame_ratio", float("nan")),
            "foot_sliding_speed_mean": sliding.details.get("contact_speed_mean_mps", float("nan")),
            "foot_sliding_speed_p95": sliding.details.get("contact_speed_p95_mps", float("nan")),
            "foot_sliding_speed_max": sliding.details.get("contact_speed_max_mps", float("nan")),
            "foot_sliding_severity": sliding.severity,
        }
    )
    row.update(_continuity_aliases(segment))
    return row


def _failed_segment_row(
    layout: MotionLayout,
    provenance: Mapping[str, str],
    local_id: int,
    global_id: int,
    start: int,
    end: int,
    error: str,
) -> dict[str, Any]:
    row = {name: "" for name in CSV_COLUMNS}
    row.update(
        {
            "schema_version": quality_metadata.QUALITY_METADATA_SCHEMA_VERSION,
            "manifest_index": layout.manifest_index,
            "motion_id": layout.manifest_index,
            "motion_key": layout.motion_key,
            "motion_path": _project_relative(layout.motion_path),
            "category": provenance["category"],
            "category_source": provenance["category_source"],
            "source_group": provenance["source_group"],
            "local_segment_id": local_id,
            "global_segment_id": global_id,
            "start_frame": start,
            "end_frame_exclusive": end,
            "fps": layout.fps,
            "num_frames": end - start,
            "motion_num_frames": layout.num_frames,
            "quality_score": 0.0,
            "quality_status": "reject",
            "hard_violation": True,
            "insufficient_metrics": True,
            "available_metric_count": 0,
            "metric_coverage": 0.0,
            "optional_metric_coverage": 0.0,
            "status_reasons": "motion_audit_error",
            "primary_trigger_metrics": "schema_error",
            "joint_limit_error": error,
        }
    )
    for metric in quality.METRIC_NAMES:
        row[f"{metric}_available"] = False
        row[f"{metric}_hard_violation"] = metric == "nonfinite_values"
    return row


def _group_status_summary(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        value = str(row.get(key, ""))
        if value:
            grouped[value][str(row["quality_status"])] += 1
            grouped[value]["total"] += 1
    output: dict[str, dict[str, float | int]] = {}
    for value in sorted(grouped):
        counts = grouped[value]
        total = counts["total"]
        output[value] = {
            "segment_count": total,
            "pass_count": counts["pass"],
            "borderline_count": counts["borderline"],
            "reject_count": counts["reject"],
            "reject_ratio": counts["reject"] / total,
        }
    return output


def _per_metric_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float | int]]:
    output: dict[str, dict[str, float | int]] = {}
    count = len(rows)
    for metric in quality.METRIC_NAMES:
        available = [row for row in rows if bool(row.get(f"{metric}_available", False))]
        raw_values = [
            float(row.get(f"{metric}_raw_value", float("nan")))
            for row in available
            if math.isfinite(float(row.get(f"{metric}_raw_value", float("nan"))))
        ]
        warning_count = sum(float(row.get(f"{metric}_severity", 0.0) or 0.0) > 0.0 for row in available)
        reject_count = sum(float(row.get(f"{metric}_severity", 0.0) or 0.0) >= 1.0 for row in available)
        hard_count = sum(bool(row.get(f"{metric}_hard_violation", False)) for row in available)
        raw_array = np.asarray(raw_values, dtype=np.float64)
        output[metric] = {
            "available_count": len(available),
            "availability_ratio": len(available) / count if count else 0.0,
            "warning_count": warning_count,
            "reject_severity_count": reject_count,
            "hard_violation_count": hard_count,
            "raw_p50": float(np.percentile(raw_array, 50.0)) if raw_array.size else None,
            "raw_p90": float(np.percentile(raw_array, 90.0)) if raw_array.size else None,
            "raw_p95": float(np.percentile(raw_array, 95.0)) if raw_array.size else None,
            "raw_p99": float(np.percentile(raw_array, 99.0)) if raw_array.size else None,
            "raw_max": float(np.max(raw_array)) if raw_array.size else None,
        }
    return output


def _select_review_rows(
    rows: Sequence[Mapping[str, Any]], *, seed: int, pass_threshold: float, reject_threshold: float
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    tie_break = {int(row["global_segment_id"]): rng.random() for row in rows}
    selected_ids: set[int] = set()
    output: list[dict[str, Any]] = []

    def add(bucket: str, candidates: Sequence[Mapping[str, Any]], sort_key) -> None:
        ordered = sorted(
            candidates,
            key=lambda row: (sort_key(row), tie_break[int(row["global_segment_id"])]),
        )
        for row in ordered:
            global_id = int(row["global_segment_id"])
            if global_id in selected_ids:
                continue
            selected_ids.add(global_id)
            play_command = (
                "env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py "
                f"--motion_file {shlex.quote(str(row['motion_path']))} "
                f"--start_frame {int(row['start_frame'])} "
                f"--end_frame_exclusive {int(row['end_frame_exclusive'])}"
            )
            video_command = (
                "python scripts/render_npz_preview.py "
                f"--motion_file {shlex.quote(str(row['motion_path']))} "
                f"--start_frame {int(row['start_frame'])} "
                f"--end_frame_exclusive {int(row['end_frame_exclusive'])} "
                "--output "
                f"/tmp/wbt_npz_previews/segment_{global_id:06d}_f{int(row['start_frame'])}_"
                f"{int(row['end_frame_exclusive'])}.mp4"
            )
            output.append(
                {
                    "review_bucket": bucket,
                    "motion_path": row["motion_path"],
                    "motion_id": row["motion_id"],
                    "local_segment_id": row["local_segment_id"],
                    "global_segment_id": global_id,
                    "start_frame": row["start_frame"],
                    "end_frame_exclusive": row["end_frame_exclusive"],
                    "quality_score": row["quality_score"],
                    "quality_status": row["quality_status"],
                    "primary_trigger_metrics": row["primary_trigger_metrics"],
                    "recommended_play_command": play_command,
                    "recommended_video_command": video_command,
                }
            )
            if sum(item["review_bucket"] == bucket for item in output) >= 20:
                break

    pass_rows = [row for row in rows if row["quality_status"] == "pass"]
    reject_rows = [row for row in rows if row["quality_status"] == "reject"]
    add("highest_pass", pass_rows, lambda row: -float(row["quality_score"]))
    add("near_pass_borderline", rows, lambda row: abs(float(row["quality_score"]) - pass_threshold))
    add("near_borderline_reject", rows, lambda row: abs(float(row["quality_score"]) - reject_threshold))
    add("lowest_reject", reject_rows, lambda row: float(row["quality_score"]))
    return output


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def main() -> None:
    args = parse_args()
    _validate_args(args)
    manifest = args.manifest.expanduser().resolve()
    if not manifest.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {manifest}")
    output_dir = args.output_dir.expanduser().resolve()
    _prepare_output_dir(output_dir, args.overwrite)

    all_keys, all_paths = quality_metadata.resolve_manifest_entries(manifest, working_directory=PROJECT_ROOT)
    original_manifest_sha256 = quality_metadata.sha256_file(manifest)
    if args.max_motions is not None:
        selected_count = min(args.max_motions, len(all_keys))
        motion_keys = all_keys[:selected_count]
        motion_paths = all_paths[:selected_count]
    else:
        motion_keys = all_keys
        motion_paths = all_paths

    normalized_manifest = output_dir / "normalized_manifest.txt"
    normalized_manifest.write_text("".join(f"{key}\n" for key in motion_keys), encoding="utf-8")
    normalized_manifest_sha256 = quality_metadata.sha256_file(normalized_manifest)
    # Full-pool metadata binds to the original manifest, so training can use the
    # canonical random6000 file directly.  Prefix pilots bind to the explicit
    # normalized subset manifest because it is the actual training pool.
    effective_manifest = normalized_manifest if args.max_motions is not None else manifest
    manifest_sha256 = quality_metadata.sha256_file(effective_manifest)

    config = quality.load_quality_config(args.quality_config)
    config["segment_length_seconds"] = float(args.segment_length_seconds)
    config_canonical = _canonical_json(config)
    quality_config_sha256 = hashlib.sha256(config_canonical.encode("utf-8")).hexdigest()
    _write_json(output_dir / "quality_config_resolved.json", config)

    layouts = [
        _inspect_motion(i, key, path)
        for i, (key, path) in enumerate(zip(motion_keys, motion_paths, strict=True))
    ]
    split_rows = _load_split_metadata(args.dataset_metadata.expanduser().resolve())
    provenance = [_provenance_for_motion(layout, split_rows) for layout in layouts]
    non_train = [
        layout.motion_key
        for layout, item in zip(layouts, provenance, strict=True)
        if item["split"] not in {"train", "unknown", ""}
    ]
    if non_train:
        raise ValueError(
            "Manifest contains Validation/Test motions according to split metadata; refusing to audit: "
            + ", ".join(non_train[:10])
        )
    unknown_train_membership = [
        layout.motion_key
        for layout, item in zip(layouts, provenance, strict=True)
        if item["split"] in {"unknown", ""}
    ]
    if args.strict and unknown_train_membership:
        raise ValueError(
            "Strict quality auditing requires split metadata proving every motion belongs to Train; "
            "membership is unknown for: "
            + ", ".join(unknown_train_membership[:10])
        )

    segment_index = sampling.FixedLengthSegmentIndex(
        [layout.num_frames for layout in layouts],
        [layout.fps for layout in layouts],
        args.segment_length_seconds,
        device="cpu",
    )
    pool_fingerprint = sampling.motion_pool_fingerprint(
        motion_paths,
        [layout.num_frames for layout in layouts],
        [layout.fps for layout in layouts],
    )

    urdf_path = _resolve_urdf_path(config, args.urdf_path)
    unique_joint_orders = {layout.joint_names for layout in layouts}
    joint_limits = {
        names: quality.parse_urdf_joint_limits(urdf_path, names) for names in unique_joint_orders
    }

    tasks: list[tuple[MotionLayout, list[tuple[int, int]], quality.JointLimitTable]] = []
    for layout in layouts:
        first = int(segment_index.motion_segment_offsets[layout.manifest_index].item())
        last = int(segment_index.motion_segment_offsets[layout.manifest_index + 1].item())
        bounds = list(
            zip(
                segment_index.segment_start_frames[first:last].tolist(),
                segment_index.segment_end_frames[first:last].tolist(),
                strict=True,
            )
        )
        tasks.append((layout, bounds, joint_limits[layout.joint_names]))

    def audit_task(task):
        layout, bounds, limits = task
        try:
            return quality.audit_motion_segments(
                layout.motion_path,
                config,
                segment_bounds=bounds,
                joint_limits=limits,
            ), None
        except Exception as exc:
            return None, f"{type(exc).__name__}: {exc}"

    rows: list[dict[str, Any]] = []
    audit_errors: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for layout, provenance_item, task, result in zip(
            layouts, provenance, tasks, executor.map(audit_task, tasks), strict=True
        ):
            audit, error = result
            first_global = int(segment_index.motion_segment_offsets[layout.manifest_index].item())
            if error is not None:
                audit_errors.append(
                    {"motion_key": layout.motion_key, "motion_path": layout.motion_path, "error": error}
                )
                if args.strict:
                    raise RuntimeError(f"Quality audit failed for '{layout.motion_path}': {error}")
                for local_id, (start, end) in enumerate(task[1]):
                    rows.append(
                        _failed_segment_row(
                            layout,
                            provenance_item,
                            local_id,
                            first_global + local_id,
                            start,
                            end,
                            error,
                        )
                    )
                continue
            assert audit is not None
            for segment in audit.segment_results:
                rows.append(
                    _segment_row(
                        layout,
                        provenance_item,
                        audit,
                        segment,
                        first_global + segment.local_segment_id,
                    )
                )

    rows.sort(key=lambda row: int(row["global_segment_id"]))
    if len(rows) != segment_index.num_segments:
        raise RuntimeError(f"Audit produced {len(rows)} rows for {segment_index.num_segments} segments.")
    _write_csv(output_dir / "segment_quality_metadata.csv", rows, CSV_COLUMNS)

    status_codes = [quality_metadata.QUALITY_STATUS_TO_CODE[str(row["quality_status"])] for row in rows]
    payload = quality_metadata.metadata_npz_payload(
        segment_schema_version=sampling.SAMPLING_STATE_VERSION,
        segment_length_seconds=args.segment_length_seconds,
        manifest_sha256=manifest_sha256,
        quality_config_sha256=quality_config_sha256,
        pool_fingerprint=pool_fingerprint,
        motion_keys=motion_keys,
        motion_lengths=[layout.num_frames for layout in layouts],
        motion_fps=[layout.fps for layout in layouts],
        motion_segment_offsets=segment_index.motion_segment_offsets.tolist(),
        global_segment_id=segment_index.metadata()["global_segment_id"].tolist(),
        motion_id=segment_index.segment_motion_ids.tolist(),
        local_segment_id=segment_index.segment_local_ids.tolist(),
        start_frame=segment_index.segment_start_frames.tolist(),
        end_frame_exclusive=segment_index.segment_end_frames.tolist(),
        quality_score=[float(row["quality_score"]) for row in rows],
        quality_status=status_codes,
    )
    metadata_path = output_dir / "segment_quality_metadata.npz"
    np.savez_compressed(metadata_path, **payload)
    metadata = quality_metadata.SegmentQualityMetadata.load(metadata_path)
    metadata.validate_against(
        manifest_path=effective_manifest,
        motion_keys=motion_keys,
        motion_lengths=[layout.num_frames for layout in layouts],
        motion_fps=[layout.fps for layout in layouts],
        motion_segment_offsets=segment_index.motion_segment_offsets.tolist(),
        segment_start_frames=segment_index.segment_start_frames.tolist(),
        segment_end_frames=segment_index.segment_end_frames.tolist(),
        segment_length_seconds=args.segment_length_seconds,
        segment_schema_version=sampling.SAMPLING_STATE_VERSION,
        pool_fingerprint=pool_fingerprint,
    )

    allowed_mask = np.asarray([status != "reject" for status in (row["quality_status"] for row in rows)], dtype=bool)
    gate_index = sampling.QualityGatedStartIndex(segment_index, allowed_mask, empty_motion_policy="exclude")
    gate_summary = gate_index.summary()
    status_counts = Counter(str(row["quality_status"]) for row in rows)
    segment_count = len(rows)
    per_metric = _per_metric_summary(rows)
    per_category = _group_status_summary(rows, "category")
    per_source_group = _group_status_summary(rows, "source_group")

    warnings_list: list[str] = []
    reject_ratio = status_counts["reject"] / segment_count
    if reject_ratio > 0.20:
        warnings_list.append(f"Reject ratio is {reject_ratio:.2%}, above the 20% manual-review threshold.")
    if gate_summary["num_empty_motions"]:
        warnings_list.append(f"{gate_summary['num_empty_motions']} motion(s) have no eligible M1 start frame.")
    if unknown_train_membership:
        warnings_list.append(
            f"Train membership is unknown for {len(unknown_train_membership)} motion(s); "
            "the output is not formal-training compatible."
        )
    required_metrics = tuple(config["status"].get("required_metrics", ()))
    for metric, summary in per_metric.items():
        if summary["availability_ratio"] < 0.80:
            warnings_list.append(
                f"Metric '{metric}' availability is {float(summary['availability_ratio']):.2%}, below 80%."
            )
        if metric in required_metrics and summary["availability_ratio"] < 1.0:
            warnings_list.append(
                f"Required metric '{metric}' availability is "
                f"{float(summary['availability_ratio']):.2%}; formal metadata should be 100%."
            )
    for metric in ("ground_penetration", "foot_sliding"):
        summary = per_metric.get(metric, {})
        if float(summary.get("availability_ratio", 0.0)) < 0.99:
            warnings_list.append(
                f"Foot metric '{metric}' availability is "
                f"{float(summary.get('availability_ratio', 0.0)):.2%}; check foot body mapping and sole offsets."
            )
    for category, summary in per_category.items():
        if summary["segment_count"] >= 10 and summary["reject_ratio"] > 0.50:
            warnings_list.append(
                f"Category '{category}' reject ratio is {float(summary['reject_ratio']):.2%}."
            )
    if status_counts["reject"]:
        reject_triggers = Counter()
        for row in rows:
            if row["quality_status"] != "reject":
                continue
            for trigger in str(row["primary_trigger_metrics"]).split(";"):
                if trigger:
                    reject_triggers[trigger] += 1
        if reject_triggers:
            metric, count = reject_triggers.most_common(1)[0]
            if count / status_counts["reject"] > 0.80:
                warnings_list.append(
                    f"Metric '{metric}' appears in {count / status_counts['reject']:.2%} of rejected segments."
                )

    motion_reject: dict[int, Counter[str]] = defaultdict(Counter)
    for row in rows:
        motion_reject[int(row["motion_id"])][str(row["quality_status"])] += 1
        motion_reject[int(row["motion_id"])]["total"] += 1
    per_motion = [
        {
            "motion_id": motion_id,
            "motion_key": motion_keys[motion_id],
            "segment_count": counts["total"],
            "pass_count": counts["pass"],
            "borderline_count": counts["borderline"],
            "reject_count": counts["reject"],
            "reject_ratio": counts["reject"] / counts["total"],
        }
        for motion_id, counts in sorted(motion_reject.items())
    ]
    top_reject_motions = sorted(
        (item for item in per_motion if item["reject_count"] > 0),
        key=lambda item: (-item["reject_ratio"], -item["reject_count"], item["motion_id"]),
    )[:50]
    reject_segments = [
        {
            "motion_id": int(row["motion_id"]),
            "motion_key": str(row["motion_key"]),
            "global_segment_id": int(row["global_segment_id"]),
            "local_segment_id": int(row["local_segment_id"]),
            "start_frame": int(row["start_frame"]),
            "end_frame_exclusive": int(row["end_frame_exclusive"]),
            "quality_score": float(row["quality_score"]),
            "status_reasons": str(row["status_reasons"]),
            "primary_trigger_metrics": str(row["primary_trigger_metrics"]),
        }
        for row in rows
        if row["quality_status"] == "reject"
    ]

    empty_motion_rows: list[dict[str, Any]] = []
    for motion_id in gate_index.empty_motion_ids.detach().cpu().tolist():
        motion_rows = [row for row in rows if int(row["motion_id"]) == int(motion_id)]
        counts = Counter(str(row["quality_status"]) for row in motion_rows)
        trigger_counts: Counter[str] = Counter()
        reason_counts: Counter[str] = Counter()
        for row in motion_rows:
            if row["quality_status"] == "reject":
                for trigger in str(row["primary_trigger_metrics"]).split(";"):
                    if trigger:
                        trigger_counts[trigger] += 1
                for reason in str(row["status_reasons"]).split(";"):
                    if reason:
                        reason_counts[reason] += 1
        empty_motion_rows.append(
            {
                "motion_id": motion_id,
                "motion_key": motion_keys[motion_id],
                "motion_path": _project_relative(motion_paths[motion_id]),
                "segment_count": len(motion_rows),
                "pass_count": counts["pass"],
                "borderline_count": counts["borderline"],
                "reject_count": counts["reject"],
                "reject_ratio": counts["reject"] / len(motion_rows) if motion_rows else 0.0,
                "eligible_start_count": int(gate_index.motion_eligible_start_counts[motion_id].item()),
                "primary_reject_metrics": ";".join(
                    f"{metric}:{count}" for metric, count in trigger_counts.most_common()
                ),
                "status_reasons": ";".join(
                    f"{reason}:{count}" for reason, count in reason_counts.most_common()
                ),
            }
        )
    _write_csv(output_dir / "empty_eligible_motions.csv", empty_motion_rows, EMPTY_MOTION_COLUMNS)

    status_cfg = config["status"]
    review_rows = _select_review_rows(
        rows,
        seed=args.seed,
        pass_threshold=float(status_cfg["pass_score_threshold"]),
        reject_threshold=float(status_cfg["reject_score_threshold"]),
    )
    _write_csv(output_dir / "quality_review_segments.csv", review_rows, REVIEW_COLUMNS)

    unknown_split_count = len(unknown_train_membership)
    summary = {
        "schema_version": quality_metadata.QUALITY_METADATA_SCHEMA_VERSION,
        "quality_algorithm_schema_version": quality.QUALITY_SCHEMA_VERSION,
        "scope": "final converted PHUMA -> WBT/G1 trajectories; Train manifest only",
        "generation_timestamp": datetime.now(timezone.utc).isoformat(),
        "determinism_note": (
            "Scores, statuses, ordered arrays, and review selection are deterministic for identical inputs/config; "
            "generation_timestamp and NPZ ZIP container bytes are excluded."
        ),
        "git_commit": _git_commit(),
        "input_manifest": str(manifest),
        "input_manifest_sha256": original_manifest_sha256,
        "effective_manifest": str(effective_manifest),
        "manifest_sha256": manifest_sha256,
        "normalized_manifest": str(normalized_manifest),
        "normalized_manifest_sha256": normalized_manifest_sha256,
        "input_manifest_motion_count": len(all_keys),
        "manifest_motion_count": len(layouts),
        "manifest_order_preserved": True,
        "selected_pool_preserved_exactly": motion_keys == all_keys[: len(motion_keys)],
        "quality_filter_removed_motion_count": 0,
        "normalized_manifest_preserves_input_pool": args.max_motions is None and motion_keys == all_keys,
        "normalized_manifest_pool_note": (
            "The input manifest is the effective training pool; normalized_manifest.txt is an unfiltered "
            "path-normalized copy for inspection."
            if args.max_motions is None
            else "normalized_manifest.txt is the ordered input-manifest prefix selected by --max-motions; "
            "it is a pilot subset, not a quality-filtered pool."
        ),
        "max_motions": args.max_motions,
        "quality_config": str(args.quality_config.expanduser().resolve()),
        "quality_config_sha256": quality_config_sha256,
        "quality_config_provisional": bool(config.get("provisional", False)),
        "quality_config_status": config.get("status", {}),
        "ground_config": config.get("ground", {}),
        "urdf_path": str(urdf_path) if urdf_path is not None else None,
        "urdf_sha256": (
            quality_metadata.sha256_file(urdf_path)
            if urdf_path is not None and urdf_path.is_file()
            else None
        ),
        "segment_length_seconds": args.segment_length_seconds,
        "segment_schema_version": sampling.SAMPLING_STATE_VERSION,
        "motion_count": len(layouts),
        "segment_count": segment_count,
        "pass_count": status_counts["pass"],
        "pass_ratio": status_counts["pass"] / segment_count,
        "borderline_count": status_counts["borderline"],
        "borderline_ratio": status_counts["borderline"] / segment_count,
        "reject_count": status_counts["reject"],
        "reject_ratio": reject_ratio,
        "empty_eligible_motion_count": gate_summary["num_empty_motions"],
        "excluded_motion_count_if_quality_gate_exclude": gate_summary["num_excluded_motions"],
        "effective_motion_count_if_quality_gate_exclude": gate_summary["num_eligible_motions"],
        "eligible_motion_ratio_if_quality_gate_exclude": gate_summary["eligible_motion_fraction"],
        "eligible_start_frame_count": gate_summary["num_eligible_start_frames"],
        "eligible_start_fraction": gate_summary["eligible_start_fraction"],
        "metadata_npz_sha256": metadata.metadata_sha256,
        "metadata_match_ok": True,
        "training_compatible": not audit_errors and not unknown_train_membership,
        "training_compatibility_note": (
            "Metadata records data facts only. Empty eligible motions are not a build failure; "
            "quality-gated training decides whether to error or exclude them via empty_motion_policy."
        ),
        "audit_errors": audit_errors,
        "unknown_split_membership_count": unknown_split_count,
        "per_metric": per_metric,
        "per_category": per_category,
        "category_statistic_source": (
            "splits_v1 metadata when available; otherwise explicitly labeled diagnostic_path_category"
        ),
        "per_source_group": per_source_group,
        "per_motion": per_motion,
        "top_reject_motions": top_reject_motions,
        "empty_eligible_motions": empty_motion_rows,
        "reject_segments": reject_segments,
        "warnings": warnings_list,
        "implemented_metrics": list(quality.METRIC_NAMES),
        "unavailable_extension_metrics": [
            "self_collision",
            "theoretical_torque",
            "power",
            "dynamic_residual",
        ],
    }
    _write_json(output_dir / "quality_summary.json", summary)

    print(
        f"[INFO]: Audited {len(layouts)} motion(s), {segment_count} segment(s): "
        f"pass={status_counts['pass']}, borderline={status_counts['borderline']}, "
        f"reject={status_counts['reject']}, empty_motions={gate_summary['num_empty_motions']}"
    )
    print(f"[INFO]: Metadata: {metadata_path}")
    if args.max_motions is not None:
        print(f"[INFO]: Use the matching normalized prefix manifest for this pilot: {effective_manifest}")
    for warning in warnings_list:
        print(f"[WARNING]: {warning}")


if __name__ == "__main__":
    main()
