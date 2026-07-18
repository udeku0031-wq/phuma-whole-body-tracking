#!/usr/bin/env python3
"""Audit converted WBT motion .npz files for local trajectory anomalies.

This script inspects the final PHUMA -> WBT/G1 robot-state trajectories, not
the original PHUMA human motion.  It is intended for post-retargeting quality
audits: frame-local jumps, velocity inconsistency, quaternion issues, and
possible foot sliding in the converted motion library.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Iterable

import numpy as np


REQUIRED_KEYS = (
    "fps",
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
)

SLICE_SUFFIX_PATTERNS = (
    re.compile(r"(?i)(?:_chunk_\d+|_chunk\d+|-chunk-\d+)$"),
    re.compile(r"(?i)(?:_clip_\d+|_clip\d{3,}|-clip-\d+)$"),
    re.compile(r"(?i)(?:_segment_\d+|_segment\d+|-segment-\d+)$"),
    re.compile(r"(?i)(?:_part_\d+|_part\d+|-part-\d+)$"),
)

CSV_COLUMNS = (
    "path",
    "relative_path",
    "category",
    "source_file",
    "source_group",
    "num_frames",
    "fps",
    "valid",
    "flags",
    "invalid_reason",
    "score",
    "max_body_step_m",
    "body_step_p99_m",
    "body_step_spike_count",
    "body_step_spike_threshold_m",
    "body_step_peak_frame",
    "body_step_peak_body",
    "max_joint_step_rad",
    "joint_step_p99_rad",
    "joint_step_spike_count",
    "joint_step_spike_threshold_rad",
    "joint_step_peak_frame",
    "joint_step_peak_joint",
    "max_quat_step_rad",
    "quat_step_p99_rad",
    "quat_step_spike_count",
    "quat_step_spike_threshold_rad",
    "quat_step_peak_frame",
    "quat_step_peak_body",
    "max_quat_norm_error",
    "body_vel_consistency_p95_mps",
    "body_vel_consistency_max_mps",
    "joint_vel_consistency_p95_radps",
    "joint_vel_consistency_max_radps",
    "foot_slide_ratio",
    "foot_slide_frames",
    "foot_slide_max_speed_mps",
)

THRESHOLD_METRICS = (
    "max_body_step_m",
    "max_joint_step_rad",
    "max_quat_step_rad",
    "body_vel_consistency_max_mps",
    "foot_slide_ratio",
)

DEFAULT_HARD_MINIMUMS = {
    "max_body_step_m": 0.12,
    "max_joint_step_rad": 0.45,
    "max_quat_step_rad": 0.70,
    "body_vel_consistency_max_mps": 2.50,
    "foot_slide_ratio": 0.25,
}


@dataclass
class AuditRecord:
    path: str
    relative_path: str
    category: str
    source_file: str
    source_group: str
    num_frames: int = 0
    fps: float = float("nan")
    valid: bool = True
    flags: tuple[str, ...] = ()
    invalid_reason: str = ""
    score: int = 0
    max_body_step_m: float = 0.0
    body_step_p99_m: float = 0.0
    body_step_spike_count: int = 0
    body_step_spike_threshold_m: float = 0.0
    body_step_peak_frame: int = -1
    body_step_peak_body: str = ""
    max_joint_step_rad: float = 0.0
    joint_step_p99_rad: float = 0.0
    joint_step_spike_count: int = 0
    joint_step_spike_threshold_rad: float = 0.0
    joint_step_peak_frame: int = -1
    joint_step_peak_joint: str = ""
    max_quat_step_rad: float = 0.0
    quat_step_p99_rad: float = 0.0
    quat_step_spike_count: int = 0
    quat_step_spike_threshold_rad: float = 0.0
    quat_step_peak_frame: int = -1
    quat_step_peak_body: str = ""
    max_quat_norm_error: float = 0.0
    body_vel_consistency_p95_mps: float = 0.0
    body_vel_consistency_max_mps: float = 0.0
    joint_vel_consistency_p95_radps: float = 0.0
    joint_vel_consistency_max_radps: float = 0.0
    foot_slide_ratio: float = 0.0
    foot_slide_frames: int = 0
    foot_slide_max_speed_mps: float = 0.0

    def to_row(self) -> dict[str, str]:
        return {
            "path": self.path,
            "relative_path": self.relative_path,
            "category": self.category,
            "source_file": self.source_file,
            "source_group": self.source_group,
            "num_frames": str(self.num_frames),
            "fps": format_float(self.fps),
            "valid": str(self.valid).lower(),
            "flags": ";".join(self.flags),
            "invalid_reason": self.invalid_reason,
            "score": str(self.score),
            "max_body_step_m": format_float(self.max_body_step_m),
            "body_step_p99_m": format_float(self.body_step_p99_m),
            "body_step_spike_count": str(self.body_step_spike_count),
            "body_step_spike_threshold_m": format_float(self.body_step_spike_threshold_m),
            "body_step_peak_frame": str(self.body_step_peak_frame),
            "body_step_peak_body": self.body_step_peak_body,
            "max_joint_step_rad": format_float(self.max_joint_step_rad),
            "joint_step_p99_rad": format_float(self.joint_step_p99_rad),
            "joint_step_spike_count": str(self.joint_step_spike_count),
            "joint_step_spike_threshold_rad": format_float(self.joint_step_spike_threshold_rad),
            "joint_step_peak_frame": str(self.joint_step_peak_frame),
            "joint_step_peak_joint": self.joint_step_peak_joint,
            "max_quat_step_rad": format_float(self.max_quat_step_rad),
            "quat_step_p99_rad": format_float(self.quat_step_p99_rad),
            "quat_step_spike_count": str(self.quat_step_spike_count),
            "quat_step_spike_threshold_rad": format_float(self.quat_step_spike_threshold_rad),
            "quat_step_peak_frame": str(self.quat_step_peak_frame),
            "quat_step_peak_body": self.quat_step_peak_body,
            "max_quat_norm_error": format_float(self.max_quat_norm_error),
            "body_vel_consistency_p95_mps": format_float(self.body_vel_consistency_p95_mps),
            "body_vel_consistency_max_mps": format_float(self.body_vel_consistency_max_mps),
            "joint_vel_consistency_p95_radps": format_float(self.joint_vel_consistency_p95_radps),
            "joint_vel_consistency_max_radps": format_float(self.joint_vel_consistency_max_radps),
            "foot_slide_ratio": format_float(self.foot_slide_ratio),
            "foot_slide_frames": str(self.foot_slide_frames),
            "foot_slide_max_speed_mps": format_float(self.foot_slide_max_speed_mps),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit final WBT/G1 .npz trajectories converted from PHUMA."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input_dir", type=Path, help="Directory containing converted .npz files.")
    source.add_argument("--manifest", type=Path, help="Text file containing .npz paths to audit.")
    parser.add_argument("--pattern", default="*.npz", help="Glob pattern used with --input_dir.")
    parser.add_argument("--data_root", type=Path, default=Path("PHUMA_wbt_motions/g1_all"))
    parser.add_argument("--project_root", type=Path, default=Path("."))
    parser.add_argument("--output_dir", type=Path, default=Path("results/phuma_wbt_quality_audit"))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None, help="Optional deterministic sample size.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle", action="store_true", help="Shuffle before applying --limit.")
    parser.add_argument("--robust_z", type=float, default=12.0)
    parser.add_argument("--foot_contact_z", type=float, default=0.04)
    parser.add_argument("--foot_slide_speed", type=float, default=0.40)
    return parser.parse_args()


def format_float(value: float) -> str:
    if not math.isfinite(float(value)):
        return str(value)
    return f"{float(value):.8g}"


def posix_text(value: str) -> str:
    return value.replace("\\", "/").strip()


def scalar_to_str(value: object) -> str:
    arr = np.asarray(value)
    if arr.shape == ():
        return str(arr.item())
    return str(arr.squeeze().item()) if arr.size == 1 else str(arr)


def strip_known_slice_suffix(stem: str) -> str:
    for pattern in SLICE_SUFFIX_PATTERNS:
        match = pattern.search(stem)
        if match:
            return stem[: match.start()]
    return stem


def without_last_suffix(name: str) -> str:
    if "." not in name:
        return name
    return name.rsplit(".", 1)[0]


def drop_phuma_data_prefix(parts: tuple[str, ...]) -> tuple[str, ...]:
    cleaned = tuple(part for part in parts if part not in ("", "/"))
    for i in range(len(cleaned) - 1):
        if cleaned[i] == "data" and cleaned[i + 1] in {"g1", "h1", "h1_2"}:
            return cleaned[i + 2 :]
    for i, part in enumerate(cleaned):
        if part in {"g1", "h1", "h1_2"}:
            return cleaned[i + 1 :]
    return cleaned


def build_group_from_posix_path(path_text: str, strip_phuma_prefix: bool) -> str | None:
    path_text = posix_text(path_text)
    if not path_text:
        return None
    parts = PurePosixPath(path_text).parts
    if strip_phuma_prefix:
        parts = drop_phuma_data_prefix(parts)
    else:
        parts = tuple(part for part in parts if part not in ("", "/"))
    if not parts:
        return None
    parent = list(parts[:-1])
    stem = strip_known_slice_suffix(without_last_suffix(parts[-1]))
    if not stem:
        return None
    return "/".join(parent + [stem])


def relative_to_data_root(path: Path, data_root: Path) -> str:
    try:
        return path.resolve().relative_to(data_root.resolve()).as_posix()
    except ValueError:
        return path.name


def project_relative(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def category_for_path(path: Path, data_root: Path) -> str:
    try:
        rel = path.resolve().relative_to(data_root.resolve())
        return rel.parts[0] if rel.parts else "__root__"
    except ValueError:
        return "__outside_data_root__"


def normalize_source_group(source_file: str, npz_path: Path, data_root: Path) -> str:
    text = posix_text(str(source_file or ""))
    if text and text.lower() not in {"none", "nan", "null"}:
        group = build_group_from_posix_path(text, strip_phuma_prefix=True)
        if group:
            return group
    fallback = relative_to_data_root(npz_path, data_root)
    group = build_group_from_posix_path(fallback, strip_phuma_prefix=False)
    return group or strip_known_slice_suffix(npz_path.stem)


def resolve_files(args: argparse.Namespace) -> list[Path]:
    if args.input_dir is not None:
        files = sorted(args.input_dir.resolve().rglob(args.pattern))
    else:
        base = args.manifest.resolve().parent
        files = []
        with args.manifest.open("r", encoding="utf-8") as f:
            for line in f:
                text = line.strip()
                if not text or text.startswith("#"):
                    continue
                path = Path(text)
                if not path.is_absolute():
                    project_path = args.project_root / path
                    path = project_path if project_path.exists() else base / path
                files.append(path.resolve())

    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(files)
    if args.limit is not None:
        files = files[: args.limit]
    return files


def robust_spike_stats(values: np.ndarray, abs_min: float, robust_z: float) -> tuple[int, float]:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    if flat.size == 0:
        return 0, float(abs_min)
    median = float(np.median(flat))
    mad = float(np.median(np.abs(flat - median)))
    sigma = 1.4826 * mad
    threshold = max(abs_min, median + robust_z * sigma)
    return int(np.count_nonzero(flat > threshold)), float(threshold)


def finite_diff_velocity(values: np.ndarray, dt: float) -> np.ndarray:
    return np.gradient(values.astype(np.float64), dt, axis=0)


def peak_frame_and_name(values: np.ndarray, names: list[str]) -> tuple[int, str]:
    if values.size == 0:
        return -1, ""
    index = np.unravel_index(int(np.argmax(values)), values.shape)
    frame = int(index[0])
    item_index = int(index[1]) if len(index) > 1 else -1
    name = names[item_index] if 0 <= item_index < len(names) else ""
    return frame, name


def audit_one(
    path: Path,
    project_root: Path,
    data_root: Path,
    robust_z: float,
    foot_contact_z: float,
    foot_slide_speed: float,
) -> AuditRecord:
    abs_path = path.resolve()
    relative_path = project_relative(abs_path, project_root)
    category = category_for_path(abs_path, data_root)
    source_file = ""
    source_group = normalize_source_group("", abs_path, data_root)
    base = AuditRecord(
        path=abs_path.as_posix(),
        relative_path=relative_path,
        category=category,
        source_file=source_file,
        source_group=source_group,
    )

    try:
        loaded = np.load(abs_path, allow_pickle=True)
    except Exception as exc:
        base.valid = False
        base.invalid_reason = f"load_error:{type(exc).__name__}:{exc}"
        base.flags = ("invalid",)
        base.score = 100
        return base

    try:
        files = set(loaded.files)
        missing = [key for key in REQUIRED_KEYS if key not in files]
        if missing:
            base.valid = False
            base.invalid_reason = f"missing_keys:{','.join(missing)}"
            base.flags = ("invalid",)
            base.score = 100
            return base

        if "source_file" in files:
            source_file = scalar_to_str(loaded["source_file"])
        source_group = normalize_source_group(source_file, abs_path, data_root)

        fps = float(np.asarray(loaded["fps"]).squeeze())
        joint_pos = np.asarray(loaded["joint_pos"], dtype=np.float64)
        joint_vel = np.asarray(loaded["joint_vel"], dtype=np.float64)
        body_pos = np.asarray(loaded["body_pos_w"], dtype=np.float64)
        body_quat = np.asarray(loaded["body_quat_w"], dtype=np.float64)
        body_lin_vel = np.asarray(loaded["body_lin_vel_w"], dtype=np.float64)
        body_ang_vel = np.asarray(loaded["body_ang_vel_w"], dtype=np.float64)
        body_names = [str(x) for x in loaded["body_names"].tolist()] if "body_names" in files else []
        joint_names = [str(x) for x in loaded["joint_names"].tolist()] if "joint_names" in files else []

        record = AuditRecord(
            path=abs_path.as_posix(),
            relative_path=relative_path,
            category=category,
            source_file=source_file,
            source_group=source_group,
            num_frames=int(joint_pos.shape[0]) if joint_pos.ndim >= 1 else 0,
            fps=fps,
        )

        shape_errors: list[str] = []
        if not math.isfinite(fps) or fps <= 0:
            shape_errors.append(f"invalid_fps:{fps}")
        if joint_pos.ndim != 2 or joint_pos.shape[0] < 3:
            shape_errors.append(f"invalid_joint_pos_shape:{joint_pos.shape}")
        if joint_vel.shape != joint_pos.shape:
            shape_errors.append(f"joint_vel_shape:{joint_vel.shape}:expected:{joint_pos.shape}")
        for name, tensor, width in (
            ("body_pos_w", body_pos, 3),
            ("body_quat_w", body_quat, 4),
            ("body_lin_vel_w", body_lin_vel, 3),
            ("body_ang_vel_w", body_ang_vel, 3),
        ):
            if tensor.ndim != 3 or tensor.shape[0] != joint_pos.shape[0] or tensor.shape[2] != width:
                shape_errors.append(f"{name}_shape:{tensor.shape}")
        for name, tensor in (
            ("joint_pos", joint_pos),
            ("joint_vel", joint_vel),
            ("body_pos_w", body_pos),
            ("body_quat_w", body_quat),
            ("body_lin_vel_w", body_lin_vel),
            ("body_ang_vel_w", body_ang_vel),
        ):
            if tensor.size and not np.isfinite(tensor).all():
                shape_errors.append(f"nonfinite:{name}")

        if shape_errors:
            record.valid = False
            record.invalid_reason = ";".join(shape_errors)
            record.flags = ("invalid",)
            record.score = 100
            return record

        dt = 1.0 / fps
        body_step = np.linalg.norm(np.diff(body_pos, axis=0), axis=-1)
        joint_step = np.abs(np.diff(joint_pos, axis=0))
        quat_norm = np.linalg.norm(body_quat, axis=-1)
        quat_unit = body_quat / np.maximum(quat_norm[..., None], 1e-12)
        quat_dot = np.abs(np.sum(quat_unit[:-1] * quat_unit[1:], axis=-1))
        quat_step = 2.0 * np.arccos(np.clip(quat_dot, 0.0, 1.0))

        body_vel_from_pos = finite_diff_velocity(body_pos, dt)
        joint_vel_from_pos = finite_diff_velocity(joint_pos, dt)
        body_vel_error = np.linalg.norm(body_vel_from_pos - body_lin_vel, axis=-1)
        joint_vel_error = np.abs(joint_vel_from_pos - joint_vel)

        record.max_body_step_m = float(np.max(body_step))
        record.body_step_p99_m = float(np.percentile(body_step, 99.0))
        record.body_step_spike_count, record.body_step_spike_threshold_m = robust_spike_stats(
            body_step, abs_min=0.12, robust_z=robust_z
        )
        record.body_step_peak_frame, record.body_step_peak_body = peak_frame_and_name(body_step, body_names)

        record.max_joint_step_rad = float(np.max(joint_step))
        record.joint_step_p99_rad = float(np.percentile(joint_step, 99.0))
        record.joint_step_spike_count, record.joint_step_spike_threshold_rad = robust_spike_stats(
            joint_step, abs_min=0.45, robust_z=robust_z
        )
        record.joint_step_peak_frame, record.joint_step_peak_joint = peak_frame_and_name(joint_step, joint_names)

        record.max_quat_step_rad = float(np.max(quat_step))
        record.quat_step_p99_rad = float(np.percentile(quat_step, 99.0))
        record.quat_step_spike_count, record.quat_step_spike_threshold_rad = robust_spike_stats(
            quat_step, abs_min=0.70, robust_z=robust_z
        )
        record.quat_step_peak_frame, record.quat_step_peak_body = peak_frame_and_name(quat_step, body_names)

        record.max_quat_norm_error = float(np.max(np.abs(quat_norm - 1.0)))
        record.body_vel_consistency_p95_mps = float(np.percentile(body_vel_error, 95.0))
        record.body_vel_consistency_max_mps = float(np.max(body_vel_error))
        record.joint_vel_consistency_p95_radps = float(np.percentile(joint_vel_error, 95.0))
        record.joint_vel_consistency_max_radps = float(np.max(joint_vel_error))

        foot_indexes = [
            body_names.index(name)
            for name in ("left_ankle_roll_link", "right_ankle_roll_link")
            if name in body_names
        ]
        if foot_indexes:
            foot_pos = body_pos[:, foot_indexes, :]
            min_z = np.min(foot_pos[:, :, 2], axis=0, keepdims=True)
            contact = foot_pos[:, :, 2] <= min_z + foot_contact_z
            foot_speed_xy = np.linalg.norm(np.diff(foot_pos[:, :, :2], axis=0), axis=-1) * fps
            slide = contact[:-1] & contact[1:] & (foot_speed_xy > foot_slide_speed)
            record.foot_slide_frames = int(np.count_nonzero(slide))
            record.foot_slide_ratio = float(record.foot_slide_frames / max(slide.size, 1))
            record.foot_slide_max_speed_mps = float(np.max(foot_speed_xy)) if foot_speed_xy.size else 0.0

        flags: list[str] = []
        if record.max_quat_norm_error > 1e-3:
            flags.append("quat_norm_error")
        if record.body_step_spike_count:
            flags.append("body_step_local_spike")
        if record.joint_step_spike_count:
            flags.append("joint_step_local_spike")
        if record.quat_step_spike_count:
            flags.append("quat_step_local_spike")
        if record.foot_slide_ratio >= 0.25:
            flags.append("possible_foot_slide")
        record.flags = tuple(flags)
        return record
    except Exception as exc:
        base.valid = False
        base.invalid_reason = f"audit_error:{type(exc).__name__}:{exc}"
        base.flags = ("invalid",)
        base.score = 100
        return base
    finally:
        loaded.close()


def robust_dataset_threshold(values: list[float], hard_minimum: float) -> float:
    arr = np.asarray([v for v in values if math.isfinite(float(v))], dtype=np.float64)
    if arr.size == 0:
        return hard_minimum
    median = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median)))
    robust = median + 8.0 * 1.4826 * mad
    percentile = float(np.percentile(arr, 99.5))
    return max(hard_minimum, min(percentile, robust) if robust > median else percentile)


def apply_dataset_flags(records: list[AuditRecord]) -> dict[str, float]:
    valid_records = [record for record in records if record.valid]
    thresholds: dict[str, float] = {}
    for metric in THRESHOLD_METRICS:
        thresholds[metric] = robust_dataset_threshold(
            [float(getattr(record, metric)) for record in valid_records],
            DEFAULT_HARD_MINIMUMS[metric],
        )

    for record in records:
        flags = list(record.flags)
        if record.valid:
            for metric, threshold in thresholds.items():
                if float(getattr(record, metric)) > threshold:
                    flags.append(f"dataset_outlier:{metric}")
        seen: set[str] = set()
        deduped = []
        for flag in flags:
            if flag not in seen:
                seen.add(flag)
                deduped.append(flag)
        record.flags = tuple(deduped)
        record.score = score_flags(record.flags)
    return thresholds


def score_flags(flags: Iterable[str]) -> int:
    score = 0
    for flag in flags:
        if flag == "invalid":
            score += 100
        elif flag.startswith("dataset_outlier"):
            score += 20
        elif flag == "possible_foot_slide":
            score += 12
        elif flag.endswith("local_spike"):
            score += 10
        elif flag == "quat_norm_error":
            score += 25
        else:
            score += 5
    return score


def write_csv(path: Path, rows: Iterable[dict[str, str]], columns: tuple[str, ...]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize(records: list[AuditRecord], thresholds: dict[str, float], args: argparse.Namespace) -> dict[str, object]:
    flag_counter: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    category_flagged: Counter[str] = Counter()
    for record in records:
        category_counts[record.category] += 1
        if record.flags:
            category_flagged[record.category] += 1
        flag_counter.update(record.flags)

    metric_summary: dict[str, dict[str, float]] = {}
    for metric in (
        "max_body_step_m",
        "max_joint_step_rad",
        "max_quat_step_rad",
        "body_vel_consistency_max_mps",
        "foot_slide_ratio",
    ):
        values = np.asarray([float(getattr(r, metric)) for r in records if r.valid], dtype=np.float64)
        if values.size:
            metric_summary[metric] = {
                "median": float(np.median(values)),
                "p95": float(np.percentile(values, 95.0)),
                "p99": float(np.percentile(values, 99.0)),
                "max": float(np.max(values)),
            }

    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "final converted PHUMA -> WBT/G1 .npz trajectories",
        "input_dir": str(args.input_dir) if args.input_dir is not None else "",
        "manifest": str(args.manifest) if args.manifest is not None else "",
        "data_root": str(args.data_root),
        "num_files": len(records),
        "num_valid": sum(1 for record in records if record.valid),
        "num_invalid": sum(1 for record in records if not record.valid),
        "num_flagged": sum(1 for record in records if record.flags),
        "thresholds": thresholds,
        "flag_counts": dict(flag_counter),
        "category_counts": dict(category_counts),
        "category_flagged": dict(category_flagged),
        "metric_summary": metric_summary,
        "args": {
            "seed": args.seed,
            "limit": args.limit,
            "shuffle": args.shuffle,
            "workers": args.workers,
            "robust_z": args.robust_z,
            "foot_contact_z": args.foot_contact_z,
            "foot_slide_speed": args.foot_slide_speed,
        },
    }


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    data_root = args.data_root.resolve()
    files = resolve_files(args)
    if not files:
        raise SystemExit("No .npz files found to audit.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO]: Auditing {len(files)} converted WBT motion file(s).", flush=True)
    if args.workers <= 1:
        records = [
            audit_one(path, project_root, data_root, args.robust_z, args.foot_contact_z, args.foot_slide_speed)
            for path in files
        ]
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            records = list(
                executor.map(
                    lambda p: audit_one(p, project_root, data_root, args.robust_z, args.foot_contact_z, args.foot_slide_speed),
                    files,
                )
            )

    thresholds = apply_dataset_flags(records)
    records = sorted(records, key=lambda r: (-r.score, r.category, r.relative_path))

    per_motion_path = args.output_dir / "per_motion_audit.csv"
    flagged_path = args.output_dir / "flagged_motions.txt"
    top_path = args.output_dir / "top_anomalies.csv"
    category_path = args.output_dir / "category_summary.csv"
    summary_path = args.output_dir / "summary.json"

    write_csv(per_motion_path, (record.to_row() for record in records), CSV_COLUMNS)

    with flagged_path.open("w", encoding="utf-8") as f:
        for record in records:
            if record.flags:
                f.write(record.relative_path + "\n")

    write_csv(top_path, (record.to_row() for record in records[: min(500, len(records))]), CSV_COLUMNS)

    by_category: dict[str, list[AuditRecord]] = defaultdict(list)
    for record in records:
        by_category[record.category].append(record)
    category_rows = []
    for category, items in sorted(by_category.items()):
        valid_items = [item for item in items if item.valid]
        category_rows.append(
            {
                "category": category,
                "num_files": str(len(items)),
                "num_valid": str(len(valid_items)),
                "num_flagged": str(sum(1 for item in items if item.flags)),
                "flagged_ratio": format_float(sum(1 for item in items if item.flags) / max(len(items), 1)),
                "median_max_body_step_m": format_float(np.median([i.max_body_step_m for i in valid_items]) if valid_items else 0.0),
                "p99_max_body_step_m": format_float(np.percentile([i.max_body_step_m for i in valid_items], 99) if valid_items else 0.0),
                "median_max_joint_step_rad": format_float(np.median([i.max_joint_step_rad for i in valid_items]) if valid_items else 0.0),
                "p99_max_joint_step_rad": format_float(np.percentile([i.max_joint_step_rad for i in valid_items], 99) if valid_items else 0.0),
                "median_foot_slide_ratio": format_float(np.median([i.foot_slide_ratio for i in valid_items]) if valid_items else 0.0),
                "p99_foot_slide_ratio": format_float(np.percentile([i.foot_slide_ratio for i in valid_items], 99) if valid_items else 0.0),
            }
        )
    write_csv(
        category_path,
        category_rows,
        (
            "category",
            "num_files",
            "num_valid",
            "num_flagged",
            "flagged_ratio",
            "median_max_body_step_m",
            "p99_max_body_step_m",
            "median_max_joint_step_rad",
            "p99_max_joint_step_rad",
            "median_foot_slide_ratio",
            "p99_foot_slide_ratio",
        ),
    )

    summary = summarize(records, thresholds, args)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[INFO]: Wrote {per_motion_path}", flush=True)
    print(f"[INFO]: Wrote {flagged_path}", flush=True)
    print(f"[INFO]: Wrote {top_path}", flush=True)
    print(f"[INFO]: Wrote {category_path}", flush=True)
    print(f"[INFO]: Wrote {summary_path}", flush=True)
    print(
        "[SUMMARY]: "
        f"valid={summary['num_valid']}/{summary['num_files']} "
        f"flagged={summary['num_flagged']} "
        f"invalid={summary['num_invalid']}",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
