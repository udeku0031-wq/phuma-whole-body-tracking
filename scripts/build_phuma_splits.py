#!/usr/bin/env python3
"""Build fixed PHUMA train/validation/test splits for WBT training.

The split unit is the original source sequence, not the converted .npz chunk.
This prevents adjacent chunks from the same source motion from leaking across
train/validation/test.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import subprocess
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Iterable

import numpy as np


SPLITS = ("train", "validation", "test")
MANIFEST_NAMES = {
    "train": "train_pool.txt",
    "validation": "validation.txt",
    "test": "test.txt",
}
GROUP_MANIFEST_NAMES = {
    "train": "train_source_groups.txt",
    "validation": "validation_source_groups.txt",
    "test": "test_source_groups.txt",
}
REQUIRED_MOTION_KEYS = (
    "fps",
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
)
CSV_COLUMNS = (
    "path",
    "relative_path",
    "category",
    "source_file",
    "source_format",
    "source_group",
    "num_frames",
    "fps",
    "split",
    "used_fallback",
    "valid",
    "invalid_reason",
)
CHECKSUM_FILES = (
    "train_pool.txt",
    "validation.txt",
    "test.txt",
    "train_source_groups.txt",
    "validation_source_groups.txt",
    "test_source_groups.txt",
    "split_config.json",
)


SLICE_SUFFIX_PATTERNS = (
    re.compile(r"(?i)(?:_chunk_\d+|_chunk\d+|-chunk-\d+)$"),
    # PHUMA uses names such as clip1_chunk_0000. Do not strip the clip1 part.
    re.compile(r"(?i)(?:_clip_\d+|_clip\d{3,}|-clip-\d+)$"),
    re.compile(r"(?i)(?:_segment_\d+|_segment\d+|-segment-\d+)$"),
    re.compile(r"(?i)(?:_part_\d+|_part\d+|-part-\d+)$"),
)


@dataclass
class MotionRecord:
    path: str
    relative_path: str
    category: str
    source_file: str
    source_format: str
    source_group: str
    num_frames: int
    fps: float
    split: str = ""
    used_fallback: bool = False
    valid: bool = True
    invalid_reason: str = ""

    def to_csv_row(self) -> dict[str, str]:
        return {
            "path": self.path,
            "relative_path": self.relative_path,
            "category": self.category,
            "source_file": self.source_file,
            "source_format": self.source_format,
            "source_group": self.source_group,
            "num_frames": str(self.num_frames),
            "fps": f"{self.fps:g}" if math.isfinite(self.fps) else str(self.fps),
            "split": self.split,
            "used_fallback": str(bool(self.used_fallback)).lower(),
            "valid": str(bool(self.valid)).lower(),
            "invalid_reason": self.invalid_reason,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build fixed, group-aware PHUMA train/validation/test manifests."
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--data-root", type=Path, default=Path("PHUMA_wbt_motions/g1_all"))
    parser.add_argument("--output-dir", type=Path, default=Path("PHUMA_wbt_motions/manifests/splits_v1"))
    parser.add_argument(
        "--split-mode",
        choices=("auto", "official", "grouped-random"),
        default="auto",
        help="auto uses official splits if they are found, otherwise grouped-random.",
    )
    parser.add_argument("--official-train-file", type=Path, default=None)
    parser.add_argument("--official-test-file", type=Path, default=None)
    parser.add_argument("--official-unseen-file", type=Path, default=None)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--path-mode",
        choices=("relative", "absolute", "data-relative"),
        default="relative",
        help="Path format used inside generated manifests.",
    )
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def bool_from_csv(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


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


def relative_to_data_root(npz_path: Path, data_root: Path) -> str:
    try:
        return npz_path.resolve().relative_to(data_root.resolve()).as_posix()
    except ValueError:
        return npz_path.name


def normalize_source_group(source_file: str, npz_path: Path, data_root: Path) -> tuple[str, bool]:
    """Return (source_group, used_fallback).

    The normal path is derived from source_file and stripped down to the
    PHUMA category-relative source path. Fallback uses the converted .npz path
    relative to data_root and applies the same explicit slice suffix stripping.
    """

    text = posix_text(str(source_file or ""))
    if text and text.lower() not in {"none", "nan", "null"}:
        group = build_group_from_posix_path(text, strip_phuma_prefix=True)
        if group:
            return group, False

    fallback = relative_to_data_root(npz_path, data_root)
    group = build_group_from_posix_path(fallback, strip_phuma_prefix=False)
    if group:
        return group, True
    return strip_known_slice_suffix(npz_path.stem), True


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


def read_motion_metadata(npz_path: Path, project_root: Path, data_root: Path) -> MotionRecord:
    abs_path = npz_path.resolve()
    category = category_for_path(abs_path, data_root)
    relative_path = project_relative(abs_path, project_root)
    source_file = ""
    source_format = ""
    source_group, used_fallback = normalize_source_group("", abs_path, data_root)
    num_frames = 0
    fps = float("nan")
    valid = True
    reasons: list[str] = []

    try:
        loaded = np.load(abs_path, allow_pickle=True)
    except Exception as exc:
        return MotionRecord(
            path=abs_path.as_posix(),
            relative_path=relative_path,
            category=category,
            source_file="",
            source_format="",
            source_group=source_group,
            num_frames=0,
            fps=float("nan"),
            used_fallback=True,
            valid=False,
            invalid_reason=f"load_error:{type(exc).__name__}:{exc}",
        )

    try:
        files = set(getattr(loaded, "files", []))
        missing_motion_keys = [key for key in REQUIRED_MOTION_KEYS if key not in files]
        if missing_motion_keys:
            valid = False
            reasons.append(f"missing_keys:{','.join(missing_motion_keys)}")

        if "source_file" in files:
            source_file = scalar_to_str(loaded["source_file"])
        if "source_format" in files:
            source_format = scalar_to_str(loaded["source_format"])
        source_group, used_fallback = normalize_source_group(source_file, abs_path, data_root)

        if "fps" in files:
            try:
                fps = float(np.asarray(loaded["fps"]).squeeze())
                if not math.isfinite(fps) or fps <= 0:
                    valid = False
                    reasons.append(f"invalid_fps:{fps}")
            except Exception as exc:
                valid = False
                reasons.append(f"invalid_fps:{type(exc).__name__}:{exc}")
        else:
            valid = False
            reasons.append("missing_fps")

        if "joint_pos" in files:
            try:
                joint_pos = np.asarray(loaded["joint_pos"])
                if joint_pos.ndim != 2:
                    valid = False
                    reasons.append(f"invalid_joint_pos_ndim:{joint_pos.shape}")
                else:
                    num_frames = int(joint_pos.shape[0])
                    if joint_pos.shape[0] <= 0:
                        valid = False
                        reasons.append("empty_joint_pos")
                    if joint_pos.shape[1] <= 0:
                        valid = False
                        reasons.append(f"invalid_joint_dim:{joint_pos.shape[1]}")
            except Exception as exc:
                valid = False
                reasons.append(f"invalid_joint_pos:{type(exc).__name__}:{exc}")
        else:
            valid = False
            reasons.append("missing_joint_pos")
    finally:
        if hasattr(loaded, "close"):
            loaded.close()

    return MotionRecord(
        path=abs_path.as_posix(),
        relative_path=relative_path,
        category=category,
        source_file=source_file,
        source_format=source_format,
        source_group=source_group,
        num_frames=num_frames,
        fps=fps,
        used_fallback=used_fallback,
        valid=valid,
        invalid_reason=";".join(reasons),
    )


def scan_metadata(project_root: Path, data_root: Path, workers: int) -> list[MotionRecord]:
    files = sorted(data_root.resolve().rglob("*.npz"))
    if workers <= 1:
        return [read_motion_metadata(path, project_root, data_root) for path in files]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(lambda p: read_motion_metadata(p, project_root, data_root), files))


def stable_shuffle(items: list[str], seed: int, namespace: str) -> list[str]:
    derived = hashlib.sha256(f"{seed}:{namespace}".encode("utf-8")).hexdigest()
    rng = random.Random(int(derived[:16], 16))
    shuffled = list(items)
    rng.shuffle(shuffled)
    return shuffled


def allocate_split_counts(n: int, ratios: tuple[float, float, float]) -> tuple[int, int, int]:
    if n <= 0:
        return (0, 0, 0)
    if n == 1:
        return (1, 0, 0)
    if n == 2:
        return (1, 1, 0) if ratios[1] >= ratios[2] else (1, 0, 1)

    required = [1 if ratio > 0 else 0 for ratio in ratios]
    required[0] = max(required[0], 1)
    while sum(required) > n:
        idx = max(range(3), key=lambda i: required[i])
        required[idx] -= 1

    remaining = n - sum(required)
    counts = required[:]
    for _ in range(remaining):
        deficits = [ratios[i] * n - counts[i] for i in range(3)]
        idx = max(range(3), key=lambda i: (deficits[i], ratios[i], -i))
        counts[idx] += 1
    return tuple(counts)  # type: ignore[return-value]


def group_records(records: Iterable[MotionRecord]) -> dict[str, list[MotionRecord]]:
    grouped: dict[str, list[MotionRecord]] = defaultdict(list)
    for record in records:
        if record.valid:
            grouped[record.source_group].append(record)
    return dict(grouped)


def split_groups_by_category(
    records: list[MotionRecord],
    ratios: tuple[float, float, float],
    seed: int,
    allowed_groups: set[str] | None = None,
    namespace_prefix: str = "",
) -> tuple[dict[str, str], list[str]]:
    groups_by_category: dict[str, set[str]] = defaultdict(set)
    for record in records:
        if not record.valid:
            continue
        if allowed_groups is not None and record.source_group not in allowed_groups:
            continue
        groups_by_category[record.category].add(record.source_group)

    group_to_split: dict[str, str] = {}
    warnings: list[str] = []
    for category in sorted(groups_by_category):
        groups = stable_shuffle(sorted(groups_by_category[category]), seed, f"{namespace_prefix}{category}")
        counts = allocate_split_counts(len(groups), ratios)
        if len(groups) < 3:
            warnings.append(
                f"category '{category}' has only {len(groups)} source_group(s); "
                "cannot cover train/validation/test without duplicating groups"
            )
        offsets = [0, counts[0], counts[0] + counts[1], sum(counts)]
        for split_name, start, end in zip(SPLITS, offsets[:-1], offsets[1:]):
            for group in groups[start:end]:
                group_to_split[group] = split_name
    return group_to_split, warnings


def assign_splits(records: list[MotionRecord], group_to_split: dict[str, str]) -> None:
    for record in records:
        record.split = group_to_split.get(record.source_group, "") if record.valid else ""


def ratio_sum_is_valid(ratios: tuple[float, float, float]) -> bool:
    return all(ratio >= 0 for ratio in ratios) and abs(sum(ratios) - 1.0) < 1e-8


def format_manifest_path(path: Path, project_root: Path, data_root: Path, mode: str) -> str:
    path = path.resolve()
    if mode == "absolute":
        return path.as_posix()
    if mode == "data-relative":
        return path.relative_to(data_root.resolve()).as_posix()
    return path.relative_to(project_root.resolve()).as_posix()


def resolve_manifest_entry(entry: str, manifest_dir: Path, project_root: Path, data_root: Path | None = None) -> Path:
    item = Path(entry)
    if item.is_absolute():
        return item.resolve()

    candidates = [
        (project_root / item),
        (manifest_dir / item),
    ]
    if data_root is not None:
        candidates.append(data_root / item)
    candidates.append(item)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return (project_root / item).resolve()


def common_official_split_candidates(project_root: Path) -> dict[str, Path]:
    names = {
        "train": {"phuma_train.txt", "train.txt", "train_split.txt"},
        "test": {"phuma_test.txt", "test.txt", "test_split.txt"},
        "unseen": {"unseen_video.txt", "unseen.txt"},
    }
    candidates: dict[str, Path] = {}
    search_roots = [project_root / "PHUMA", project_root]
    for root in search_roots:
        if not root.exists():
            continue
        for path in root.rglob("*.txt"):
            parts = {part.lower() for part in path.parts}
            if "data" in parts and "splits" not in parts and "split" not in parts:
                continue
            lower_name = path.name.lower()
            parent_is_split = path.parent.name.lower() in {"split", "splits"}
            for split_name, split_names in names.items():
                if lower_name in split_names or (parent_is_split and split_name in lower_name):
                    candidates.setdefault(split_name, path)
    return candidates


def canonical_source_key(text: str) -> str:
    text = posix_text(text)
    if not text:
        return ""
    stemmed = without_last_suffix(text)
    return stemmed.strip("/")


def read_official_lines(path: Path) -> list[str]:
    lines: list[str] = []
    with path.open() as f:
        for line in f:
            item = line.strip()
            if item and not item.startswith("#"):
                lines.append(item)
    return lines


def build_source_lookup(records: list[MotionRecord]) -> tuple[dict[str, set[str]], set[str]]:
    source_to_groups: dict[str, set[str]] = defaultdict(set)
    groups = set()
    for record in records:
        if not record.valid:
            continue
        groups.add(record.source_group)
        if record.source_file:
            source_to_groups[canonical_source_key(record.source_file)].add(record.source_group)
        source_to_groups[canonical_source_key(record.relative_path)].add(record.source_group)
    return dict(source_to_groups), groups


def map_official_entries_to_groups(
    entries: list[str], records: list[MotionRecord], data_root: Path
) -> tuple[set[str], list[str]]:
    source_to_groups, all_groups = build_source_lookup(records)
    mapped: set[str] = set()
    failures: list[str] = []
    source_keys = list(source_to_groups)

    for entry in entries:
        entry_key = canonical_source_key(entry)
        matches = set(source_to_groups.get(entry_key, set()))
        if not matches:
            suffix_matches = [
                group
                for key in source_keys
                if key.endswith("/" + entry_key) or entry_key.endswith("/" + key)
                for group in source_to_groups[key]
            ]
            matches = set(suffix_matches)
        if not matches:
            group = build_group_from_posix_path(entry, strip_phuma_prefix=True)
            if group in all_groups:
                matches = {group}
        if len(matches) == 1:
            mapped.update(matches)
        else:
            failures.append(entry if matches else f"{entry} (unmapped)")
    return mapped, failures


def choose_split_mode(
    args: argparse.Namespace, project_root: Path, records: list[MotionRecord]
) -> tuple[str, dict[str, Path], dict[str, int], list[str]]:
    official_files = common_official_split_candidates(project_root)
    if args.official_train_file:
        official_files["train"] = args.official_train_file
    if args.official_test_file:
        official_files["test"] = args.official_test_file
    if args.official_unseen_file:
        official_files["unseen"] = args.official_unseen_file

    official_files = {key: path for key, path in official_files.items() if path.exists()}
    stats = {
        "official_train_entries": 0,
        "official_test_entries": 0,
        "official_unseen_entries": 0,
        "official_mapped_groups": 0,
        "official_mapping_failures": 0,
    }
    warnings: list[str] = []

    has_required_official = "train" in official_files and ("test" in official_files or "unseen" in official_files)
    if args.split_mode == "official" and not has_required_official:
        raise RuntimeError("official split mode requested, but no usable official train/test files were found.")
    if args.split_mode == "auto" and not has_required_official:
        warnings.append("No PHUMA official split files found; falling back to grouped-random.")
        return "grouped-random", official_files, stats, warnings
    if args.split_mode == "grouped-random":
        return "grouped-random", official_files, stats, warnings

    return "official", official_files, stats, warnings


def build_official_split(
    records: list[MotionRecord],
    official_files: dict[str, Path],
    ratios: tuple[float, float, float],
    seed: int,
    data_root: Path,
) -> tuple[dict[str, str], dict[str, int], list[str]]:
    warnings: list[str] = []
    stats = {
        "official_train_entries": 0,
        "official_test_entries": 0,
        "official_unseen_entries": 0,
        "official_mapped_groups": 0,
        "official_mapping_failures": 0,
    }

    train_entries = read_official_lines(official_files["train"])
    test_entries = read_official_lines(official_files["test"]) if "test" in official_files else []
    unseen_entries = read_official_lines(official_files["unseen"]) if "unseen" in official_files else []
    stats["official_train_entries"] = len(train_entries)
    stats["official_test_entries"] = len(test_entries)
    stats["official_unseen_entries"] = len(unseen_entries)

    train_groups, train_failures = map_official_entries_to_groups(train_entries, records, data_root)
    test_groups, test_failures = map_official_entries_to_groups(test_entries + unseen_entries, records, data_root)
    stats["official_mapped_groups"] = len(train_groups | test_groups)
    stats["official_mapping_failures"] = len(train_failures) + len(test_failures)
    if train_failures or test_failures:
        warnings.append(
            f"Official split mapping failed for {len(train_failures) + len(test_failures)} entrie(s)."
        )

    all_valid_groups = set(group_records(records))
    missing = all_valid_groups - train_groups - test_groups
    if missing:
        warnings.append(f"Official split did not cover {len(missing)} source_group(s).")
    if not train_groups or not test_groups or missing or stats["official_mapping_failures"]:
        raise RuntimeError("official split files could not be mapped reliably to the converted PHUMA dataset.")

    train_groups -= test_groups
    train_val_total = ratios[0] + ratios[1]
    val_within_train = ratios[1] / train_val_total if train_val_total > 0 else 0.0
    train_within_train = 1.0 - val_within_train
    train_val_assignment, small_warnings = split_groups_by_category(
        records,
        (train_within_train, val_within_train, 0.0),
        seed,
        allowed_groups=train_groups,
        namespace_prefix="official-train-val:",
    )
    warnings.extend(small_warnings)
    for group in test_groups:
        train_val_assignment[group] = "test"
    return train_val_assignment, stats, warnings


def manifest_paths_by_split(
    records: list[MotionRecord], project_root: Path, data_root: Path, path_mode: str
) -> dict[str, list[str]]:
    by_split: dict[str, list[str]] = {split: [] for split in SPLITS}
    for record in sorted((r for r in records if r.valid and r.split in SPLITS), key=lambda r: r.relative_path):
        by_split[record.split].append(format_manifest_path(Path(record.path), project_root, data_root, path_mode))
    return by_split


def source_groups_by_split(records: list[MotionRecord]) -> dict[str, list[str]]:
    by_split: dict[str, set[str]] = {split: set() for split in SPLITS}
    for record in records:
        if record.valid and record.split in SPLITS:
            by_split[record.split].add(record.source_group)
    return {split: sorted(groups) for split, groups in by_split.items()}


def run_integrity_checks(
    records: list[MotionRecord],
    project_root: Path,
    data_root: Path,
    manifest_dir: Path | None = None,
    manifest_entries: dict[str, list[str]] | None = None,
) -> dict[str, object]:
    valid_records = [record for record in records if record.valid]
    path_to_record = {Path(record.path).resolve(): record for record in valid_records}

    split_to_paths: dict[str, list[Path]] = {split: [] for split in SPLITS}
    unknown_manifest_paths: list[str] = []
    missing_manifest_paths: list[str] = []
    if manifest_entries is not None:
        assert manifest_dir is not None
        for split, entries in manifest_entries.items():
            for entry in entries:
                path = resolve_manifest_entry(entry, manifest_dir, project_root, data_root)
                if not path.exists():
                    missing_manifest_paths.append(entry)
                    continue
                split_to_paths[split].append(path.resolve())
                if path.resolve() not in path_to_record:
                    unknown_manifest_paths.append(entry)
        for record in records:
            record.split = ""
        for split, paths in split_to_paths.items():
            for path in paths:
                if path in path_to_record:
                    path_to_record[path].split = split
    else:
        for record in valid_records:
            if record.split in SPLITS:
                split_to_paths[record.split].append(Path(record.path).resolve())

    split_to_path_sets = {split: set(paths) for split, paths in split_to_paths.items()}
    split_to_groups: dict[str, set[str]] = {split: set() for split in SPLITS}
    for split, paths in split_to_path_sets.items():
        for path in paths:
            if path in path_to_record:
                split_to_groups[split].add(path_to_record[path].source_group)

    file_intersections = {
        "train_validation": len(split_to_path_sets["train"] & split_to_path_sets["validation"]),
        "train_test": len(split_to_path_sets["train"] & split_to_path_sets["test"]),
        "validation_test": len(split_to_path_sets["validation"] & split_to_path_sets["test"]),
    }
    group_intersections = {
        "train_validation": len(split_to_groups["train"] & split_to_groups["validation"]),
        "train_test": len(split_to_groups["train"] & split_to_groups["test"]),
        "validation_test": len(split_to_groups["validation"] & split_to_groups["test"]),
    }

    all_assigned_paths = [path for paths in split_to_paths.values() for path in paths]
    assigned_set = set(all_assigned_paths)
    valid_path_set = set(path_to_record)
    duplicate_assigned_files = len(all_assigned_paths) - len(assigned_set)
    unassigned_files = sorted(valid_path_set - assigned_set)
    extra_files = sorted(assigned_set - valid_path_set)

    split_by_group: dict[str, set[str]] = defaultdict(set)
    files_by_group: dict[str, list[str]] = defaultdict(list)
    for record in valid_records:
        if record.split:
            split_by_group[record.source_group].add(record.split)
            files_by_group[record.source_group].append(record.relative_path)
    leaked_groups = {
        group: sorted(files_by_group[group]) for group, splits in split_by_group.items() if len(splits) > 1
    }

    internal_duplicates = {
        split: len(paths) - len(set(paths)) for split, paths in split_to_paths.items()
    }
    missing_existing_paths = [
        record.relative_path for record in valid_records if record.split in SPLITS and not Path(record.path).exists()
    ]

    ok = (
        not any(file_intersections.values())
        and not any(group_intersections.values())
        and not duplicate_assigned_files
        and not unassigned_files
        and not extra_files
        and not leaked_groups
        and not any(internal_duplicates.values())
        and not missing_existing_paths
        and not missing_manifest_paths
        and not unknown_manifest_paths
    )

    return {
        "ok": ok,
        "file_intersections": file_intersections,
        "source_group_intersections": group_intersections,
        "duplicate_assigned_files": duplicate_assigned_files,
        "unassigned_files": len(unassigned_files),
        "extra_assigned_files": len(extra_files),
        "internal_manifest_duplicates": internal_duplicates,
        "missing_existing_paths": len(missing_existing_paths),
        "missing_manifest_paths": len(missing_manifest_paths),
        "unknown_manifest_paths": len(unknown_manifest_paths),
        "source_group_leak_count": len(leaked_groups),
        "source_group_leak_examples": dict(list(leaked_groups.items())[:10]),
    }


def split_ratios(counts: dict[str, int]) -> dict[str, float]:
    total = sum(counts.values())
    if total == 0:
        return {split: 0.0 for split in SPLITS}
    return {split: counts.get(split, 0) / total for split in SPLITS}


def category_distribution(records: list[MotionRecord]) -> dict[str, object]:
    categories = sorted({record.category for record in records if record.valid})
    distribution: dict[str, object] = {}
    for category in categories:
        category_records = [record for record in records if record.valid and record.category == category]
        item: dict[str, object] = {
            "total": summarize_records(category_records),
            "splits": {},
        }
        for split in SPLITS:
            item["splits"][split] = summarize_records([r for r in category_records if r.split == split])
        distribution[category] = item
    return distribution


def summarize_records(records: list[MotionRecord]) -> dict[str, float | int]:
    frames = sum(record.num_frames for record in records)
    duration = sum(record.num_frames / record.fps for record in records if record.fps > 0 and math.isfinite(record.fps))
    return {
        "files": len(records),
        "source_groups": len({record.source_group for record in records}),
        "frames": frames,
        "duration_sec": duration,
    }


def build_report(
    records: list[MotionRecord],
    split_mode_used: str,
    official_stats: dict[str, int],
    warnings: list[str],
    checks: dict[str, object],
) -> dict[str, object]:
    valid_records = [record for record in records if record.valid]
    invalid_records = [record for record in records if not record.valid]
    file_counts = Counter(record.split for record in valid_records if record.split in SPLITS)
    group_counts = {split: 0 for split in SPLITS}
    for split, groups in source_groups_by_split(valid_records).items():
        group_counts[split] = len(groups)

    return {
        "split_mode_used": split_mode_used,
        "total_files": len(records),
        "valid_files": len(valid_records),
        "invalid_files": len(invalid_records),
        "total_source_groups": len({record.source_group for record in valid_records}),
        "fallback_files": sum(1 for record in records if record.used_fallback),
        "split_file_counts": {split: file_counts.get(split, 0) for split in SPLITS},
        "split_source_group_counts": group_counts,
        "split_file_ratios": split_ratios({split: file_counts.get(split, 0) for split in SPLITS}),
        "split_source_group_ratios": split_ratios(group_counts),
        "category_distribution": category_distribution(valid_records),
        "integrity_checks": checks,
        "official_mapping": official_stats,
        "warnings": warnings,
    }


def git_commit(project_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def build_config(
    args: argparse.Namespace,
    project_root: Path,
    data_root: Path,
    output_dir: Path,
    split_mode_used: str,
    official_files: dict[str, Path],
    records: list[MotionRecord],
) -> dict[str, object]:
    return {
        "split_version": "splits_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data_root": data_root.as_posix(),
        "project_root": project_root.as_posix(),
        "output_dir": output_dir.as_posix(),
        "split_mode_requested": args.split_mode,
        "split_mode_used": split_mode_used,
        "official_split_files": {key: path.as_posix() for key, path in official_files.items()},
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "path_mode": args.path_mode,
        "total_npz_files": len(records),
        "valid_npz_files": sum(1 for record in records if record.valid),
        "invalid_npz_files": sum(1 for record in records if not record.valid),
        "normalization_rules": {
            "source_file_priority": True,
            "fallback": "converted npz path relative to data_root",
            "strip_suffixes": [
                "_chunk_0000",
                "_chunk0000",
                "-chunk-0000",
                "_clip_0000",
                "_clip0000",
                "-clip-0000",
                "_segment_0000",
                "_segment0000",
                "-segment-0000",
                "_part_0000",
                "_part0000",
                "-part-0000",
            ],
            "does_not_strip_plain_trailing_digits": True,
            "keeps_parent_directories": True,
        },
        "git_commit": git_commit(project_root),
    }


def output_exists(output_dir: Path) -> bool:
    important = ["metadata.csv", "train_pool.txt", "validation.txt", "test.txt", "split_config.json"]
    return any((output_dir / name).exists() for name in important)


def write_csv(records: list[MotionRecord], output_path: Path) -> None:
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for record in sorted(records, key=lambda item: item.relative_path):
            writer.writerow(record.to_csv_row())


def write_manifest(path: Path, lines: list[str]) -> None:
    with path.open("w") as f:
        for line in lines:
            f.write(f"{line}\n")


def write_invalid_files(records: list[MotionRecord], output_path: Path) -> None:
    invalid = [record for record in records if not record.valid]
    with output_path.open("w") as f:
        for record in sorted(invalid, key=lambda item: item.relative_path):
            f.write(f"{record.relative_path}\t{record.invalid_reason}\n")


def write_json(path: Path, data: object) -> None:
    with path.open("w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, sort_keys=True)
        f.write("\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_checksums(output_dir: Path) -> None:
    with (output_dir / "checksums.sha256").open("w") as f:
        for name in CHECKSUM_FILES:
            path = output_dir / name
            f.write(f"{sha256_file(path)}  {name}\n")


def write_readme(output_dir: Path, split_mode_used: str, report: dict[str, object]) -> None:
    file_counts = report["split_file_counts"]
    group_counts = report["split_source_group_counts"]
    text = f"""# PHUMA WBT Fixed Split v1

This directory contains the fixed PHUMA split used by the local whole_body_tracking + PHUMA project.

## Why group-aware splitting is required

PHUMA motions are often chunked from one original sequence, for example:

```text
Apink_Mr_Chu_chunk_0000.npz
Apink_Mr_Chu_chunk_0001.npz
Apink_Mr_Chu_chunk_0002.npz
```

Adjacent chunks are highly similar. Splitting each `.npz` independently would leak nearly identical motion into train,
validation, and test. This split therefore assigns all chunks from the same `source_group` to exactly one split.

## Source group

`source_group` is derived from the converted `.npz` field `source_file` when available. Paths are normalized to POSIX
format, PHUMA `data/g1` prefixes are removed, file extensions are removed, and only explicit slice suffixes such as
`_chunk_0000`, `_clip_0000`, `_segment_0000`, and `_part_0000` are stripped. Plain trailing numbers are kept.

## Split mode

Actual mode used: `{split_mode_used}`.

`official` uses PHUMA official train/test files when they can be mapped reliably. `grouped-random` uses a fixed seed and
category-stratified source-group splitting. `auto` tries official first and falls back to grouped-random.

## Files

- `metadata.csv`: one row per converted `.npz`, including category, source_group, split, validity, frame count, and fps.
- `train_pool.txt`: training pool manifest.
- `validation.txt`: validation manifest for checkpoint/model-selection experiments.
- `test.txt`: final test manifest. Do not tune on this file.
- `*_source_groups.txt`: source-group lists for each split.
- `split_report.json`: counts, category distribution, and leakage checks.
- `split_config.json`: reproducibility metadata.
- `checksums.sha256`: checksums for the split manifests/config.
- `invalid_files.txt`: invalid or corrupted files excluded from all splits.

## Counts

```text
train      files={file_counts['train']} source_groups={group_counts['train']}
validation files={file_counts['validation']} source_groups={group_counts['validation']}
test       files={file_counts['test']} source_groups={group_counts['test']}
```

## Regenerate

```bash
python scripts/build_phuma_splits.py \\
  --project-root . \\
  --data-root PHUMA_wbt_motions/g1_all \\
  --output-dir PHUMA_wbt_motions/manifests/splits_v1 \\
  --split-mode auto \\
  --train-ratio 0.8 \\
  --val-ratio 0.1 \\
  --test-ratio 0.1 \\
  --seed 42 \\
  --path-mode relative \\
  --strict \\
  --force
```

## Validate

```bash
python scripts/build_phuma_splits.py \\
  --project-root . \\
  --data-root PHUMA_wbt_motions/g1_all \\
  --output-dir PHUMA_wbt_motions/manifests/splits_v1 \\
  --validate-only \\
  --strict
```

The default refuses to overwrite an existing fixed split. Use `--force` only when intentionally replacing the split.

Future Random-6000 subsets, quality filtering, curriculum learning, direct mixed training, validation, and final testing
should all be derived from this fixed split. Existing models trained before this split may have already seen data that is
now in `test.txt`, so they should not be reported as formal held-out test results.
"""
    (output_dir / "README.md").write_text(text)


def write_outputs(
    records: list[MotionRecord],
    args: argparse.Namespace,
    project_root: Path,
    data_root: Path,
    output_dir: Path,
    config: dict[str, object],
    report: dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(records, output_dir / "metadata.csv")

    manifests = manifest_paths_by_split(records, project_root, data_root, args.path_mode)
    groups = source_groups_by_split(records)
    for split in SPLITS:
        write_manifest(output_dir / MANIFEST_NAMES[split], manifests[split])
        write_manifest(output_dir / GROUP_MANIFEST_NAMES[split], groups[split])

    write_invalid_files(records, output_dir / "invalid_files.txt")
    write_json(output_dir / "split_config.json", config)
    write_json(output_dir / "split_report.json", report)
    write_readme(output_dir, str(config["split_mode_used"]), report)
    write_checksums(output_dir)


def read_metadata_csv(path: Path, project_root: Path) -> list[MotionRecord]:
    records: list[MotionRecord] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            abs_path = Path(row.get("path", ""))
            relative_path = row.get("relative_path", "")
            if relative_path:
                candidate = (project_root / relative_path).resolve()
                if candidate.exists():
                    abs_path = candidate
            records.append(
                MotionRecord(
                    path=abs_path.resolve().as_posix(),
                    relative_path=relative_path or project_relative(abs_path, project_root),
                    category=row.get("category", ""),
                    source_file=row.get("source_file", ""),
                    source_format=row.get("source_format", ""),
                    source_group=row.get("source_group", ""),
                    num_frames=int(float(row.get("num_frames", "0") or 0)),
                    fps=float(row.get("fps", "nan") or "nan"),
                    split=row.get("split", ""),
                    used_fallback=bool_from_csv(row.get("used_fallback", "")),
                    valid=bool_from_csv(row.get("valid", "")),
                    invalid_reason=row.get("invalid_reason", ""),
                )
            )
    return records


def read_manifest_entries(output_dir: Path) -> dict[str, list[str]]:
    entries: dict[str, list[str]] = {}
    for split, name in MANIFEST_NAMES.items():
        path = output_dir / name
        if not path.exists():
            raise FileNotFoundError(f"Missing manifest: {path}")
        with path.open() as f:
            entries[split] = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    return entries


def validate_existing_split(args: argparse.Namespace, project_root: Path, data_root: Path, output_dir: Path) -> int:
    metadata_path = output_dir / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata.csv: {metadata_path}")
    records = read_metadata_csv(metadata_path, project_root)
    manifest_entries = read_manifest_entries(output_dir)
    checks = run_integrity_checks(records, project_root, data_root, output_dir, manifest_entries)
    invalid_count = sum(1 for record in records if not record.valid)
    print(json.dumps(checks, indent=2, ensure_ascii=False, sort_keys=True))
    if args.strict and invalid_count:
        print(f"[ERROR] strict mode: metadata contains {invalid_count} invalid file(s).", file=sys.stderr)
        return 1
    if not checks["ok"]:
        print("[ERROR] split validation failed.", file=sys.stderr)
        return 1
    print("[INFO] split validation passed.")
    return 0


def summarize_for_stdout(report: dict[str, object]) -> None:
    file_counts = report["split_file_counts"]
    group_counts = report["split_source_group_counts"]
    checks = report["integrity_checks"]
    print("[INFO] split generation complete")
    print(f"[INFO] split_mode_used: {report['split_mode_used']}")
    for split in SPLITS:
        print(f"[INFO] {split}: files={file_counts[split]}, source_groups={group_counts[split]}")
    print(f"[INFO] integrity ok: {checks['ok']}")
    print(f"[INFO] invalid_files: {report['invalid_files']}, fallback_files: {report['fallback_files']}")


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    data_root = (project_root / args.data_root).resolve() if not args.data_root.is_absolute() else args.data_root.resolve()
    output_dir = (
        (project_root / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir.resolve()
    )
    ratios = (args.train_ratio, args.val_ratio, args.test_ratio)

    if not ratio_sum_is_valid(ratios):
        print("[ERROR] --train-ratio + --val-ratio + --test-ratio must equal 1.0", file=sys.stderr)
        return 2
    if not data_root.exists():
        print(f"[ERROR] data root does not exist: {data_root}", file=sys.stderr)
        return 2
    if args.validate_only:
        return validate_existing_split(args, project_root, data_root, output_dir)
    if output_exists(output_dir) and not args.force:
        print(
            f"[ERROR] fixed split already exists in {output_dir}. Use --force only if you intend to replace it.",
            file=sys.stderr,
        )
        return 2
    if output_exists(output_dir) and args.force:
        print("WARNING: existing fixed dataset split is being replaced", file=sys.stderr)

    print(f"[INFO] scanning converted PHUMA motions under {data_root}")
    records = scan_metadata(project_root, data_root, args.workers)
    print(f"[INFO] scanned {len(records)} .npz file(s)")

    warnings: list[str] = []
    official_stats: dict[str, int] = {
        "official_train_entries": 0,
        "official_test_entries": 0,
        "official_unseen_entries": 0,
        "official_mapped_groups": 0,
        "official_mapping_failures": 0,
    }
    try:
        split_mode_used, official_files, mode_stats, mode_warnings = choose_split_mode(args, project_root, records)
        official_stats.update(mode_stats)
        warnings.extend(mode_warnings)
        if split_mode_used == "official":
            group_to_split, official_stats, official_warnings = build_official_split(
                records, official_files, ratios, args.seed, data_root
            )
            warnings.extend(official_warnings)
        else:
            group_to_split, split_warnings = split_groups_by_category(records, ratios, args.seed)
            warnings.extend(split_warnings)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    assign_splits(records, group_to_split)
    checks = run_integrity_checks(records, project_root, data_root)
    invalid_count = sum(1 for record in records if not record.valid)
    if not checks["ok"]:
        print("[ERROR] generated split failed integrity checks", file=sys.stderr)
        print(json.dumps(checks, indent=2, ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return 1
    if args.strict and invalid_count:
        print(f"[ERROR] strict mode: found {invalid_count} invalid file(s).", file=sys.stderr)
        output_dir.mkdir(parents=True, exist_ok=True)
        write_invalid_files(records, output_dir / "invalid_files.txt")
        return 1

    config = build_config(args, project_root, data_root, output_dir, split_mode_used, official_files, records)
    report = build_report(records, split_mode_used, official_stats, warnings, checks)
    write_outputs(records, args, project_root, data_root, output_dir, config, report)

    summarize_for_stdout(report)
    if warnings:
        print("[INFO] warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
