#!/usr/bin/env python3
"""Build fixed nested random training subsets from the PHUMA train pool."""

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
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence


DEFAULT_DATA_ROOT = Path("PHUMA_wbt_motions/g1_all")
DEFAULT_TRAIN_POOL = Path("PHUMA_wbt_motions/manifests/splits_v1/train_pool.txt")
DEFAULT_METADATA = Path("PHUMA_wbt_motions/manifests/splits_v1/metadata.csv")
DEFAULT_VALIDATION = Path("PHUMA_wbt_motions/manifests/splits_v1/validation.txt")
DEFAULT_TEST = Path("PHUMA_wbt_motions/manifests/splits_v1/test.txt")
DEFAULT_OUTPUT_DIR = Path("PHUMA_wbt_motions/manifests/experiments/random_seed42")
DEFAULT_SIZES = (3000, 6000, 12000)
SAMPLING_METHOD = "uniform_random_file_sampling"
SAMPLING_NOTES = (
    "No category balancing",
    "No quality filtering",
    "No source-group deduplication",
    "No difficulty weighting",
)
SAMPLING_METADATA_COLUMNS = (
    "random_rank",
    "path",
    "category",
    "source_group",
    "num_frames",
    "fps",
    "duration_sec",
    "in_random3000",
    "in_random6000",
    "in_random12000",
)
REQUIRED_METADATA_FIELDS = ("category", "source_group", "num_frames", "fps")
CHECKSUM_REQUIRED_NAMES = (
    "random_order_seed{seed}.txt",
    "sampling_config.json",
)
SLICE_SUFFIX_PATTERNS = (
    re.compile(r"(?i)(?:_chunk_\d+|_chunk\d+|-chunk-\d+)$"),
    re.compile(r"(?i)(?:_clip_\d+|_clip\d{3,}|-clip-\d+)$"),
    re.compile(r"(?i)(?:_segment_\d+|_segment\d+|-segment-\d+)$"),
    re.compile(r"(?i)(?:_part_\d+|_part\d+|-part-\d+)$"),
)


@dataclass(frozen=True)
class MotionEntry:
    raw_path: str
    abs_path: Path
    output_path: str
    category: str
    source_group: str
    num_frames: int
    fps: float
    exists: bool
    metadata_matched: bool

    @property
    def duration_sec(self) -> float:
        if self.num_frames <= 0 or not math.isfinite(self.fps) or self.fps <= 0:
            return 0.0
        return float(self.num_frames) / float(self.fps)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build fixed nested random PHUMA train subsets from splits_v1/train_pool.txt."
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--train-pool", type=Path, default=DEFAULT_TRAIN_POOL)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--validation-manifest", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--test-manifest", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sizes", type=str, default=",".join(str(size) for size in DEFAULT_SIZES))
    parser.add_argument(
        "--path-mode",
        choices=("relative", "absolute", "data-relative"),
        default="relative",
        help="Path format used inside generated random manifests.",
    )
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args(argv)


def posix_text(value: object) -> str:
    return str(value).replace("\\", "/").strip()


def clean_manifest_line(line: str) -> str:
    return posix_text(line)


def abs_key(path: Path) -> str:
    return path.resolve(strict=False).as_posix()


def resolve_cli_path(project_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else project_root / path


def parse_sizes(value: str) -> list[int]:
    raw_parts = [part.strip() for part in value.split(",") if part.strip()]
    if not raw_parts:
        raise ValueError("--sizes must contain at least one positive integer.")
    sizes: list[int] = []
    for part in raw_parts:
        try:
            size = int(part)
        except ValueError as exc:
            raise ValueError(f"--sizes contains a non-integer value: {part!r}") from exc
        if size <= 0:
            raise ValueError(f"--sizes values must be positive, got {size}.")
        sizes.append(size)
    if len(set(sizes)) != len(sizes):
        raise ValueError(f"--sizes contains duplicate values: {value}")
    return sorted(sizes)


def read_manifest(path: Path) -> list[str]:
    entries: list[str] = []
    with path.open() as f:
        for line in f:
            item = clean_manifest_line(line)
            if not item or item.startswith("#"):
                continue
            entries.append(item)
    return entries


def write_lines(path: Path, entries: Iterable[str]) -> None:
    path.write_text("".join(f"{entry}\n" for entry in entries))


def strip_known_slice_suffix(stem: str) -> str:
    for pattern in SLICE_SUFFIX_PATTERNS:
        match = pattern.search(stem)
        if match:
            return stem[: match.start()]
    return stem


def derive_category_and_group(abs_path: Path, data_root: Path) -> tuple[str, str]:
    try:
        rel = abs_path.resolve(strict=False).relative_to(data_root.resolve(strict=False))
        category = rel.parts[0] if rel.parts else "__root__"
        parent = list(rel.parts[:-1])
        stem = strip_known_slice_suffix(abs_path.stem)
        source_group = "/".join(parent + [stem]) if parent else stem
        return category, source_group
    except ValueError:
        return "__outside_data_root__", strip_known_slice_suffix(abs_path.stem)


def resolve_manifest_entry(entry: str, manifest_path: Path, project_root: Path, data_root: Path) -> Path:
    text = posix_text(entry)
    path = Path(text)
    if path.is_absolute():
        return path.resolve(strict=False)

    manifest_dir = manifest_path.parent
    candidates: list[Path] = []
    if text.startswith("PHUMA_wbt_motions/") or text.startswith("./PHUMA_wbt_motions/"):
        candidates.append(project_root / text)
    candidates.extend(
        [
            manifest_dir / text,
            project_root / text,
            data_root / text,
            Path.cwd() / text,
        ]
    )
    seen: set[str] = set()
    unique_candidates: list[Path] = []
    for candidate in candidates:
        key = candidate.resolve(strict=False).as_posix()
        if key not in seen:
            seen.add(key)
            unique_candidates.append(candidate)
    for candidate in unique_candidates:
        if candidate.exists():
            return candidate.resolve(strict=False)
    return unique_candidates[0].resolve(strict=False)


def emit_manifest_path(abs_path: Path, project_root: Path, data_root: Path, path_mode: str) -> str:
    resolved = abs_path.resolve(strict=False)
    if path_mode == "absolute":
        return resolved.as_posix()
    if path_mode == "data-relative":
        try:
            return resolved.relative_to(data_root.resolve(strict=False)).as_posix()
        except ValueError:
            return resolved.as_posix()
    try:
        return resolved.relative_to(project_root.resolve(strict=False)).as_posix()
    except ValueError:
        return resolved.as_posix()


def metadata_path_keys(abs_path: Path, project_root: Path, data_root: Path) -> list[str]:
    resolved = abs_path.resolve(strict=False)
    keys = [resolved.as_posix()]
    try:
        keys.append(resolved.relative_to(project_root.resolve(strict=False)).as_posix())
    except ValueError:
        pass
    try:
        keys.append(resolved.relative_to(data_root.resolve(strict=False)).as_posix())
    except ValueError:
        pass
    return keys


def add_metadata_key(index: dict[str, dict[str, str]], key: str, row: dict[str, str]) -> None:
    text = posix_text(key)
    if not text:
        return
    index.setdefault(text, row)
    if text.startswith("./"):
        index.setdefault(text[2:], row)


def load_metadata_index(
    metadata_path: Path, project_root: Path, data_root: Path
) -> tuple[dict[str, dict[str, str]], list[str], bool]:
    warnings: list[str] = []
    index: dict[str, dict[str, str]] = {}
    if not metadata_path.exists():
        warnings.append(f"metadata file is missing: {metadata_path}")
        return index, warnings, False

    with metadata_path.open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or ())
        if "path" not in fieldnames and "relative_path" not in fieldnames:
            warnings.append("metadata.csv must include at least one of path or relative_path.")
            return index, warnings, False
        missing = [field for field in REQUIRED_METADATA_FIELDS if field not in fieldnames]
        if missing:
            warnings.append(f"metadata.csv is missing required fields: {missing}")
            return index, warnings, False

        for row in reader:
            for column in ("path", "relative_path"):
                if not row.get(column):
                    continue
                raw = posix_text(row[column])
                add_metadata_key(index, raw, row)
                row_abs = resolve_manifest_entry(raw, metadata_path, project_root, data_root)
                for key in metadata_path_keys(row_abs, project_root, data_root):
                    add_metadata_key(index, key, row)
    return index, warnings, True


def parse_int(value: object, default: int = 0) -> int:
    try:
        if value is None or str(value).strip() == "":
            return default
        return int(float(str(value)))
    except ValueError:
        return default


def parse_float(value: object, default: float = float("nan")) -> float:
    try:
        if value is None or str(value).strip() == "":
            return default
        return float(str(value))
    except ValueError:
        return default


def scan_npz_record(abs_path: Path, data_root: Path) -> tuple[str, str, int, float]:
    category, source_group = derive_category_and_group(abs_path, data_root)
    num_frames = 0
    fps = float("nan")
    if abs_path.exists():
        try:
            import numpy as np

            loaded = np.load(abs_path, allow_pickle=True)
            try:
                if hasattr(loaded, "files"):
                    if "joint_pos" in loaded.files:
                        num_frames = int(np.asarray(loaded["joint_pos"]).shape[0])
                    if "fps" in loaded.files:
                        fps = float(np.asarray(loaded["fps"]).squeeze())
            finally:
                if hasattr(loaded, "close"):
                    loaded.close()
        except Exception:
            pass
    return category, source_group, num_frames, fps


def metadata_for_path(
    raw_path: str,
    abs_path: Path,
    metadata_index: dict[str, dict[str, str]],
    metadata_ok: bool,
    project_root: Path,
    data_root: Path,
) -> tuple[str, str, int, float, bool]:
    row = None
    lookup_keys = [posix_text(raw_path), *metadata_path_keys(abs_path, project_root, data_root)]
    for key in lookup_keys:
        row = metadata_index.get(key)
        if row is not None:
            break
    if row is None:
        category, source_group, num_frames, fps = scan_npz_record(abs_path, data_root) if not metadata_ok else (
            *derive_category_and_group(abs_path, data_root),
            0,
            float("nan"),
        )
        return category, source_group, num_frames, fps, False
    return (
        str(row.get("category", "")),
        str(row.get("source_group", "")),
        parse_int(row.get("num_frames")),
        parse_float(row.get("fps")),
        True,
    )


def entries_to_records(
    entries: Sequence[str],
    manifest_path: Path,
    project_root: Path,
    data_root: Path,
    path_mode: str,
    metadata_index: dict[str, dict[str, str]],
    metadata_ok: bool,
) -> list[MotionEntry]:
    records: list[MotionEntry] = []
    for raw_path in entries:
        abs_path = resolve_manifest_entry(raw_path, manifest_path, project_root, data_root)
        category, source_group, num_frames, fps, metadata_matched = metadata_for_path(
            raw_path, abs_path, metadata_index, metadata_ok, project_root, data_root
        )
        records.append(
            MotionEntry(
                raw_path=raw_path,
                abs_path=abs_path,
                output_path=emit_manifest_path(abs_path, project_root, data_root, path_mode),
                category=category,
                source_group=source_group,
                num_frames=num_frames,
                fps=fps,
                exists=abs_path.exists(),
                metadata_matched=metadata_matched,
            )
        )
    return records


def duplicate_count(values: Iterable[str]) -> int:
    counts = Counter(values)
    return sum(count - 1 for count in counts.values() if count > 1)


def duplicate_examples(values: Iterable[str], limit: int = 10) -> list[str]:
    counts = Counter(values)
    return sorted([value for value, count in counts.items() if count > 1])[:limit]


def summarize_records(records: Sequence[MotionEntry]) -> dict[str, object]:
    category_counts = Counter(record.category for record in records)
    source_group_counts = Counter(record.source_group for record in records)
    total_frames = sum(record.num_frames for record in records)
    total_duration = sum(record.duration_sec for record in records)
    total_files = len(records)
    return {
        "files": total_files,
        "unique_source_groups": len(source_group_counts),
        "total_frames": total_frames,
        "total_duration_sec": round(total_duration, 6),
        "category_file_counts": dict(sorted(category_counts.items())),
        "category_file_ratios": {
            category: round(count / total_files, 8) if total_files else 0.0
            for category, count in sorted(category_counts.items())
        },
        "source_group_chunk_counts": dict(sorted(source_group_counts.items())),
        "max_source_group_chunk_count": max(source_group_counts.values(), default=0),
    }


def record_abs_keys(records: Sequence[MotionEntry]) -> list[str]:
    return [abs_key(record.abs_path) for record in records]


def record_group_set(records: Sequence[MotionEntry]) -> set[str]:
    return {record.source_group for record in records if record.source_group}


def intersection_examples(left: set[str], right: set[str], limit: int = 10) -> list[str]:
    return sorted(left.intersection(right))[:limit]


def read_sampling_metadata_duplicate_count(path: Path) -> tuple[int, list[str], bool]:
    if not path.exists():
        return 0, [], False
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if "path" not in (reader.fieldnames or []):
            return 0, [], False
        paths = [posix_text(row.get("path", "")) for row in reader if row.get("path")]
    return duplicate_count(paths), duplicate_examples(paths), True


def build_report(
    *,
    seed: int,
    sizes: Sequence[int],
    pool_records: Sequence[MotionEntry],
    order_records: Sequence[MotionEntry],
    subset_records: dict[int, list[MotionEntry]],
    validation_records: Sequence[MotionEntry],
    test_records: Sequence[MotionEntry],
    warnings: Sequence[str],
    sampling_metadata_path: Path | None = None,
) -> dict[str, object]:
    pool_keys = set(record_abs_keys(pool_records))
    order_keys = record_abs_keys(order_records)
    validation_keys = set(record_abs_keys(validation_records))
    test_keys = set(record_abs_keys(test_records))
    validation_groups = record_group_set(validation_records)
    test_groups = record_group_set(test_records)

    nested_checks: dict[str, object] = {}
    previous_size: int | None = None
    previous_keys: set[str] | None = None
    for size in sizes:
        subset_keys_list = record_abs_keys(subset_records[size])
        subset_keys = set(subset_keys_list)
        nested_checks[f"random{size}_matches_order_prefix"] = subset_keys_list == order_keys[:size]
        if previous_size is not None and previous_keys is not None:
            nested_checks[f"random{previous_size}_strict_subset_of_random{size}"] = (
                previous_keys.issubset(subset_keys) and previous_keys != subset_keys
            )
        previous_size = size
        previous_keys = subset_keys

    file_intersections: dict[str, dict[str, object]] = {}
    source_group_intersections: dict[str, dict[str, object]] = {}
    subset_duplicate_counts: dict[str, int] = {}
    subset_outside_counts: dict[str, int] = {}
    subset_missing_counts: dict[str, int] = {}
    subset_metadata_failed_counts: dict[str, int] = {}
    for size, records in subset_records.items():
        name = f"random{size}"
        keys = record_abs_keys(records)
        key_set = set(keys)
        group_set = record_group_set(records)
        subset_duplicate_counts[name] = duplicate_count(keys)
        subset_outside_counts[name] = len(key_set.difference(pool_keys))
        subset_missing_counts[name] = sum(1 for record in records if not record.exists)
        subset_metadata_failed_counts[name] = sum(1 for record in records if not record.metadata_matched)
        file_intersections[name] = {
            "validation": len(key_set.intersection(validation_keys)),
            "validation_examples": intersection_examples(key_set, validation_keys),
            "test": len(key_set.intersection(test_keys)),
            "test_examples": intersection_examples(key_set, test_keys),
        }
        source_group_intersections[name] = {
            "validation": len(group_set.intersection(validation_groups)),
            "validation_examples": intersection_examples(group_set, validation_groups),
            "test": len(group_set.intersection(test_groups)),
            "test_examples": intersection_examples(group_set, test_groups),
        }

    sampling_metadata_duplicate_count = 0
    sampling_metadata_duplicate_examples: list[str] = []
    sampling_metadata_present = sampling_metadata_path is None
    if sampling_metadata_path is not None:
        (
            sampling_metadata_duplicate_count,
            sampling_metadata_duplicate_examples,
            sampling_metadata_present,
        ) = read_sampling_metadata_duplicate_count(sampling_metadata_path)

    all_records = list(pool_records) + list(order_records) + [
        record for records in subset_records.values() for record in records
    ]
    integrity_checks = {
        "train_pool_duplicate_paths_count": duplicate_count(record_abs_keys(pool_records)),
        "train_pool_duplicate_examples": duplicate_examples(record_abs_keys(pool_records)),
        "random_order_duplicate_paths_count": duplicate_count(order_keys),
        "random_order_duplicate_examples": duplicate_examples(order_keys),
        "random_order_contains_train_pool": set(order_keys) == pool_keys and len(order_keys) == len(pool_records),
        "random_order_outside_train_pool_count": len(set(order_keys).difference(pool_keys)),
        "subset_duplicate_paths_count": subset_duplicate_counts,
        "subset_outside_train_pool_count": subset_outside_counts,
        "validation_file_intersections": {
            name: data["validation"] for name, data in file_intersections.items()
        },
        "test_file_intersections": {name: data["test"] for name, data in file_intersections.items()},
        "validation_source_group_intersections": {
            name: data["validation"] for name, data in source_group_intersections.items()
        },
        "test_source_group_intersections": {
            name: data["test"] for name, data in source_group_intersections.items()
        },
        "file_intersection_details": file_intersections,
        "source_group_intersection_details": source_group_intersections,
        "missing_files_count": sum(1 for record in all_records if not record.exists),
        "missing_files_examples": [record.output_path for record in all_records if not record.exists][:10],
        "metadata_match_failed_count": sum(1 for record in all_records if not record.metadata_matched),
        "metadata_match_failed_examples": [
            record.output_path for record in all_records if not record.metadata_matched
        ][:10],
        "subset_missing_files_count": subset_missing_counts,
        "subset_metadata_match_failed_count": subset_metadata_failed_counts,
        "sampling_metadata_present": sampling_metadata_present,
        "sampling_metadata_duplicate_paths_count": sampling_metadata_duplicate_count,
        "sampling_metadata_duplicate_examples": sampling_metadata_duplicate_examples,
    }

    expected_file_counts = {f"random{size}": size for size in sizes}
    actual_file_counts = {f"random{size}": len(records) for size, records in subset_records.items()}
    return {
        "train_pool_file_count": len(pool_records),
        "train_pool_unique_source_group_count": len(record_group_set(pool_records)),
        "random_seed": seed,
        "sampling_method": SAMPLING_METHOD,
        "requested_sizes": list(sizes),
        "expected_subset_file_counts": expected_file_counts,
        "actual_subset_file_counts": actual_file_counts,
        "train_pool_summary": summarize_records(pool_records),
        "subsets": {f"random{size}": summarize_records(records) for size, records in subset_records.items()},
        "nested_checks": nested_checks,
        "integrity_checks": integrity_checks,
        "warnings": list(warnings),
    }


def integrity_errors(report: dict[str, object], strict_metadata: bool) -> list[str]:
    errors: list[str] = []
    expected_counts = report["expected_subset_file_counts"]
    actual_counts = report["actual_subset_file_counts"]
    if expected_counts != actual_counts:
        errors.append(f"subset file counts mismatch: expected {expected_counts}, got {actual_counts}")

    nested_checks = report["nested_checks"]
    for name, ok in nested_checks.items():
        if not ok:
            errors.append(f"nested check failed: {name}")

    checks = report["integrity_checks"]
    simple_zero_checks = (
        "train_pool_duplicate_paths_count",
        "random_order_duplicate_paths_count",
        "random_order_outside_train_pool_count",
        "missing_files_count",
        "sampling_metadata_duplicate_paths_count",
    )
    for name in simple_zero_checks:
        if checks[name] != 0:
            errors.append(f"{name} = {checks[name]}")
    if not checks["random_order_contains_train_pool"]:
        errors.append("random_order_seed file does not contain exactly the Train Pool paths.")
    if not checks["sampling_metadata_present"]:
        errors.append("sampling_metadata.csv is missing or lacks a path column.")

    for name in (
        "subset_duplicate_paths_count",
        "subset_outside_train_pool_count",
        "validation_file_intersections",
        "test_file_intersections",
        "validation_source_group_intersections",
        "test_source_group_intersections",
    ):
        bad = {key: value for key, value in checks[name].items() if value != 0}
        if bad:
            errors.append(f"{name} has non-zero values: {bad}")

    if strict_metadata and checks["metadata_match_failed_count"] != 0:
        errors.append(f"metadata_match_failed_count = {checks['metadata_match_failed_count']}")
    return errors


def git_commit(project_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return "unknown"
    return result.stdout.strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_sampling_config(
    *,
    project_root: Path,
    train_pool_path: Path,
    output_dir: Path,
    seed: int,
    sizes: Sequence[int],
    path_mode: str,
    pool_size: int,
) -> dict[str, object]:
    split_config_path = train_pool_path.parent / "split_config.json"
    split_config_checksum = sha256_file(split_config_path) if split_config_path.exists() else None
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "project_root": project_root.as_posix(),
        "train_pool_manifest": train_pool_path.as_posix(),
        "output_dir": output_dir.as_posix(),
        "split_version": train_pool_path.parent.name,
        "split_config_checksum": split_config_checksum,
        "seed": seed,
        "requested_sizes": list(sizes),
        "train_pool_size": pool_size,
        "path_mode": path_mode,
        "sampling_method": SAMPLING_METHOD,
        "sampling_notes": list(SAMPLING_NOTES),
        "git_commit": git_commit(project_root),
    }


def write_sampling_metadata(path: Path, ordered_records: Sequence[MotionEntry], sizes: Sequence[int]) -> None:
    subset_members = {size: set(record_abs_keys(ordered_records[:size])) for size in sizes}
    dynamic_columns = tuple(f"in_random{size}" for size in sizes)
    columns = (
        "random_rank",
        "path",
        "category",
        "source_group",
        "num_frames",
        "fps",
        "duration_sec",
        *dynamic_columns,
    )
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for rank, record in enumerate(ordered_records, start=1):
            key = abs_key(record.abs_path)
            row = {
                "random_rank": rank,
                "path": record.output_path,
                "category": record.category,
                "source_group": record.source_group,
                "num_frames": record.num_frames,
                "fps": f"{record.fps:g}" if math.isfinite(record.fps) else "",
                "duration_sec": f"{record.duration_sec:.6f}",
            }
            for size in sizes:
                row[f"in_random{size}"] = str(key in subset_members[size]).lower()
            writer.writerow(row)


def write_json(path: Path, data: dict[str, object]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def write_checksums(output_dir: Path, seed: int, sizes: Sequence[int]) -> None:
    names = [template.format(seed=seed) for template in CHECKSUM_REQUIRED_NAMES]
    names.extend(f"random{size}_seed{seed}.txt" for size in sizes)
    names.extend(["sampling_metadata.csv", "sampling_report.json", "README.md"])
    lines = []
    for name in sorted(dict.fromkeys(names)):
        path = output_dir / name
        if path.exists():
            lines.append(f"{sha256_file(path)}  {name}")
    write_lines(output_dir / "checksums.sha256", lines)


def read_existing_random_manifests(output_dir: Path, seed: int, sizes: Sequence[int]) -> tuple[list[str], dict[int, list[str]]]:
    order_path = output_dir / f"random_order_seed{seed}.txt"
    if not order_path.exists():
        raise FileNotFoundError(f"missing random order manifest: {order_path}")
    order_entries = read_manifest(order_path)
    subset_entries: dict[int, list[str]] = {}
    for size in sizes:
        subset_path = output_dir / f"random{size}_seed{seed}.txt"
        if not subset_path.exists():
            raise FileNotFoundError(f"missing random subset manifest: {subset_path}")
        subset_entries[size] = read_manifest(subset_path)
    return order_entries, subset_entries


def make_readme(seed: int, sizes: Sequence[int], report: dict[str, object]) -> str:
    main_size = 6000 if 6000 in sizes else sizes[len(sizes) // 2]
    max_iterations = 34000
    command = f"""cd ~/whole_body_tracking_new
conda activate hybrid_robot

env -u PYTHONPATH -u LD_LIBRARY_PATH \\
python scripts/rsl_rl/train.py \\
  --task Tracking-Flat-G1-v0 \\
  --motion_file PHUMA_wbt_motions/manifests/experiments/random_seed42/random{main_size}_seed{seed}.txt \\
  --headless \\
  --logger wandb \\
  --log_project_name whole_body_tracking_phuma \\
  --run_name direct_random{main_size}_seed{seed}_env3072 \\
  --num_envs 3072 \\
  --max_iterations {max_iterations}
"""
    subset_lines = []
    for size in sizes:
        summary = report["subsets"][f"random{size}"]
        subset_lines.append(
            f"- Random-{size}: files={summary['files']} source_groups={summary['unique_source_groups']} "
            f"frames={summary['total_frames']} duration_sec={summary['total_duration_sec']}"
        )
    return f"""# Random Seed {seed} Direct Mixed Training Baseline

This directory contains fixed nested random training subsets derived only from
`PHUMA_wbt_motions/manifests/splits_v1/train_pool.txt`.

Sampling method: `{SAMPLING_METHOD}`.

Important exclusions:

- No category balancing
- No quality filtering
- No source-group deduplication
- No difficulty weighting

The full random order is stored in `random_order_seed{seed}.txt`. Future larger
or smaller random subsets for this experiment should be prefixes of that file.

## Subsets

{chr(10).join(subset_lines)}

Nested checks and leakage checks are recorded in `sampling_report.json`.

## Direct Mixed Training Command

Run this manually from a fresh policy initialization. Do not add `--resume`,
`--load_run`, or `--checkpoint`.

```bash
{command}```

This command loads the complete Random-{main_size} motion library from iteration
1. Motion id sampling and start-frame sampling remain the current project logic
implemented in `MotionCommand`.

## W&B Config

`scripts/rsl_rl/train.py` records direct-mixed metadata when the manifest sits
next to `sampling_config.json`, including sampling method, split version,
manifest name, train size, train-pool size, sampling seed, training seed,
`num_envs`, `max_iterations`, `resume=false`, and `curriculum=false`.

## Checkpoints

The G1 PPO config currently sets `save_interval = 500` in
`source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/agents/rsl_rl_ppo_cfg.py`.
That is already more frequent than every 2000 iterations, so the 34000-iteration
run should keep periodic checkpoints such as `model_10000.pt`, `model_20000.pt`,
and `model_30000.pt`, plus the final checkpoint produced by RSL-RL at the end of
learning. If you want exactly every 2000 iterations, change `save_interval` from
`500` to `2000` in that config before launching.

With `num_envs=3072` and the current G1 PPO setting `num_steps_per_env=24`, each
iteration collects `3072 * 24 = 73728` environment steps. Keep `num_envs=3072`
for later curriculum comparisons.
"""


def prepare_context(args: argparse.Namespace) -> dict[str, object]:
    project_root = args.project_root.resolve(strict=False)
    data_root = project_root / DEFAULT_DATA_ROOT
    train_pool_path = resolve_cli_path(project_root, args.train_pool).resolve(strict=False)
    metadata_path = resolve_cli_path(project_root, args.metadata).resolve(strict=False)
    validation_path = resolve_cli_path(project_root, args.validation_manifest).resolve(strict=False)
    test_path = resolve_cli_path(project_root, args.test_manifest).resolve(strict=False)
    output_dir = resolve_cli_path(project_root, args.output_dir).resolve(strict=False)
    sizes = parse_sizes(args.sizes)

    metadata_index, metadata_warnings, metadata_ok = load_metadata_index(metadata_path, project_root, data_root)
    train_entries = read_manifest(train_pool_path)
    validation_entries = read_manifest(validation_path) if validation_path.exists() else []
    test_entries = read_manifest(test_path) if test_path.exists() else []
    pool_records = entries_to_records(
        train_entries, train_pool_path, project_root, data_root, args.path_mode, metadata_index, metadata_ok
    )
    validation_records = entries_to_records(
        validation_entries, validation_path, project_root, data_root, args.path_mode, metadata_index, metadata_ok
    )
    test_records = entries_to_records(
        test_entries, test_path, project_root, data_root, args.path_mode, metadata_index, metadata_ok
    )

    if sizes[-1] > len(pool_records):
        raise ValueError(f"largest requested size {sizes[-1]} exceeds Train Pool size {len(pool_records)}.")

    return {
        "project_root": project_root,
        "data_root": data_root,
        "train_pool_path": train_pool_path,
        "metadata_path": metadata_path,
        "validation_path": validation_path,
        "test_path": test_path,
        "output_dir": output_dir,
        "sizes": sizes,
        "metadata_warnings": metadata_warnings,
        "metadata_ok": metadata_ok,
        "pool_records": pool_records,
        "validation_records": validation_records,
        "test_records": test_records,
    }


def generate(args: argparse.Namespace) -> dict[str, object]:
    context = prepare_context(args)
    output_dir: Path = context["output_dir"]
    sizes: list[int] = context["sizes"]
    pool_records: list[MotionEntry] = list(context["pool_records"])

    if output_dir.exists() and not args.force:
        raise FileExistsError(f"{output_dir} already exists; use --force to replace fixed random experiment manifests.")
    if output_dir.exists() and args.force:
        print("WARNING: fixed random experiment manifests are being replaced", file=sys.stderr)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    ordered_records = list(pool_records)
    rng.shuffle(ordered_records)
    subset_records = {size: ordered_records[:size] for size in sizes}
    warnings = list(context["metadata_warnings"])
    report = build_report(
        seed=args.seed,
        sizes=sizes,
        pool_records=pool_records,
        order_records=ordered_records,
        subset_records=subset_records,
        validation_records=context["validation_records"],
        test_records=context["test_records"],
        warnings=warnings,
    )
    errors = integrity_errors(report, strict_metadata=args.strict)
    if errors:
        raise ValueError("fixed random subset generation failed:\n  - " + "\n  - ".join(errors))

    write_lines(output_dir / f"random_order_seed{args.seed}.txt", [record.output_path for record in ordered_records])
    for size, records in subset_records.items():
        write_lines(output_dir / f"random{size}_seed{args.seed}.txt", [record.output_path for record in records])

    config = build_sampling_config(
        project_root=context["project_root"],
        train_pool_path=context["train_pool_path"],
        output_dir=output_dir,
        seed=args.seed,
        sizes=sizes,
        path_mode=args.path_mode,
        pool_size=len(pool_records),
    )
    write_sampling_metadata(output_dir / "sampling_metadata.csv", ordered_records, sizes)
    write_json(output_dir / "sampling_config.json", config)
    write_json(output_dir / "sampling_report.json", report)
    (output_dir / "README.md").write_text(make_readme(args.seed, sizes, report))
    write_checksums(output_dir, args.seed, sizes)
    return report


def validate_existing(args: argparse.Namespace) -> dict[str, object]:
    context = prepare_context(args)
    output_dir: Path = context["output_dir"]
    sizes: list[int] = context["sizes"]
    order_entries, subset_entries = read_existing_random_manifests(output_dir, args.seed, sizes)

    order_path = output_dir / f"random_order_seed{args.seed}.txt"
    order_records = entries_to_records(
        order_entries,
        order_path,
        context["project_root"],
        context["data_root"],
        args.path_mode,
        load_metadata_index(context["metadata_path"], context["project_root"], context["data_root"])[0],
        context["metadata_ok"],
    )
    metadata_index, _, metadata_ok = load_metadata_index(
        context["metadata_path"], context["project_root"], context["data_root"]
    )
    subset_records = {
        size: entries_to_records(
            entries,
            output_dir / f"random{size}_seed{args.seed}.txt",
            context["project_root"],
            context["data_root"],
            args.path_mode,
            metadata_index,
            metadata_ok,
        )
        for size, entries in subset_entries.items()
    }
    report = build_report(
        seed=args.seed,
        sizes=sizes,
        pool_records=context["pool_records"],
        order_records=order_records,
        subset_records=subset_records,
        validation_records=context["validation_records"],
        test_records=context["test_records"],
        warnings=context["metadata_warnings"],
        sampling_metadata_path=output_dir / "sampling_metadata.csv",
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.validate_only:
            report = validate_existing(args)
            errors = integrity_errors(report, strict_metadata=True)
            if errors:
                print("random subset validation failed:", file=sys.stderr)
                for error in errors:
                    print(f"  - {error}", file=sys.stderr)
                return 1
            print("random subset validation passed.")
            return 0

        report = generate(args)
        sizes = parse_sizes(args.sizes)
        print(
            f"Generated random seed {args.seed} subsets at {args.output_dir}: "
            + ", ".join(f"random{size}={report['actual_subset_file_counts'][f'random{size}']}" for size in sizes)
        )
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
