"""Utilities for deterministic WBT checkpoint evaluation.

This module intentionally avoids Isaac Sim imports so checkpoint selection,
result aggregation, and validation can be unit-tested quickly.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


CHECKPOINT_RE = re.compile(r"^model_(\d+)\.pt$")
SLICE_SUFFIX_RE = re.compile(r"(?i)(?:_chunk_\d+|_chunk\d+|-chunk-\d+)$")
DEFAULT_JOINT_COUNT = 29

PER_MOTION_COLUMNS = (
    "motion_path",
    "category",
    "source_group",
    "num_frames",
    "completed_frames",
    "episode_steps",
    "success",
    "completion_ratio",
    "body_position_error_m",
    "joint_position_error_l2_rad",
    "joint_position_error_rms_rad",
    "termination_reason",
    "checkpoint",
)

SUMMARY_COLUMNS = (
    "category",
    "num_motions",
    "num_success",
    "num_failure",
    "success_rate",
    "mean_completion_ratio",
    "mean_body_position_error_m",
    "mean_joint_position_error_l2_rad",
    "mean_joint_position_error_rms_rad",
)

SOURCE_GROUP_COLUMNS = (
    "source_group",
    "category",
    "num_motions",
    "num_success",
    "num_failure",
    "success_rate",
    "mean_completion_ratio",
    "mean_body_position_error_m",
    "mean_joint_position_error_l2_rad",
    "mean_joint_position_error_rms_rad",
)

COMPARISON_COLUMNS = (
    "checkpoint",
    "load_run",
    "iteration",
    "micro_success_rate",
    "macro_success_rate",
    "mean_completion_ratio",
    "mean_body_position_error_m",
    "mean_joint_position_error_l2_rad",
    "mean_joint_position_error_rms_rad",
    "num_failures",
)


def parse_checkpoint_iteration(path_or_name: str | Path) -> int | None:
    """Return the integer iteration from a model checkpoint filename."""
    match = CHECKPOINT_RE.match(Path(path_or_name).name)
    return int(match.group(1)) if match else None


def is_final_test_manifest(path: str | Path | None) -> bool:
    return path is not None and Path(path).name == "test.txt"


def should_reject_final_test(path: str | Path | None, confirmed: bool) -> bool:
    return is_final_test_manifest(path) and not confirmed


def list_checkpoints(run_dir: str | Path, pattern: str = "model_*.pt") -> list[Path]:
    """List checkpoints in a run directory sorted by iteration."""
    paths = [path for path in Path(run_dir).glob(pattern) if parse_checkpoint_iteration(path) is not None]
    return sorted(paths, key=lambda path: parse_checkpoint_iteration(path) or -1)


def select_checkpoints(
    available: Iterable[str | Path],
    specs: str | Iterable[str] = "10000,20000,30000,final",
) -> list[Path]:
    """Select real checkpoints by nearest requested iteration plus final.

    Numeric specs select the real checkpoint with minimum absolute distance.
    ``final``/``last`` select the checkpoint with the largest iteration.
    Duplicate resolved paths are returned once, preserving spec order.
    """
    paths = [Path(path) for path in available]
    parsed = [(path, parse_checkpoint_iteration(path)) for path in paths]
    parsed = [(path, iteration) for path, iteration in parsed if iteration is not None]
    if not parsed:
        raise ValueError("No model_*.pt checkpoints were found.")

    if isinstance(specs, str):
        spec_items = [item.strip() for item in specs.split(",") if item.strip()]
    else:
        spec_items = [str(item).strip() for item in specs if str(item).strip()]
    if not spec_items:
        spec_items = ["10000", "20000", "30000", "final"]

    selected: list[Path] = []
    seen: set[Path] = set()
    for spec in spec_items:
        if spec.lower() in {"final", "last", "latest"}:
            path, _ = max(parsed, key=lambda item: item[1])
        else:
            try:
                target = int(spec)
            except ValueError as exc:
                raise ValueError(f"Invalid checkpoint spec '{spec}'. Use an integer iteration or 'final'.") from exc
            path, _ = min(parsed, key=lambda item: (abs((item[1] or 0) - target), item[1] or 0))
        resolved = path.resolve()
        if resolved not in seen:
            selected.append(path)
            seen.add(resolved)
    return selected


def choose_best_checkpoint(rows: list[dict[str, object]], epsilon: float = 0.002) -> dict[str, object]:
    """Choose a validation checkpoint using the requested tie-break rules."""
    if not rows:
        raise ValueError("Cannot select a best checkpoint from zero rows.")

    def f(row: dict[str, object], key: str) -> float:
        return float(row.get(key, 0.0))

    def iteration(row: dict[str, object]) -> int:
        value = row.get("iteration")
        if value is not None and str(value) != "":
            return int(value)
        parsed = parse_checkpoint_iteration(str(row.get("checkpoint", "")))
        return parsed if parsed is not None else 10**18

    best = rows[0]
    for row in rows[1:]:
        macro_delta = f(row, "macro_success_rate") - f(best, "macro_success_rate")
        if macro_delta > epsilon:
            best = row
            continue
        if abs(macro_delta) > epsilon:
            continue

        micro_delta = f(row, "micro_success_rate") - f(best, "micro_success_rate")
        if micro_delta > epsilon:
            best = row
            continue
        if abs(micro_delta) > epsilon:
            continue

        completion_delta = f(row, "mean_completion_ratio") - f(best, "mean_completion_ratio")
        if completion_delta > epsilon:
            best = row
            continue
        if abs(completion_delta) > epsilon:
            continue

        body_delta = f(row, "mean_body_position_error_m") - f(best, "mean_body_position_error_m")
        if body_delta < -epsilon:
            best = row
            continue
        if abs(body_delta) > epsilon:
            continue

        if iteration(row) < iteration(best):
            best = row
    return best


def load_manifest_entries(manifest: str | Path) -> list[str]:
    """Load non-empty, non-comment manifest entries."""
    entries: list[str] = []
    with Path(manifest).open() as f:
        for line in f:
            item = line.strip()
            if item and not item.startswith("#"):
                entries.append(item)
    return entries


def resolve_manifest_entry(entry: str, manifest: str | Path, project_root: str | Path) -> Path:
    """Resolve a manifest entry the same way MotionLoader does, plus project-root relative paths."""
    item = Path(entry)
    if item.is_absolute():
        return item.resolve()

    manifest_dir = Path(manifest).resolve().parent
    root = Path(project_root).resolve()
    candidates = [manifest_dir / entry, root / entry, Path(entry).resolve()]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return (root / entry).resolve()


def canonical_manifest_set(manifest: str | Path, project_root: str | Path) -> set[str]:
    return {
        resolve_manifest_entry(entry, manifest, project_root).as_posix()
        for entry in load_manifest_entries(manifest)
    }


def validate_probe_subset(
    probe_manifest: str | Path,
    validation_manifest: str | Path,
    train_manifest: str | Path,
    test_manifest: str | Path,
    project_root: str | Path,
    expected_count: int = 500,
) -> dict[str, object]:
    """Validate that a fixed probe is a duplicate-free subset of validation only."""
    probe_entries = load_manifest_entries(probe_manifest)
    probe_set = canonical_manifest_set(probe_manifest, project_root)
    validation_set = canonical_manifest_set(validation_manifest, project_root)
    train_set = canonical_manifest_set(train_manifest, project_root)
    test_set = canonical_manifest_set(test_manifest, project_root)

    duplicate_count = len(probe_entries) - len(probe_set)
    missing_from_validation = sorted(probe_set.difference(validation_set))
    train_overlap = sorted(probe_set.intersection(train_set))
    test_overlap = sorted(probe_set.intersection(test_set))
    ok = (
        len(probe_entries) == expected_count
        and duplicate_count == 0
        and not missing_from_validation
        and not train_overlap
        and not test_overlap
    )
    return {
        "ok": ok,
        "probe_manifest": str(probe_manifest),
        "num_entries": len(probe_entries),
        "expected_count": expected_count,
        "num_unique": len(probe_set),
        "duplicate_count": duplicate_count,
        "missing_from_validation": len(missing_from_validation),
        "train_overlap": len(train_overlap),
        "test_overlap": len(test_overlap),
    }


def project_relative(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def infer_category(path: Path) -> str:
    parts = path.as_posix().split("/")
    for marker in ("g1_all", "g1_single", "g1_subset20"):
        if marker in parts:
            index = parts.index(marker)
            if index + 1 < len(parts):
                return parts[index + 1]
    return path.parent.name or "__unknown__"


def infer_source_group(path: Path) -> str:
    stem = SLICE_SUFFIX_RE.sub("", path.stem)
    category = infer_category(path)
    parts = path.as_posix().split("/")
    if "g1_all" in parts:
        rel_parts = parts[parts.index("g1_all") + 1 : -1]
        return "/".join(rel_parts + [stem])
    return f"{category}/{stem}"


def load_metadata_lookup(project_root: Path) -> dict[str, dict[str, str]]:
    metadata_path = project_root / "PHUMA_wbt_motions" / "manifests" / "splits_v1" / "metadata.csv"
    lookup: dict[str, dict[str, str]] = {}
    if not metadata_path.exists():
        return lookup

    with metadata_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data = {
                "category": row.get("category", ""),
                "source_group": row.get("source_group", ""),
                "num_frames": row.get("num_frames", ""),
            }
            rel = row.get("relative_path", "")
            abs_text = row.get("path", "")
            if rel:
                lookup[rel] = data
                lookup[(project_root / rel).resolve().as_posix()] = data
            if abs_text:
                lookup[Path(abs_text).resolve().as_posix()] = data
    return lookup


def motion_info(path: Path, project_root: Path, metadata_lookup: dict[str, dict[str, str]]) -> tuple[str, str]:
    abs_key = path.resolve().as_posix()
    rel_key = project_relative(path, project_root)
    item = metadata_lookup.get(abs_key) or metadata_lookup.get(rel_key)
    if item:
        return item.get("category") or infer_category(path), item.get("source_group") or infer_source_group(path)
    return infer_category(path), infer_source_group(path)


def clamp_completion_ratio(completed_frames: int | float, num_frames: int | float) -> float:
    if float(num_frames) <= 0.0:
        return 0.0
    return max(0.0, min(float(completed_frames) / float(num_frames), 1.0))


def make_result_row(
    *,
    motion_path: Path,
    category: str,
    source_group: str,
    num_frames: int,
    completed_steps: int,
    body_error_sum: float,
    joint_error_sum: float,
    metric_count: int,
    success: bool,
    termination_reason: str,
    checkpoint: str,
    project_root: Path,
    joint_count: int = DEFAULT_JOINT_COUNT,
) -> dict[str, object]:
    completed_frames = min(max(completed_steps + 1, 0), max(num_frames, 0))
    completion_ratio = clamp_completion_ratio(completed_frames, num_frames)
    denom = max(metric_count, 1)
    joint_l2 = body_l2 = 0.0
    body_l2 = body_error_sum / denom
    joint_l2 = joint_error_sum / denom
    joint_rms = joint_l2 / math.sqrt(max(joint_count, 1))
    return {
        "motion_path": project_relative(motion_path, project_root),
        "category": category,
        "source_group": source_group,
        "num_frames": int(num_frames),
        "completed_frames": int(completed_frames),
        "episode_steps": int(completed_steps),
        "success": int(success),
        "completion_ratio": f"{completion_ratio:.6f}",
        "body_position_error_m": f"{body_l2:.6f}",
        "joint_position_error_l2_rad": f"{joint_l2:.6f}",
        "joint_position_error_rms_rad": f"{joint_rms:.6f}",
        "termination_reason": termination_reason,
        "checkpoint": checkpoint,
    }


def load_per_motion_csv(path: str | Path) -> list[dict[str, object]]:
    csv_path = Path(path)
    if not csv_path.exists():
        return []
    with csv_path.open(newline="") as f:
        return list(csv.DictReader(f))


def _float(row: dict[str, object], key: str) -> float:
    value = row.get(key, 0.0)
    if value in (None, ""):
        return 0.0
    return float(value)


def _int(row: dict[str, object], key: str) -> int:
    value = row.get(key, 0)
    if value in (None, ""):
        return 0
    return int(float(value))


def summarize_rows(rows: list[dict[str, object]], checkpoint: str = "") -> dict[str, object]:
    if not rows:
        return {
            "checkpoint": checkpoint,
            "num_motions": 0,
            "num_success": 0,
            "num_failure": 0,
            "micro_success_rate": 0.0,
            "macro_success_rate": 0.0,
            "mean_completion_ratio": 0.0,
            "mean_body_position_error_m": 0.0,
            "mean_joint_position_error_l2_rad": 0.0,
            "mean_joint_position_error_rms_rad": 0.0,
            "termination_reason_counts": {},
            "num_categories": 0,
            "num_source_groups": 0,
        }

    category_summary = group_summary(rows, "category")
    num_success = sum(_int(row, "success") for row in rows)
    num_motions = len(rows)
    return {
        "checkpoint": checkpoint,
        "num_motions": num_motions,
        "num_success": num_success,
        "num_failure": num_motions - num_success,
        "micro_success_rate": num_success / num_motions,
        "macro_success_rate": sum(_float(row, "success_rate") for row in category_summary) / len(category_summary),
        "mean_completion_ratio": sum(_float(row, "completion_ratio") for row in rows) / num_motions,
        "mean_body_position_error_m": sum(_float(row, "body_position_error_m") for row in rows) / num_motions,
        "mean_joint_position_error_l2_rad": sum(_float(row, "joint_position_error_l2_rad") for row in rows) / num_motions,
        "mean_joint_position_error_rms_rad": sum(_float(row, "joint_position_error_rms_rad") for row in rows) / num_motions,
        "termination_reason_counts": dict(Counter(str(row.get("termination_reason", "")) for row in rows)),
        "num_categories": len(category_summary),
        "num_source_groups": len({str(row.get("source_group", "")) for row in rows}),
    }


def group_summary(rows: list[dict[str, object]], group_key: str) -> list[dict[str, object]]:
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(group_key, ""))].append(row)

    summary_rows: list[dict[str, object]] = []
    for group_value in sorted(groups):
        items = groups[group_value]
        num_motions = len(items)
        num_success = sum(_int(row, "success") for row in items)
        out: dict[str, object] = {
            group_key: group_value,
            "num_motions": num_motions,
            "num_success": num_success,
            "num_failure": num_motions - num_success,
            "success_rate": f"{num_success / max(num_motions, 1):.6f}",
            "mean_completion_ratio": f"{sum(_float(row, 'completion_ratio') for row in items) / max(num_motions, 1):.6f}",
            "mean_body_position_error_m": f"{sum(_float(row, 'body_position_error_m') for row in items) / max(num_motions, 1):.6f}",
            "mean_joint_position_error_l2_rad": f"{sum(_float(row, 'joint_position_error_l2_rad') for row in items) / max(num_motions, 1):.6f}",
            "mean_joint_position_error_rms_rad": f"{sum(_float(row, 'joint_position_error_rms_rad') for row in items) / max(num_motions, 1):.6f}",
        }
        if group_key == "source_group":
            categories = sorted({str(row.get("category", "")) for row in items})
            out["category"] = categories[0] if len(categories) == 1 else "|".join(categories)
        summary_rows.append(out)
    return summary_rows


def validate_results_exactly_once(rows: list[dict[str, object]], expected_motion_paths: Iterable[str]) -> dict[str, object]:
    expected = [str(path) for path in expected_motion_paths]
    expected_set = set(expected)
    observed = [str(row.get("motion_path", "")) for row in rows]
    counts = Counter(observed)
    duplicates = sorted(path for path, count in counts.items() if count > 1)
    missing = sorted(expected_set.difference(counts))
    unexpected = sorted(set(observed).difference(expected_set))
    return {
        "ok": not duplicates and not missing and not unexpected and len(rows) == len(expected),
        "expected_count": len(expected),
        "observed_count": len(rows),
        "duplicates": duplicates,
        "missing": missing,
        "unexpected": unexpected,
    }


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(project_root: str | Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            text=True,
            capture_output=True,
        )
    except Exception:
        return "unknown"
    return result.stdout.strip()


def atomic_write_text(path: Path, text: str) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(text)
    os.replace(tmp_path, path)


def write_csv(path: Path, rows: list[dict[str, object]], columns: Iterable[str]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp_path, path)


def write_evaluation_outputs(
    output_dir: str | Path,
    rows: list[dict[str, object]],
    *,
    checkpoint: str,
    evaluation_config: dict[str, object],
    expected_motion_paths: Iterable[str] | None = None,
) -> dict[str, object]:
    """Write per-motion, grouped summaries, summary JSON, config, and failures."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    write_csv(out / "per_motion.csv", rows, PER_MOTION_COLUMNS)
    category_rows = group_summary(rows, "category")
    source_group_rows = group_summary(rows, "source_group")
    write_csv(out / "category_summary.csv", category_rows, SUMMARY_COLUMNS)
    write_csv(out / "source_group_summary.csv", source_group_rows, SOURCE_GROUP_COLUMNS)

    failures = [row for row in rows if _int(row, "success") == 0]
    failure_lines = [
        f"{row.get('motion_path','')}\t{row.get('termination_reason','')}\tcompletion={row.get('completion_ratio','')}"
        for row in failures
    ]
    atomic_write_text(out / "failures.txt", "\n".join(failure_lines) + ("\n" if failure_lines else ""))

    summary = summarize_rows(rows, checkpoint=checkpoint)
    summary.update(
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "randomization_disabled": bool(evaluation_config.get("disable_randomization", False)),
        }
    )
    summary.update(evaluation_config)
    if expected_motion_paths is not None:
        summary["manifest_integrity"] = validate_results_exactly_once(rows, expected_motion_paths)

    atomic_write_text(out / "summary.json", json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    atomic_write_text(out / "evaluation_config.json", json.dumps(evaluation_config, indent=2, ensure_ascii=False) + "\n")
    return summary


def comparison_row_from_summary(summary: dict[str, object]) -> dict[str, object]:
    checkpoint = str(summary.get("checkpoint", ""))
    iteration = parse_checkpoint_iteration(checkpoint)
    return {
        "checkpoint": checkpoint,
        "load_run": str(summary.get("load_run", "")),
        "iteration": iteration if iteration is not None else "",
        "micro_success_rate": f"{float(summary.get('micro_success_rate', 0.0)):.6f}",
        "macro_success_rate": f"{float(summary.get('macro_success_rate', 0.0)):.6f}",
        "mean_completion_ratio": f"{float(summary.get('mean_completion_ratio', 0.0)):.6f}",
        "mean_body_position_error_m": f"{float(summary.get('mean_body_position_error_m', 0.0)):.6f}",
        "mean_joint_position_error_l2_rad": f"{float(summary.get('mean_joint_position_error_l2_rad', 0.0)):.6f}",
        "mean_joint_position_error_rms_rad": f"{float(summary.get('mean_joint_position_error_rms_rad', 0.0)):.6f}",
        "num_failures": int(summary.get("num_failure", 0)),
    }


def _same_path_text(left: str | Path | None, right: str | Path | None) -> bool:
    if left is None or right is None:
        return False
    left_text = str(left)
    right_text = str(right)
    if left_text == right_text:
        return True
    try:
        return Path(left_text).resolve() == Path(right_text).resolve()
    except Exception:
        return False


def summary_is_complete(
    summary: dict[str, object],
    *,
    expected_manifest: str | Path | None = None,
    expected_checkpoint: str | Path | None = None,
) -> bool:
    """Return whether an evaluation summary represents a complete manifest run."""
    integrity = summary.get("manifest_integrity", {})
    if not isinstance(integrity, dict) or not bool(integrity.get("ok", False)):
        return False

    expected_count = int(integrity.get("expected_count", -1))
    observed_count = int(integrity.get("observed_count", -2))
    num_motions = int(summary.get("num_motions", -3))
    if expected_count <= 0 or observed_count != expected_count or num_motions != expected_count:
        return False

    if expected_manifest is not None and not (
        _same_path_text(summary.get("manifest"), expected_manifest)
        or _same_path_text(summary.get("manifest_path"), expected_manifest)
    ):
        return False

    if expected_checkpoint is not None:
        actual_checkpoint = Path(str(summary.get("checkpoint", ""))).name
        expected_checkpoint_name = Path(str(expected_checkpoint)).name
        if actual_checkpoint != expected_checkpoint_name:
            return False

    return True
