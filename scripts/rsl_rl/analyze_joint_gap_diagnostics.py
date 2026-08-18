"""Build the STEP 3 Joint-Specific Gap offline diagnostic report."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch

import evaluation_utils as eval_utils
import joint_diagnostics as joint_diag


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "joint_gap_stage1_diagnostics"
M7RAW_RUN = PROJECT_ROOT / "logs/rsl_rl/g1_flat/2026-08-05_15-31-20_formal_v1_ablation_M7Raw_n6000_trainseed42"
AFULL_RUN = PROJECT_ROOT / "logs/rsl_rl/g1_flat/2026-08-14_22-43-51_formal_v2_A_raw_motion_gap_segment_n6000_trainseed42"
VALIDATION_MANIFEST = PROJECT_ROOT / "PHUMA_wbt_motions/manifests/splits_v1/validation.txt"
TRAIN_MANIFEST = PROJECT_ROOT / "PHUMA_wbt_motions/manifests/experiments/random_seed42/random6000_seed42.txt"
DIFFICULTY_METADATA = PROJECT_ROOT / "outputs/module2_difficulty_random6000_seed42_v1/segment_difficulty_metadata.npz"
CLUSTER_METADATA = PROJECT_ROOT / "outputs/module4_clusters_random6000_seed42_v1/motion_cluster_metadata.npz"


def _csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def _write_json(path: str | Path, payload: Mapping[str, object]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    os.replace(tmp, destination)


def _float(row: Mapping[str, object], key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    if value in (None, ""):
        return default
    return float(value)


def _int(row: Mapping[str, object], key: str, default: int = 0) -> int:
    value = row.get(key, default)
    if value in (None, ""):
        return default
    return int(float(value))


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(args: Sequence[str]) -> str:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def _rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape[0], dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < sorted_values.size:
        end = start + 1
        while end < sorted_values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        average_rank = 0.5 * (start + end - 1) + 1.0
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    mask = np.isfinite(left) & np.isfinite(right)
    if int(np.count_nonzero(mask)) < 2:
        return 0.0
    x = left[mask] - float(np.mean(left[mask]))
    y = right[mask] - float(np.mean(right[mask]))
    denom = math.sqrt(float(np.sum(x * x)) * float(np.sum(y * y)))
    return float(np.sum(x * y) / denom) if denom > 0.0 else 0.0


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    mask = np.isfinite(left) & np.isfinite(right)
    if int(np.count_nonzero(mask)) < 2:
        return 0.0
    return _pearson(_rankdata(left[mask]), _rankdata(right[mask]))


def _load_npz_array(path: Path, key: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        return np.asarray(data[key])


def _load_checkpoint_sampling_state(path: Path) -> Mapping[str, object]:
    checkpoint = torch.load(path, map_location="cpu")
    infos = checkpoint.get("infos", {})
    state = infos.get("sampling_state") if isinstance(infos, Mapping) else None
    if not isinstance(state, Mapping):
        raise ValueError(f"Checkpoint has no sampling_state: {path}")
    return state


def _tensor_array(value) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _segment_probability(sampling_state: Mapping[str, object]) -> tuple[np.ndarray, np.ndarray]:
    online = sampling_state.get("online_learning")
    if not isinstance(online, Mapping):
        raise ValueError("sampling_state.online_learning is missing.")
    sampler = online.get("sampler")
    if not isinstance(sampler, Mapping):
        raise ValueError("online_learning.sampler is missing.")
    segment_motion_ids = _tensor_array(sampler["segment_motion_ids"]).astype(np.int64)
    segment_probability = _tensor_array(sampler["segment_probability"]).astype(np.float64)
    if "global_segment_probability" in sampler:
        probability = _tensor_array(sampler["global_segment_probability"]).astype(np.float64)
    else:
        motion_probability = _tensor_array(sampler["motion_probability"]).astype(np.float64)
        probability = motion_probability[segment_motion_ids] * segment_probability
    return probability, segment_motion_ids


def _load_motion_metadata(manifest: Path) -> tuple[list[str], list[str], list[str]]:
    lookup = eval_utils.load_metadata_lookup(PROJECT_ROOT)
    motion_paths: list[str] = []
    categories: list[str] = []
    source_groups: list[str] = []
    for entry in eval_utils.load_manifest_entries(manifest):
        path = eval_utils.resolve_manifest_entry(entry, manifest, PROJECT_ROOT)
        motion_paths.append(eval_utils.project_relative(path, PROJECT_ROOT))
        category, source_group = eval_utils.motion_info(path, PROJECT_ROOT, lookup)
        categories.append(category)
        source_groups.append(source_group)
    return motion_paths, categories, source_groups


def _comparison_per_joint(m7_dir: Path, afull_dir: Path, output_dir: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    m7_rows = {_int(row, "joint_index"): row for row in _csv_rows(m7_dir / "per_joint_summary.csv")}
    af_rows = {_int(row, "joint_index"): row for row in _csv_rows(afull_dir / "per_joint_summary.csv")}
    rows: list[dict[str, object]] = []
    for index in sorted(set(m7_rows) & set(af_rows)):
        m7 = m7_rows[index]
        af = af_rows[index]
        m7_abs = _float(m7, "mean_abs_error")
        af_abs = _float(af, "mean_abs_error")
        m7_rms = _float(m7, "rms_error")
        af_rms = _float(af, "rms_error")
        improvement = m7_rms - af_rms
        rows.append(
            {
                "joint_index": index,
                "joint_name": m7["joint_name"],
                "joint_group": m7["joint_group"],
                "M7Raw_mean_abs_error": f"{m7_abs:.9f}",
                "AFull_mean_abs_error": f"{af_abs:.9f}",
                "delta_abs": f"{af_abs - m7_abs:.9f}",
                "M7Raw_rms_error": f"{m7_rms:.9f}",
                "AFull_rms_error": f"{af_rms:.9f}",
                "delta_rms": f"{af_rms - m7_rms:.9f}",
                "rms_improvement": f"{improvement:.9f}",
                "relative_improvement_percent": f"{(improvement / m7_rms * 100.0) if m7_rms else 0.0:.6f}",
            }
        )
    rows.sort(key=lambda row: _float(row, "rms_improvement"), reverse=True)
    joint_diag.write_csv(
        output_dir / "per_joint_validation_comparison.csv",
        rows,
        tuple(rows[0].keys()) if rows else (),
    )
    improved = sum(1 for row in rows if _float(row, "rms_improvement") > 0.0)
    degraded = sum(1 for row in rows if _float(row, "rms_improvement") < 0.0)
    return rows, {"num_improved_joints": improved, "num_degraded_joints": degraded}


def _group_comparison(m7_dir: Path, afull_dir: Path, output_dir: Path) -> list[dict[str, object]]:
    m7_rows = _csv_rows(m7_dir / "per_joint_summary.csv")
    af_rows = {_int(row, "joint_index"): row for row in _csv_rows(afull_dir / "per_joint_summary.csv")}
    grouped: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    counts: dict[str, int] = defaultdict(int)
    for m7 in m7_rows:
        index = _int(m7, "joint_index")
        af = af_rows[index]
        group = str(m7["joint_group"])
        counts[group] += 1
        grouped[group]["m7_abs_sum"] += _float(m7, "abs_error_sum")
        grouped[group]["af_abs_sum"] += _float(af, "abs_error_sum")
        grouped[group]["m7_sq_sum"] += _float(m7, "squared_error_sum")
        grouped[group]["af_sq_sum"] += _float(af, "squared_error_sum")
        grouped[group]["m7_count"] += _float(m7, "sample_count")
        grouped[group]["af_count"] += _float(af, "sample_count")

    rows: list[dict[str, object]] = []
    for group in sorted(grouped):
        item = grouped[group]
        m7_abs = item["m7_abs_sum"] / max(item["m7_count"], 1.0)
        af_abs = item["af_abs_sum"] / max(item["af_count"], 1.0)
        m7_rms = math.sqrt(item["m7_sq_sum"] / max(item["m7_count"], 1.0))
        af_rms = math.sqrt(item["af_sq_sum"] / max(item["af_count"], 1.0))
        rows.append(
            {
                "joint_group": group,
                "num_joints": counts[group],
                "M7Raw_mean_abs_error": f"{m7_abs:.9f}",
                "AFull_mean_abs_error": f"{af_abs:.9f}",
                "delta_abs": f"{af_abs - m7_abs:.9f}",
                "M7Raw_rms_error": f"{m7_rms:.9f}",
                "AFull_rms_error": f"{af_rms:.9f}",
                "delta_rms": f"{af_rms - m7_rms:.9f}",
                "relative_delta_percent": f"{((af_rms - m7_rms) / m7_rms * 100.0) if m7_rms else 0.0:.6f}",
            }
        )
    rows.sort(key=lambda row: _float(row, "delta_rms"))
    joint_diag.write_csv(output_dir / "joint_group_comparison.csv", rows, tuple(rows[0].keys()) if rows else ())
    return rows


def _paired_motion_analysis(m7_dir: Path, afull_dir: Path, output_dir: Path, tolerance: float) -> tuple[list[dict[str, object]], dict[str, int]]:
    m7_rows = {row["motion_path"]: row for row in _csv_rows(m7_dir / "per_motion.csv")}
    af_rows = {row["motion_path"]: row for row in _csv_rows(afull_dir / "per_motion.csv")}
    rows: list[dict[str, object]] = []
    counts = Counter()
    for motion_path in sorted(set(m7_rows) & set(af_rows)):
        m7 = m7_rows[motion_path]
        af = af_rows[motion_path]
        m7_joint = _float(m7, "joint_position_error_l2_rad")
        af_joint = _float(af, "joint_position_error_l2_rad")
        delta = af_joint - m7_joint
        m7_success = _int(m7, "success")
        af_success = _int(af, "success")
        m7_completion = _float(m7, "completion_ratio")
        af_completion = _float(af, "completion_ratio")
        if delta < -tolerance:
            counts["afull_joint_better"] += 1
        elif delta > tolerance:
            counts["m7raw_joint_better"] += 1
        else:
            counts["joint_same"] += 1
        if delta < -tolerance and (af_success < m7_success or af_completion + tolerance < m7_completion):
            counts["joint_improved_success_or_completion_down"] += 1
        if delta < -tolerance and af_success >= m7_success and af_completion + tolerance >= m7_completion:
            counts["joint_improved_no_success_drop"] += 1
        rows.append(
            {
                "motion_path": motion_path,
                "category": m7["category"],
                "source_group": m7["source_group"],
                "M7Raw_joint_error": f"{m7_joint:.9f}",
                "AFull_joint_error": f"{af_joint:.9f}",
                "delta_joint_error": f"{delta:.9f}",
                "M7Raw_success": m7_success,
                "AFull_success": af_success,
                "M7Raw_completion": f"{m7_completion:.9f}",
                "AFull_completion": f"{af_completion:.9f}",
                "delta_completion": f"{af_completion - m7_completion:.9f}",
            }
        )
    joint_diag.write_csv(output_dir / "paired_motion_joint_analysis.csv", rows, tuple(rows[0].keys()) if rows else ())
    return rows, dict(counts)


def _category_analysis(paired_rows: Sequence[Mapping[str, object]], output_dir: Path) -> list[dict[str, object]]:
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in paired_rows:
        grouped[str(row["category"])].append(row)
    rows: list[dict[str, object]] = []
    for category in sorted(grouped):
        items = grouped[category]
        count = len(items)
        m7_joint = sum(_float(row, "M7Raw_joint_error") for row in items) / max(count, 1)
        af_joint = sum(_float(row, "AFull_joint_error") for row in items) / max(count, 1)
        m7_success = sum(_int(row, "M7Raw_success") for row in items) / max(count, 1)
        af_success = sum(_int(row, "AFull_success") for row in items) / max(count, 1)
        rows.append(
            {
                "category": category,
                "num_motions": count,
                "M7Raw_joint_error": f"{m7_joint:.9f}",
                "AFull_joint_error": f"{af_joint:.9f}",
                "delta_joint_error": f"{af_joint - m7_joint:.9f}",
                "M7Raw_success": f"{m7_success:.9f}",
                "AFull_success": f"{af_success:.9f}",
                "success_delta": f"{af_success - m7_success:.9f}",
            }
        )
    rows.sort(key=lambda row: (_float(row, "delta_joint_error"), -_float(row, "success_delta")))
    joint_diag.write_csv(output_dir / "per_category_joint_analysis.csv", rows, tuple(rows[0].keys()) if rows else ())
    return rows


def _generic_gap_analysis(
    *,
    afull_gap_checkpoint: Path,
    difficulty_metadata: Path,
    cluster_metadata: Path,
    train_manifest: Path,
    output_dir: Path,
) -> tuple[list[dict[str, object]], dict[str, float]]:
    state = _load_checkpoint_sampling_state(afull_gap_checkpoint)
    online = state["online_learning"]
    statistics = online["statistics"]
    gap_result = online["gap_result"]
    segment_error = online["segment_error_result"]
    if gap_result is None:
        raise ValueError(f"A-full checkpoint has no gap_result: {afull_gap_checkpoint}")
    local_gap = _tensor_array(gap_result["local_gap"]).astype(np.float64)
    valid = _tensor_array(gap_result["global_valid"]).astype(bool)
    joint_rms = _tensor_array(statistics["segment_joint_error_ema"]).astype(np.float64)
    composite_error = _tensor_array(segment_error["error"]).astype(np.float64)
    body_component = _tensor_array(segment_error["contributions"]["body"]).astype(np.float64)
    segment_motion_ids = _tensor_array(online["sampler"]["segment_motion_ids"]).astype(np.int64)
    difficulty_bin = _load_npz_array(difficulty_metadata, "difficulty_bin").astype(np.int64)
    with np.load(cluster_metadata, allow_pickle=False) as data:
        motion_cluster_ids = np.asarray(data["cluster_id"], dtype=np.int64)
    cluster_ids = motion_cluster_ids[segment_motion_ids]
    motion_paths, categories, source_groups = _load_motion_metadata(train_manifest)

    pearson = _pearson(local_gap[valid], joint_rms[valid])
    spearman = _spearman(local_gap[valid], joint_rms[valid])
    valid_indices = np.flatnonzero(valid)
    sorted_valid = valid_indices[np.argsort(local_gap[valid_indices])[::-1]]
    top1 = set(sorted_valid[: max(1, int(math.ceil(sorted_valid.size * 0.01)))].tolist())
    top5 = set(sorted_valid[: max(1, int(math.ceil(sorted_valid.size * 0.05)))].tolist())

    def subset_row(name: str, indices: np.ndarray) -> dict[str, object]:
        if indices.size == 0:
            indices = np.asarray([], dtype=np.int64)
        return {
            "subset": name,
            "num_segments": int(indices.size),
            "local_gap_mean": f"{float(np.mean(local_gap[indices])) if indices.size else 0.0:.9f}",
            "joint_RMS_mean": f"{float(np.mean(joint_rms[indices])) if indices.size else 0.0:.9f}",
            "composite_error_mean": f"{float(np.mean(composite_error[indices])) if indices.size else 0.0:.9f}",
            "body_component_mean": f"{float(np.mean(body_component[indices])) if indices.size else 0.0:.9f}",
            "difficulty_bin_mean": f"{float(np.mean(difficulty_bin[indices])) if indices.size else 0.0:.9f}",
            "dominant_cluster": int(Counter(cluster_ids[indices].tolist()).most_common(1)[0][0]) if indices.size else "",
            "pearson_Glocal_jointRMS": f"{pearson:.9f}",
            "spearman_Glocal_jointRMS": f"{spearman:.9f}",
        }

    rows = [
        subset_row("all_valid_segments", valid_indices),
        subset_row("top1pct_G_local", np.asarray(sorted(top1), dtype=np.int64)),
        subset_row("top5pct_G_local", np.asarray(sorted(top5), dtype=np.int64)),
    ]
    joint_diag.write_csv(output_dir / "generic_gap_vs_joint_error.csv", rows, tuple(rows[0].keys()))
    return rows, {"pearson": pearson, "spearman": spearman}


def _top_boosted_segments(
    *,
    m7_sampling_checkpoint: Path,
    afull_sampling_checkpoint: Path,
    afull_gap_checkpoint: Path,
    difficulty_metadata: Path,
    cluster_metadata: Path,
    train_manifest: Path,
    output_dir: Path,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    m7_state = _load_checkpoint_sampling_state(m7_sampling_checkpoint)
    af_state = _load_checkpoint_sampling_state(afull_sampling_checkpoint)
    af_gap_state = _load_checkpoint_sampling_state(afull_gap_checkpoint)
    m7_prob, m7_segment_motion_ids = _segment_probability(m7_state)
    af_prob, af_segment_motion_ids = _segment_probability(af_state)
    if m7_prob.shape != af_prob.shape or not np.array_equal(m7_segment_motion_ids, af_segment_motion_ids):
        raise ValueError("Sampling checkpoints do not share the same segment layout.")

    online_gap = af_gap_state["online_learning"]
    gap_result = online_gap["gap_result"]
    if gap_result is None:
        raise ValueError("A-full gap checkpoint has no gap_result.")
    local_gap = _tensor_array(gap_result["local_gap"]).astype(np.float64)
    stats = online_gap["statistics"]
    joint_rms = _tensor_array(stats["segment_joint_error_ema"]).astype(np.float64)
    composite_error = _tensor_array(online_gap["segment_error_result"]["error"]).astype(np.float64)
    difficulty_bin = _load_npz_array(difficulty_metadata, "difficulty_bin").astype(np.int64)
    with np.load(cluster_metadata, allow_pickle=False) as data:
        cluster_ids = np.asarray(data["cluster_id"], dtype=np.int64)
    motion_paths, categories, source_groups = _load_motion_metadata(train_manifest)

    ratio = (af_prob + 1.0e-12) / (m7_prob + 1.0e-12)
    order = np.argsort(ratio)[::-1]
    top5_count = max(1, int(math.ceil(order.size * 0.05)))
    top1_count = max(1, int(math.ceil(order.size * 0.01)))
    rows: list[dict[str, object]] = []
    for rank, segment_id in enumerate(order[:top5_count], start=1):
        motion_id = int(af_segment_motion_ids[segment_id])
        rows.append(
            {
                "boost_rank": rank,
                "top_band": "top1pct" if rank <= top1_count else "top5pct",
                "global_segment_id": int(segment_id),
                "motion_id": motion_id,
                "local_segment_id": int(segment_id - np.searchsorted(af_segment_motion_ids, motion_id, side="left")),
                "motion_path": motion_paths[motion_id],
                "category": categories[motion_id],
                "source_group": source_groups[motion_id],
                "cluster": int(cluster_ids[motion_id]),
                "difficulty_bin": int(difficulty_bin[segment_id]),
                "M7Raw_probability": f"{m7_prob[segment_id]:.12e}",
                "AFull_probability": f"{af_prob[segment_id]:.12e}",
                "probability_ratio": f"{ratio[segment_id]:.9f}",
                "generic_G_local": f"{local_gap[segment_id]:.9f}",
                "joint_RMS": f"{joint_rms[segment_id]:.9f}",
                "composite_error": f"{composite_error[segment_id]:.9f}",
            }
        )
    joint_diag.write_csv(output_dir / "top_boosted_segments.csv", rows, tuple(rows[0].keys()) if rows else ())
    summary = {
        "top1_count": top1_count,
        "top5_count": top5_count,
        "top1_joint_RMS_mean": float(np.mean([_float(row, "joint_RMS") for row in rows[:top1_count]])) if rows else 0.0,
        "top5_joint_RMS_mean": float(np.mean([_float(row, "joint_RMS") for row in rows])) if rows else 0.0,
        "all_joint_RMS_mean": float(np.mean(joint_rms[np.isfinite(joint_rms)])),
        "sampling_comparison_checkpoints": {
            "M7Raw": str(m7_sampling_checkpoint),
            "AFull": str(afull_sampling_checkpoint),
        },
    }
    return rows, summary


def _dispersion_summary(eval_dir: Path, label: str) -> dict[str, object]:
    rows = _csv_rows(eval_dir / "per_segment_joint_diagnostics.csv")
    ratios: list[float] = []
    cvs: list[float] = []
    for row in rows:
        mean = _float(row, "mean_joint_rms_error")
        maximum = _float(row, "max_joint_rms_error")
        if mean > 0.0:
            ratios.append(maximum / mean)
        cvs.append(_float(row, "joint_error_cv"))
    ratio_array = np.asarray(ratios, dtype=np.float64)
    cv_array = np.asarray(cvs, dtype=np.float64)
    return {
        "label": label,
        "num_segments": int(len(rows)),
        "max_to_mean_p50": float(np.quantile(ratio_array, 0.50)) if ratio_array.size else 0.0,
        "max_to_mean_p90": float(np.quantile(ratio_array, 0.90)) if ratio_array.size else 0.0,
        "max_to_mean_p95": float(np.quantile(ratio_array, 0.95)) if ratio_array.size else 0.0,
        "max_to_mean_p99": float(np.quantile(ratio_array, 0.99)) if ratio_array.size else 0.0,
        "fraction_max_to_mean_ge_2": float(np.mean(ratio_array >= 2.0)) if ratio_array.size else 0.0,
        "cv_p90": float(np.quantile(cv_array, 0.90)) if cv_array.size else 0.0,
        "fraction_cv_ge_1": float(np.mean(cv_array >= 1.0)) if cv_array.size else 0.0,
    }


def _copy_mapping(m7_dir: Path, output_dir: Path) -> dict[str, object]:
    rows = _csv_rows(m7_dir / "joint_mapping.csv")
    joint_diag.write_joint_mapping_csv(output_dir / "joint_mapping.csv", rows)
    return {
        "joint_count": len(rows),
        "joint_mapping_hash": joint_diag.mapping_hash(rows),
        "all_reference_match": all(_int(row, "reference_matches_robot") == 1 for row in rows),
        "all_action_match": all(_int(row, "action_matches_robot") == 1 for row in rows),
        "all_evaluator_match": all(_int(row, "evaluator_matches_robot") == 1 for row in rows),
    }


def _report(
    *,
    output_dir: Path,
    per_joint: Sequence[Mapping[str, object]],
    groups: Sequence[Mapping[str, object]],
    paired_counts: Mapping[str, int],
    categories: Sequence[Mapping[str, object]],
    correlations: Mapping[str, float],
    boost_summary: Mapping[str, object],
    dispersion: Sequence[Mapping[str, object]],
    mapping_summary: Mapping[str, object],
) -> str:
    top_improved = list(per_joint[:10])
    top_degraded = sorted(per_joint, key=lambda row: _float(row, "rms_improvement"))[:10]
    improved = sum(1 for row in per_joint if _float(row, "rms_improvement") > 0.0)
    degraded = sum(1 for row in per_joint if _float(row, "rms_improvement") < 0.0)

    if improved >= int(math.ceil(len(per_joint) * 0.65)):
        improvement_shape = "broad improvement"
    elif improved > degraded:
        improvement_shape = "mixed improvement with more improved than degraded joints"
    else:
        improvement_shape = "mixed or concentrated improvement"

    spearman = float(correlations.get("spearman", 0.0))
    top5_joint = float(boost_summary.get("top5_joint_RMS_mean", 0.0))
    all_joint = float(boost_summary.get("all_joint_RMS_mean", 0.0))
    dispersion_nonzero = any(float(item.get("fraction_max_to_mean_ge_2", 0.0)) > 0.05 for item in dispersion)
    support_score = 0
    if improved > degraded:
        support_score += 1
    if spearman > 0.20:
        support_score += 1
    if all_joint > 0.0 and top5_joint > all_joint * 1.05:
        support_score += 1
    if dispersion_nonzero:
        support_score += 1
    if int(paired_counts.get("joint_improved_no_success_drop", 0)) > 0:
        support_score += 1
    conclusion = "SUPPORTED" if support_score >= 4 else "PARTIALLY SUPPORTED" if support_score >= 2 else "NOT SUPPORTED"

    lines: list[str] = []
    lines.append("# Joint-Specific Gap Stage 1 Diagnostics")
    lines.append("")
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    lines.append("")
    lines.append("## Joint Mapping")
    lines.append("")
    lines.append(f"- joint_count: {mapping_summary['joint_count']}")
    lines.append(f"- reference/robot/action/evaluator order all match: {mapping_summary['all_reference_match'] and mapping_summary['all_action_match'] and mapping_summary['all_evaluator_match']}")
    lines.append("")
    lines.append("## Top 10 Improved Joints")
    lines.append("")
    for row in top_improved:
        lines.append(
            f"- {row['joint_index']} {row['joint_name']} ({row['joint_group']}): "
            f"rms improvement {float(row['rms_improvement']):.6f} rad "
            f"({float(row['relative_improvement_percent']):.2f}%)"
        )
    lines.append("")
    lines.append("## Top 10 Degraded Joints")
    lines.append("")
    for row in top_degraded:
        lines.append(
            f"- {row['joint_index']} {row['joint_name']} ({row['joint_group']}): "
            f"rms change {float(row['delta_rms']):.6f} rad"
        )
    lines.append("")
    lines.append("## Group Summary")
    lines.append("")
    for row in groups[:8]:
        lines.append(
            f"- {row['joint_group']}: delta_rms={float(row['delta_rms']):.6f}, "
            f"relative_delta={float(row['relative_delta_percent']):.2f}%"
        )
    lines.append("")
    lines.append("## Paired Motion Summary")
    lines.append("")
    for key in (
        "afull_joint_better",
        "m7raw_joint_better",
        "joint_same",
        "joint_improved_success_or_completion_down",
        "joint_improved_no_success_drop",
    ):
        lines.append(f"- {key}: {int(paired_counts.get(key, 0))}")
    lines.append("")
    lines.append("## Generic Gap Relationship")
    lines.append("")
    lines.append(f"- Pearson(G_local, segment_joint_RMS): {float(correlations.get('pearson', 0.0)):.4f}")
    lines.append(f"- Spearman(G_local, segment_joint_RMS): {spearman:.4f}")
    lines.append(f"- top5 boosted joint_RMS_mean: {top5_joint:.6f}")
    lines.append(f"- all segment joint_RMS_mean: {all_joint:.6f}")
    lines.append("")
    lines.append("## Scalar RMS Masking")
    lines.append("")
    for item in dispersion:
        lines.append(
            f"- {item['label']}: max/mean p90={float(item['max_to_mean_p90']):.3f}, "
            f"fraction max/mean>=2={float(item['fraction_max_to_mean_ge_2']):.3f}, "
            f"fraction CV>=1={float(item['fraction_cv_ge_1']):.3f}"
        )
    lines.append("")
    lines.append("## Required Answers")
    lines.append("")
    lines.append(f"Q1. A-full Joint L2 gains come mainly from: {', '.join(str(row['joint_name']) for row in top_improved[:5])}.")
    lines.append(f"Q2. Improvement shape: {improvement_shape} ({improved} improved, {degraded} degraded).")
    lines.append(f"Q3. Generic Segment Gap vs joint error: Spearman={spearman:.4f}.")
    lines.append(
        "Q4. A-full high-sampling segments are "
        + ("more joint-deficit than average." if all_joint > 0.0 and top5_joint > all_joint * 1.05 else "not clearly more joint-deficit than average.")
    )
    lines.append(
        "Q5. Joint precision gains with Success/Completion cost: "
        f"{int(paired_counts.get('joint_improved_success_or_completion_down', 0))}; "
        f"without cost: {int(paired_counts.get('joint_improved_no_success_drop', 0))}."
    )
    lines.append(
        "Q6. Scalar joint RMS masking evidence: "
        + ("yes, high-dispersion segments are nonzero." if dispersion_nonzero else "weak in these diagnostics.")
    )
    lines.append(f"Q7. Final conclusion: {conclusion}.")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("Cannot reconstruct historical per-segment per-joint EMA from old checkpoints; old checkpoints only store scalar segment_joint_error_ema.")

    text = "\n".join(lines) + "\n"
    (output_dir / "joint_gap_diagnostics.md").write_text(text)
    return conclusion


def _required_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required diagnostic input is missing: {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--m7raw_eval_dir", type=Path, default=DEFAULT_OUTPUT_DIR / "eval/M7Raw_model_33999")
    parser.add_argument("--afull_eval_dir", type=Path, default=DEFAULT_OUTPUT_DIR / "eval/AFull_model_25000")
    parser.add_argument("--m7raw_checkpoint", type=Path, default=M7RAW_RUN / "model_33999.pt")
    parser.add_argument("--afull_checkpoint", type=Path, default=AFULL_RUN / "model_25000.pt")
    parser.add_argument("--m7raw_sampling_checkpoint", type=Path, default=M7RAW_RUN / "model_25000.pt")
    parser.add_argument("--afull_sampling_checkpoint", type=Path, default=AFULL_RUN / "model_25000.pt")
    parser.add_argument("--afull_gap_checkpoint", type=Path, default=AFULL_RUN / "model_25000.pt")
    parser.add_argument("--validation_manifest", type=Path, default=VALIDATION_MANIFEST)
    parser.add_argument("--train_manifest", type=Path, default=TRAIN_MANIFEST)
    parser.add_argument("--difficulty_metadata", type=Path, default=DIFFICULTY_METADATA)
    parser.add_argument("--cluster_metadata", type=Path, default=CLUSTER_METADATA)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--same_joint_tolerance", type=float, default=0.001)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in (
        args.m7raw_eval_dir / "per_joint_summary.csv",
        args.afull_eval_dir / "per_joint_summary.csv",
        args.m7raw_eval_dir / "per_motion.csv",
        args.afull_eval_dir / "per_motion.csv",
        args.m7raw_eval_dir / "per_segment_joint_diagnostics.csv",
        args.afull_eval_dir / "per_segment_joint_diagnostics.csv",
        args.m7raw_checkpoint,
        args.afull_checkpoint,
        args.m7raw_sampling_checkpoint,
        args.afull_sampling_checkpoint,
        args.afull_gap_checkpoint,
        args.validation_manifest,
        args.train_manifest,
        args.difficulty_metadata,
        args.cluster_metadata,
    ):
        _required_file(path)

    mapping_summary = _copy_mapping(args.m7raw_eval_dir, output_dir)
    per_joint_rows, joint_summary = _comparison_per_joint(args.m7raw_eval_dir, args.afull_eval_dir, output_dir)
    group_rows = _group_comparison(args.m7raw_eval_dir, args.afull_eval_dir, output_dir)
    paired_rows, paired_counts = _paired_motion_analysis(
        args.m7raw_eval_dir, args.afull_eval_dir, output_dir, args.same_joint_tolerance
    )
    category_rows = _category_analysis(paired_rows, output_dir)
    gap_rows, correlations = _generic_gap_analysis(
        afull_gap_checkpoint=args.afull_gap_checkpoint,
        difficulty_metadata=args.difficulty_metadata,
        cluster_metadata=args.cluster_metadata,
        train_manifest=args.train_manifest,
        output_dir=output_dir,
    )
    boost_rows, boost_summary = _top_boosted_segments(
        m7_sampling_checkpoint=args.m7raw_sampling_checkpoint,
        afull_sampling_checkpoint=args.afull_sampling_checkpoint,
        afull_gap_checkpoint=args.afull_gap_checkpoint,
        difficulty_metadata=args.difficulty_metadata,
        cluster_metadata=args.cluster_metadata,
        train_manifest=args.train_manifest,
        output_dir=output_dir,
    )
    dispersion = [
        _dispersion_summary(args.m7raw_eval_dir, "M7Raw"),
        _dispersion_summary(args.afull_eval_dir, "AFull"),
    ]
    conclusion = _report(
        output_dir=output_dir,
        per_joint=per_joint_rows,
        groups=group_rows,
        paired_counts=paired_counts,
        categories=category_rows,
        correlations=correlations,
        boost_summary=boost_summary,
        dispersion=dispersion,
        mapping_summary=mapping_summary,
    )

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git(["rev-parse", "HEAD"]),
        "branch": _git(["branch", "--show-current"]),
        "M7Raw_checkpoint": str(args.m7raw_checkpoint),
        "M7Raw_checkpoint_sha256": _sha256(args.m7raw_checkpoint),
        "AFull_checkpoint": str(args.afull_checkpoint),
        "AFull_checkpoint_sha256": _sha256(args.afull_checkpoint),
        "M7Raw_sampling_checkpoint": str(args.m7raw_sampling_checkpoint),
        "AFull_sampling_checkpoint": str(args.afull_sampling_checkpoint),
        "validation_manifest": str(args.validation_manifest),
        "validation_manifest_sha256": _sha256(args.validation_manifest),
        "train_manifest": str(args.train_manifest),
        "train_manifest_sha256": _sha256(args.train_manifest),
        "seed": args.seed,
        "deterministic": True,
        "randomization_disabled": True,
        "joint_count": mapping_summary["joint_count"],
        "joint_mapping_hash": mapping_summary["joint_mapping_hash"],
        "evaluator_config": {
            "joint_diagnostics_opt_in": True,
            "same_joint_tolerance": args.same_joint_tolerance,
        },
        "summary": {
            **joint_summary,
            "paired_motion_counts": paired_counts,
            "generic_gap_correlations": correlations,
            "boost_summary": boost_summary,
            "dispersion": dispersion,
            "final_conclusion": conclusion,
        },
        "outputs": {
            "joint_mapping": str(output_dir / "joint_mapping.csv"),
            "per_joint_validation_comparison": str(output_dir / "per_joint_validation_comparison.csv"),
            "joint_group_comparison": str(output_dir / "joint_group_comparison.csv"),
            "paired_motion_joint_analysis": str(output_dir / "paired_motion_joint_analysis.csv"),
            "per_category_joint_analysis": str(output_dir / "per_category_joint_analysis.csv"),
            "generic_gap_vs_joint_error": str(output_dir / "generic_gap_vs_joint_error.csv"),
            "top_boosted_segments": str(output_dir / "top_boosted_segments.csv"),
            "report": str(output_dir / "joint_gap_diagnostics.md"),
        },
    }
    _write_json(output_dir / "diagnostic_manifest.json", manifest)
    print(f"[INFO] wrote {output_dir / 'joint_gap_diagnostics.md'}")
    print(f"[INFO] conclusion={conclusion}")


if __name__ == "__main__":
    main()
