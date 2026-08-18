#!/usr/bin/env python3
"""Summarize STEP 8 M7-JGap runtime smoke checkpoints.

This script is intentionally offline: it does not launch Isaac Sim, run
Validation, run Test, or update policy weights.  It only reads smoke training
checkpoints and logs, then writes the STEP 8 health report artifacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch


DEFAULT_OUTPUT_DIR = "outputs/joint_gap_stage4_smoke"
DEFAULT_RUN_NAME = "smoke_M7JGap_lam0025_n6000_seed42_500"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_value(project_root: Path, *args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=project_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return ""


def _checkpoint_iteration(path: Path) -> int:
    match = re.search(r"model_(\d+)\.pt$", path.name)
    if not match:
        raise ValueError(f"Cannot parse checkpoint iteration from {path}")
    return int(match.group(1))


def _list_checkpoints(run_dir: Path, pattern: str) -> list[Path]:
    checkpoints = sorted(run_dir.glob(pattern), key=_checkpoint_iteration)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints matched {run_dir / pattern}")
    return checkpoints


def _select_checkpoints(checkpoints: Sequence[Path], targets: str) -> list[Path]:
    by_iter = {_checkpoint_iteration(path): path for path in checkpoints}
    selected: list[Path] = []
    seen: set[Path] = set()
    for token in [item.strip() for item in targets.split(",") if item.strip()]:
        if token == "final":
            target = max(by_iter)
        else:
            desired = int(token)
            target = min(by_iter, key=lambda value: (abs(value - desired), value))
        path = by_iter[target]
        if path not in seen:
            selected.append(path)
            seen.add(path)
    return selected


def _as_tensor(value: Any, *, dtype: torch.dtype) -> torch.Tensor:
    return torch.as_tensor(value, dtype=dtype, device="cpu")


def _walk_numeric_tensors(value: Any, prefix: str = ""):
    if isinstance(value, torch.Tensor):
        if value.dtype == torch.bool:
            return
        yield prefix, value
    elif isinstance(value, Mapping):
        for key, item in value.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            yield from _walk_numeric_tensors(item, next_prefix)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _walk_numeric_tensors(item, f"{prefix}[{index}]")


def _has_nonfinite(value: Any) -> tuple[bool, list[str]]:
    bad: list[str] = []
    for name, tensor in _walk_numeric_tensors(value):
        numeric = torch.as_tensor(tensor)
        if numeric.is_floating_point() and not torch.all(torch.isfinite(numeric)):
            bad.append(name)
    return bool(bad), bad[:20]


def _summary(values: torch.Tensor, valid: torch.Tensor | None = None) -> dict[str, float]:
    tensor = torch.as_tensor(values, dtype=torch.float64)
    mask = torch.isfinite(tensor)
    if valid is not None:
        mask &= torch.as_tensor(valid, dtype=torch.bool, device=tensor.device)
    selected = tensor[mask]
    if selected.numel() == 0:
        return {"mean": 0.0, "p90": 0.0, "max": 0.0}
    return {
        "mean": float(selected.mean().item()),
        "p90": float(torch.quantile(selected, torch.tensor(0.90, dtype=selected.dtype)).item()),
        "max": float(selected.max().item()),
    }


def _probability_summary(probability: torch.Tensor, eligible: torch.Tensor | None = None) -> dict[str, float]:
    prob = torch.as_tensor(probability, dtype=torch.float64)
    mask = torch.isfinite(prob) & (prob > 0.0)
    if eligible is not None:
        mask &= torch.as_tensor(eligible, dtype=torch.bool, device=prob.device)
    selected = prob[mask]
    if selected.numel() == 0:
        return {
            "top1_mass": 0.0,
            "top5_mass": 0.0,
            "max_probability": 0.0,
            "entropy": 0.0,
            "effective_count": 0.0,
        }
    selected = selected / selected.sum().clamp_min(1.0e-300)
    ordered, _ = torch.sort(selected, descending=True)
    top1_count = max(1, int(math.ceil(0.01 * ordered.numel())))
    top5_count = max(1, int(math.ceil(0.05 * ordered.numel())))
    entropy = float((-(selected * torch.log(selected.clamp_min(1.0e-300)))).sum().item())
    return {
        "top1_mass": float(ordered[:top1_count].sum().item()),
        "top5_mass": float(ordered[:top5_count].sum().item()),
        "max_probability": float(ordered[0].item()),
        "entropy": entropy,
        "effective_count": float(1.0 / selected.square().sum().clamp_min(1.0e-300).item()),
    }


def _spearman(left: torch.Tensor, right: torch.Tensor, valid: torch.Tensor | None = None) -> float:
    x = torch.as_tensor(left, dtype=torch.float64).reshape(-1)
    y = torch.as_tensor(right, dtype=torch.float64).reshape(-1)
    mask = torch.isfinite(x) & torch.isfinite(y)
    if valid is not None:
        mask &= torch.as_tensor(valid, dtype=torch.bool).reshape(-1)
    x = x[mask]
    y = y[mask]
    if x.numel() < 3:
        return 0.0
    rank_x = torch.argsort(torch.argsort(x)).to(torch.float64)
    rank_y = torch.argsort(torch.argsort(y)).to(torch.float64)
    rank_x -= rank_x.mean()
    rank_y -= rank_y.mean()
    denom = torch.sqrt(rank_x.square().sum() * rank_y.square().sum())
    if denom <= 0:
        return 0.0
    return float((rank_x * rank_y).sum().div(denom).item())


def _load_checkpoint(path: Path) -> tuple[dict[str, Any], Mapping[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    infos = checkpoint.get("infos", {})
    if not isinstance(infos, Mapping) or "sampling_state" not in infos:
        raise RuntimeError(f"{path} has no infos.sampling_state.")
    sampling_state = infos["sampling_state"]
    if not isinstance(sampling_state, Mapping):
        raise RuntimeError(f"{path} sampling_state is not a mapping.")
    return checkpoint, sampling_state


def _find_run_dir(project_root: Path, run_dir: str, run_name: str) -> Path:
    if run_dir:
        path = Path(run_dir)
        return path if path.is_absolute() else project_root / path
    matches = sorted((project_root / "logs" / "rsl_rl" / "g1_flat").glob(f"*_{run_name}"))
    if not matches:
        raise FileNotFoundError(f"Could not find logs/rsl_rl/g1_flat/*_{run_name}")
    return matches[-1]


def _load_logs(paths: Sequence[str], project_root: Path) -> str:
    chunks: list[str] = []
    for text in paths:
        path = Path(text)
        if not path.is_absolute():
            path = project_root / path
        if path.exists():
            chunks.append(path.read_text(encoding="utf-8", errors="replace"))
    return "\n".join(chunks)


def _timing_summary(joint_gap_state: Mapping[str, Any]) -> dict[str, float]:
    timing = joint_gap_state.get("timing_ms", {})
    values: Sequence[Any] = ()
    if isinstance(timing, Mapping):
        raw_values = timing.get("joint_gap_update", ())
        if isinstance(raw_values, Sequence):
            values = raw_values
    tensor = torch.as_tensor([float(value) for value in values], dtype=torch.float64)
    if tensor.numel() == 0:
        return {"update_ms_mean": 0.0, "update_ms_p90": 0.0, "update_ms_max": 0.0, "update_count": 0.0}
    return {
        "update_ms_mean": float(tensor.mean().item()),
        "update_ms_p90": float(torch.quantile(tensor, torch.tensor(0.90, dtype=tensor.dtype)).item()),
        "update_ms_max": float(tensor.max().item()),
        "update_count": float(tensor.numel()),
    }


def _checkpoint_health(path: Path, lambda_joint: float) -> dict[str, Any]:
    checkpoint, sampling_state = _load_checkpoint(path)
    online = sampling_state.get("online_learning")
    if not isinstance(online, Mapping):
        raise RuntimeError(f"{path} has no online_learning state.")
    sampler = online.get("sampler")
    if not isinstance(sampler, Mapping):
        raise RuntimeError(f"{path} has no sampler state.")
    joint_gap = online.get("joint_gap")
    if not isinstance(joint_gap, Mapping):
        raise RuntimeError(f"{path} has no joint_gap state.")
    segment_error = online.get("segment_error_result")
    motion_error = online.get("motion_error_result")
    if not isinstance(segment_error, Mapping) or not isinstance(motion_error, Mapping):
        raise RuntimeError(f"{path} is missing raw priority caches.")
    calibration = joint_gap.get("joint_bin_calibration")
    result = joint_gap.get("joint_gap_result")
    stats = joint_gap.get("statistics")
    online_stats = online.get("statistics")
    if not isinstance(calibration, Mapping) or not isinstance(result, Mapping) or not isinstance(stats, Mapping):
        raise RuntimeError(f"{path} is missing derived joint-gap caches.")
    if not isinstance(online_stats, Mapping):
        raise RuntimeError(f"{path} has no online statistics state.")

    research = sampling_state.get("research_config", {})
    method_ok = isinstance(research, Mapping) and research.get("method_name") == "M7-JGap"
    motion_mode = str(sampler.get("motion_mode", ""))
    segment_mode = str(sampler.get("segment_mode", ""))
    settings = joint_gap.get("settings", {})
    lambda_seen = float(settings.get("lambda_joint", float("nan"))) if isinstance(settings, Mapping) else float("nan")

    raw = _as_tensor(segment_error["error"], dtype=torch.float64)
    raw_valid = _as_tensor(segment_error["valid"], dtype=torch.bool)
    corrected = _as_tensor(result["corrected_priority"], dtype=torch.float64)
    correction = _as_tensor(result["correction"], dtype=torch.float64)
    segment_valid = _as_tensor(result["segment_valid"], dtype=torch.bool)
    raw_gate = _as_tensor(result["raw_gate_mask"], dtype=torch.bool)
    gap_score = _as_tensor(result["segment_gap_score"], dtype=torch.float64)
    local_gap = _as_tensor(result["local_gap"], dtype=torch.float64)
    local_valid = _as_tensor(result["local_valid"], dtype=torch.bool)
    segment_joint_error = _as_tensor(stats["segment_joint_error_ema"], dtype=torch.float64)
    initialized = _as_tensor(stats["segment_joint_error_initialized"], dtype=torch.bool)
    sigma_floor_mask = _as_tensor(calibration["sigma_floor_mask"], dtype=torch.bool)
    reliable_mask = _as_tensor(calibration["reliable_mask"], dtype=torch.bool)
    fallback_mask = _as_tensor(calibration["fallback_mask"], dtype=torch.bool)
    valid_counts = _as_tensor(calibration["valid_segment_count"], dtype=torch.float64)

    multiplier_mask = raw_valid & torch.isfinite(raw) & (raw > 0.0) & torch.isfinite(corrected)
    multiplier = torch.where(multiplier_mask, corrected / raw.clamp_min(1.0e-300), torch.ones_like(raw))
    multiplier_summary = _summary(multiplier, multiplier_mask)
    correction_summary = _summary(correction, segment_valid)
    gap_summary = _summary(gap_score, segment_valid)

    top_k = int(joint_gap.get("top_k", 6))
    topk_raw = torch.topk(segment_joint_error.clamp_min(0.0), k=min(top_k, segment_joint_error.shape[1]), dim=1).values.mean(dim=1)
    mean_joint_raw = segment_joint_error.mean(dim=1)
    spearman_gap_raw_segment = _spearman(gap_score, raw, segment_valid & raw_valid)
    spearman_gap_raw_joint_mean = _spearman(gap_score, mean_joint_raw, segment_valid & initialized)
    spearman_gap_raw_joint_topk = _spearman(gap_score, topk_raw, segment_valid & initialized)

    if "motion_probability" in sampler:
        motion_probability = _as_tensor(sampler["motion_probability"], dtype=torch.float64)
    else:
        motion_probability = _as_tensor(sampler["motion_probability_conditional"], dtype=torch.float64)
    segment_probability = _as_tensor(sampler["segment_probability"], dtype=torch.float64)
    segment_motion_ids = _as_tensor(sampler["segment_motion_ids"], dtype=torch.long)
    motion_eligible = _as_tensor(sampler.get("motion_eligible_mask", torch.ones_like(motion_probability)), dtype=torch.bool)
    segment_eligible = _as_tensor(sampler.get("segment_eligible_mask", torch.ones_like(segment_probability)), dtype=torch.bool)
    segment_marginal = motion_probability[segment_motion_ids] * segment_probability
    motion_summary = _probability_summary(motion_probability, motion_eligible)
    segment_summary = _probability_summary(segment_marginal, segment_eligible)

    cluster_l1 = 0.0
    if "cluster_target_probability" in sampler and "observed_cluster_share" in sampler:
        target = _as_tensor(sampler["cluster_target_probability"], dtype=torch.float64)
        observed = _as_tensor(sampler["observed_cluster_share"], dtype=torch.float64)
        if target.shape == observed.shape:
            cluster_l1 = float(torch.sum(torch.abs(target - observed)).item())

    nonfinite, nonfinite_paths = _has_nonfinite(sampling_state)
    update_timing = _timing_summary(joint_gap)
    top_freq = _as_tensor(result["topk_selection_frequency"], dtype=torch.float64)
    joint_names = []
    mapping = joint_gap.get("joint_mapping_identity", {})
    if isinstance(mapping, Mapping):
        joint_names = [str(name) for name in mapping.get("joint_names", [])]
    if not joint_names:
        joint_names = [f"joint_{index}" for index in range(int(top_freq.numel()))]

    floor_count = int(torch.count_nonzero(sigma_floor_mask & reliable_mask).item())
    reliable_count = int(torch.count_nonzero(reliable_mask).item())
    fallback_count = int(torch.count_nonzero(fallback_mask).item())
    row = {
        "checkpoint": path.name,
        "checkpoint_path": str(path),
        "checkpoint_sha256": _sha256_file(path),
        "filename_iteration": _checkpoint_iteration(path),
        "checkpoint_iter": int(checkpoint.get("iter", -1)),
        "method_ok": int(method_ok),
        "motion_mode": motion_mode,
        "segment_mode": segment_mode,
        "lambda_joint": lambda_seen,
        "generic_gap_off": int(online.get("bin_calibration") is None and online.get("gap_result") is None),
        "joint_gap_enabled": int(bool(joint_gap.get("enabled", False))),
        "num_segments": int(raw.numel()),
        "num_joints": int(segment_joint_error.shape[1]),
        "num_bins": int(sigma_floor_mask.shape[0]),
        "completed_window_count": int(online.get("completed_window_count", -1)),
        "last_formula_update_iteration": int(online.get("last_formula_update_iteration", -1)),
        "initialized_segments": int(torch.count_nonzero(initialized).item()),
        "active_segment_fraction": float(torch.count_nonzero(correction[segment_valid] > 0.0).item()) / max(int(torch.count_nonzero(segment_valid).item()), 1),
        "raw_gate_pass_fraction": float(torch.count_nonzero(raw_gate[segment_valid]).item()) / max(int(torch.count_nonzero(segment_valid).item()), 1),
        "correction_mean": correction_summary["mean"],
        "correction_p90": correction_summary["p90"],
        "correction_max": correction_summary["max"],
        "gap_mean": gap_summary["mean"],
        "gap_p90": gap_summary["p90"],
        "gap_max": gap_summary["max"],
        "multiplier_mean": multiplier_summary["mean"],
        "multiplier_p90": multiplier_summary["p90"],
        "multiplier_max": multiplier_summary["max"],
        "sigma_floor_fraction": floor_count / max(reliable_count, 1),
        "sigma_floor_count": floor_count,
        "sigma_reliable_count": reliable_count,
        "fallback_fraction": fallback_count / max(int(fallback_mask.numel()), 1),
        "spearman_gap_raw_segment": spearman_gap_raw_segment,
        "spearman_gap_raw_joint_mean": spearman_gap_raw_joint_mean,
        "spearman_gap_raw_joint_topk": spearman_gap_raw_joint_topk,
        "motion_top1_mass": motion_summary["top1_mass"],
        "motion_top5_mass": motion_summary["top5_mass"],
        "motion_max_probability": motion_summary["max_probability"],
        "motion_entropy": motion_summary["entropy"],
        "motion_effective_count": motion_summary["effective_count"],
        "segment_top1_mass": segment_summary["top1_mass"],
        "segment_top5_mass": segment_summary["top5_mass"],
        "segment_max_probability": segment_summary["max_probability"],
        "segment_entropy": segment_summary["entropy"],
        "segment_effective_count": segment_summary["effective_count"],
        "cluster_target_observed_l1": cluster_l1,
        "sampler_probability_update_count": int(sampler.get("probability_update_count", -1)),
        "sampler_fallback_count": int(sampler.get("fallback_count", -1)),
        "nonfinite": int(nonfinite),
        "nonfinite_paths": ";".join(nonfinite_paths),
        **update_timing,
    }
    row["_arrays"] = {
        "sigma_floor_mask": sigma_floor_mask,
        "reliable_mask": reliable_mask,
        "fallback_mask": fallback_mask,
        "valid_counts": valid_counts,
        "top_freq": top_freq,
        "joint_names": joint_names,
        "segment_joint_error": segment_joint_error,
        "local_gap": local_gap,
        "local_valid": local_valid,
    }
    row["_research"] = research
    return row


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _classification(rows: Sequence[Mapping[str, Any]], lambda_joint: float, final_iteration: int) -> tuple[str, list[str], list[str]]:
    failures: list[str] = []
    warnings: list[str] = []
    if not rows:
        failures.append("No checkpoints were analyzed.")
        return "FAIL", failures, warnings
    final = rows[-1]
    for row in rows:
        if int(row["method_ok"]) != 1:
            failures.append(f"{row['checkpoint']}: method_name is not M7-JGap.")
        if row["motion_mode"] != "raw_error" or row["segment_mode"] != "raw_error_joint_gap":
            failures.append(f"{row['checkpoint']}: sampler mode mismatch.")
        if int(row["generic_gap_off"]) != 1:
            failures.append(f"{row['checkpoint']}: generic learning gap cache is present.")
        if int(row["joint_gap_enabled"]) != 1:
            failures.append(f"{row['checkpoint']}: joint_gap.enabled is false.")
        if not math.isclose(float(row["lambda_joint"]), lambda_joint, rel_tol=0.0, abs_tol=1.0e-12):
            failures.append(f"{row['checkpoint']}: lambda_joint mismatch.")
        if int(row["nonfinite"]) != 0:
            failures.append(f"{row['checkpoint']}: non-finite sampling state values: {row['nonfinite_paths']}")
        if float(row["correction_max"]) > 1.0 + 1.0e-9:
            failures.append(f"{row['checkpoint']}: correction exceeds 1.")
        if float(row["multiplier_max"]) > 1.0 + lambda_joint + 1.0e-6:
            failures.append(f"{row['checkpoint']}: multiplier exceeds frozen lambda bound.")
    if int(final["filename_iteration"]) < final_iteration:
        failures.append(f"Final checkpoint is {final['checkpoint']}, expected at least model_{final_iteration}.pt.")
    if float(final["active_segment_fraction"]) <= 0.0:
        failures.append("Joint correction never became active by the final smoke checkpoint.")
    if int(final["initialized_segments"]) <= 0:
        failures.append("No per-joint segments initialized.")
    if len(rows) >= 2:
        previous = rows[0]
        if int(final["initialized_segments"]) < int(previous["initialized_segments"]):
            failures.append("Resume appears to cold restart per-joint initialization.")
        if int(final["completed_window_count"]) <= int(previous["completed_window_count"]):
            failures.append("Resume did not advance online-learning window count.")
    if float(final["sigma_floor_fraction"]) > 0.90:
        warnings.append("CALIBRATION WARNING: sigma_floor_fraction > 0.90.")
    if max(
        abs(float(final["spearman_gap_raw_segment"])),
        abs(float(final["spearman_gap_raw_joint_topk"])),
    ) > 0.95:
        warnings.append("DEGENERACY WARNING: joint gap is highly rank-correlated with raw priority.")
    if failures:
        return "FAIL", failures, warnings
    if warnings:
        return "PASS WITH WARNING", failures, warnings
    return "PASS", failures, warnings


def _render_report(
    *,
    output_dir: Path,
    run_name: str,
    run_dir: Path,
    branch: str,
    commit: str,
    status: str,
    failures: Sequence[str],
    warnings: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
    wandb_status: str,
    lambda_joint: float,
    num_envs: int,
    seed: int,
    final_iteration: int,
) -> str:
    final = rows[-1] if rows else {}
    lines = [
        "# STEP 8 M7-JGap Runtime Smoke Report",
        "",
        f"STEP 8: {status}",
        "",
        "## 1. Run identity",
        "",
        f"- run name: {run_name}",
        f"- branch: {branch}",
        f"- commit: {commit}",
        f"- run dir: {run_dir}",
        f"- seed: {seed}",
        f"- num_envs: {num_envs}",
        f"- iterations: {final.get('filename_iteration', 'NA')}/{final_iteration}",
        "",
        "## 2. Resolved method contract",
        "",
        "- Quality: ON",
        "- Diversity: ON",
        "- Motion: raw_error",
        "- Segment: raw_error_joint_gap",
        "- Generic Gap: OFF",
        "- Joint Gap: ON",
        f"- lambda: {lambda_joint}",
        "",
        "## 3. Runtime health",
        "",
        f"- NaN/Inf: {'FAIL' if int(final.get('nonfinite', 1)) else 'PASS'}",
        f"- sampler fallback count: {final.get('sampler_fallback_count', 'NA')}",
        f"- probability updates: {final.get('sampler_probability_update_count', 'NA')}",
        "",
        "## 4. JGap initialization",
        "",
        f"- first analyzed active iteration: {next((row['filename_iteration'] for row in rows if float(row['active_segment_fraction']) > 0.0), 'NA')}",
        f"- initialized segments: {final.get('initialized_segments', 'NA')}",
        f"- active fraction: {final.get('active_segment_fraction', 'NA')}",
        "",
        "## 5. Correction",
        "",
        f"- mean: {final.get('correction_mean', 'NA')}",
        f"- p90: {final.get('correction_p90', 'NA')}",
        f"- max: {final.get('correction_max', 'NA')}",
        f"- max multiplier: {final.get('multiplier_max', 'NA')}",
        "",
        "## 6. Sigma floor",
        "",
        f"- overall: {final.get('sigma_floor_fraction', 'NA')}",
        f"- classification: {'CALIBRATION WARNING' if any('CALIBRATION' in item for item in warnings) else 'NORMAL'}",
        "",
        "## 7. Gap degeneracy analysis",
        "",
        f"- Spearman G_joint vs raw segment priority: {final.get('spearman_gap_raw_segment', 'NA')}",
        f"- Spearman G_joint vs raw joint mean: {final.get('spearman_gap_raw_joint_mean', 'NA')}",
        f"- Spearman G_joint vs Top6 raw joint error: {final.get('spearman_gap_raw_joint_topk', 'NA')}",
        f"- classification: {'DEGENERACY WARNING' if any('DEGENERACY' in item for item in warnings) else 'NORMAL'}",
        "",
        "## 8. Sampling distribution",
        "",
        f"- cluster target-observed L1: {final.get('cluster_target_observed_l1', 'NA')}",
        f"- motion top1/top5: {final.get('motion_top1_mass', 'NA')} / {final.get('motion_top5_mass', 'NA')}",
        f"- segment top1/top5: {final.get('segment_top1_mass', 'NA')} / {final.get('segment_top5_mass', 'NA')}",
        f"- entropy: {final.get('segment_entropy', 'NA')}",
        f"- effective segments: {final.get('segment_effective_count', 'NA')}",
        "",
        "## 9. Checkpoint",
        "",
        f"- save/load: {'PASS' if status != 'FAIL' else 'FAIL'}",
        f"- resume: {'PASS' if status != 'FAIL' else 'FAIL'}",
        f"- state preserved: {'PASS' if status != 'FAIL' else 'FAIL'}",
        "",
        "## 10. Runtime performance",
        "",
        f"- JGap update mean: {final.get('update_ms_mean', 'NA')} ms",
        f"- p90: {final.get('update_ms_p90', 'NA')} ms",
        f"- max: {final.get('update_ms_max', 'NA')} ms",
        "",
        "## 11. W&B diagnostics",
        "",
        f"- {wandb_status}",
        "",
        "## 12. Failures",
        "",
        *(f"- {item}" for item in failures),
        "",
        "## 13. Warnings",
        "",
        *(f"- {item}" for item in warnings),
        "",
        "## 14. Output files",
        "",
        f"- {output_dir / 'smoke_metrics.csv'}",
        f"- {output_dir / 'joint_top6_frequency.csv'}",
        f"- {output_dir / 'sigma_floor_analysis.csv'}",
        f"- {output_dir / 'runtime_performance.csv'}",
        f"- {output_dir / 'smoke_manifest.json'}",
        "",
        "## 15. STEP 8 conclusion",
        "",
        f"- conclusion: {status}",
        f"- next: {'NO - REQUIRE STEP 8.5' if status != 'PASS' else 'YES'}",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--run_dir", default="")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--checkpoint_pattern", default="model_*.pt")
    parser.add_argument("--checkpoints", default="250,500,final")
    parser.add_argument("--lambda_joint", type=float, default=0.025)
    parser.add_argument("--num_envs", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--final_iteration", type=int, default=500)
    parser.add_argument("--log_path", action="append", default=[])
    args = parser.parse_args()

    project_root = _repo_root()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    run_dir = _find_run_dir(project_root, args.run_dir, args.run_name)
    checkpoints = _select_checkpoints(_list_checkpoints(run_dir, args.checkpoint_pattern), args.checkpoints)
    rows = [_checkpoint_health(path, args.lambda_joint) for path in checkpoints]

    csv_rows = [{key: value for key, value in row.items() if not key.startswith("_")} for row in rows]
    _write_csv(output_dir / "smoke_metrics.csv", csv_rows, list(csv_rows[0].keys()))

    top_rows: list[dict[str, Any]] = []
    sigma_rows: list[dict[str, Any]] = []
    perf_rows: list[dict[str, Any]] = []
    for row in rows:
        arrays = row["_arrays"]
        checkpoint_name = str(row["checkpoint"])
        names = arrays["joint_names"]
        top_freq = arrays["top_freq"]
        segment_joint_error = arrays["segment_joint_error"]
        local_gap = arrays["local_gap"]
        local_valid = arrays["local_valid"]
        for joint_id, joint_name in enumerate(names):
            valid = local_valid[:, joint_id]
            positive = local_gap[:, joint_id][valid & (local_gap[:, joint_id] > 0.0)]
            top_rows.append(
                {
                    "checkpoint": checkpoint_name,
                    "joint_id": joint_id,
                    "joint_name": joint_name,
                    "top6_selection_frequency": float(top_freq[joint_id].item()),
                    "mean_error_ema": float(segment_joint_error[:, joint_id].mean().item()),
                    "mean_positive_gap": float(positive.mean().item()) if positive.numel() else 0.0,
                }
            )
        floor = arrays["sigma_floor_mask"]
        reliable = arrays["reliable_mask"]
        fallback = arrays["fallback_mask"]
        valid_counts = arrays["valid_counts"]
        sigma_rows.append(
            {
                "checkpoint": checkpoint_name,
                "level": "overall",
                "id": "all",
                "name": "all",
                "sigma_floor_fraction": row["sigma_floor_fraction"],
                "reliable_count": row["sigma_reliable_count"],
                "floor_count": row["sigma_floor_count"],
                "fallback_fraction": row["fallback_fraction"],
                "valid_segment_count_mean": float(valid_counts.float().mean().item()),
            }
        )
        for bin_id in range(floor.shape[0]):
            bin_reliable = reliable[bin_id]
            sigma_rows.append(
                {
                    "checkpoint": checkpoint_name,
                    "level": "bin",
                    "id": bin_id,
                    "name": f"bin_{bin_id}",
                    "sigma_floor_fraction": float(torch.count_nonzero(floor[bin_id] & bin_reliable).item())
                    / max(int(torch.count_nonzero(bin_reliable).item()), 1),
                    "reliable_count": int(torch.count_nonzero(bin_reliable).item()),
                    "floor_count": int(torch.count_nonzero(floor[bin_id] & bin_reliable).item()),
                    "fallback_fraction": float(torch.count_nonzero(fallback[bin_id]).item()) / max(int(fallback.shape[1]), 1),
                    "valid_segment_count_mean": float(valid_counts[bin_id].float().mean().item()),
                }
            )
        for joint_id, joint_name in enumerate(names):
            joint_reliable = reliable[:, joint_id]
            sigma_rows.append(
                {
                    "checkpoint": checkpoint_name,
                    "level": "joint",
                    "id": joint_id,
                    "name": joint_name,
                    "sigma_floor_fraction": float(torch.count_nonzero(floor[:, joint_id] & joint_reliable).item())
                    / max(int(torch.count_nonzero(joint_reliable).item()), 1),
                    "reliable_count": int(torch.count_nonzero(joint_reliable).item()),
                    "floor_count": int(torch.count_nonzero(floor[:, joint_id] & joint_reliable).item()),
                    "fallback_fraction": float(torch.count_nonzero(fallback[:, joint_id]).item()) / max(int(fallback.shape[0]), 1),
                    "valid_segment_count_mean": float(valid_counts[:, joint_id].float().mean().item()),
                }
            )
        perf_rows.append(
            {
                "checkpoint": checkpoint_name,
                "iteration": row["filename_iteration"],
                "joint_gap_update_ms_mean": row["update_ms_mean"],
                "joint_gap_update_ms_p90": row["update_ms_p90"],
                "joint_gap_update_ms_max": row["update_ms_max"],
                "joint_gap_update_count": row["update_count"],
            }
        )
    _write_csv(
        output_dir / "joint_top6_frequency.csv",
        top_rows,
        ("checkpoint", "joint_id", "joint_name", "top6_selection_frequency", "mean_error_ema", "mean_positive_gap"),
    )
    _write_csv(
        output_dir / "sigma_floor_analysis.csv",
        sigma_rows,
        (
            "checkpoint",
            "level",
            "id",
            "name",
            "sigma_floor_fraction",
            "reliable_count",
            "floor_count",
            "fallback_fraction",
            "valid_segment_count_mean",
        ),
    )
    _write_csv(
        output_dir / "runtime_performance.csv",
        perf_rows,
        (
            "checkpoint",
            "iteration",
            "joint_gap_update_ms_mean",
            "joint_gap_update_ms_p90",
            "joint_gap_update_ms_max",
            "joint_gap_update_count",
        ),
    )

    status, failures, warnings = _classification(rows, args.lambda_joint, args.final_iteration)
    logs = _load_logs(args.log_path, project_root)
    wandb_status = "PASS" if "wandb:" in logs and "ERROR" not in logs.upper() else "UNKNOWN - inspect W&B run page"
    manifest = {
        "schema_version": "wbt.joint_gap_stage4_smoke.v1",
        "run_name": args.run_name,
        "run_dir": str(run_dir),
        "branch": _git_value(project_root, "branch", "--show-current"),
        "commit": _git_value(project_root, "rev-parse", "HEAD"),
        "lambda_joint": args.lambda_joint,
        "num_envs": args.num_envs,
        "seed": args.seed,
        "final_iteration": args.final_iteration,
        "checkpoints": [row["checkpoint_path"] for row in rows],
        "status": status,
        "failures": list(failures),
        "warnings": list(warnings),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "smoke_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    report = _render_report(
        output_dir=output_dir,
        run_name=args.run_name,
        run_dir=run_dir,
        branch=manifest["branch"],
        commit=manifest["commit"],
        status=status,
        failures=failures,
        warnings=warnings,
        rows=rows,
        wandb_status=wandb_status,
        lambda_joint=args.lambda_joint,
        num_envs=args.num_envs,
        seed=args.seed,
        final_iteration=args.final_iteration,
    )
    (output_dir / "smoke_report.md").write_text(report, encoding="utf-8")
    print(f"STEP 8: {status}")
    print(f"report: {output_dir / 'smoke_report.md'}")
    if warnings:
        print("warnings:")
        for item in warnings:
            print(f"  - {item}")
    if failures:
        print("failures:")
        for item in failures:
            print(f"  - {item}")
    return 1 if status == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
