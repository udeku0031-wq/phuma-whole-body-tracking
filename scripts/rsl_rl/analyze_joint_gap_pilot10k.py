#!/usr/bin/env python3
"""Analyze STEP 9 M7-JGap 10k pilot against the strict M7-Raw 10k reference.

The analysis is local and offline: it reads TensorBoard event files and saved
training checkpoints.  It does not launch Isaac Sim, run Validation, run Test,
or modify the policy.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


DEFAULT_RUN_NAME = "pilot_M7JGap_lam0025_n6000_trainseed42_10k"
DEFAULT_REFERENCE_RUN_DIR = "logs/rsl_rl/g1_flat/2026-08-05_15-31-20_formal_v1_ablation_M7Raw_n6000_trainseed42"
DEFAULT_OUTPUT_DIR = "outputs/joint_gap_stage5_pilot10k"
DEFAULT_CHECKPOINTS = "1000,3000,5000,7000,10000,final"
DEFAULT_TOP6_CHECKPOINTS = {1000, 5000, 10000}
DEFAULT_LAMBDA = 0.025
EXPECTED_MIN_FINAL_ITERATION = 9999

SCALAR_TAGS = (
    "Train/mean_reward",
    "Train/mean_episode_length",
    "Episode_Reward/motion_global_anchor_pos",
    "Episode_Reward/motion_global_anchor_ori",
    "Episode_Reward/motion_body_pos",
    "Episode_Reward/motion_body_ori",
    "Episode_Reward/motion_body_lin_vel",
    "Episode_Reward/motion_body_ang_vel",
    "Episode_Reward/undesired_contacts",
    "Metrics/motion/error_body_pos",
    "Metrics/motion/error_joint_pos",
    "Metrics/motion/error_body_lin_vel",
    "Metrics/motion/error_body_ang_vel",
    "Metrics/motion/sampling_entropy",
    "Metrics/motion/sampling_top1_prob",
    "Metrics/motion/sampling_top1_bin",
    "Loss/value_function",
    "Loss/surrogate",
    "Loss/entropy",
    "Loss/learning_rate",
    "Perf/total_fps",
    "sampling/probability_update_count",
    "sampling/fallback_count",
    "sampling/target_observed_l1",
    "sampling/max_motion_probability",
    "sampling/max_segment_probability",
    "sampling/segment_entropy",
    "sampling/effective_segment_count",
    "joint_gap/active_segment_fraction",
    "joint_gap/correction_mean",
    "joint_gap/correction_p90",
    "joint_gap/correction_max",
    "joint_gap/segment_top1_mass",
    "joint_gap/segment_top5_mass",
    "joint_gap/effective_segment_count",
    "joint_gap/sigma_floor_fraction",
    "joint_gap/update_ms_mean",
    "joint_gap/update_ms_p90",
)

LOWER_IS_BETTER = {
    "Episode_Reward/undesired_contacts",
    "Metrics/motion/error_body_pos",
    "Metrics/motion/error_joint_pos",
    "Metrics/motion/error_body_lin_vel",
    "Metrics/motion/error_body_ang_vel",
    "Loss/value_function",
    "sampling/fallback_count",
    "sampling/target_observed_l1",
    "sampling/max_motion_probability",
    "sampling/max_segment_probability",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_smoke_module(project_root: Path):
    path = project_root / "scripts" / "rsl_rl" / "analyze_joint_gap_smoke.py"
    spec = importlib.util.spec_from_file_location("wbt_joint_gap_smoke_analysis_for_pilot", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
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
    if not path.stem.startswith("model_"):
        raise ValueError(f"Cannot parse checkpoint iteration from {path}")
    return int(path.stem.split("_", 1)[1])


def _find_run_dir(project_root: Path, run_dir: str, run_name: str) -> Path:
    if run_dir:
        path = Path(run_dir)
        return path if path.is_absolute() else project_root / path
    matches = sorted((project_root / "logs" / "rsl_rl" / "g1_flat").glob(f"*_{run_name}"))
    if not matches:
        raise FileNotFoundError(f"Could not find logs/rsl_rl/g1_flat/*_{run_name}")
    return matches[-1]


def _event_files(run_dir: Path) -> list[Path]:
    files = sorted(run_dir.glob("events.out.tfevents*"))
    if not files:
        raise FileNotFoundError(f"No TensorBoard event file found in {run_dir}")
    return files


def _load_scalars(run_dir: Path) -> dict[str, list[tuple[int, float]]]:
    events: dict[str, list[tuple[int, float]]] = {}
    for path in _event_files(run_dir):
        accumulator = EventAccumulator(str(path), size_guidance={"scalars": 0})
        accumulator.Reload()
        for tag in accumulator.Tags().get("scalars", []):
            values = [(int(item.step), float(item.value)) for item in accumulator.Scalars(tag)]
            if not values:
                continue
            events.setdefault(tag, []).extend(values)
    for tag, values in events.items():
        dedup: dict[int, float] = {}
        for step, value in values:
            dedup[step] = value
        events[tag] = sorted(dedup.items())
    return events


def _latest_step(events: Mapping[str, Sequence[tuple[int, float]]]) -> int:
    steps = [values[-1][0] for values in events.values() if values]
    if not steps:
        return -1
    return max(steps)


def _window_mean(values: Sequence[tuple[int, float]], *, end_step: int, window: int) -> float | None:
    selected = [value for step, value in values if end_step - window < step <= end_step and math.isfinite(value)]
    if not selected:
        return None
    return float(sum(selected) / len(selected))


def _rolling_best(values: Sequence[tuple[int, float]], *, window: int, higher_is_better: bool) -> tuple[int, float] | None:
    finite = [(step, value) for step, value in values if math.isfinite(value)]
    if not finite:
        return None
    best: tuple[int, float] | None = None
    for index, (step, _value) in enumerate(finite):
        selected = [value for item_step, value in finite[: index + 1] if step - window < item_step <= step]
        if len(selected) < max(1, window // 2):
            continue
        mean_value = float(sum(selected) / len(selected))
        if best is None:
            best = (step, mean_value)
        elif higher_is_better and mean_value > best[1]:
            best = (step, mean_value)
        elif not higher_is_better and mean_value < best[1]:
            best = (step, mean_value)
    return best


def _scalar_rows(
    *,
    label: str,
    events: Mapping[str, Sequence[tuple[int, float]]],
    compare_to: Mapping[str, Sequence[tuple[int, float]]] | None,
    end_step: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for tag in SCALAR_TAGS:
        values = events.get(tag, ())
        if not values:
            rows.append({"run": label, "tag": tag, "present": 0})
            continue
        final = next((value for step, value in reversed(values) if step <= end_step), None)
        last500 = _window_mean(values, end_step=end_step, window=500)
        last1000 = _window_mean(values, end_step=end_step, window=1000)
        best = _rolling_best(values, window=500, higher_is_better=tag not in LOWER_IS_BETTER)
        ref_last500 = None
        ref_last1000 = None
        if compare_to is not None and tag in compare_to:
            ref_last500 = _window_mean(compare_to[tag], end_step=end_step, window=500)
            ref_last1000 = _window_mean(compare_to[tag], end_step=end_step, window=1000)
        rows.append(
            {
                "run": label,
                "tag": tag,
                "present": 1,
                "final": "" if final is None else f"{final:.12g}",
                "last500": "" if last500 is None else f"{last500:.12g}",
                "last1000": "" if last1000 is None else f"{last1000:.12g}",
                "best_rolling500_step": "" if best is None else best[0],
                "best_rolling500": "" if best is None else f"{best[1]:.12g}",
                "reference_last500": "" if ref_last500 is None else f"{ref_last500:.12g}",
                "reference_last1000": "" if ref_last1000 is None else f"{ref_last1000:.12g}",
                "delta_last500_vs_reference": (
                    ""
                    if last500 is None or ref_last500 is None
                    else f"{last500 - ref_last500:.12g}"
                ),
                "delta_last1000_vs_reference": (
                    ""
                    if last1000 is None or ref_last1000 is None
                    else f"{last1000 - ref_last1000:.12g}"
                ),
            }
        )
    return rows


def _select_checkpoints(run_dir: Path, requested: str) -> list[Path]:
    available = sorted(run_dir.glob("model_*.pt"), key=_checkpoint_iteration)
    if not available:
        raise FileNotFoundError(f"No checkpoints found in {run_dir}")
    by_iter = {_checkpoint_iteration(path): path for path in available}
    selected: list[Path] = []
    seen: set[Path] = set()
    for token in [item.strip() for item in requested.split(",") if item.strip()]:
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


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(columns or rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _reference_identity(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    sampling_state = checkpoint["infos"]["sampling_state"]
    online = sampling_state["online_learning"]
    sampler = online["sampler"]
    return {
        "checkpoint": str(path),
        "checkpoint_sha256": _sha256(path),
        "checkpoint_iter": int(checkpoint.get("iter", -1)),
        "method_name": sampling_state.get("research_config", {}).get("method_name"),
        "motion_mode": sampler.get("motion_mode"),
        "segment_mode": sampler.get("segment_mode"),
        "probability_update_count": int(sampler.get("probability_update_count", -1)),
    }


def _classification(
    *,
    pilot_rows: Sequence[Mapping[str, Any]],
    scalar_rows: Sequence[Mapping[str, Any]],
    final_iteration: int,
    lambda_joint: float,
) -> tuple[str, list[str], list[str]]:
    failures: list[str] = []
    warnings: list[str] = []
    if not pilot_rows:
        failures.append("No pilot checkpoints were analyzed.")
        return "FAIL", failures, warnings
    final = pilot_rows[-1]
    if int(final["filename_iteration"]) < final_iteration:
        failures.append(
            f"Final pilot checkpoint is {final['checkpoint']}, expected at least model_{final_iteration}.pt."
        )
    if final["motion_mode"] != "raw_error" or final["segment_mode"] != "raw_error_joint_gap":
        failures.append("Pilot sampler mode is not raw_error/raw_error_joint_gap.")
    if int(final["method_ok"]) != 1:
        failures.append("Pilot method_name is not M7-JGap.")
    if int(final["generic_gap_off"]) != 1:
        failures.append("Generic learning-gap state is present in the pilot.")
    if not math.isclose(float(final["lambda_joint"]), lambda_joint, rel_tol=0.0, abs_tol=1.0e-12):
        failures.append("Pilot lambda does not match frozen STEP 7 value.")
    if int(final["nonfinite"]) != 0:
        failures.append(f"Pilot contains non-finite sampling state: {final['nonfinite_paths']}")
    if int(final["sampler_probability_update_count"]) <= 0:
        failures.append("Pilot sampler probability_update_count did not advance after warmup.")
    if float(final["active_segment_fraction"]) <= 0.0:
        failures.append("Joint-Gap correction is inactive at the final checkpoint.")
    if float(final["multiplier_max"]) > 1.0 + lambda_joint + 1.0e-6:
        failures.append("Joint-Gap multiplier exceeds the frozen lambda bound.")
    if float(final["sampler_fallback_count"]) > 0.0:
        warnings.append("Sampler fallback count is nonzero.")
    if float(final["sigma_floor_fraction"]) > 0.90:
        warnings.append("sigma_floor_fraction > 0.90.")
    if float(final["segment_top1_mass"]) > 0.10 or float(final["segment_top5_mass"]) > 0.35:
        warnings.append("Segment probability concentration is high.")
    reward_row = next((row for row in scalar_rows if row.get("tag") == "Train/mean_reward"), None)
    if reward_row and reward_row.get("delta_last1000_vs_reference") not in ("", None):
        if float(reward_row["delta_last1000_vs_reference"]) < -0.5:
            warnings.append("Train/mean_reward last1000 is materially below M7-Raw 10k.")
    if failures:
        return "FAIL", failures, warnings
    if warnings:
        return "PASS WITH WARNING", failures, warnings
    return "PASS", failures, warnings


def _render_report(
    *,
    output_dir: Path,
    status: str,
    failures: Sequence[str],
    warnings: Sequence[str],
    run_name: str,
    run_dir: Path,
    reference_identity: Mapping[str, Any],
    pilot_rows: Sequence[Mapping[str, Any]],
    scalar_rows: Sequence[Mapping[str, Any]],
    latest_step: int,
) -> str:
    final = pilot_rows[-1] if pilot_rows else {}

    def scalar(tag: str, field: str) -> str:
        row = next((item for item in scalar_rows if item.get("tag") == tag), None)
        if not row:
            return "NA"
        value = row.get(field, "")
        return "NA" if value == "" else str(value)

    lines = [
        "# STEP 9 M7-JGap 10k Pilot Report",
        "",
        f"STEP 9: {status}",
        "",
        "## 1. Run identity",
        "",
        f"- run name: {run_name}",
        f"- run dir: {run_dir}",
        f"- branch: {_git_value(_repo_root(), 'branch', '--show-current')}",
        f"- commit: {_git_value(_repo_root(), 'rev-parse', 'HEAD')}",
        f"- latest scalar step: {latest_step}",
        "",
        "## 2. Strict 10k reference",
        "",
        f"- checkpoint: {reference_identity['checkpoint']}",
        f"- checkpoint iter: {reference_identity['checkpoint_iter']}",
        f"- method: {reference_identity['method_name']}",
        f"- motion/segment: {reference_identity['motion_mode']} / {reference_identity['segment_mode']}",
        "",
        "## 3. Pilot contract",
        "",
        "- Quality: ON",
        "- Diversity: ON",
        "- Difficulty: ON",
        "- Motion: raw_error",
        "- Segment: raw_error_joint_gap",
        "- Generic Gap: OFF",
        f"- Joint Gap lambda: {final.get('lambda_joint', 'NA')}",
        "",
        "## 4. Training window comparison vs M7-Raw 10k",
        "",
        f"- mean reward last1000: {scalar('Train/mean_reward', 'last1000')} (delta {scalar('Train/mean_reward', 'delta_last1000_vs_reference')})",
        f"- episode length last1000: {scalar('Train/mean_episode_length', 'last1000')} (delta {scalar('Train/mean_episode_length', 'delta_last1000_vs_reference')})",
        f"- body pos error last1000: {scalar('Metrics/motion/error_body_pos', 'last1000')} (delta {scalar('Metrics/motion/error_body_pos', 'delta_last1000_vs_reference')})",
        f"- joint pos error last1000: {scalar('Metrics/motion/error_joint_pos', 'last1000')} (delta {scalar('Metrics/motion/error_joint_pos', 'delta_last1000_vs_reference')})",
        f"- best reward rolling500: {scalar('Train/mean_reward', 'best_rolling500')} @ {scalar('Train/mean_reward', 'best_rolling500_step')}",
        f"- best joint error rolling500: {scalar('Metrics/motion/error_joint_pos', 'best_rolling500')} @ {scalar('Metrics/motion/error_joint_pos', 'best_rolling500_step')}",
        "",
        "## 5. Sampling and JGap health",
        "",
        f"- sampler probability updates: {final.get('sampler_probability_update_count', 'NA')}",
        f"- sampler fallback count: {final.get('sampler_fallback_count', 'NA')}",
        f"- cluster target-observed L1: {final.get('cluster_target_observed_l1', 'NA')}",
        f"- segment top1/top5: {final.get('segment_top1_mass', 'NA')} / {final.get('segment_top5_mass', 'NA')}",
        f"- effective segments: {final.get('segment_effective_count', 'NA')}",
        f"- active segment fraction: {final.get('active_segment_fraction', 'NA')}",
        f"- correction mean/p90/max: {final.get('correction_mean', 'NA')} / {final.get('correction_p90', 'NA')} / {final.get('correction_max', 'NA')}",
        f"- multiplier max: {final.get('multiplier_max', 'NA')}",
        "",
        "## 6. Calibration and degeneracy",
        "",
        f"- sigma floor fraction: {final.get('sigma_floor_fraction', 'NA')}",
        f"- Spearman gap vs raw segment: {final.get('spearman_gap_raw_segment', 'NA')}",
        f"- Spearman gap vs mean joint raw: {final.get('spearman_gap_raw_joint_mean', 'NA')}",
        f"- Spearman gap vs top-k joint raw: {final.get('spearman_gap_raw_joint_topk', 'NA')}",
        "",
        "## 7. Runtime overhead",
        "",
        f"- JGap update mean/p90/max ms: {final.get('update_ms_mean', 'NA')} / {final.get('update_ms_p90', 'NA')} / {final.get('update_ms_max', 'NA')}",
        f"- total fps last1000: {scalar('Perf/total_fps', 'last1000')} (delta {scalar('Perf/total_fps', 'delta_last1000_vs_reference')})",
        "",
        "## 8. Output files",
        "",
        f"- {output_dir / 'training_window_comparison.csv'}",
        f"- {output_dir / 'joint_gap_trajectory.csv'}",
        f"- {output_dir / 'joint_top6_frequency.csv'}",
        f"- {output_dir / 'sampling_concentration.csv'}",
        f"- {output_dir / 'runtime_performance.csv'}",
        f"- {output_dir / 'pilot_manifest.json'}",
        "",
        "## 9. STEP 10 gate",
        "",
        f"- can enter fixed validation_probe500: {'YES' if status in {'PASS', 'PASS WITH WARNING'} else 'NO'}",
    ]
    if failures:
        lines.extend(["", "## Failures", ""])
        lines.extend(f"- {item}" for item in failures)
    if warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {item}" for item in warnings)
    return "\n".join(lines) + "\n"


def run_analysis(
    *,
    run_name: str,
    run_dir: Path,
    reference_run_dir: Path,
    output_dir: Path,
    checkpoints: str,
    lambda_joint: float,
    final_iteration: int,
    project_root: Path | None = None,
) -> dict[str, Any]:
    project_root = project_root or _repo_root()
    run_dir = _find_run_dir(project_root, str(run_dir) if str(run_dir) != "." else "", run_name)
    reference_run_dir = reference_run_dir if reference_run_dir.is_absolute() else project_root / reference_run_dir
    output_dir = output_dir if output_dir.is_absolute() else project_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    smoke = _load_smoke_module(project_root)
    selected = _select_checkpoints(run_dir, checkpoints)
    pilot_health = [smoke._checkpoint_health(path, lambda_joint) for path in selected]
    health_rows = [{key: value for key, value in row.items() if not key.startswith("_")} for row in pilot_health]
    _write_csv(output_dir / "joint_gap_trajectory.csv", health_rows)

    degeneracy_rows = [
        {
            "checkpoint": row["checkpoint"],
            "iteration": row["filename_iteration"],
            "spearman_gap_raw_segment": row["spearman_gap_raw_segment"],
            "spearman_gap_raw_joint_mean": row["spearman_gap_raw_joint_mean"],
            "spearman_gap_raw_joint_topk": row["spearman_gap_raw_joint_topk"],
            "sigma_floor_fraction": row["sigma_floor_fraction"],
            "active_segment_fraction": row["active_segment_fraction"],
            "correction_max": row["correction_max"],
        }
        for row in pilot_health
    ]
    _write_csv(output_dir / "sampling_concentration.csv", degeneracy_rows)

    top6_rows: list[dict[str, Any]] = []
    sigma_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    for row in pilot_health:
        arrays = row["_arrays"]
        iteration = int(row["filename_iteration"])
        names = arrays["joint_names"]
        top_freq = arrays["top_freq"]
        if iteration in DEFAULT_TOP6_CHECKPOINTS or row is pilot_health[-1]:
            for joint_id, joint_name in enumerate(names):
                top6_rows.append(
                    {
                        "checkpoint": row["checkpoint"],
                        "iteration": iteration,
                        "joint_id": joint_id,
                        "joint_name": joint_name,
                        "top6_selection_frequency": float(top_freq[joint_id].item()),
                    }
                )
        floor = arrays["sigma_floor_mask"]
        reliable = arrays["reliable_mask"]
        fallback = arrays["fallback_mask"]
        valid_counts = arrays["valid_counts"]
        sigma_rows.append(
            {
                "checkpoint": row["checkpoint"],
                "iteration": iteration,
                "level": "overall",
                "id": "all",
                "name": "all",
                "sigma_floor_fraction": row["sigma_floor_fraction"],
                "fallback_fraction": row["fallback_fraction"],
                "reliable_count": row["sigma_reliable_count"],
                "floor_count": row["sigma_floor_count"],
                "valid_segment_count_mean": float(valid_counts.float().mean().item()),
            }
        )
        for bin_id in range(floor.shape[0]):
            bin_reliable = reliable[bin_id]
            sigma_rows.append(
                {
                    "checkpoint": row["checkpoint"],
                    "iteration": iteration,
                    "level": "bin",
                    "id": bin_id,
                    "name": f"bin_{bin_id}",
                    "sigma_floor_fraction": float(torch.count_nonzero(floor[bin_id] & bin_reliable).item())
                    / max(int(torch.count_nonzero(bin_reliable).item()), 1),
                    "fallback_fraction": float(torch.count_nonzero(fallback[bin_id]).item()) / max(int(fallback.shape[1]), 1),
                    "reliable_count": int(torch.count_nonzero(bin_reliable).item()),
                    "floor_count": int(torch.count_nonzero(floor[bin_id] & bin_reliable).item()),
                    "valid_segment_count_mean": float(valid_counts[bin_id].float().mean().item()),
                }
            )
        runtime_rows.append(
            {
                "checkpoint": row["checkpoint"],
                "iteration": iteration,
                "joint_gap_update_ms_mean": row["update_ms_mean"],
                "joint_gap_update_ms_p90": row["update_ms_p90"],
                "joint_gap_update_ms_max": row["update_ms_max"],
                "joint_gap_update_count": row["update_count"],
            }
        )
    _write_csv(output_dir / "joint_top6_frequency.csv", top6_rows)
    _write_csv(output_dir / "sigma_floor_trajectory.csv", sigma_rows)
    _write_csv(output_dir / "runtime_performance.csv", runtime_rows)

    pilot_events = _load_scalars(run_dir)
    reference_events = _load_scalars(reference_run_dir)
    pilot_latest_step = _latest_step(pilot_events)
    compare_step = min(max(pilot_latest_step, 0), 10000)
    scalar_rows = _scalar_rows(
        label="M7-JGap-pilot10k",
        events=pilot_events,
        compare_to=reference_events,
        end_step=compare_step,
    )
    _write_csv(output_dir / "training_window_comparison.csv", scalar_rows)

    reference_checkpoint = min(reference_run_dir.glob("model_*.pt"), key=lambda path: (abs(_checkpoint_iteration(path) - 10000), _checkpoint_iteration(path)))
    reference_identity = _reference_identity(reference_checkpoint)
    if (
        reference_identity["method_name"] != "M7-Raw"
        or reference_identity["motion_mode"] != "raw_error"
        or reference_identity["segment_mode"] != "raw_error"
    ):
        raise RuntimeError("Strict 10k reference is not M7-Raw raw_error/raw_error.")

    status, failures, warnings = _classification(
        pilot_rows=health_rows,
        scalar_rows=scalar_rows,
        final_iteration=final_iteration,
        lambda_joint=lambda_joint,
    )
    manifest = {
        "schema_version": "wbt.joint_gap_stage5_pilot10k.v1",
        "status": status,
        "failures": list(failures),
        "warnings": list(warnings),
        "run_name": run_name,
        "run_dir": str(run_dir),
        "branch": _git_value(project_root, "branch", "--show-current"),
        "commit": _git_value(project_root, "rev-parse", "HEAD"),
        "lambda_joint": lambda_joint,
        "latest_scalar_step": pilot_latest_step,
        "analysis_compare_step": compare_step,
        "checkpoints": [row["checkpoint_path"] for row in health_rows],
        "reference": reference_identity,
    }
    (output_dir / "pilot_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    report = _render_report(
        output_dir=output_dir,
        status=status,
        failures=failures,
        warnings=warnings,
        run_name=run_name,
        run_dir=run_dir,
        reference_identity=reference_identity,
        pilot_rows=health_rows,
        scalar_rows=scalar_rows,
        latest_step=pilot_latest_step,
    )
    (output_dir / "pilot_training_report.md").write_text(report, encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--run_dir", default="")
    parser.add_argument("--reference_run_dir", default=DEFAULT_REFERENCE_RUN_DIR)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--checkpoints", default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--lambda_joint", type=float, default=DEFAULT_LAMBDA)
    parser.add_argument("--final_iteration", type=int, default=EXPECTED_MIN_FINAL_ITERATION)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run_analysis(
        run_name=args.run_name,
        run_dir=Path(args.run_dir or "."),
        reference_run_dir=Path(args.reference_run_dir),
        output_dir=Path(args.output_dir),
        checkpoints=args.checkpoints,
        lambda_joint=args.lambda_joint,
        final_iteration=args.final_iteration,
    )
    print(f"STEP 9: {manifest['status']}")
    print(f"report: {Path(args.output_dir) / 'pilot_training_report.md'}")
    if manifest["warnings"]:
        print("warnings:")
        for item in manifest["warnings"]:
            print(f"  - {item}")
    if manifest["failures"]:
        print("failures:")
        for item in manifest["failures"]:
            print(f"  - {item}")
    return 0 if manifest["status"] in {"PASS", "PASS WITH WARNING"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
