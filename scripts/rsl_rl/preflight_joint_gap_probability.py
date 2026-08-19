#!/usr/bin/env python3
"""STEP 9 preflight: prove Joint Gap changes P(segment | motion).

This script is intentionally offline.  It reads a STEP 8 M7-JGap smoke
checkpoint, replays one post-warmup sampler probability rebuild with raw
segment priorities and one with the checkpoint's joint-gap-corrected
priorities, then checks that only the segment-conditional distribution changes.
It does not launch Isaac Sim, train, validate, or test a policy.
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
import types
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


DEFAULT_CHECKPOINT = (
    "logs/rsl_rl/g1_flat/"
    "2026-08-19_00-22-32_smoke_M7JGap_lam0025_n6000_seed42_500/"
    "model_498.pt"
)
DEFAULT_OUTPUT_DIR = "outputs/joint_gap_stage5_pilot10k"
DEFAULT_LAMBDA = 0.025
PROBABILITY_ATOL = 1.0e-12


@dataclass(frozen=True)
class PreflightState:
    checkpoint_path: Path
    checkpoint_iteration: int
    checkpoint_sha256: str
    research_config: Mapping[str, Any]
    sampler_state: Mapping[str, Any]
    motion_score: torch.Tensor
    motion_valid: torch.Tensor
    raw_segment_score: torch.Tensor
    raw_segment_valid: torch.Tensor
    corrected_segment_score: torch.Tensor
    correction: torch.Tensor
    joint_gap_valid: torch.Tensor
    motion_sample_count: torch.Tensor
    segment_sample_count: torch.Tensor


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
    stem = path.stem
    if not stem.startswith("model_"):
        return -1
    return int(stem.split("_", 1)[1])


def _as_tensor(value: Any, *, dtype: torch.dtype) -> torch.Tensor:
    return torch.as_tensor(value, dtype=dtype, device="cpu")


def _load_utils_modules(repo_root: Path, prefix: str = "wbt_joint_gap_preflight") -> dict[str, Any]:
    for name in list(sys.modules):
        if name == prefix or name.startswith(prefix + "."):
            del sys.modules[name]
    package_dir = repo_root / "source" / "whole_body_tracking" / "whole_body_tracking"
    root_package = types.ModuleType(prefix)
    root_package.__path__ = [str(package_dir)]
    utils_package = types.ModuleType(prefix + ".utils")
    utils_package.__path__ = [str(package_dir / "utils")]
    sys.modules[prefix] = root_package
    sys.modules[prefix + ".utils"] = utils_package
    modules: dict[str, Any] = {}
    for module_name in ("adaptive_sampling", "diversity_sampling"):
        path = package_dir / "utils" / f"{module_name}.py"
        spec = importlib.util.spec_from_file_location(prefix + ".utils." + module_name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Unable to load {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        modules[module_name] = module
    return modules


def _load_state(path: Path) -> PreflightState:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    try:
        sampling_state = checkpoint["infos"]["sampling_state"]
        research_config = sampling_state["research_config"]
        online = sampling_state["online_learning"]
        sampler = online["sampler"]
        stats = online["statistics"]
        motion_error = online["motion_error_result"]
        segment_error = online["segment_error_result"]
        joint_gap = online["joint_gap"]
        joint_gap_result = joint_gap["joint_gap_result"]
    except KeyError as exc:
        raise RuntimeError(f"{path} does not contain the STEP 8 JGap sampling state.") from exc
    if research_config.get("method_name") != "M7-JGap":
        raise RuntimeError(f"{path} is not an M7-JGap checkpoint.")
    if sampler.get("motion_mode") != "raw_error" or sampler.get("segment_mode") != "raw_error_joint_gap":
        raise RuntimeError(f"{path} does not use raw_error/raw_error_joint_gap sampling.")
    if online.get("bin_calibration") is not None or online.get("gap_result") is not None:
        raise RuntimeError(f"{path} unexpectedly contains generic learning-gap state.")
    return PreflightState(
        checkpoint_path=path,
        checkpoint_iteration=int(checkpoint.get("iter", _checkpoint_iteration(path))),
        checkpoint_sha256=_sha256_file(path),
        research_config=research_config,
        sampler_state=sampler,
        motion_score=_as_tensor(motion_error["error"], dtype=torch.float64),
        motion_valid=_as_tensor(motion_error["valid"], dtype=torch.bool),
        raw_segment_score=_as_tensor(segment_error["error"], dtype=torch.float64),
        raw_segment_valid=_as_tensor(segment_error["valid"], dtype=torch.bool),
        corrected_segment_score=_as_tensor(joint_gap_result["corrected_priority"], dtype=torch.float64),
        correction=_as_tensor(joint_gap_result["correction"], dtype=torch.float64),
        joint_gap_valid=_as_tensor(joint_gap_result["segment_valid"], dtype=torch.bool),
        motion_sample_count=_as_tensor(stats["motion_sample_count"], dtype=torch.float64),
        segment_sample_count=_as_tensor(stats["segment_sample_count"], dtype=torch.float64),
    )


def _sampler_settings(state: PreflightState) -> Mapping[str, Any]:
    online = state.research_config.get("online_learning", {})
    if not isinstance(online, Mapping):
        raise RuntimeError("research_config.online_learning is missing.")
    return online


def _make_sampler(modules: Mapping[str, Any], state: PreflightState):
    sampler_state = state.sampler_state
    settings = _sampler_settings(state)
    return modules["diversity_sampling"].DiversityConstrainedSampler(
        _as_tensor(sampler_state["segment_motion_ids"], dtype=torch.long),
        _as_tensor(sampler_state["segment_start_frames"], dtype=torch.long),
        _as_tensor(sampler_state["segment_end_frames"], dtype=torch.long),
        _as_tensor(sampler_state["motion_lengths"], dtype=torch.long),
        motion_cluster_ids=_as_tensor(sampler_state["motion_cluster_ids"], dtype=torch.long),
        motion_eligible_mask=_as_tensor(sampler_state["motion_eligible_mask"], dtype=torch.bool),
        segment_eligible_mask=_as_tensor(sampler_state["segment_eligible_mask"], dtype=torch.bool),
        motion_mode="raw_error",
        segment_mode="raw_error_joint_gap",
        warmup_iterations=int(settings.get("warmup_iterations", 1000)),
        probability_update_interval=int(settings.get("probability_update_interval", 50)),
        uniform_mix=float(settings.get("uniform_mix", 0.15)),
        temperature=float(settings.get("temperature", 1.0)),
        under_sampling_weight=float(settings.get("under_sampling_weight", 0.25)),
        motion_probability_cap=float(settings.get("motion_probability_cap", 0.02)),
        segment_probability_cap=float(settings.get("segment_probability_cap", 1.0)),
        score_clip=float(settings.get("score_clip", 10.0)),
        sampler_seed=int(settings.get("sampler_seed", 42)),
        config_hash=str(sampler_state["config_hash"]),
        num_clusters=int(sampler_state["num_clusters"]),
        minimum_budget_fraction_of_uniform=float(sampler_state["minimum_budget_fraction_of_uniform"]),
        cluster_size_exponent=float(sampler_state["cluster_size_exponent"]),
        budget_mode=str(sampler_state["budget_mode"]),
        cluster_metadata_hash=str(sampler_state.get("cluster_metadata_hash", "")),
        cluster_profile_sha256=str(sampler_state.get("cluster_profile_sha256", "")),
        cluster_schema_version=str(sampler_state.get("cluster_schema_version", "")),
        device="cpu",
    )


def _update_sampler(modules: Mapping[str, Any], state: PreflightState, segment_score: torch.Tensor, iteration: int):
    sampler = _make_sampler(modules, state)
    changed = sampler.update_probabilities(
        iteration,
        motion_score=state.motion_score,
        motion_score_valid=state.motion_valid,
        segment_score=segment_score,
        segment_score_valid=state.raw_segment_valid,
        motion_sample_count=state.motion_sample_count,
        segment_sample_count=state.segment_sample_count,
    )
    if not changed:
        raise RuntimeError(f"Post-warmup probability rebuild unexpectedly returned False at iteration {iteration}.")
    return sampler


def _tensor_hash(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous().to(torch.float64)
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()


def _probability_summary(probability: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, float]:
    prob = probability.detach().cpu().to(torch.float64)
    if mask is not None:
        prob = prob[mask.detach().cpu().to(torch.bool)]
    prob = prob[torch.isfinite(prob) & (prob > 0.0)]
    if prob.numel() == 0:
        return {"entropy": 0.0, "effective_count": 0.0, "top1_mass": 0.0, "top5_mass": 0.0, "max_probability": 0.0}
    prob = prob / prob.sum().clamp_min(1.0e-300)
    ordered = torch.sort(prob, descending=True).values
    top1 = max(1, int(math.ceil(0.01 * ordered.numel())))
    top5 = max(1, int(math.ceil(0.05 * ordered.numel())))
    entropy = float((-(prob * torch.log(prob.clamp_min(1.0e-300)))).sum().item())
    return {
        "entropy": entropy,
        "effective_count": float(1.0 / prob.square().sum().clamp_min(1.0e-300).item()),
        "top1_mass": float(ordered[:top1].sum().item()),
        "top5_mass": float(ordered[:top5].sum().item()),
        "max_probability": float(ordered[0].item()),
    }


def _per_motion_tv_rows(
    *,
    segment_motion_ids: torch.Tensor,
    raw_probability: torch.Tensor,
    jgap_probability: torch.Tensor,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    rows: list[dict[str, Any]] = []
    tv_values: list[float] = []
    num_motions = int(segment_motion_ids.max().item()) + 1 if segment_motion_ids.numel() else 0
    for motion_id in range(num_motions):
        mask = segment_motion_ids == motion_id
        if not torch.any(mask):
            continue
        raw = raw_probability[mask].to(torch.float64)
        jgap = jgap_probability[mask].to(torch.float64)
        if float(raw.sum().item()) <= 0.0 and float(jgap.sum().item()) <= 0.0:
            continue
        tv = 0.5 * torch.sum(torch.abs(raw - jgap))
        rows.append(
            {
                "motion_id": motion_id,
                "num_segments": int(torch.count_nonzero(mask).item()),
                "tv": float(tv.item()),
                "raw_max": float(raw.max().item()),
                "jgap_max": float(jgap.max().item()),
            }
        )
        tv_values.append(float(tv.item()))
    values = torch.tensor(tv_values, dtype=torch.float64)
    summary = {
        "mean_motion_tv": float(values.mean().item()) if values.numel() else 0.0,
        "p50_motion_tv": float(torch.quantile(values, torch.tensor(0.50, dtype=torch.float64)).item()) if values.numel() else 0.0,
        "p90_motion_tv": float(torch.quantile(values, torch.tensor(0.90, dtype=torch.float64)).item()) if values.numel() else 0.0,
        "p95_motion_tv": float(torch.quantile(values, torch.tensor(0.95, dtype=torch.float64)).item()) if values.numel() else 0.0,
        "max_motion_tv": float(values.max().item()) if values.numel() else 0.0,
    }
    return rows, summary


def _max_abs_diff(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.numel() == 0:
        return 0.0
    return float(torch.max(torch.abs(left.to(torch.float64) - right.to(torch.float64))).item())


def _nonfinite_paths(value: Any, prefix: str = "") -> list[str]:
    paths: list[str] = []
    if isinstance(value, torch.Tensor):
        if value.is_floating_point() and not torch.all(torch.isfinite(value)):
            paths.append(prefix)
    elif isinstance(value, Mapping):
        for key, item in value.items():
            paths.extend(_nonfinite_paths(item, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            paths.extend(_nonfinite_paths(item, f"{prefix}[{index}]"))
    return paths


def run_preflight(
    *,
    checkpoint: Path,
    output_dir: Path,
    lambda_joint: float = DEFAULT_LAMBDA,
    update_iteration: int | None = None,
    project_root: Path | None = None,
) -> dict[str, Any]:
    project_root = project_root or _repo_root()
    checkpoint = checkpoint if checkpoint.is_absolute() else project_root / checkpoint
    output_dir = output_dir if output_dir.is_absolute() else project_root / output_dir
    state = _load_state(checkpoint)
    modules = _load_utils_modules(project_root)
    settings = _sampler_settings(state)
    forced_iteration = int(update_iteration if update_iteration is not None else settings.get("warmup_iterations", 1000))
    if forced_iteration < int(settings.get("warmup_iterations", 1000)):
        raise ValueError("update_iteration must be at or after warmup_iterations.")

    raw_sampler = _update_sampler(modules, state, state.raw_segment_score, forced_iteration)
    jgap_sampler = _update_sampler(modules, state, state.corrected_segment_score, forced_iteration)

    segment_motion_ids = _as_tensor(state.sampler_state["segment_motion_ids"], dtype=torch.long)
    segment_eligible = _as_tensor(state.sampler_state["segment_eligible_mask"], dtype=torch.bool)
    raw_global = raw_sampler.motion_probability[segment_motion_ids] * raw_sampler.segment_probability
    jgap_global = jgap_sampler.motion_probability[segment_motion_ids] * jgap_sampler.segment_probability
    segment_abs_diff = torch.abs(jgap_sampler.segment_probability - raw_sampler.segment_probability)
    changed_mask = segment_abs_diff > PROBABILITY_ATOL
    active_mask = state.raw_segment_valid & state.joint_gap_valid & torch.isfinite(state.correction) & (state.correction > 0.0)
    valid_multiplier = (
        state.raw_segment_valid
        & torch.isfinite(state.raw_segment_score)
        & (state.raw_segment_score > 0.0)
        & torch.isfinite(state.corrected_segment_score)
    )
    multiplier = torch.ones_like(state.raw_segment_score)
    multiplier[valid_multiplier] = state.corrected_segment_score[valid_multiplier] / state.raw_segment_score[valid_multiplier].clamp_min(1.0e-300)

    per_motion_rows, per_motion_summary = _per_motion_tv_rows(
        segment_motion_ids=segment_motion_ids,
        raw_probability=raw_sampler.segment_probability,
        jgap_probability=jgap_sampler.segment_probability,
    )
    raw_summary = _probability_summary(raw_global, segment_eligible)
    jgap_summary = _probability_summary(jgap_global, segment_eligible)
    cluster_diff = _max_abs_diff(raw_sampler.cluster_probability, jgap_sampler.cluster_probability)
    motion_conditional_diff = _max_abs_diff(
        raw_sampler.motion_probability_conditional,
        jgap_sampler.motion_probability_conditional,
    )
    motion_global_diff = _max_abs_diff(raw_sampler.motion_probability, jgap_sampler.motion_probability)
    segment_diff = _max_abs_diff(raw_sampler.segment_probability, jgap_sampler.segment_probability)
    global_segment_diff = _max_abs_diff(raw_global, jgap_global)
    extra_mass_on_active = float(torch.clamp(jgap_global[active_mask] - raw_global[active_mask], min=0.0).sum().item())
    net_mass_on_active = float((jgap_global[active_mask] - raw_global[active_mask]).sum().item())

    failures: list[str] = []
    warnings: list[str] = []
    if _nonfinite_paths(state.research_config) or _nonfinite_paths(state.sampler_state):
        failures.append("Checkpoint sampling state contains non-finite values.")
    if not math.isclose(float(lambda_joint), float(settings["joint_gap"]["lambda_joint"]), rel_tol=0.0, abs_tol=1.0e-12):
        failures.append("Requested lambda does not match checkpoint joint_gap.lambda_joint.")
    if cluster_diff > PROBABILITY_ATOL:
        failures.append(f"P(c) changed under JGap replay: max diff {cluster_diff:.6g}.")
    if motion_conditional_diff > PROBABILITY_ATOL or motion_global_diff > PROBABILITY_ATOL:
        failures.append(
            "P(m|c) or P(m) changed under JGap replay: "
            f"conditional {motion_conditional_diff:.6g}, global {motion_global_diff:.6g}."
        )
    if int(torch.count_nonzero(active_mask).item()) == 0:
        failures.append("JGap correction is inactive in the smoke checkpoint.")
    if int(torch.count_nonzero(changed_mask & segment_eligible).item()) == 0:
        failures.append("P(segment|motion) did not change after applying JGap corrected priorities.")
    if raw_sampler.fallback_count != jgap_sampler.fallback_count:
        warnings.append("Sampler fallback count differs between raw and JGap probability rebuilds.")

    if failures:
        status = "BLOCKED" if any("P(segment|motion)" in item or "inactive" in item for item in failures) else "FAIL"
    elif warnings:
        status = "PASS WITH WARNING"
    else:
        status = "PASS"

    report: dict[str, Any] = {
        "status": status,
        "failures": failures,
        "warnings": warnings,
        "checkpoint": str(checkpoint),
        "checkpoint_iteration": state.checkpoint_iteration,
        "checkpoint_sha256": state.checkpoint_sha256,
        "branch": _git_value(project_root, "branch", "--show-current"),
        "commit": _git_value(project_root, "rev-parse", "HEAD"),
        "method_name": state.research_config.get("method_name"),
        "motion_mode": state.sampler_state.get("motion_mode"),
        "segment_mode": state.sampler_state.get("segment_mode"),
        "warmup_iterations": int(settings.get("warmup_iterations", 1000)),
        "forced_update_iteration": forced_iteration,
        "smoke_sampler_probability_updates": int(state.sampler_state.get("probability_update_count", -1)),
        "raw_rebuild_probability_updates": raw_sampler.probability_update_count,
        "jgap_rebuild_probability_updates": jgap_sampler.probability_update_count,
        "lambda_joint": float(lambda_joint),
        "active_corrected_segments": int(torch.count_nonzero(active_mask).item()),
        "eligible_segments": int(torch.count_nonzero(segment_eligible).item()),
        "changed_conditional_segments": int(torch.count_nonzero(changed_mask & segment_eligible).item()),
        "max_abs_diff_cluster_probability": cluster_diff,
        "max_abs_diff_motion_probability_conditional": motion_conditional_diff,
        "max_abs_diff_motion_probability": motion_global_diff,
        "max_abs_diff_segment_probability_conditional": segment_diff,
        "max_abs_diff_global_segment_probability": global_segment_diff,
        "raw_segment_probability_hash": _tensor_hash(raw_sampler.segment_probability),
        "jgap_segment_probability_hash": _tensor_hash(jgap_sampler.segment_probability),
        "raw_global_segment_probability_hash": _tensor_hash(raw_global),
        "jgap_global_segment_probability_hash": _tensor_hash(jgap_global),
        "correction_mean": float(state.correction[active_mask].mean().item()) if torch.any(active_mask) else 0.0,
        "correction_p90": float(torch.quantile(state.correction[active_mask], torch.tensor(0.90, dtype=torch.float64)).item()) if torch.any(active_mask) else 0.0,
        "correction_max": float(state.correction[active_mask].max().item()) if torch.any(active_mask) else 0.0,
        "multiplier_mean": float(multiplier[valid_multiplier].mean().item()) if torch.any(valid_multiplier) else 1.0,
        "multiplier_max": float(multiplier[valid_multiplier].max().item()) if torch.any(valid_multiplier) else 1.0,
        "extra_global_mass_on_active_corrected_segments": extra_mass_on_active,
        "net_global_mass_on_active_corrected_segments": net_mass_on_active,
        "raw_global_segment_summary": raw_summary,
        "jgap_global_segment_summary": jgap_summary,
        "per_motion_tv": per_motion_summary,
        "raw_sampler_fallback_count": raw_sampler.fallback_count,
        "jgap_sampler_fallback_count": jgap_sampler.fallback_count,
    }
    _write_outputs(output_dir, report, per_motion_rows)
    return report


def _write_outputs(output_dir: Path, report: Mapping[str, Any], per_motion_rows: Sequence[Mapping[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "preflight_probability.json"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    csv_path = output_dir / "preflight_probability_per_motion.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["motion_id", "num_segments", "tv", "raw_max", "jgap_max"])
        writer.writeheader()
        writer.writerows(per_motion_rows)
    status = str(report["status"])
    lines = [
        "# STEP 9 JGap Probability Preflight",
        "",
        f"STEP 9 preflight: {status}",
        "",
        "## Identity",
        "",
        f"- branch: {report['branch']}",
        f"- commit: {report['commit']}",
        f"- checkpoint: {report['checkpoint']}",
        f"- checkpoint iteration: {report['checkpoint_iteration']}",
        f"- checkpoint sha256: {report['checkpoint_sha256']}",
        f"- method: {report['method_name']}",
        f"- motion mode: {report['motion_mode']}",
        f"- segment mode: {report['segment_mode']}",
        "",
        "## Smoke Probability Counter",
        "",
        f"- smoke sampler probability updates: {report['smoke_sampler_probability_updates']}",
        f"- warmup iterations: {report['warmup_iterations']}",
        f"- forced replay iteration: {report['forced_update_iteration']}",
        "- interpretation: the STEP 8 500-iteration smoke stayed inside warmup, so zero runtime sampler probability updates are expected.",
        "",
        "## Distribution Replay",
        "",
        f"- raw replay updates: {report['raw_rebuild_probability_updates']}",
        f"- jgap replay updates: {report['jgap_rebuild_probability_updates']}",
        f"- active corrected segments: {report['active_corrected_segments']} / {report['eligible_segments']}",
        f"- changed conditional segments: {report['changed_conditional_segments']}",
        f"- max |Δ P(c)|: {report['max_abs_diff_cluster_probability']:.12g}",
        f"- max |Δ P(m|c)|: {report['max_abs_diff_motion_probability_conditional']:.12g}",
        f"- max |Δ P(m)|: {report['max_abs_diff_motion_probability']:.12g}",
        f"- max |Δ P(s|m)|: {report['max_abs_diff_segment_probability_conditional']:.12g}",
        f"- max |Δ P(c,m,s)|: {report['max_abs_diff_global_segment_probability']:.12g}",
        f"- raw P(s|m) hash: {report['raw_segment_probability_hash']}",
        f"- jgap P(s|m) hash: {report['jgap_segment_probability_hash']}",
        "",
        "## Correction",
        "",
        f"- lambda: {report['lambda_joint']}",
        f"- correction mean/p90/max: {report['correction_mean']:.12g} / {report['correction_p90']:.12g} / {report['correction_max']:.12g}",
        f"- multiplier mean/max: {report['multiplier_mean']:.12g} / {report['multiplier_max']:.12g}",
        f"- positive global mass shifted onto corrected segments: {report['extra_global_mass_on_active_corrected_segments']:.12g}",
        f"- net global mass on corrected segments: {report['net_global_mass_on_active_corrected_segments']:.12g}",
        "",
        "## Concentration",
        "",
        f"- raw top1/top5/max: {report['raw_global_segment_summary']['top1_mass']:.12g} / {report['raw_global_segment_summary']['top5_mass']:.12g} / {report['raw_global_segment_summary']['max_probability']:.12g}",
        f"- jgap top1/top5/max: {report['jgap_global_segment_summary']['top1_mass']:.12g} / {report['jgap_global_segment_summary']['top5_mass']:.12g} / {report['jgap_global_segment_summary']['max_probability']:.12g}",
        f"- mean/p95/max motion TV: {report['per_motion_tv']['mean_motion_tv']:.12g} / {report['per_motion_tv']['p95_motion_tv']:.12g} / {report['per_motion_tv']['max_motion_tv']:.12g}",
    ]
    if report["failures"]:
        lines.extend(["", "## Failures", ""])
        lines.extend(f"- {item}" for item in report["failures"])
    if report["warnings"]:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {item}" for item in report["warnings"])
    (output_dir / "preflight_probability_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--lambda_joint", type=float, default=DEFAULT_LAMBDA)
    parser.add_argument("--update_iteration", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run_preflight(
        checkpoint=Path(args.checkpoint),
        output_dir=Path(args.output_dir),
        lambda_joint=args.lambda_joint,
        update_iteration=args.update_iteration,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] in {"PASS", "PASS WITH WARNING"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
