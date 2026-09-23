#!/usr/bin/env python3
"""Offline STEP 7 lambda replay for Joint-Specific Gap.

This script never launches Isaac Sim, trains a policy, updates PPO, runs
Validation, or runs Test.  It consumes frozen-policy proxy snapshots collected
on the training motion library and M7-Raw checkpoint sampling state, then
measures how the fixed lambda candidates perturb the segment sampling
distribution.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import re
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch


DEFAULT_RUN_DIR = "logs/rsl_rl/g1_flat/2026-08-05_15-31-20_formal_v1_ablation_M7Raw_n6000_trainseed42"
DEFAULT_TRAIN_MANIFEST = "PHUMA_wbt_motions/manifests/experiments/random_seed42/random6000_seed42.txt"
DEFAULT_QUALITY_METADATA = "outputs/module1_quality_random6000_seed42_original_v1/segment_quality_metadata.npz"
DEFAULT_DIFFICULTY_METADATA = "outputs/module2_difficulty_random6000_seed42_v1/segment_difficulty_metadata.npz"
DEFAULT_CLUSTER_METADATA = "outputs/module4_clusters_random6000_seed42_v1/motion_cluster_metadata.npz"
DEFAULT_SNAPSHOT_DIR = "outputs/joint_gap_stage3_lambda_replay/snapshots"
DEFAULT_OUTPUT_DIR = "outputs/joint_gap_stage3_lambda_replay"
M7_RAW_BASELINE_COMMIT = "9591b69138f25846b2ba44214ffd29cbaa022ef0"
STEP5_IMPLEMENTATION_COMMIT = "e4343d2"
STEP6_IDENTITY_COMMIT = "245fd75"
LAMBDA_CANDIDATES = (0.0, 0.025, 0.050, 0.100)
SLICE_SUFFIX_RE = re.compile(r"(?i)(?:_chunk_\d+|_chunk\d+|-chunk-\d+)$")
REBUILD_AUDIT_ATOL = 1.0e-8


class MissingSnapshotsError(RuntimeError):
    def __init__(self, missing: Sequence[Path]) -> None:
        super().__init__("Missing STEP 7 snapshot(s): " + ", ".join(str(path) for path in missing))
        self.missing = list(missing)


@dataclass(frozen=True)
class CheckpointState:
    path: Path
    iteration: int
    checkpoint_sha256: str
    research_config: dict[str, Any]
    motion_lengths: torch.Tensor
    segment_motion_ids: torch.Tensor
    segment_start_frames: torch.Tensor
    segment_end_frames: torch.Tensor
    motion_eligible_mask: torch.Tensor
    segment_eligible_mask: torch.Tensor
    motion_cluster_ids: torch.Tensor
    raw_cluster_probability: torch.Tensor
    raw_motion_probability_conditional: torch.Tensor
    raw_motion_probability: torch.Tensor
    raw_segment_probability: torch.Tensor
    raw_motion_score: torch.Tensor
    raw_motion_valid: torch.Tensor
    raw_segment_priority: torch.Tensor
    raw_segment_valid: torch.Tensor
    motion_sample_count: torch.Tensor
    segment_sample_count: torch.Tensor
    sampler_state: dict[str, Any]


@dataclass(frozen=True)
class Snapshot:
    path: Path
    checkpoint_iteration: int
    checkpoint_path: str
    checkpoint_sha256: str
    train_manifest_sha256: str
    segment_joint_error: torch.Tensor
    segment_observation_count: torch.Tensor
    eligible_mask: torch.Tensor
    observed_mask: torch.Tensor
    difficulty_bin: torch.Tensor
    motion_cluster_id: torch.Tensor
    motion_segment_offsets: torch.Tensor
    motion_id: torch.Tensor
    local_segment_id: torch.Tensor
    start_frame: torch.Tensor
    end_frame_exclusive: torch.Tensor
    joint_names: list[str]
    joint_mapping_hash: str
    categories: list[str]
    source_groups: list[str]
    proxy_statistic: str
    proxy_limitation: str


@dataclass(frozen=True)
class JointGapDetails:
    calibration: Any
    global_gap: torch.Tensor
    global_valid: torch.Tensor
    local_gap: torch.Tensor
    local_valid: torch.Tensor
    topk_values: torch.Tensor
    topk_indices: torch.Tensor
    topk_selection_frequency: torch.Tensor
    raw_gate_mask: torch.Tensor
    segment_gap_score: torch.Tensor
    segment_valid: torch.Tensor
    correction: torch.Tensor
    corrected_priority: torch.Tensor
    timings_ms: dict[str, float]


@dataclass(frozen=True)
class ReplayConfig:
    project_root: Path
    run_dir: Path
    snapshots_dir: Path
    output_dir: Path
    checkpoints: str
    checkpoint_pattern: str
    train_manifest: Path
    quality_metadata: Path
    difficulty_metadata: Path
    cluster_metadata: Path
    write_freeze_doc: bool = True


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _absolute(path_text: str | os.PathLike[str], project_root: Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else project_root / path


def _sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(project_root: Path) -> str:
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return ""


def _load_utils_modules(repo_root: Path, prefix: str = "wbt_lambda_replay") -> dict[str, Any]:
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
    for module_name in ("adaptive_sampling", "diversity_sampling", "learning_gap", "joint_gap"):
        path = package_dir / "utils" / f"{module_name}.py"
        spec = importlib.util.spec_from_file_location(prefix + ".utils." + module_name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Unable to load {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        modules[module_name] = module
    return modules


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


def _select_checkpoints(available: Sequence[Path], targets: str) -> list[Path]:
    by_iter = {_checkpoint_iteration(path): path for path in available}
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
    if 33999 in by_iter and by_iter[33999] not in seen:
        selected.append(by_iter[33999])
    return selected


def _snapshot_path(snapshots_dir: Path, checkpoint: Path) -> Path:
    return snapshots_dir / f"m7raw_ckpt_{_checkpoint_iteration(checkpoint)}_joint_snapshot.npz"


def _as_tensor(value: Any, *, dtype: torch.dtype) -> torch.Tensor:
    return torch.as_tensor(value, dtype=dtype, device="cpu")


def _load_checkpoint_state(path: Path) -> CheckpointState:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    try:
        sampling_state = checkpoint["infos"]["sampling_state"]
        online = sampling_state["online_learning"]
        sampler = online["sampler"]
        stats = online["statistics"]
        segment_error = online["segment_error_result"]
        motion_error = online["motion_error_result"]
    except KeyError as exc:
        raise RuntimeError(f"{path} does not contain the M7-Raw sampling state required by STEP 7.") from exc
    if segment_error is None or motion_error is None:
        raise RuntimeError(f"{path} is missing raw segment/motion priority caches.")
    if sampler.get("motion_mode") != "raw_error" or sampler.get("segment_mode") != "raw_error":
        raise RuntimeError(f"{path} is not an M7-Raw raw_error/raw_error checkpoint.")
    return CheckpointState(
        path=path,
        iteration=_checkpoint_iteration(path),
        checkpoint_sha256=_sha256_file(path),
        research_config=dict(sampling_state.get("research_config", {})),
        motion_lengths=_as_tensor(sampler["motion_lengths"], dtype=torch.long),
        segment_motion_ids=_as_tensor(sampler["segment_motion_ids"], dtype=torch.long),
        segment_start_frames=_as_tensor(sampler["segment_start_frames"], dtype=torch.long),
        segment_end_frames=_as_tensor(sampler["segment_end_frames"], dtype=torch.long),
        motion_eligible_mask=_as_tensor(sampler["motion_eligible_mask"], dtype=torch.bool),
        segment_eligible_mask=_as_tensor(sampler["segment_eligible_mask"], dtype=torch.bool),
        motion_cluster_ids=_as_tensor(sampler["motion_cluster_ids"], dtype=torch.long),
        raw_cluster_probability=_as_tensor(sampler["cluster_probability"], dtype=torch.float64),
        raw_motion_probability_conditional=_as_tensor(sampler["motion_probability_conditional"], dtype=torch.float64),
        raw_motion_probability=_as_tensor(sampler["motion_probability"], dtype=torch.float64),
        raw_segment_probability=_as_tensor(sampler["segment_probability"], dtype=torch.float64),
        raw_motion_score=_as_tensor(motion_error["error"], dtype=torch.float64),
        raw_motion_valid=_as_tensor(motion_error["valid"], dtype=torch.bool),
        raw_segment_priority=_as_tensor(segment_error["error"], dtype=torch.float64),
        raw_segment_valid=_as_tensor(segment_error["valid"], dtype=torch.bool),
        motion_sample_count=_as_tensor(stats["motion_sample_count"], dtype=torch.float64),
        segment_sample_count=_as_tensor(stats["segment_sample_count"], dtype=torch.float64),
        sampler_state=dict(sampler),
    )


def _string_array(npz: Any, name: str, fallback: Sequence[str] | None = None) -> list[str]:
    if name not in npz.files:
        return [str(item) for item in (fallback or [])]
    return np.asarray(npz[name]).astype(str).tolist()


def _load_snapshot(path: Path, manifest_sha256: str) -> Snapshot:
    with np.load(path, allow_pickle=False) as data:
        schema = str(np.asarray(data["schema_version"]).item())
        if schema != "wbt.joint_gap.proxy_snapshot.v1":
            raise RuntimeError(f"{path} has unsupported snapshot schema {schema!r}.")
        snapshot_manifest_sha = str(np.asarray(data["train_manifest_sha256"]).item())
        if snapshot_manifest_sha != manifest_sha256:
            raise RuntimeError(f"{path} was not collected on the requested training manifest.")
        segment_joint_error = _as_tensor(data["segment_joint_error"], dtype=torch.float64)
        segment_observation_count = _as_tensor(data["segment_observation_count"], dtype=torch.long)
        eligible_mask = _as_tensor(data["eligible_mask"], dtype=torch.bool)
        observed_mask = _as_tensor(data["observed_mask"], dtype=torch.bool)
        difficulty_bin = _as_tensor(data["difficulty_bin"], dtype=torch.long)
        motion_id = _as_tensor(data["motion_id"], dtype=torch.long)
        categories = _string_array(data, "category")
        source_groups = _string_array(data, "source_group")
        if not categories:
            categories = ["__unknown__"] * int(torch.max(motion_id).item() + 1)
        if not source_groups:
            source_groups = ["__unknown__"] * len(categories)
        return Snapshot(
            path=path,
            checkpoint_iteration=int(np.asarray(data["checkpoint_iteration"]).item()),
            checkpoint_path=str(np.asarray(data["checkpoint_path"]).item()),
            checkpoint_sha256=str(np.asarray(data["checkpoint_sha256"]).item()),
            train_manifest_sha256=snapshot_manifest_sha,
            segment_joint_error=segment_joint_error,
            segment_observation_count=segment_observation_count,
            eligible_mask=eligible_mask,
            observed_mask=observed_mask,
            difficulty_bin=difficulty_bin,
            motion_cluster_id=_as_tensor(data["motion_cluster_id"], dtype=torch.long),
            motion_segment_offsets=_as_tensor(data["motion_segment_offsets"], dtype=torch.long),
            motion_id=motion_id,
            local_segment_id=_as_tensor(data["local_segment_id"], dtype=torch.long),
            start_frame=_as_tensor(data["start_frame"], dtype=torch.long),
            end_frame_exclusive=_as_tensor(data["end_frame_exclusive"], dtype=torch.long),
            joint_names=_string_array(data, "joint_names"),
            joint_mapping_hash=str(np.asarray(data["joint_mapping_hash"]).item()),
            categories=categories,
            source_groups=source_groups,
            proxy_statistic=str(np.asarray(data["proxy_statistic"]).item()),
            proxy_limitation=str(np.asarray(data["proxy_limitation"]).item()),
        )


def _validate_pair(state: CheckpointState, snapshot: Snapshot) -> None:
    num_segments = int(state.segment_motion_ids.numel())
    num_motions = int(state.motion_lengths.numel())
    if snapshot.checkpoint_iteration != state.iteration:
        raise RuntimeError(f"{snapshot.path} iteration does not match {state.path}.")
    if snapshot.checkpoint_sha256 != state.checkpoint_sha256:
        raise RuntimeError(f"{snapshot.path} checkpoint SHA does not match {state.path}.")
    if snapshot.segment_joint_error.shape[0] != num_segments:
        raise RuntimeError(f"{snapshot.path} segment count does not match checkpoint layout.")
    if snapshot.motion_id.shape != state.segment_motion_ids.shape or not torch.equal(snapshot.motion_id, state.segment_motion_ids):
        raise RuntimeError(f"{snapshot.path} segment motion IDs do not match checkpoint layout.")
    if snapshot.motion_cluster_id.shape != state.motion_cluster_ids.shape or not torch.equal(snapshot.motion_cluster_id, state.motion_cluster_ids):
        raise RuntimeError(f"{snapshot.path} cluster IDs do not match checkpoint layout.")
    if snapshot.motion_segment_offsets.numel() != num_motions + 1:
        raise RuntimeError(f"{snapshot.path} motion_segment_offsets does not match checkpoint motion count.")
    if snapshot.eligible_mask.shape != state.segment_eligible_mask.shape:
        raise RuntimeError(f"{snapshot.path} eligibility shape does not match checkpoint layout.")


def _metadata_identity(config: ReplayConfig) -> dict[str, str]:
    return {
        "train_manifest_path": str(config.train_manifest.resolve()),
        "train_manifest_sha256": _sha256_file(config.train_manifest),
        "quality_metadata_path": str(config.quality_metadata.resolve()),
        "quality_metadata_sha256": _sha256_file(config.quality_metadata),
        "difficulty_metadata_path": str(config.difficulty_metadata.resolve()),
        "difficulty_metadata_sha256": _sha256_file(config.difficulty_metadata),
        "cluster_metadata_path": str(config.cluster_metadata.resolve()),
        "cluster_metadata_sha256": _sha256_file(config.cluster_metadata),
    }


def _sampler_settings(state: CheckpointState) -> dict[str, Any]:
    online = dict(state.research_config.get("online_learning", {}))
    return {
        "warmup_iterations": int(online.get("warmup_iterations", 1000)),
        "probability_update_interval": int(online.get("probability_update_interval", 50)),
        "uniform_mix": float(online.get("uniform_mix", 0.15)),
        "temperature": float(online.get("temperature", 1.0)),
        "under_sampling_weight": float(online.get("under_sampling_weight", 0.25)),
        "motion_probability_cap": float(online.get("motion_probability_cap", 0.02)),
        "segment_probability_cap": float(online.get("segment_probability_cap", 1.0)),
        "score_clip": float(online.get("score_clip", 10.0)),
        "sampler_seed": int(online.get("sampler_seed", 42)),
        "min_segment_observations": int(online.get("min_segment_observations", 32)),
        "min_bin_valid_segments": int(online.get("min_bin_valid_segments", 32)),
        "sigma_floor": float(online.get("sigma_floor", 0.10)),
        "gap_clip": float(online.get("gap_clip", 5.0)),
        "bin_observation_weighted": bool(online.get("bin_observation_weighted", False)),
    }


def _make_sampler(modules: dict[str, Any], state: CheckpointState, *, segment_mode: str):
    settings = _sampler_settings(state)
    sampler_state = state.sampler_state
    return modules["diversity_sampling"].DiversityConstrainedSampler(
        state.segment_motion_ids,
        state.segment_start_frames,
        state.segment_end_frames,
        state.motion_lengths,
        motion_cluster_ids=state.motion_cluster_ids,
        motion_eligible_mask=state.motion_eligible_mask,
        segment_eligible_mask=state.segment_eligible_mask,
        motion_mode="raw_error",
        segment_mode=segment_mode,
        warmup_iterations=settings["warmup_iterations"],
        probability_update_interval=settings["probability_update_interval"],
        uniform_mix=settings["uniform_mix"],
        temperature=settings["temperature"],
        under_sampling_weight=settings["under_sampling_weight"],
        motion_probability_cap=settings["motion_probability_cap"],
        segment_probability_cap=settings["segment_probability_cap"],
        score_clip=settings["score_clip"],
        sampler_seed=settings["sampler_seed"],
        config_hash=str(sampler_state["config_hash"]),
        num_clusters=int(sampler_state["num_clusters"]),
        minimum_budget_fraction_of_uniform=float(sampler_state["minimum_budget_fraction_of_uniform"]),
        cluster_size_exponent=float(sampler_state["cluster_size_exponent"]),
        budget_mode=str(sampler_state["budget_mode"]),
        cluster_metadata_hash=str(sampler_state["cluster_metadata_hash"]),
        cluster_profile_sha256=str(sampler_state["cluster_profile_sha256"]),
        cluster_schema_version=str(sampler_state["cluster_schema_version"]),
        device="cpu",
    )


def _quantile(values: torch.Tensor, q: float) -> float:
    selected = values.detach().cpu().to(torch.float64)
    selected = selected[torch.isfinite(selected)]
    if selected.numel() == 0:
        return 0.0
    return float(torch.quantile(selected, torch.tensor(q, dtype=torch.float64)).item())


def _probability_stats(probability: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, float | int]:
    prob = probability.detach().cpu().to(torch.float64)
    if mask is not None:
        prob = prob[mask.detach().cpu().to(torch.bool)]
    prob = prob[torch.isfinite(prob) & (prob > 0.0)]
    if prob.numel() == 0:
        return {"entropy": 0.0, "effective_count": 0.0, "top1_mass": 0.0, "top5_mass": 0.0, "max_probability": 0.0}
    total = prob.sum()
    if total > 0.0:
        prob = prob / total
    ordered = torch.sort(prob, descending=True).values
    top1_count = max(1, int(math.ceil(0.01 * ordered.numel())))
    top5_count = max(1, int(math.ceil(0.05 * ordered.numel())))
    return {
        "entropy": float((-(prob * torch.log(prob.clamp_min(1.0e-300))).sum()).item()),
        "effective_count": float(1.0 / torch.sum(prob.square()).item()),
        "top1_mass": float(ordered[:top1_count].sum().item()),
        "top5_mass": float(ordered[:top5_count].sum().item()),
        "max_probability": float(prob.max().item()),
    }


def _tv_distance(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.detach().cpu().to(torch.float64)
    right = right.detach().cpu().to(torch.float64)
    return float(0.5 * torch.sum(torch.abs(left - right)).item())


def _js_divergence(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.detach().cpu().to(torch.float64)
    right = right.detach().cpu().to(torch.float64)
    left = left / left.sum().clamp_min(1.0e-300)
    right = right / right.sum().clamp_min(1.0e-300)
    midpoint = 0.5 * (left + right)
    left_term = torch.where(left > 0.0, left * torch.log(left.clamp_min(1.0e-300) / midpoint.clamp_min(1.0e-300)), torch.zeros_like(left))
    right_term = torch.where(right > 0.0, right * torch.log(right.clamp_min(1.0e-300) / midpoint.clamp_min(1.0e-300)), torch.zeros_like(right))
    return float((0.5 * left_term.sum() + 0.5 * right_term.sum()).item())


def _average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape[0], dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < sorted_values.shape[0]:
        end = start + 1
        while end < sorted_values.shape[0] and sorted_values[end] == sorted_values[start]:
            end += 1
        rank = 0.5 * (start + end - 1) + 1.0
        ranks[order[start:end]] = rank
        start = end
    return ranks


def _spearman(left: torch.Tensor, right: torch.Tensor) -> float:
    left_np = left.detach().cpu().to(torch.float64).numpy()
    right_np = right.detach().cpu().to(torch.float64).numpy()
    mask = np.isfinite(left_np) & np.isfinite(right_np)
    left_np = left_np[mask]
    right_np = right_np[mask]
    if left_np.shape[0] < 2:
        return 1.0
    left_rank = _average_ranks(left_np)
    right_rank = _average_ranks(right_np)
    left_std = float(left_rank.std())
    right_std = float(right_rank.std())
    if left_std == 0.0 or right_std == 0.0:
        return 1.0 if np.array_equal(left_np, right_np) else 0.0
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def _top_overlap(left: torch.Tensor, right: torch.Tensor, fraction: float | None = None, count: int | None = None) -> float:
    if count is None:
        count = max(1, int(math.ceil(float(fraction) * int(left.numel()))))
    count = min(int(count), int(left.numel()), int(right.numel()))
    if count <= 0:
        return 1.0
    left_top = set(torch.topk(left, k=count).indices.detach().cpu().tolist())
    right_top = set(torch.topk(right, k=count).indices.detach().cpu().tolist())
    return len(left_top & right_top) / float(count)


def _compute_joint_gap_details(
    modules: dict[str, Any],
    *,
    state: CheckpointState,
    snapshot: Snapshot,
    lambda_joint: float,
) -> JointGapDetails:
    joint_gap = modules["joint_gap"]
    settings = _sampler_settings(state)
    segment_valid = (
        snapshot.eligible_mask
        & snapshot.observed_mask
        & (snapshot.segment_observation_count >= settings["min_segment_observations"])
    )
    weights = snapshot.segment_observation_count if settings["bin_observation_weighted"] else None
    t0 = time.perf_counter()
    calibration = joint_gap.compute_joint_bin_statistics(
        snapshot.segment_joint_error,
        segment_valid,
        snapshot.difficulty_bin,
        num_bins=10,
        min_bin_valid_segments=settings["min_bin_valid_segments"],
        sigma_floor=settings["sigma_floor"],
        observation_weights=weights,
    )
    t1 = time.perf_counter()
    global_gap, global_valid, _ = joint_gap.compute_joint_global_gap(
        snapshot.segment_joint_error,
        segment_valid,
        snapshot.difficulty_bin,
        calibration,
        gap_clip=settings["gap_clip"],
    )
    t2 = time.perf_counter()
    local = joint_gap.compute_joint_local_gap(
        global_gap,
        global_valid,
        state.segment_motion_ids,
        num_motions=int(state.motion_lengths.numel()),
        gap_clip=settings["gap_clip"],
    )
    t3 = time.perf_counter()
    topk = joint_gap.aggregate_topk_joint_gap(local.local_gap, local.local_valid, top_k=6)
    t4 = time.perf_counter()
    raw_gate, _, _ = joint_gap.compute_raw_gate(
        state.raw_segment_priority,
        state.raw_segment_valid,
        state.segment_motion_ids,
        num_motions=int(state.motion_lengths.numel()),
    )
    finite_score = torch.isfinite(topk.segment_gap_score)
    segment_valid_final = topk.segment_valid & state.raw_segment_valid & finite_score
    correction = torch.where(
        raw_gate & segment_valid_final,
        torch.tanh(torch.clamp(topk.segment_gap_score, min=0.0)),
        torch.zeros_like(state.raw_segment_priority),
    )
    correction = torch.nan_to_num(correction, nan=0.0, posinf=0.0, neginf=0.0)
    correction = torch.clamp(correction, 0.0, 1.0 - 1.0e-12)
    if float(lambda_joint) == 0.0:
        corrected = state.raw_segment_priority
    else:
        corrected = state.raw_segment_priority * (1.0 + float(lambda_joint) * correction)
    corrected = torch.where(state.raw_segment_valid & torch.isfinite(corrected), corrected, state.raw_segment_priority)
    authoritative = joint_gap.compute_joint_gap_correction(
        raw_priority=state.raw_segment_priority,
        raw_valid=state.raw_segment_valid,
        segment_joint_error=snapshot.segment_joint_error,
        segment_joint_valid=segment_valid,
        segment_motion_ids=state.segment_motion_ids,
        difficulty_bin=snapshot.difficulty_bin,
        calibration=calibration,
        num_motions=int(state.motion_lengths.numel()),
        top_k=6,
        lambda_joint=float(lambda_joint),
        gap_clip=settings["gap_clip"],
    )
    if not torch.equal(authoritative.corrected_priority, corrected) or not torch.equal(authoritative.correction, correction):
        raise RuntimeError("Joint-gap replay formula drifted from the STEP 5 authoritative function.")
    t5 = time.perf_counter()
    return JointGapDetails(
        calibration=calibration,
        global_gap=global_gap,
        global_valid=global_valid,
        local_gap=local.local_gap,
        local_valid=local.local_valid,
        topk_values=topk.topk_values,
        topk_indices=topk.topk_indices,
        topk_selection_frequency=topk.topk_selection_frequency,
        raw_gate_mask=raw_gate,
        segment_gap_score=topk.segment_gap_score,
        segment_valid=segment_valid_final,
        correction=correction,
        corrected_priority=corrected,
        timings_ms={
            "joint_calibration_ms": (t1 - t0) * 1000.0,
            "global_gap_ms": (t2 - t1) * 1000.0,
            "local_median_ms": (t3 - t2) * 1000.0,
            "top6_ms": (t4 - t3) * 1000.0,
            "authoritative_check_and_correction_ms": (t5 - t4) * 1000.0,
        },
    )


def _probabilities_for_lambda(
    modules: dict[str, Any],
    state: CheckpointState,
    corrected_priority: torch.Tensor,
) -> tuple[Any, dict[str, torch.Tensor], float]:
    sampler = _make_sampler(modules, state, segment_mode="raw_error_joint_gap")
    iteration = int(state.sampler_state.get("last_probability_update_iteration", state.iteration))
    start = time.perf_counter()
    changed = sampler.update_probabilities(
        iteration,
        motion_score=state.raw_motion_score,
        motion_score_valid=state.raw_motion_valid,
        segment_score=corrected_priority,
        segment_score_valid=state.raw_segment_valid,
        motion_sample_count=state.motion_sample_count,
        segment_sample_count=state.segment_sample_count,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    if not changed:
        raise RuntimeError("Lambda replay expected a probability update.")
    for name, rebuilt, frozen in (
        ("cluster_probability", sampler.cluster_probability, state.raw_cluster_probability),
        (
            "motion_probability_conditional",
            sampler.motion_probability_conditional,
            state.raw_motion_probability_conditional,
        ),
        ("motion_probability", sampler.motion_probability, state.raw_motion_probability),
    ):
        max_diff = float(torch.max(torch.abs(rebuilt - frozen)).item()) if rebuilt.numel() else 0.0
        if max_diff > REBUILD_AUDIT_ATOL:
            raise RuntimeError(f"{name} changed during lambda replay rebuild; max_diff={max_diff:.6g}.")
    segment_conditional = sampler.segment_probability
    if torch.equal(corrected_priority, state.raw_segment_priority):
        segment_conditional = state.raw_segment_probability
    return sampler, {
        "cluster": state.raw_cluster_probability,
        "motion_conditional": state.raw_motion_probability_conditional,
        "motion": state.raw_motion_probability,
        "segment_conditional": segment_conditional,
        "global_segment": state.raw_motion_probability[state.segment_motion_ids] * segment_conditional,
    }, elapsed_ms


def _relative_change(new: float, old: float) -> float:
    if old == 0.0:
        return 0.0 if new == 0.0 else float("inf")
    return (new / old) - 1.0


def _per_motion_rows(
    *,
    checkpoint_iteration: int,
    lambda_joint: float,
    state: CheckpointState,
    raw_segment_probability: torch.Tensor,
    new_segment_probability: torch.Tensor,
) -> tuple[list[dict[str, object]], dict[str, float]]:
    rows: list[dict[str, object]] = []
    tv_values: list[float] = []
    for motion_id in range(int(state.motion_lengths.numel())):
        mask = state.segment_motion_ids == motion_id
        if not torch.any(mask):
            continue
        raw = raw_segment_probability[mask]
        new = new_segment_probability[mask]
        if float(raw.sum().item()) <= 0.0 and float(new.sum().item()) <= 0.0:
            continue
        tv = _tv_distance(raw, new)
        js = _js_divergence(raw, new)
        rank = _spearman(raw, new)
        raw_stats = _probability_stats(raw)
        new_stats = _probability_stats(new)
        count = int(torch.count_nonzero(mask).item())
        tv_values.append(tv)
        rows.append(
            {
                "checkpoint": checkpoint_iteration,
                "lambda": f"{lambda_joint:.3f}",
                "motion_id": motion_id,
                "num_segments": count,
                "TV": f"{tv:.12g}",
                "JS": f"{js:.12g}",
                "rank_correlation": f"{rank:.12g}",
                "raw_entropy": f"{raw_stats['entropy']:.12g}",
                "new_entropy": f"{new_stats['entropy']:.12g}",
                "raw_top1": f"{raw_stats['top1_mass']:.12g}",
                "new_top1": f"{new_stats['top1_mass']:.12g}",
            }
        )
    tv_tensor = torch.tensor(tv_values, dtype=torch.float64)
    summary = {
        "mean_motion_tv": float(tv_tensor.mean().item()) if tv_tensor.numel() else 0.0,
        "median_motion_tv": _quantile(tv_tensor, 0.50),
        "p90_motion_tv": _quantile(tv_tensor, 0.90),
        "p95_motion_tv": _quantile(tv_tensor, 0.95),
        "p99_motion_tv": _quantile(tv_tensor, 0.99),
        "max_motion_tv": float(tv_tensor.max().item()) if tv_tensor.numel() else 0.0,
    }
    return rows, summary


def _segment_count_group_summary(state: CheckpointState, per_motion_rows: Sequence[dict[str, object]]) -> dict[str, float]:
    rows_by_motion = {int(row["motion_id"]): float(row["TV"]) for row in per_motion_rows}
    segment_counts = torch.bincount(state.segment_motion_ids, minlength=int(state.motion_lengths.numel()))
    summaries: dict[str, float] = {}
    groups = {
        "count1": torch.where(segment_counts == 1)[0].tolist(),
        "count2": torch.where(segment_counts == 2)[0].tolist(),
        "short": torch.where(segment_counts <= 2)[0].tolist(),
        "long": torch.where(segment_counts >= 8)[0].tolist(),
    }
    for name, motion_ids in groups.items():
        values = [rows_by_motion[motion_id] for motion_id in motion_ids if motion_id in rows_by_motion]
        summaries[f"{name}_mean_tv"] = float(np.mean(values)) if values else 0.0
        summaries[f"{name}_max_tv"] = float(np.max(values)) if values else 0.0
    return summaries


def _candidate_row(
    *,
    state: CheckpointState,
    snapshot: Snapshot,
    lambda_joint: float,
    details: JointGapDetails,
    probabilities: dict[str, torch.Tensor],
    probability_rebuild_ms: float,
    per_motion_summary: dict[str, float],
) -> dict[str, object]:
    eligible = state.segment_eligible_mask
    raw_global = state.raw_motion_probability[state.segment_motion_ids] * state.raw_segment_probability
    new_global = probabilities["global_segment"]
    raw_stats = _probability_stats(raw_global, eligible)
    new_stats = _probability_stats(new_global, eligible)
    active = eligible & (details.correction > 0.0)
    active_boost = float(lambda_joint) * details.correction[active]
    multiplicative = 1.0 + float(lambda_joint) * details.correction[eligible]
    raw_gate_fraction = (
        float(torch.count_nonzero(details.raw_gate_mask & eligible).item())
        / max(int(torch.count_nonzero(eligible).item()), 1)
    )
    active_fraction = (
        float(torch.count_nonzero(active).item())
        / max(int(torch.count_nonzero(eligible).item()), 1)
    )
    segment_cap = _sampler_settings(state)["segment_probability_cap"]
    cap_hits_raw = int(torch.count_nonzero(state.raw_segment_probability[eligible] >= segment_cap - 1.0e-12).item())
    cap_hits_new = int(torch.count_nonzero(probabilities["segment_conditional"][eligible] >= segment_cap - 1.0e-12).item())
    cluster_exact = bool(torch.equal(probabilities["cluster"], state.raw_cluster_probability))
    motion_conditional_exact = bool(torch.equal(probabilities["motion_conditional"], state.raw_motion_probability_conditional))
    motion_exact = bool(torch.equal(probabilities["motion"], state.raw_motion_probability))
    entropy_change = _relative_change(float(new_stats["entropy"]), float(raw_stats["entropy"]))
    top1_change = _relative_change(float(new_stats["top1_mass"]), float(raw_stats["top1_mass"]))
    top5_change = _relative_change(float(new_stats["top5_mass"]), float(raw_stats["top5_mass"]))
    spearman = _spearman(raw_global[eligible], new_global[eligible])
    measurable = _quantile(active_boost, 0.90) >= 0.01 if float(lambda_joint) > 0.0 else False
    cap_pathology = cap_hits_new > cap_hits_raw
    safety_pass = (
        cluster_exact
        and motion_conditional_exact
        and motion_exact
        and entropy_change >= -0.01
        and top1_change <= 0.05
        and top5_change <= 0.05
        and per_motion_summary["median_motion_tv"] <= 0.01
        and per_motion_summary["p95_motion_tv"] <= 0.03
        and spearman >= 0.98
        and not cap_pathology
    )
    timings = dict(details.timings_ms)
    timings["probability_rebuild_ms"] = probability_rebuild_ms
    return {
        "checkpoint": state.iteration,
        "lambda": f"{lambda_joint:.3f}",
        "mean_correction_factor": f"{float(multiplicative.mean().item()):.12g}",
        "p50_correction_factor": f"{_quantile(multiplicative, 0.50):.12g}",
        "p90_correction_factor": f"{_quantile(multiplicative, 0.90):.12g}",
        "p95_correction_factor": f"{_quantile(multiplicative, 0.95):.12g}",
        "max_correction_factor": f"{float(multiplicative.max().item()) if multiplicative.numel() else 1.0:.12g}",
        "active_correction_fraction": f"{active_fraction:.12g}",
        "raw_gate_pass_fraction": f"{raw_gate_fraction:.12g}",
        "p90_active_boost": f"{_quantile(active_boost, 0.90):.12g}",
        "raw_segment_entropy": f"{raw_stats['entropy']:.12g}",
        "segment_entropy": f"{new_stats['entropy']:.12g}",
        "entropy_change": f"{entropy_change:.12g}",
        "raw_effective_segment_count": f"{raw_stats['effective_count']:.12g}",
        "effective_segment_count": f"{new_stats['effective_count']:.12g}",
        "raw_top1_mass": f"{raw_stats['top1_mass']:.12g}",
        "top1_mass": f"{new_stats['top1_mass']:.12g}",
        "top1_change": f"{top1_change:.12g}",
        "raw_top5_mass": f"{raw_stats['top5_mass']:.12g}",
        "top5_mass": f"{new_stats['top5_mass']:.12g}",
        "top5_change": f"{top5_change:.12g}",
        "max_segment_probability": f"{new_stats['max_probability']:.12g}",
        "global_tv": f"{_tv_distance(raw_global, new_global):.12g}",
        "global_js": f"{_js_divergence(raw_global, new_global):.12g}",
        "spearman": f"{spearman:.12g}",
        "top100_overlap": f"{_top_overlap(raw_global[eligible], new_global[eligible], count=100):.12g}",
        "top1pct_overlap": f"{_top_overlap(raw_global[eligible], new_global[eligible], fraction=0.01):.12g}",
        "top5pct_overlap": f"{_top_overlap(raw_global[eligible], new_global[eligible], fraction=0.05):.12g}",
        "mean_motion_tv": f"{per_motion_summary['mean_motion_tv']:.12g}",
        "median_motion_tv": f"{per_motion_summary['median_motion_tv']:.12g}",
        "p90_motion_tv": f"{per_motion_summary['p90_motion_tv']:.12g}",
        "p95_motion_tv": f"{per_motion_summary['p95_motion_tv']:.12g}",
        "p99_motion_tv": f"{per_motion_summary['p99_motion_tv']:.12g}",
        "max_motion_tv": f"{per_motion_summary['max_motion_tv']:.12g}",
        "cap_hits_raw": cap_hits_raw,
        "cap_hits_new": cap_hits_new,
        "cap_hit_delta": cap_hits_new - cap_hits_raw,
        "waterfill_pathology": int(cap_pathology),
        "sigma_floor_fraction": f"{float(details.calibration.sigma_floor_mask.to(torch.float64).mean().item()):.12g}",
        "cluster_exact": int(cluster_exact),
        "motion_conditional_exact": int(motion_conditional_exact),
        "motion_exact": int(motion_exact),
        "safety_pass": int(safety_pass),
        "measurable": int(measurable),
        "status": "PASS" if safety_pass and (measurable or float(lambda_joint) == 0.0) else ("TOO_WEAK" if safety_pass else "FAIL"),
        **{name: f"{value:.12g}" for name, value in timings.items()},
    }


def _top_boosted_rows(
    *,
    state: CheckpointState,
    snapshot: Snapshot,
    lambda_joint: float,
    details: JointGapDetails,
    probabilities: dict[str, torch.Tensor],
) -> list[dict[str, object]]:
    eligible_ids = torch.where(state.segment_eligible_mask & (details.correction > 0.0))[0]
    if eligible_ids.numel() == 0:
        return []
    count = max(1, int(math.ceil(0.01 * int(torch.count_nonzero(state.segment_eligible_mask).item()))))
    top_ids = eligible_ids[torch.topk(details.correction[eligible_ids], k=min(count, int(eligible_ids.numel()))).indices]
    raw_global = state.raw_motion_probability[state.segment_motion_ids] * state.raw_segment_probability
    new_global = probabilities["global_segment"]
    rows: list[dict[str, object]] = []
    for segment_id in top_ids.detach().cpu().tolist():
        motion_id = int(state.segment_motion_ids[segment_id].item())
        top_indices = details.topk_indices[segment_id].detach().cpu().tolist()
        top_values = details.topk_values[segment_id].detach().cpu().tolist()
        top_names = [
            snapshot.joint_names[index] if index < len(snapshot.joint_names) else f"joint_{index}"
            for index in top_indices
        ]
        rows.append(
            {
                "checkpoint": state.iteration,
                "lambda": f"{lambda_joint:.3f}",
                "global_segment_id": segment_id,
                "motion_id": motion_id,
                "segment_id": int(snapshot.local_segment_id[segment_id].item()),
                "category": snapshot.categories[motion_id] if motion_id < len(snapshot.categories) else "__unknown__",
                "cluster": int(state.motion_cluster_ids[motion_id].item()),
                "difficulty_bin": int(snapshot.difficulty_bin[segment_id].item()),
                "raw_priority": f"{float(state.raw_segment_priority[segment_id].item()):.12g}",
                "joint_gap": f"{float(details.segment_gap_score[segment_id].item()):.12g}",
                "correction": f"{float(details.correction[segment_id].item()):.12g}",
                "new_priority": f"{float(details.corrected_priority[segment_id].item()):.12g}",
                "raw_probability": f"{float(raw_global[segment_id].item()):.12g}",
                "new_probability": f"{float(new_global[segment_id].item()):.12g}",
                "top6_joint_indices": ",".join(str(item) for item in top_indices),
                "top6_joint_names": ",".join(top_names),
                "top6_joint_gaps": ",".join(f"{float(value):.6g}" for value in top_values),
            }
        )
    return rows


def _joint_frequency_rows(*, state: CheckpointState, snapshot: Snapshot, details: JointGapDetails) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    active_segments = max(int(torch.count_nonzero(details.segment_gap_score > 0.0).item()), 1)
    for joint_index in range(snapshot.segment_joint_error.shape[1]):
        positive = details.local_valid[:, joint_index] & (details.local_gap[:, joint_index] > 0.0)
        rows.append(
            {
                "checkpoint": state.iteration,
                "joint_index": joint_index,
                "joint_name": snapshot.joint_names[joint_index] if joint_index < len(snapshot.joint_names) else f"joint_{joint_index}",
                "top6_frequency": f"{float(details.topk_selection_frequency[joint_index].item()):.12g}",
                "top6_count_estimate": f"{float(details.topk_selection_frequency[joint_index].item()) * active_segments:.12g}",
                "mean_positive_gap": f"{float(details.local_gap[positive, joint_index].mean().item()) if torch.any(positive) else 0.0:.12g}",
                "mean_abs_error": f"{float(snapshot.segment_joint_error[snapshot.eligible_mask, joint_index].mean().item()):.12g}",
            }
        )
    return rows


def _group_budget(details: JointGapDetails, snapshot: Snapshot, state: CheckpointState) -> dict[str, Any]:
    eligible = state.segment_eligible_mask
    correction = torch.where(eligible, details.correction, torch.zeros_like(details.correction))
    total = float(correction.sum().item())
    result: dict[str, Any] = {"total_positive_correction": total}
    for name, ids, count in (
        ("difficulty_bin", snapshot.difficulty_bin, 10),
        ("cluster", state.motion_cluster_ids[state.segment_motion_ids], int(state.raw_cluster_probability.numel())),
    ):
        rows = []
        for index in range(count):
            mask = eligible & (ids == index)
            mass = float(correction[mask].sum().item())
            rows.append({"id": index, "correction_share": 0.0 if total <= 0.0 else mass / total, "eligible_segments": int(torch.count_nonzero(mask).item())})
        result[name] = rows
    category_rows = []
    for category in sorted(set(snapshot.categories)):
        motion_mask = torch.tensor([item == category for item in snapshot.categories], dtype=torch.bool)
        if motion_mask.numel() < int(state.motion_lengths.numel()):
            motion_mask = torch.nn.functional.pad(motion_mask, (0, int(state.motion_lengths.numel()) - motion_mask.numel()))
        mask = eligible & motion_mask[state.segment_motion_ids]
        mass = float(correction[mask].sum().item())
        category_rows.append({"category": category, "correction_share": 0.0 if total <= 0.0 else mass / total, "eligible_segments": int(torch.count_nonzero(mask).item())})
    result["category"] = category_rows
    return result


def _aggregate_lambda_status(candidate_rows: Sequence[dict[str, object]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    selected: float | None = None
    for lambda_joint in LAMBDA_CANDIDATES[1:]:
        label = f"{lambda_joint:.3f}"
        rows = [row for row in candidate_rows if row["lambda"] == label]
        safety = bool(rows) and all(int(row["safety_pass"]) == 1 for row in rows)
        measurable = bool(rows) and all(int(row["measurable"]) == 1 for row in rows)
        if safety and measurable:
            status = "PASS"
            if selected is None:
                selected = lambda_joint
        elif safety:
            status = "TOO_WEAK"
        else:
            status = "FAIL"
        result[label] = {
            "status": status,
            "snapshots": len(rows),
            "safety_pass": safety,
            "measurable": measurable,
        }
    return {
        "selected_lambda": selected,
        "status_by_lambda": result,
        "overall_status": "PASS" if selected is not None else "NO SAFE NONZERO LAMBDA",
    }


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_report(config: ReplayConfig, report: dict[str, Any]) -> None:
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "lambda_candidate_comparison.csv", report["candidate_rows"])
    _write_csv(output_dir / "per_motion_lambda_perturbation.csv", report["per_motion_rows"])
    _write_csv(output_dir / "joint_top6_frequency.csv", report["joint_frequency_rows"])
    _write_csv(output_dir / "top_jointgap_boosted_segments.csv", report["top_boosted_rows"])
    manifest = {
        "status": report["selection"]["overall_status"],
        "selected_lambda": report["selection"]["selected_lambda"],
        "candidate_set": [f"{value:.3f}" for value in LAMBDA_CANDIDATES],
        "baseline_commit": M7_RAW_BASELINE_COMMIT,
        "step5_implementation_commit": STEP5_IMPLEMENTATION_COMMIT,
        "step6_identity_commit": STEP6_IDENTITY_COMMIT,
        "metadata_identity": report["metadata_identity"],
        "checkpoints": report["checkpoints"],
        "snapshots": report["snapshots"],
        "safety_rule": report["safety_rule"],
        "status_by_lambda": report["selection"]["status_by_lambda"],
        "proxy_limitation": report["proxy_limitation"],
        "sigma_floor": report["sigma_floor_summary"],
        "group_budget": report["group_budget"],
        "performance": report["performance"],
    }
    (output_dir / "lambda_replay_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Joint-Specific Gap Lambda Replay",
        "",
        f"status: {report['selection']['overall_status']}",
        f"selected_lambda: {report['selection']['selected_lambda']}",
        "",
        "These snapshots are frozen-policy deterministic diagnostic rollouts on the training motion library.",
        "They are not reconstructed M7-Raw historical per-joint EMA state; STEP 7 only calibrates sampling perturbation.",
        "",
        "## Checkpoints",
    ]
    for item in report["checkpoints"]:
        lines.append(f"- model_{item['iteration']}.pt: {item['path']} sha256={item['sha256']}")
    lines.extend(["", "## Snapshot Coverage"])
    for item in report["snapshots"]:
        lines.append(
            "- "
            f"model_{item['checkpoint_iteration']}.pt: "
            f"eligible={item['eligible_segments']} observed={item['observed_segments']} "
            f"cold={item['cold_segments']} sha256={item['sha256']}"
        )
    lines.extend(["", "## Candidate Summary", ""])
    summary_columns = [
        "lambda",
        "status",
        "mean_boost",
        "p90_active_boost",
        "entropy_change",
        "top1_change",
        "top5_change",
        "median_motion_tv",
        "p95_motion_tv",
        "spearman",
        "cap_hit_delta",
    ]
    lines.append("| " + " | ".join(summary_columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(summary_columns)) + " |")
    for lambda_joint in LAMBDA_CANDIDATES[1:]:
        label = f"{lambda_joint:.3f}"
        rows = [row for row in report["candidate_rows"] if row["lambda"] == label]
        if not rows:
            continue
        status = report["selection"]["status_by_lambda"][label]["status"]
        aggregate = {
            "mean_boost": float(np.mean([float(row["mean_correction_factor"]) - 1.0 for row in rows])),
            "p90_active_boost": float(np.min([float(row["p90_active_boost"]) for row in rows])),
            "entropy_change": float(np.min([float(row["entropy_change"]) for row in rows])),
            "top1_change": float(np.max([float(row["top1_change"]) for row in rows])),
            "top5_change": float(np.max([float(row["top5_change"]) for row in rows])),
            "median_motion_tv": float(np.max([float(row["median_motion_tv"]) for row in rows])),
            "p95_motion_tv": float(np.max([float(row["p95_motion_tv"]) for row in rows])),
            "spearman": float(np.min([float(row["spearman"]) for row in rows])),
            "cap_hit_delta": int(np.max([int(row["cap_hit_delta"]) for row in rows])),
        }
        lines.append(
            "| "
            + " | ".join(
                [
                    label,
                    status,
                    f"{aggregate['mean_boost']:.6g}",
                    f"{aggregate['p90_active_boost']:.6g}",
                    f"{aggregate['entropy_change']:.6g}",
                    f"{aggregate['top1_change']:.6g}",
                    f"{aggregate['top5_change']:.6g}",
                    f"{aggregate['median_motion_tv']:.6g}",
                    f"{aggregate['p95_motion_tv']:.6g}",
                    f"{aggregate['spearman']:.6g}",
                    str(aggregate["cap_hit_delta"]),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Selection Reason", report["selection_reason"], ""])
    lines.extend(["## Sigma Floor"])
    for checkpoint, item in report["sigma_floor_summary"].items():
        warning = " CALIBRATION WARNING" if item["calibration_warning"] else ""
        lines.append(f"- model_{checkpoint}.pt: overall_fraction={item['overall_fraction']:.6g}{warning}")
    lines.append("")
    lines.extend(["## Performance Timings"])
    for key, value in report["performance"].items():
        lines.append(f"- {key}: {value:.6g} ms")
    (output_dir / "lambda_replay_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if config.write_freeze_doc and report["selection"]["selected_lambda"] is not None:
        _write_freeze_doc(config.project_root / "docs" / "joint_specific_gap_lambda_freeze.md", report)


def _write_freeze_doc(path: Path, report: dict[str, Any]) -> None:
    selected = report["selection"]["selected_lambda"]
    lines = [
        "# Joint-Specific Gap Lambda Freeze",
        "",
        f"Selected lambda_joint: {selected:.3f}",
        "",
        "Candidate set: 0.000, 0.025, 0.050, 0.100.",
        "",
        "Frozen-policy checkpoints:",
    ]
    for item in report["checkpoints"]:
        lines.append(f"- model_{item['iteration']}.pt: {item['sha256']}")
    lines.extend(
        [
            "",
            "Snapshot coverage:",
        ]
    )
    for item in report["snapshots"]:
        lines.append(
            "- "
            f"model_{item['checkpoint_iteration']}.pt: "
            f"eligible={item['eligible_segments']} observed={item['observed_segments']} cold={item['cold_segments']}"
        )
    lines.extend(
        [
            "",
            "Selection uses only training-distribution sampling perturbation statistics from frozen-policy proxy snapshots.",
            "No Validation, Test, reward, success, completion, or Joint L2 metrics are used.",
            "",
            "Proxy limitation: snapshots are deterministic frozen-policy diagnostic rollouts on the training motion library, not reconstructed M7-Raw historical per-joint EMA state.",
            "",
            "Safety rule: cluster and motion probabilities must be exact, entropy drop <= 1%, Top1/Top5 mass increase <= 5%, median per-motion TV <= 0.01, p95 per-motion TV <= 0.03, Spearman >= 0.98, no new cap pathology, and p90 active boost >= 1%.",
            "",
            "Selection reason:",
            report["selection_reason"],
            "",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _blocker_report(config: ReplayConfig, missing: Sequence[Path]) -> dict[str, Any]:
    metadata_identity = _metadata_identity(config)
    report = {
        "selection": {
            "overall_status": "FAIL",
            "selected_lambda": None,
            "status_by_lambda": {},
        },
        "metadata_identity": metadata_identity,
        "checkpoints": [],
        "snapshots": [],
        "candidate_rows": [],
        "per_motion_rows": [],
        "joint_frequency_rows": [],
        "top_boosted_rows": [],
        "safety_rule": _safety_rule(),
        "proxy_limitation": (
            "STEP 7 requires frozen-policy proxy snapshots collected on the training motion library. "
            "No synthetic, validation, or reconstructed historical EMA substitute was used."
        ),
        "sigma_floor_summary": {},
        "group_budget": {},
        "performance": {},
        "selection_reason": "BLOCKER: missing training-manifest frozen-policy proxy snapshots.",
        "missing_snapshots": [str(path) for path in missing],
    }
    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "lambda_replay_manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (config.output_dir / "lambda_replay_report.md").write_text(
        "# Joint-Specific Gap Lambda Replay\n\n"
        "status: FAIL\n\n"
        "BLOCKER: missing training-manifest frozen-policy proxy snapshots.\n",
        encoding="utf-8",
    )
    return report


def _safety_rule() -> dict[str, object]:
    return {
        "cluster_probability": "exact unchanged",
        "motion_probability": "exact unchanged",
        "entropy_drop_max_fraction": 0.01,
        "top1_mass_relative_increase_max": 0.05,
        "top5_mass_relative_increase_max": 0.05,
        "per_motion_tv_median_max": 0.01,
        "per_motion_tv_p95_max": 0.03,
        "spearman_min": 0.98,
        "cap_pathology": "no new cap hit increase",
        "measurable_correction": "p90(lambda*C | C>0) >= 0.01",
        "selection": "choose the smallest nonzero lambda that is safe and measurable on all snapshots",
    }


def _selection_reason(selection: dict[str, Any]) -> str:
    selected = selection["selected_lambda"]
    status_by_lambda = selection["status_by_lambda"]
    if selected is None:
        return "No nonzero lambda satisfied every pre-frozen sampling-safety and measurable-correction criterion on all snapshots."
    label = f"{selected:.3f}"
    skipped = [
        f"{lambda_joint:.3f}"
        for lambda_joint in LAMBDA_CANDIDATES[1:]
        if lambda_joint > selected
    ]
    reason = (
        f"lambda_joint={label} is the smallest nonzero candidate that is safe and measurable on every selected snapshot."
    )
    if skipped:
        reason += " Larger candidates were not chosen because STEP 7 freezes the minimum effective intervention, not the largest safe perturbation."
    too_weak = [
        label
        for label, item in status_by_lambda.items()
        if item["status"] == "TOO_WEAK"
    ]
    if too_weak:
        reason += f" Weaker safe candidates were too weak: {', '.join(too_weak)}."
    return reason


def run_replay(config: ReplayConfig) -> dict[str, Any]:
    modules = _load_utils_modules(config.project_root)
    metadata_identity = _metadata_identity(config)
    checkpoints = _select_checkpoints(_list_checkpoints(config.run_dir, config.checkpoint_pattern), config.checkpoints)
    missing = [_snapshot_path(config.snapshots_dir, checkpoint) for checkpoint in checkpoints if not _snapshot_path(config.snapshots_dir, checkpoint).exists()]
    if missing:
        report = _blocker_report(config, missing)
        raise MissingSnapshotsError(missing)

    candidate_rows: list[dict[str, object]] = []
    per_motion_rows_all: list[dict[str, object]] = []
    top_boosted_rows: list[dict[str, object]] = []
    joint_frequency_rows: list[dict[str, object]] = []
    checkpoints_out: list[dict[str, object]] = []
    snapshots_out: list[dict[str, object]] = []
    sigma_floor_summary: dict[str, object] = {}
    group_budget: dict[str, object] = {}
    performance_accumulator: dict[str, list[float]] = {}

    for checkpoint in checkpoints:
        state = _load_checkpoint_state(checkpoint)
        snapshot = _load_snapshot(_snapshot_path(config.snapshots_dir, checkpoint), metadata_identity["train_manifest_sha256"])
        _validate_pair(state, snapshot)
        checkpoints_out.append(
            {
                "iteration": state.iteration,
                "path": str(state.path.resolve()),
                "sha256": state.checkpoint_sha256,
            }
        )
        snapshots_out.append(
            {
                "checkpoint_iteration": snapshot.checkpoint_iteration,
                "path": str(snapshot.path.resolve()),
                "sha256": _sha256_file(snapshot.path),
                "eligible_segments": int(torch.count_nonzero(snapshot.eligible_mask).item()),
                "observed_segments": int(torch.count_nonzero(snapshot.observed_mask & snapshot.eligible_mask).item()),
                "cold_segments": int(torch.count_nonzero(snapshot.eligible_mask & ~snapshot.observed_mask).item()),
                "joint_mapping_hash": snapshot.joint_mapping_hash,
                "proxy_statistic": snapshot.proxy_statistic,
            }
        )
        lambda_zero_checked = False
        details_for_frequency: JointGapDetails | None = None
        for lambda_joint in LAMBDA_CANDIDATES:
            details = _compute_joint_gap_details(modules, state=state, snapshot=snapshot, lambda_joint=lambda_joint)
            sampler, probabilities, prob_ms = _probabilities_for_lambda(modules, state, details.corrected_priority)
            per_motion_rows, per_motion_summary = _per_motion_rows(
                checkpoint_iteration=state.iteration,
                lambda_joint=lambda_joint,
                state=state,
                raw_segment_probability=state.raw_segment_probability,
                new_segment_probability=probabilities["segment_conditional"],
            )
            row = _candidate_row(
                state=state,
                snapshot=snapshot,
                lambda_joint=lambda_joint,
                details=details,
                probabilities=probabilities,
                probability_rebuild_ms=prob_ms,
                per_motion_summary=per_motion_summary,
            )
            row.update(_segment_count_group_summary(state, per_motion_rows))
            candidate_rows.append(row)
            per_motion_rows_all.extend(per_motion_rows)
            if float(lambda_joint) > 0.0:
                top_boosted_rows.extend(
                    _top_boosted_rows(
                        state=state,
                        snapshot=snapshot,
                        lambda_joint=lambda_joint,
                        details=details,
                        probabilities=probabilities,
                    )
                )
            if float(lambda_joint) == 0.0:
                lambda_zero_checked = (
                    int(row["cluster_exact"]) == 1
                    and int(row["motion_conditional_exact"]) == 1
                    and int(row["motion_exact"]) == 1
                    and float(row["global_tv"]) == 0.0
                )
                details_for_frequency = details
            for key, value in details.timings_ms.items():
                performance_accumulator.setdefault(key, []).append(float(value))
            performance_accumulator.setdefault("probability_rebuild_ms", []).append(float(prob_ms))
            _ = sampler
        if not lambda_zero_checked:
            raise RuntimeError(f"lambda=0 did not reproduce M7-Raw probabilities for {checkpoint.name}.")
        assert details_for_frequency is not None
        joint_frequency_rows.extend(_joint_frequency_rows(state=state, snapshot=snapshot, details=details_for_frequency))
        sigma_floor = details_for_frequency.calibration.sigma_floor_mask
        sigma_floor_summary[str(state.iteration)] = {
            "overall_fraction": float(sigma_floor.to(torch.float64).mean().item()),
            "mask_10x29": sigma_floor.to(torch.int8).detach().cpu().tolist(),
            "calibration_warning": bool(float(sigma_floor.to(torch.float64).mean().item()) > 0.75),
        }
        group_budget[str(state.iteration)] = _group_budget(details_for_frequency, snapshot, state)

    selection = _aggregate_lambda_status(candidate_rows)
    report = {
        "selection": selection,
        "metadata_identity": metadata_identity,
        "checkpoints": checkpoints_out,
        "snapshots": snapshots_out,
        "candidate_rows": candidate_rows,
        "per_motion_rows": per_motion_rows_all,
        "joint_frequency_rows": joint_frequency_rows,
        "top_boosted_rows": top_boosted_rows,
        "safety_rule": _safety_rule(),
        "proxy_limitation": (
            "Snapshots come from frozen M7-Raw policy deterministic diagnostic rollouts on the training motion library. "
            "They are proxy statistics, not reconstructed M7-Raw historical per-joint EMA state."
        ),
        "sigma_floor_summary": sigma_floor_summary,
        "group_budget": group_budget,
        "performance": {
            key: float(np.mean(values))
            for key, values in sorted(performance_accumulator.items())
        },
        "selection_reason": _selection_reason(selection),
    }
    _write_report(config, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", "--run_dir", dest="run_dir", default=DEFAULT_RUN_DIR)
    parser.add_argument("--checkpoints", default="10000,20000,30000,final")
    parser.add_argument("--checkpoint-pattern", "--checkpoint_pattern", dest="checkpoint_pattern", default="model_*.pt")
    parser.add_argument("--snapshots-dir", "--snapshots_dir", dest="snapshots_dir", default=DEFAULT_SNAPSHOT_DIR)
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-manifest", "--train_manifest", dest="train_manifest", default=DEFAULT_TRAIN_MANIFEST)
    parser.add_argument("--quality-metadata", "--quality_metadata", dest="quality_metadata", default=DEFAULT_QUALITY_METADATA)
    parser.add_argument("--difficulty-metadata", "--difficulty_metadata", dest="difficulty_metadata", default=DEFAULT_DIFFICULTY_METADATA)
    parser.add_argument("--cluster-metadata", "--cluster_metadata", dest="cluster_metadata", default=DEFAULT_CLUSTER_METADATA)
    parser.add_argument("--no-freeze-doc", action="store_true", help="Do not write docs/joint_specific_gap_lambda_freeze.md.")
    return parser


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    project_root = _repo_root()
    config = ReplayConfig(
        project_root=project_root,
        run_dir=_absolute(args.run_dir, project_root),
        snapshots_dir=_absolute(args.snapshots_dir, project_root),
        output_dir=_absolute(args.output_dir, project_root),
        checkpoints=args.checkpoints,
        checkpoint_pattern=args.checkpoint_pattern,
        train_manifest=_absolute(args.train_manifest, project_root),
        quality_metadata=_absolute(args.quality_metadata, project_root),
        difficulty_metadata=_absolute(args.difficulty_metadata, project_root),
        cluster_metadata=_absolute(args.cluster_metadata, project_root),
        write_freeze_doc=not args.no_freeze_doc,
    )
    try:
        report = run_replay(config)
    except MissingSnapshotsError as exc:
        print(json.dumps({"status": "FAIL", "missing_snapshots": [str(path) for path in exc.missing]}, indent=2))
        return 2
    print(
        json.dumps(
            {
                "status": report["selection"]["overall_status"],
                "selected_lambda": report["selection"]["selected_lambda"],
                "output_dir": str(config.output_dir),
            },
            indent=2,
        )
    )
    return 0 if report["selection"]["selected_lambda"] is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
