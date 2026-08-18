#!/usr/bin/env python3
"""Offline M7-JGap identity replay.

This script does not launch Isaac Sim, train a policy, run Validation, or run
formal Test.  It compares the old M7-Raw sampler implementation against the
current code path and the current M7-JGap lambda=0 path on a deterministic
large synthetic state.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


DEFAULT_OLD_COMMIT = "9591b69138f25846b2ba44214ffd29cbaa022ef0"


@dataclass(frozen=True)
class ReplayState:
    motion_lengths: torch.Tensor
    segment_motion_ids: torch.Tensor
    segment_start_frames: torch.Tensor
    segment_end_frames: torch.Tensor
    motion_cluster_ids: torch.Tensor
    motion_eligible_mask: torch.Tensor
    segment_eligible_mask: torch.Tensor
    motion_score: torch.Tensor
    motion_score_valid: torch.Tensor
    segment_score: torch.Tensor
    segment_score_valid: torch.Tensor
    motion_sample_count: torch.Tensor
    segment_sample_count: torch.Tensor
    difficulty_bins: torch.Tensor
    segment_joint_error: torch.Tensor
    segment_joint_valid: torch.Tensor


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _utils_dir(repo_root: Path) -> Path:
    return repo_root / "source" / "whole_body_tracking" / "whole_body_tracking" / "utils"


def _load_utils_modules(repo_root: Path, prefix: str, *, include_joint_gap: bool) -> dict[str, Any]:
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
    for module_name in ("adaptive_sampling", "diversity_sampling", "learning_gap"):
        path = package_dir / "utils" / f"{module_name}.py"
        spec = importlib.util.spec_from_file_location(prefix + ".utils." + module_name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Unable to load {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        modules[module_name] = module
    if include_joint_gap:
        path = package_dir / "utils" / "joint_gap.py"
        spec = importlib.util.spec_from_file_location(prefix + ".utils.joint_gap", path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Unable to load {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        modules["joint_gap"] = module
    return modules


def _tensor_sha256(tensor: torch.Tensor) -> str:
    data = tensor.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def _trace_sha256(cluster_ids: torch.Tensor, motion_ids: torch.Tensor, segment_ids: torch.Tensor, starts: torch.Tensor) -> str:
    trace = torch.stack(
        [
            cluster_ids.to(torch.int64),
            motion_ids.to(torch.int64),
            segment_ids.to(torch.int64),
            starts.to(torch.int64),
        ],
        dim=1,
    )
    return _tensor_sha256(trace)


def _comparison(name: str, left: torch.Tensor, right: torch.Tensor, *, atol: float = 0.0) -> dict[str, Any]:
    left = left.detach().cpu()
    right = right.detach().cpu()
    if left.shape != right.shape:
        return {
            "name": name,
            "shape_equal": False,
            "exact_equal": False,
            "allclose": False,
            "max_abs_diff": float("inf"),
            "mean_abs_diff": float("inf"),
            "num_nonzero_diff": -1,
        }
    diff = (left.to(torch.float64) - right.to(torch.float64)).abs()
    exact = torch.equal(left, right)
    return {
        "name": name,
        "shape_equal": True,
        "exact_equal": bool(exact),
        "allclose": bool(torch.allclose(left, right, atol=atol, rtol=0.0)),
        "max_abs_diff": float(diff.max().item()) if diff.numel() else 0.0,
        "mean_abs_diff": float(diff.mean().item()) if diff.numel() else 0.0,
        "num_nonzero_diff": int(torch.count_nonzero(left != right).item()),
    }


def _distribution_stats(probability: torch.Tensor) -> dict[str, float | int]:
    probability = probability.detach().cpu().to(torch.float64)
    positive = probability[probability > 0.0]
    ordered = torch.sort(positive, descending=True).values
    top1_count = max(1, int(math.ceil(0.01 * ordered.numel()))) if ordered.numel() else 0
    top5_count = max(1, int(math.ceil(0.05 * ordered.numel()))) if ordered.numel() else 0
    entropy = float((-(positive * torch.log(positive.clamp_min(1.0e-300))).sum()).item()) if positive.numel() else 0.0
    effective = float(1.0 / torch.sum(probability.square()).item()) if torch.any(probability > 0.0) else 0.0
    return {
        "entropy": entropy,
        "top1_mass": float(ordered[:top1_count].sum().item()) if top1_count else 0.0,
        "top5_mass": float(ordered[:top5_count].sum().item()) if top5_count else 0.0,
        "effective_count": effective,
        "max_probability": float(probability.max().item()) if probability.numel() else 0.0,
    }


def _build_replay_state(*, seed: int = 123, num_motions: int = 6000, num_segments: int = 21575) -> ReplayState:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    counts = torch.ones(num_motions, dtype=torch.long)
    extra = torch.randint(num_motions, (num_segments - num_motions,), generator=generator)
    counts += torch.bincount(extra, minlength=num_motions)
    segment_motion_ids = torch.repeat_interleave(torch.arange(num_motions, dtype=torch.long), counts)
    local_ids = torch.cat([torch.arange(int(count.item()), dtype=torch.long) for count in counts])
    segment_start_frames = local_ids * 6
    segment_end_frames = segment_start_frames + 5
    motion_lengths = counts * 6 + 12

    motion_ids = torch.arange(num_motions, dtype=torch.long)
    segment_ids = torch.arange(num_segments, dtype=torch.long)
    motion_cluster_ids = ((motion_ids * 5 + torch.div(motion_ids, 17, rounding_mode="floor")) % 8).to(torch.long)

    segment_eligible_mask = ((segment_ids * 37) % 29) != 0
    first_segment = torch.cumsum(counts, dim=0) - counts
    segment_eligible_mask[first_segment] = True
    motion_eligible_count = torch.zeros(num_motions, dtype=torch.long)
    motion_eligible_count.scatter_add_(0, segment_motion_ids, segment_eligible_mask.to(torch.long))
    motion_eligible_mask = motion_eligible_count > 0

    motion_score = (
        torch.sin(motion_ids.to(torch.float64) * 0.017) * 1.7
        + torch.cos(motion_ids.to(torch.float64) * 0.003) * 0.3
        + (motion_ids % 11).to(torch.float64) / 7.0
    ).abs()
    motion_score[motion_ids % 503 == 0] = 40.0
    motion_score[motion_ids % 997 == 0] = 0.0
    motion_score_valid = motion_eligible_mask & (((motion_ids * 19) % 31) != 0)

    segment_score = (
        torch.cos(segment_ids.to(torch.float64) * 0.013).abs()
        + (segment_ids % 17).to(torch.float64) / 20.0
    )
    segment_score[segment_ids % 101 == 0] = 0.0
    segment_score[segment_ids % 113 == 0] = 0.75
    segment_score[segment_ids % 1201 == 0] = 35.0
    segment_score_valid = segment_eligible_mask & (((segment_ids * 7) % 37) != 0)
    fallback_motions = (motion_ids % 113) == 0
    segment_score_valid[fallback_motions[segment_motion_ids]] = False

    motion_sample_count = torch.randint(0, 400, (num_motions,), generator=generator, dtype=torch.long)
    segment_sample_count = torch.randint(0, 200, (num_segments,), generator=generator, dtype=torch.long)
    cold_segments = (segment_ids % 43) == 0
    segment_sample_count[cold_segments] = 0

    difficulty_bins = ((segment_motion_ids * 3 + local_ids * 7) % 10).to(torch.long)
    joint_ids = torch.arange(29, dtype=torch.float32)
    segment_joint_error = (
        0.05
        + torch.rand((num_segments, 29), generator=generator, dtype=torch.float32) * 1.5
        + difficulty_bins.to(torch.float32)[:, None] * 0.02
        + (joint_ids[None, :] % 5) * 0.01
    )
    segment_joint_error[segment_ids % 977 == 0] = 0.0
    segment_joint_error[segment_ids % 1231 == 0, 0] = float("inf")
    segment_joint_error[segment_ids % 1297 == 0, 1] = float("nan")
    segment_joint_valid = segment_eligible_mask & (segment_sample_count >= 32) & (((segment_ids * 11) % 41) != 0)

    return ReplayState(
        motion_lengths=motion_lengths,
        segment_motion_ids=segment_motion_ids,
        segment_start_frames=segment_start_frames,
        segment_end_frames=segment_end_frames,
        motion_cluster_ids=motion_cluster_ids,
        motion_eligible_mask=motion_eligible_mask,
        segment_eligible_mask=segment_eligible_mask,
        motion_score=motion_score,
        motion_score_valid=motion_score_valid,
        segment_score=segment_score,
        segment_score_valid=segment_score_valid,
        motion_sample_count=motion_sample_count,
        segment_sample_count=segment_sample_count,
        difficulty_bins=difficulty_bins,
        segment_joint_error=segment_joint_error,
        segment_joint_valid=segment_joint_valid,
    )


def _make_sampler(modules: dict[str, Any], state: ReplayState, *, segment_mode: str, seed: int):
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
        warmup_iterations=0,
        probability_update_interval=1,
        uniform_mix=0.15,
        temperature=1.0,
        under_sampling_weight=0.25,
        motion_probability_cap=0.02,
        segment_probability_cap=1.0,
        score_clip=10.0,
        sampler_seed=seed,
        config_hash="joint-gap-identity-replay",
        num_clusters=8,
        minimum_budget_fraction_of_uniform=0.5,
        cluster_size_exponent=0.5,
        cluster_metadata_hash="synthetic-cluster",
        cluster_profile_sha256="synthetic-profile",
        cluster_schema_version="synthetic",
        device="cpu",
    )


def _compute_jgap_score(modules: dict[str, Any], state: ReplayState, *, lambda_joint: float, force_cold: bool = False):
    joint_gap = modules["joint_gap"]
    valid = torch.zeros_like(state.segment_joint_valid) if force_cold else state.segment_joint_valid
    calibration = joint_gap.compute_joint_bin_statistics(
        state.segment_joint_error,
        valid,
        state.difficulty_bins,
        num_bins=10,
        min_bin_valid_segments=32,
        sigma_floor=0.10,
        observation_weights=state.segment_sample_count,
    )
    correction = joint_gap.compute_joint_gap_correction(
        raw_priority=state.segment_score,
        raw_valid=state.segment_score_valid,
        segment_joint_error=state.segment_joint_error,
        segment_joint_valid=valid,
        segment_motion_ids=state.segment_motion_ids,
        difficulty_bin=state.difficulty_bins,
        calibration=calibration,
        num_motions=int(state.motion_lengths.numel()),
        top_k=6,
        lambda_joint=lambda_joint,
        gap_clip=5.0,
    )
    return correction


def _update_sampler(sampler: Any, state: ReplayState, *, segment_score: torch.Tensor) -> None:
    changed = sampler.update_probabilities(
        0,
        motion_score=state.motion_score,
        motion_score_valid=state.motion_score_valid,
        segment_score=segment_score,
        segment_score_valid=state.segment_score_valid,
        motion_sample_count=state.motion_sample_count,
        segment_sample_count=state.segment_sample_count,
    )
    if not changed:
        raise RuntimeError("Synthetic replay expected a probability update at iteration 0.")


def _probabilities(sampler: Any, state: ReplayState) -> dict[str, torch.Tensor]:
    joint_segment_probability = sampler.motion_probability[state.segment_motion_ids] * sampler.segment_probability
    return {
        "cluster": sampler.cluster_probability,
        "motion_conditional": sampler.motion_probability_conditional,
        "motion": sampler.motion_probability,
        "segment_conditional": sampler.segment_probability,
        "joint_segment": joint_segment_probability,
    }


def _sample_trace(sampler: Any, state: ReplayState, *, num_assignments: int) -> dict[str, Any]:
    before = _tensor_sha256(sampler.generator.get_state())
    motion_ids, segment_ids, starts = sampler.sample(num_assignments)
    cluster_ids = state.motion_cluster_ids[motion_ids]
    after = _tensor_sha256(sampler.generator.get_state())
    return {
        "cluster_ids": cluster_ids,
        "motion_ids": motion_ids,
        "segment_ids": segment_ids,
        "starts": starts,
        "before_rng_sha256": before,
        "after_rng_sha256": after,
        "trace_sha256": _trace_sha256(cluster_ids, motion_ids, segment_ids, starts),
    }


def _trace_compare(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    labels = ("cluster_ids", "motion_ids", "segment_ids", "starts")
    equal_by_field = {label: bool(torch.equal(left[label], right[label])) for label in labels}
    identical = all(equal_by_field.values())
    first_mismatch = -1
    if not identical:
        size = int(left["motion_ids"].numel())
        mismatch = torch.zeros(size, dtype=torch.bool)
        for label in labels:
            mismatch |= left[label] != right[label]
        first_mismatch = int(torch.where(mismatch)[0][0].item())
    return {
        "identical": identical,
        "first_mismatch_index": first_mismatch,
        "field_identical": equal_by_field,
        "left_trace_sha256": left["trace_sha256"],
        "right_trace_sha256": right["trace_sha256"],
        "before_rng_identical": left["before_rng_sha256"] == right["before_rng_sha256"],
        "after_rng_identical": left["after_rng_sha256"] == right["after_rng_sha256"],
        "left_final_rng_sha256": left["after_rng_sha256"],
        "right_final_rng_sha256": right["after_rng_sha256"],
    }


def _top_ranking_equal(left: torch.Tensor, right: torch.Tensor, *, k: int) -> bool:
    left_top = torch.topk(left, k=min(k, left.numel())).indices
    right_top = torch.topk(right, k=min(k, right.numel())).indices
    return bool(torch.equal(left_top, right_top))


def _run_identity(
    old_modules: dict[str, Any],
    new_modules: dict[str, Any],
    *,
    num_assignments: int,
    num_motions: int = 6000,
    num_segments: int = 21575,
) -> dict[str, Any]:
    state = _build_replay_state(num_motions=num_motions, num_segments=num_segments)
    old = _make_sampler(old_modules, state, segment_mode="raw_error", seed=42)
    new_a = _make_sampler(new_modules, state, segment_mode="raw_error", seed=42)
    new_b = _make_sampler(new_modules, state, segment_mode="raw_error_joint_gap", seed=42)

    _update_sampler(old, state, segment_score=state.segment_score)
    _update_sampler(new_a, state, segment_score=state.segment_score)
    jgap = _compute_jgap_score(new_modules, state, lambda_joint=0.0)
    _update_sampler(new_b, state, segment_score=jgap.corrected_priority)

    old_prob = _probabilities(old, state)
    new_a_prob = _probabilities(new_a, state)
    new_b_prob = _probabilities(new_b, state)

    old_trace = _sample_trace(old, state, num_assignments=num_assignments)
    new_a_trace = _sample_trace(new_a, state, num_assignments=num_assignments)
    new_b_trace = _sample_trace(new_b, state, num_assignments=num_assignments)

    cold = _compute_jgap_score(new_modules, state, lambda_joint=0.1, force_cold=True)
    partial = _compute_jgap_score(new_modules, state, lambda_joint=0.0)
    cold_sampler = _make_sampler(new_modules, state, segment_mode="raw_error_joint_gap", seed=42)
    partial_sampler = _make_sampler(new_modules, state, segment_mode="raw_error_joint_gap", seed=42)
    raw_sampler = _make_sampler(new_modules, state, segment_mode="raw_error", seed=42)
    _update_sampler(cold_sampler, state, segment_score=cold.corrected_priority)
    _update_sampler(partial_sampler, state, segment_score=partial.corrected_priority)
    _update_sampler(raw_sampler, state, segment_score=state.segment_score)

    return {
        "state": {
            "num_motions": int(state.motion_lengths.numel()),
            "num_segments": int(state.segment_motion_ids.numel()),
            "num_clusters": 8,
            "eligible_segments": int(torch.count_nonzero(state.segment_eligible_mask).item()),
            "ineligible_segments": int(torch.count_nonzero(~state.segment_eligible_mask).item()),
            "fallback_motion_count": int(torch.count_nonzero(torch.bincount(state.segment_motion_ids[~state.segment_score_valid], minlength=state.motion_lengths.numel()) > 0).item()),
            "warm_joint_segments": int(torch.count_nonzero(state.segment_joint_valid).item()),
            "cold_joint_segments": int(torch.count_nonzero(~state.segment_joint_valid).item()),
        },
        "probability": {
            "new_a_vs_old": {
                key: _comparison(key, new_a_prob[key], old_prob[key])
                for key in old_prob
            },
            "new_b_vs_old": {
                key: _comparison(key, new_b_prob[key], old_prob[key])
                for key in old_prob
            },
            "new_b_vs_new_a": {
                key: _comparison(key, new_b_prob[key], new_a_prob[key])
                for key in old_prob
            },
        },
        "priority": {
            "new_b_corrected_vs_raw": _comparison("segment_priority", jgap.corrected_priority, state.segment_score),
            "cold_lambda_positive_corrected_vs_raw": _comparison("segment_priority_cold", cold.corrected_priority, state.segment_score),
            "partial_lambda_zero_corrected_vs_raw": _comparison("segment_priority_partial", partial.corrected_priority, state.segment_score),
            "cold_correction_max": float(cold.correction.max().item()),
            "partial_correction_max": float(partial.correction.max().item()),
        },
        "cluster_probabilities": {
            "old": [float(value) for value in old.cluster_probability.tolist()],
            "new_a": [float(value) for value in new_a.cluster_probability.tolist()],
            "new_b": [float(value) for value in new_b.cluster_probability.tolist()],
        },
        "ranking": {
            "motion_top10_new_a_old_equal": _top_ranking_equal(new_a_prob["motion"], old_prob["motion"], k=10),
            "motion_top10_new_b_old_equal": _top_ranking_equal(new_b_prob["motion"], old_prob["motion"], k=10),
            "segment_top100_new_a_old_equal": _top_ranking_equal(new_a_prob["joint_segment"], old_prob["joint_segment"], k=100),
            "segment_top100_new_b_old_equal": _top_ranking_equal(new_b_prob["joint_segment"], old_prob["joint_segment"], k=100),
            "motion_top1_old": {
                "id": int(torch.argmax(old_prob["motion"]).item()),
                "probability": float(old_prob["motion"].max().item()),
            },
            "segment_top1_old": {
                "id": int(torch.argmax(old_prob["joint_segment"]).item()),
                "probability": float(old_prob["joint_segment"].max().item()),
            },
        },
        "distribution_stats": {
            "old_motion": _distribution_stats(old_prob["motion"]),
            "new_a_motion": _distribution_stats(new_a_prob["motion"]),
            "new_b_motion": _distribution_stats(new_b_prob["motion"]),
            "old_segment": _distribution_stats(old_prob["joint_segment"]),
            "new_a_segment": _distribution_stats(new_a_prob["joint_segment"]),
            "new_b_segment": _distribution_stats(new_b_prob["joint_segment"]),
        },
        "trace": {
            "num_assignments": num_assignments,
            "old": {
                "trace_sha256": old_trace["trace_sha256"],
                "before_rng_sha256": old_trace["before_rng_sha256"],
                "after_rng_sha256": old_trace["after_rng_sha256"],
            },
            "new_a": {
                "trace_sha256": new_a_trace["trace_sha256"],
                "before_rng_sha256": new_a_trace["before_rng_sha256"],
                "after_rng_sha256": new_a_trace["after_rng_sha256"],
            },
            "new_b": {
                "trace_sha256": new_b_trace["trace_sha256"],
                "before_rng_sha256": new_b_trace["before_rng_sha256"],
                "after_rng_sha256": new_b_trace["after_rng_sha256"],
            },
            "new_a_vs_old": _trace_compare(new_a_trace, old_trace),
            "new_b_vs_old": _trace_compare(new_b_trace, old_trace),
            "new_b_vs_new_a": _trace_compare(new_b_trace, new_a_trace),
        },
        "sampler_counters": {
            "old": old.metrics(),
            "new_a": new_a.metrics(),
            "new_b": new_b.metrics(),
            "cold_lambda_positive": cold_sampler.metrics(),
            "partial_lambda_zero": partial_sampler.metrics(),
            "raw_reference": raw_sampler.metrics(),
        },
        "cold_state_identity": {
            "probability": {
                key: _comparison(key, _probabilities(cold_sampler, state)[key], _probabilities(raw_sampler, state)[key])
                for key in old_prob
            },
            "trace": _trace_compare(
                _sample_trace(cold_sampler, state, num_assignments=num_assignments),
                _sample_trace(raw_sampler, state, num_assignments=num_assignments),
            ),
        },
        "partial_lambda_zero_identity": {
            "probability": {
                key: _comparison(key, _probabilities(partial_sampler, state)[key], _probabilities(raw_sampler, state)[key])
                for key in old_prob
            },
        },
    }


def _performance(
    new_modules: dict[str, Any],
    repeats: int = 5,
    *,
    num_motions: int = 6000,
    num_segments: int = 21575,
) -> dict[str, float]:
    state = _build_replay_state(num_motions=num_motions, num_segments=num_segments)
    raw_times = []
    jgap_times = []
    for _ in range(repeats):
        sampler = _make_sampler(new_modules, state, segment_mode="raw_error", seed=42)
        start = time.perf_counter()
        _update_sampler(sampler, state, segment_score=state.segment_score)
        raw_times.append((time.perf_counter() - start) * 1000.0)

        sampler = _make_sampler(new_modules, state, segment_mode="raw_error_joint_gap", seed=42)
        start = time.perf_counter()
        jgap = _compute_jgap_score(new_modules, state, lambda_joint=0.0)
        _update_sampler(sampler, state, segment_score=jgap.corrected_priority)
        jgap_times.append((time.perf_counter() - start) * 1000.0)
    raw_ms = sum(raw_times) / len(raw_times)
    jgap_ms = sum(jgap_times) / len(jgap_times)
    return {
        "m7_raw_ms": raw_ms,
        "m7_jgap_lambda0_ms": jgap_ms,
        "absolute_overhead_ms": jgap_ms - raw_ms,
        "relative_overhead_percent": ((jgap_ms / raw_ms) - 1.0) * 100.0 if raw_ms > 0.0 else 0.0,
    }


def _pass_status(report: dict[str, Any]) -> bool:
    probability_sections = (
        report["identity"]["probability"]["new_a_vs_old"],
        report["identity"]["probability"]["new_b_vs_old"],
        report["identity"]["probability"]["new_b_vs_new_a"],
        report["identity"]["cold_state_identity"]["probability"],
        report["identity"]["partial_lambda_zero_identity"]["probability"],
    )
    probability_ok = all(
        item["exact_equal"]
        for section in probability_sections
        for item in section.values()
    )
    trace_ok = (
        report["identity"]["trace"]["new_a_vs_old"]["identical"]
        and report["identity"]["trace"]["new_b_vs_old"]["identical"]
        and report["identity"]["trace"]["new_b_vs_new_a"]["identical"]
        and report["identity"]["cold_state_identity"]["trace"]["identical"]
    )
    priority_ok = (
        report["identity"]["priority"]["new_b_corrected_vs_raw"]["exact_equal"]
        and report["identity"]["priority"]["cold_lambda_positive_corrected_vs_raw"]["exact_equal"]
        and report["identity"]["priority"]["partial_lambda_zero_corrected_vs_raw"]["exact_equal"]
        and report["identity"]["priority"]["cold_correction_max"] == 0.0
    )
    ranking_ok = all(
        value for key, value in report["identity"]["ranking"].items() if key.endswith("_equal")
    )
    return bool(probability_ok and trace_ok and priority_ok and ranking_ok)


def _write_reports(report: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "identity_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    trace = report["identity"]["trace"]
    (output_dir / "assignment_trace_hashes.json").write_text(
        json.dumps(
            {
                "num_assignments": trace["num_assignments"],
                "old": trace["old"],
                "new_a": trace["new_a"],
                "new_b": trace["new_b"],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    lines = [
        "# Joint Gap Identity Replay",
        "",
        f"status: {report['status']}",
        f"old_commit: {report['old_commit']}",
        f"num_assignments: {trace['num_assignments']}",
        "",
        "## Probability",
    ]
    for section_name in ("new_a_vs_old", "new_b_vs_old", "new_b_vs_new_a"):
        lines.append(f"### {section_name}")
        for name, item in report["identity"]["probability"][section_name].items():
            lines.append(
                f"- {name}: exact={item['exact_equal']} max_abs_diff={item['max_abs_diff']:.3g} "
                f"nonzero={item['num_nonzero_diff']}"
            )
    lines.extend(
        [
            "",
            "## Trace",
            f"- NEW-A vs OLD: {trace['new_a_vs_old']['identical']} {trace['new_a_vs_old']['left_trace_sha256']}",
            f"- NEW-B vs OLD: {trace['new_b_vs_old']['identical']} {trace['new_b_vs_old']['left_trace_sha256']}",
            f"- NEW-B vs NEW-A: {trace['new_b_vs_new_a']['identical']} {trace['new_b_vs_new_a']['left_trace_sha256']}",
            "",
            "## Performance",
            f"- M7-Raw: {report['performance']['m7_raw_ms']:.3f} ms",
            f"- M7-JGap lambda=0: {report['performance']['m7_jgap_lambda0_ms']:.3f} ms",
            f"- overhead: {report['performance']['relative_overhead_percent']:.2f}%",
        ]
    )
    (output_dir / "identity_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-worktree", required=True, help="Path to a checkout at the old M7-Raw commit.")
    parser.add_argument("--old-commit", default=DEFAULT_OLD_COMMIT)
    parser.add_argument("--num-assignments", type=int, default=100_000)
    parser.add_argument("--output-dir", default="outputs/joint_gap_stage2_identity")
    args = parser.parse_args()

    repo_root = _repo_root()
    old_root = Path(args.old_worktree).resolve()
    if not (_utils_dir(old_root) / "diversity_sampling.py").exists():
        raise FileNotFoundError(f"Old worktree does not contain diversity_sampling.py: {old_root}")
    old_modules = _load_utils_modules(old_root, "wbt_old_identity", include_joint_gap=False)
    new_modules = _load_utils_modules(repo_root, "wbt_new_identity", include_joint_gap=True)

    identity = _run_identity(old_modules, new_modules, num_assignments=args.num_assignments)
    report = {
        "status": "PASS",
        "old_commit": args.old_commit,
        "old_worktree": str(old_root),
        "new_repo": str(repo_root),
        "identity": identity,
        "performance": _performance(new_modules),
    }
    report["status"] = "PASS" if _pass_status(report) else "FAIL"
    _write_reports(report, repo_root / args.output_dir)
    print(json.dumps({"status": report["status"], "report": args.output_dir}, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
