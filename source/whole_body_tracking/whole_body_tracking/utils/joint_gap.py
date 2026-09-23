"""Joint-specific difficulty-calibrated segment-priority correction.

This module is intentionally pure tensor math.  It has no environment access,
no logging side effects, and no random draws.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

try:  # Package import used at runtime.
    from .learning_gap import segmented_quantile
except ImportError:  # Direct-file import used by CPU-only unit tests.
    from learning_gap import segmented_quantile


JOINT_GAP_SCHEMA_VERSION = "wbt.joint_gap.v1"


@dataclass(frozen=True)
class JointBinCalibrationResult:
    mean: torch.Tensor
    sigma: torch.Tensor
    valid_segment_count: torch.Tensor
    fallback_mask: torch.Tensor
    reliable_mask: torch.Tensor
    sigma_floor_mask: torch.Tensor
    global_mean: torch.Tensor
    global_sigma: torch.Tensor
    global_valid_count: torch.Tensor


@dataclass(frozen=True)
class JointLocalGapResult:
    local_gap: torch.Tensor
    local_valid: torch.Tensor
    motion_center: torch.Tensor
    motion_valid_count: torch.Tensor


@dataclass(frozen=True)
class JointTopKResult:
    segment_gap_score: torch.Tensor
    segment_valid: torch.Tensor
    topk_values: torch.Tensor
    topk_indices: torch.Tensor
    topk_selection_frequency: torch.Tensor


@dataclass(frozen=True)
class JointGapCorrectionResult:
    corrected_priority: torch.Tensor
    correction: torch.Tensor
    raw_gate_mask: torch.Tensor
    segment_gap_score: torch.Tensor
    segment_valid: torch.Tensor
    topk_selection_frequency: torch.Tensor
    global_gap: torch.Tensor
    global_valid: torch.Tensor
    local_gap: torch.Tensor
    local_valid: torch.Tensor


def _require_vector(name: str, value: torch.Tensor, *, size: int | None = None) -> torch.Tensor:
    tensor = torch.as_tensor(value)
    if tensor.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if size is not None and tensor.numel() != size:
        raise ValueError(f"{name} must contain {size} values.")
    return tensor


def _require_matrix(
    name: str,
    value: torch.Tensor,
    *,
    rows: int | None = None,
    columns: int | None = None,
) -> torch.Tensor:
    tensor = torch.as_tensor(value)
    if tensor.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional.")
    if rows is not None and tensor.shape[0] != rows:
        raise ValueError(f"{name} must contain {rows} rows.")
    if columns is not None and tensor.shape[1] != columns:
        raise ValueError(f"{name} must contain {columns} columns.")
    return tensor


def _positive_finite(name: str, value: float) -> float:
    resolved = float(value)
    if not math.isfinite(resolved) or resolved <= 0.0:
        raise ValueError(f"{name} must be finite and greater than zero.")
    return resolved


def _nonnegative_finite(name: str, value: float) -> float:
    resolved = float(value)
    if not math.isfinite(resolved) or resolved < 0.0:
        raise ValueError(f"{name} must be finite and non-negative.")
    return resolved


def compute_joint_bin_statistics(
    segment_joint_error: torch.Tensor,
    segment_valid: torch.Tensor,
    difficulty_bin: torch.Tensor,
    *,
    num_bins: int,
    min_bin_valid_segments: int,
    sigma_floor: float,
    observation_weights: torch.Tensor | None = None,
) -> JointBinCalibrationResult:
    """Estimate per-difficulty-bin, per-joint current-policy error statistics."""

    if num_bins < 1 or min_bin_valid_segments < 1:
        raise ValueError("num_bins and min_bin_valid_segments must be positive.")
    floor = _positive_finite("sigma_floor", sigma_floor)
    error = _require_matrix("segment_joint_error", segment_joint_error).to(torch.float64)
    num_segments, num_joints = error.shape
    valid_segment = _require_vector("segment_valid", segment_valid, size=num_segments).to(
        device=error.device, dtype=torch.bool
    )
    bins = _require_vector("difficulty_bin", difficulty_bin, size=num_segments).to(
        device=error.device, dtype=torch.long
    )
    if bins.numel() and (torch.any(bins < 0) or torch.any(bins >= num_bins)):
        raise ValueError("difficulty_bin values are outside the configured range.")
    if observation_weights is None:
        weights = torch.ones(num_segments, dtype=torch.float64, device=error.device)
        valid_weight = torch.ones(num_segments, dtype=torch.bool, device=error.device)
    else:
        weights = _require_vector("observation_weights", observation_weights, size=num_segments).to(
            device=error.device, dtype=torch.float64
        )
        valid_weight = torch.isfinite(weights) & (weights > 0.0)

    valid = valid_segment[:, None] & valid_weight[:, None] & torch.isfinite(error)
    joint_ids = torch.arange(num_joints, dtype=torch.long, device=error.device)
    group_ids = bins[:, None] * num_joints + joint_ids[None, :]
    flat_valid = valid.reshape(-1)
    flat_group_ids = group_ids.reshape(-1)
    flat_error = error.reshape(-1)
    flat_weights = weights[:, None].expand_as(error).reshape(-1)
    num_groups = int(num_bins) * int(num_joints)

    counts = torch.bincount(flat_group_ids[flat_valid], minlength=num_groups).reshape(num_bins, num_joints)
    weight_sums = torch.zeros(num_groups, dtype=torch.float64, device=error.device)
    weighted_sum = torch.zeros_like(weight_sums)
    weighted_square_sum = torch.zeros_like(weight_sums)
    if torch.any(flat_valid):
        selected_groups = flat_group_ids[flat_valid]
        selected_weights = flat_weights[flat_valid]
        selected_error = flat_error[flat_valid]
        weight_sums.scatter_add_(0, selected_groups, selected_weights)
        weighted_sum.scatter_add_(0, selected_groups, selected_weights * selected_error)
        weighted_square_sum.scatter_add_(0, selected_groups, selected_weights * selected_error.square())

    weight_sums = weight_sums.reshape(num_bins, num_joints)
    weighted_sum = weighted_sum.reshape(num_bins, num_joints)
    weighted_square_sum = weighted_square_sum.reshape(num_bins, num_joints)
    direct_mean = torch.where(weight_sums > 0.0, weighted_sum / weight_sums.clamp_min(1.0e-300), 0.0)
    direct_variance = torch.where(
        weight_sums > 0.0,
        weighted_square_sum / weight_sums.clamp_min(1.0e-300) - direct_mean.square(),
        0.0,
    ).clamp_min(0.0)
    direct_sigma_unfloored = direct_variance.sqrt()

    global_counts = counts.sum(dim=0).to(torch.long)
    global_weight = weight_sums.sum(dim=0)
    global_sum = weighted_sum.sum(dim=0)
    global_square_sum = weighted_square_sum.sum(dim=0)
    global_mean = torch.where(global_weight > 0.0, global_sum / global_weight.clamp_min(1.0e-300), 0.0)
    global_variance = torch.where(
        global_weight > 0.0,
        global_square_sum / global_weight.clamp_min(1.0e-300) - global_mean.square(),
        0.0,
    ).clamp_min(0.0)
    global_sigma_unfloored = global_variance.sqrt()

    direct_reliable = counts >= int(min_bin_valid_segments)
    global_reliable = global_counts >= int(min_bin_valid_segments)
    fallback_mask = ~direct_reliable & global_reliable[None, :]
    reliable = direct_reliable | global_reliable[None, :]
    selected_mean = torch.where(direct_reliable, direct_mean, global_mean[None, :].expand_as(direct_mean))
    selected_sigma_unfloored = torch.where(
        direct_reliable,
        direct_sigma_unfloored,
        global_sigma_unfloored[None, :].expand_as(direct_sigma_unfloored),
    )
    sigma_floor_mask = reliable & (selected_sigma_unfloored < floor)
    mean = torch.where(reliable, selected_mean, torch.zeros_like(selected_mean))
    sigma = torch.where(
        reliable,
        selected_sigma_unfloored.clamp_min(floor),
        torch.ones_like(selected_sigma_unfloored),
    )
    global_sigma = torch.where(
        global_reliable,
        global_sigma_unfloored.clamp_min(floor),
        torch.ones_like(global_sigma_unfloored),
    )
    global_mean = torch.where(global_reliable, global_mean, torch.zeros_like(global_mean))
    return JointBinCalibrationResult(
        mean=mean,
        sigma=sigma,
        valid_segment_count=counts.to(torch.long),
        fallback_mask=fallback_mask,
        reliable_mask=reliable,
        sigma_floor_mask=sigma_floor_mask,
        global_mean=global_mean,
        global_sigma=global_sigma,
        global_valid_count=global_counts,
    )


def compute_joint_global_gap(
    segment_joint_error: torch.Tensor,
    segment_valid: torch.Tensor,
    difficulty_bin: torch.Tensor,
    calibration: JointBinCalibrationResult,
    *,
    gap_clip: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return clipped per-joint global gap, validity mask and clip mask."""

    clip = _positive_finite("gap_clip", gap_clip)
    error = _require_matrix("segment_joint_error", segment_joint_error).to(torch.float64)
    num_segments, num_joints = error.shape
    valid_segment = _require_vector("segment_valid", segment_valid, size=num_segments).to(
        device=error.device, dtype=torch.bool
    )
    bins = _require_vector("difficulty_bin", difficulty_bin, size=num_segments).to(
        device=error.device, dtype=torch.long
    )
    if calibration.mean.shape != calibration.sigma.shape or calibration.mean.shape[1] != num_joints:
        raise ValueError("Joint calibration tensors do not match the joint layout.")
    if bins.numel() and (torch.any(bins < 0) or torch.any(bins >= calibration.mean.shape[0])):
        raise ValueError("difficulty_bin values are outside the calibration range.")

    mean = calibration.mean[bins]
    sigma = calibration.sigma[bins]
    reliable = calibration.reliable_mask[bins]
    raw_gap = (error - mean) / sigma.clamp_min(1.0e-300)
    finite = torch.isfinite(error) & torch.isfinite(raw_gap) & torch.isfinite(sigma) & (sigma > 0.0)
    valid = valid_segment[:, None] & reliable & finite
    clipped = valid & (raw_gap.abs() > clip)
    gap = torch.where(valid, torch.clamp(raw_gap, -clip, clip), torch.zeros_like(raw_gap))
    gap = torch.nan_to_num(gap, nan=0.0, posinf=clip, neginf=-clip)
    return gap, valid, clipped


def compute_joint_local_gap(
    global_gap: torch.Tensor,
    global_valid: torch.Tensor,
    segment_motion_ids: torch.Tensor,
    *,
    num_motions: int,
    gap_clip: float,
) -> JointLocalGapResult:
    """Center per-joint global gaps by the within-motion median."""

    if num_motions < 1:
        raise ValueError("num_motions must be positive.")
    clip = _positive_finite("gap_clip", gap_clip)
    gap = _require_matrix("global_gap", global_gap).to(torch.float64)
    num_segments, num_joints = gap.shape
    valid = _require_matrix("global_valid", global_valid, rows=num_segments, columns=num_joints).to(
        device=gap.device, dtype=torch.bool
    )
    motion_ids = _require_vector("segment_motion_ids", segment_motion_ids, size=num_segments).to(
        device=gap.device, dtype=torch.long
    )
    if motion_ids.numel() and (torch.any(motion_ids < 0) or torch.any(motion_ids >= num_motions)):
        raise ValueError("segment_motion_ids are outside the motion range.")

    centers = torch.zeros(num_motions, num_joints, dtype=torch.float64, device=gap.device)
    counts = torch.zeros(num_motions, num_joints, dtype=torch.long, device=gap.device)
    for joint_id in range(num_joints):
        center, count = segmented_quantile(
            gap[:, joint_id],
            motion_ids,
            valid[:, joint_id],
            num_groups=num_motions,
            quantile=0.50,
        )
        centers[:, joint_id] = center
        counts[:, joint_id] = count

    local_valid = valid & (counts[motion_ids] > 1)
    local = torch.where(
        local_valid,
        torch.clamp(gap - centers[motion_ids], -clip, clip),
        torch.zeros_like(gap),
    )
    local = torch.nan_to_num(local, nan=0.0, posinf=clip, neginf=-clip)
    return JointLocalGapResult(
        local_gap=local,
        local_valid=local_valid,
        motion_center=centers,
        motion_valid_count=counts,
    )


def aggregate_topk_joint_gap(
    local_gap: torch.Tensor,
    local_valid: torch.Tensor,
    *,
    top_k: int,
) -> JointTopKResult:
    """Aggregate positive local gaps by mean of the largest K joints."""

    gap = _require_matrix("local_gap", local_gap).to(torch.float64)
    valid = _require_matrix("local_valid", local_valid, rows=gap.shape[0], columns=gap.shape[1]).to(
        device=gap.device, dtype=torch.bool
    )
    num_segments, num_joints = gap.shape
    if not 1 <= int(top_k) <= num_joints:
        raise ValueError("top_k must be in [1, num_joints].")
    positive = torch.where(valid, torch.clamp(gap, min=0.0), torch.zeros_like(gap))
    values, indices = torch.topk(positive, k=int(top_k), dim=1)
    score = values.mean(dim=1)
    segment_valid = valid.any(dim=1)

    selected_positive = values > 0.0
    counts = torch.zeros(num_joints, dtype=torch.float64, device=gap.device)
    if torch.any(selected_positive):
        counts.scatter_add_(
            0,
            indices[selected_positive],
            torch.ones(int(torch.count_nonzero(selected_positive).item()), dtype=torch.float64, device=gap.device),
        )
    active_segments = torch.count_nonzero(score > 0.0).to(torch.float64).clamp_min(1.0)
    return JointTopKResult(
        segment_gap_score=score,
        segment_valid=segment_valid,
        topk_values=values,
        topk_indices=indices,
        topk_selection_frequency=counts / active_segments,
    )


def compute_raw_gate(
    raw_priority: torch.Tensor,
    raw_valid: torch.Tensor,
    segment_motion_ids: torch.Tensor,
    *,
    num_motions: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return within-motion median raw-priority gate using >= ties."""

    if num_motions < 1:
        raise ValueError("num_motions must be positive.")
    raw = _require_vector("raw_priority", raw_priority).to(torch.float64)
    valid = _require_vector("raw_valid", raw_valid, size=raw.numel()).to(device=raw.device, dtype=torch.bool)
    motion_ids = _require_vector("segment_motion_ids", segment_motion_ids, size=raw.numel()).to(
        device=raw.device, dtype=torch.long
    )
    if motion_ids.numel() and (torch.any(motion_ids < 0) or torch.any(motion_ids >= num_motions)):
        raise ValueError("segment_motion_ids are outside the motion range.")
    valid = valid & torch.isfinite(raw)
    median, count = segmented_quantile(raw, motion_ids, valid, num_groups=num_motions, quantile=0.50)
    gate = valid & (count[motion_ids] > 0) & (raw >= median[motion_ids])
    return gate, median, count


def compute_joint_gap_correction(
    *,
    raw_priority: torch.Tensor,
    raw_valid: torch.Tensor,
    segment_joint_error: torch.Tensor,
    segment_joint_valid: torch.Tensor,
    segment_motion_ids: torch.Tensor,
    difficulty_bin: torch.Tensor,
    calibration: JointBinCalibrationResult,
    num_motions: int,
    top_k: int,
    lambda_joint: float,
    gap_clip: float,
) -> JointGapCorrectionResult:
    """Apply the frozen JGap formula to raw segment priority."""

    strength = _nonnegative_finite("lambda_joint", lambda_joint)
    raw = _require_vector("raw_priority", raw_priority).to(torch.float64)
    raw_valid_tensor = _require_vector("raw_valid", raw_valid, size=raw.numel()).to(
        device=raw.device, dtype=torch.bool
    )
    global_gap, global_valid, _ = compute_joint_global_gap(
        segment_joint_error,
        segment_joint_valid,
        difficulty_bin,
        calibration,
        gap_clip=gap_clip,
    )
    local = compute_joint_local_gap(
        global_gap,
        global_valid,
        segment_motion_ids,
        num_motions=num_motions,
        gap_clip=gap_clip,
    )
    topk = aggregate_topk_joint_gap(local.local_gap, local.local_valid, top_k=top_k)
    gate, _, _ = compute_raw_gate(
        raw,
        raw_valid_tensor,
        segment_motion_ids,
        num_motions=num_motions,
    )
    finite_score = torch.isfinite(topk.segment_gap_score)
    segment_valid = topk.segment_valid & raw_valid_tensor & finite_score
    correction = torch.where(
        gate & segment_valid,
        torch.tanh(torch.clamp(topk.segment_gap_score, min=0.0)),
        torch.zeros_like(raw),
    )
    correction = torch.nan_to_num(correction, nan=0.0, posinf=0.0, neginf=0.0)
    correction = torch.clamp(correction, 0.0, 1.0 - 1.0e-12)
    if strength == 0.0:
        corrected = raw
    else:
        corrected = raw * (1.0 + strength * correction)
    corrected = torch.where(raw_valid_tensor & torch.isfinite(corrected), corrected, raw)
    return JointGapCorrectionResult(
        corrected_priority=corrected,
        correction=correction,
        raw_gate_mask=gate,
        segment_gap_score=topk.segment_gap_score,
        segment_valid=segment_valid,
        topk_selection_frequency=topk.topk_selection_frequency,
        global_gap=global_gap,
        global_valid=global_valid,
        local_gap=local.local_gap,
        local_valid=local.local_valid,
    )


def finite_joint_gap_summary(
    values: torch.Tensor,
    valid: torch.Tensor,
) -> dict[str, float]:
    """Return low-cardinality summaries for W&B-safe diagnostics."""

    tensor = torch.as_tensor(values, dtype=torch.float64)
    mask = torch.as_tensor(valid, dtype=torch.bool, device=tensor.device)
    if tensor.shape != mask.shape:
        raise ValueError("values and valid must have the same shape.")
    selected = tensor[mask & torch.isfinite(tensor)]
    if selected.numel() == 0:
        return {"mean": 0.0, "p90": 0.0, "max": 0.0}
    return {
        "mean": float(selected.mean().item()),
        "p90": float(torch.quantile(selected, torch.tensor(0.90, dtype=selected.dtype, device=selected.device)).item()),
        "max": float(selected.max().item()),
    }
