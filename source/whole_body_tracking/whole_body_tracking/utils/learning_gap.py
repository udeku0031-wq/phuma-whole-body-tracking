"""Raw tracking error and difficulty-calibrated learning-gap formulas."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SegmentErrorResult:
    error: torch.Tensor
    valid: torch.Tensor
    contributions: dict[str, torch.Tensor]
    active_weight: torch.Tensor


@dataclass(frozen=True)
class MotionErrorResult:
    error: torch.Tensor
    valid: torch.Tensor
    segment_mean: torch.Tensor
    segment_p90: torch.Tensor
    contributions: dict[str, torch.Tensor]


@dataclass(frozen=True)
class BinCalibrationResult:
    mean: torch.Tensor
    sigma: torch.Tensor
    valid_segment_count: torch.Tensor
    fallback_mask: torch.Tensor
    reliable_mask: torch.Tensor
    global_mean: torch.Tensor
    global_sigma: torch.Tensor


@dataclass(frozen=True)
class GapResult:
    global_gap: torch.Tensor
    global_valid: torch.Tensor
    motion_gap: torch.Tensor
    motion_valid: torch.Tensor
    local_gap: torch.Tensor
    motion_positive_mean: torch.Tensor
    motion_positive_p90: torch.Tensor
    clipped_mask: torch.Tensor
    contributions: dict[str, torch.Tensor]


def _require_vector(name: str, value: torch.Tensor, *, size: int | None = None) -> torch.Tensor:
    tensor = torch.as_tensor(value)
    if tensor.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if size is not None and tensor.numel() != size:
        raise ValueError(f"{name} must contain {size} values.")
    return tensor


def _validate_positive_finite(name: str, value: float) -> float:
    resolved = float(value)
    if not math.isfinite(resolved) or resolved <= 0.0:
        raise ValueError(f"{name} must be finite and greater than zero.")
    return resolved


def compute_segment_error(
    component_ema: Mapping[str, torch.Tensor],
    component_initialized: Mapping[str, torch.Tensor],
    segment_step_count: torch.Tensor,
    *,
    min_segment_observations: int,
    body_position_scale_m: float,
    joint_position_scale_rad: float,
    orientation_scale_rad: float,
    weights: Mapping[str, float],
    component_clip: float,
) -> SegmentErrorResult:
    """Compute fixed-unit segment error without using offline difficulty."""

    names = ("body", "joint", "orientation", "termination", "completion", "success")
    missing = [name for name in names if name not in component_ema or name not in component_initialized]
    if missing:
        raise ValueError(f"Segment error components are missing: {missing}")
    if min_segment_observations < 1:
        raise ValueError("min_segment_observations must be positive.")
    clip = _validate_positive_finite("component_clip", component_clip)
    scales = {
        "body": _validate_positive_finite("body_position_scale_m", body_position_scale_m),
        "joint": _validate_positive_finite("joint_position_scale_rad", joint_position_scale_rad),
        "orientation": _validate_positive_finite("orientation_scale_rad", orientation_scale_rad),
    }

    step_count = _require_vector("segment_step_count", segment_step_count)
    size = step_count.numel()
    device = step_count.device
    dtype = torch.float64
    numerator = torch.zeros(size, dtype=dtype, device=device)
    denominator = torch.zeros_like(numerator)
    contributions: dict[str, torch.Tensor] = {}
    active_tracking = torch.zeros(size, dtype=torch.bool, device=device)

    for name in names:
        value = _require_vector(name, component_ema[name], size=size).to(device=device, dtype=dtype)
        initialized = _require_vector(
            f"{name}_initialized", component_initialized[name], size=size
        ).to(device=device, dtype=torch.bool)
        weight = float(weights.get(name, 0.0))
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError(f"Segment error weight '{name}' must be finite and non-negative.")
        if name in scales:
            normalized = value / scales[name]
            active_tracking |= initialized & torch.isfinite(value)
        elif name == "termination":
            normalized = value
        else:
            normalized = 1.0 - value
        normalized = torch.clamp(normalized, 0.0, clip)
        active = initialized & torch.isfinite(normalized) & (weight > 0.0)
        contribution = torch.where(active, normalized * weight, torch.zeros_like(normalized))
        contributions[name] = contribution
        numerator += contribution
        denominator += active.to(dtype) * weight

    finite_denominator = denominator > 0.0
    error = torch.where(finite_denominator, numerator / denominator.clamp_min(torch.finfo(dtype).tiny), 0.0)
    error = torch.nan_to_num(error, nan=0.0, posinf=clip, neginf=0.0)
    valid = (step_count >= min_segment_observations) & active_tracking & finite_denominator
    return SegmentErrorResult(error=error, valid=valid, contributions=contributions, active_weight=denominator)


def segmented_quantile(
    values: torch.Tensor,
    group_ids: torch.Tensor,
    valid: torch.Tensor,
    *,
    num_groups: int,
    quantile: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return an unweighted quantile per group without a Python group loop."""

    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1].")
    values = _require_vector("values", values)
    group_ids = _require_vector("group_ids", group_ids, size=values.numel()).to(
        device=values.device, dtype=torch.long
    )
    valid = _require_vector("valid", valid, size=values.numel()).to(device=values.device, dtype=torch.bool)
    valid &= torch.isfinite(values)
    if group_ids.numel() and (torch.any(group_ids < 0) or torch.any(group_ids >= num_groups)):
        raise ValueError("group_ids are outside the configured group range.")
    output = torch.zeros(num_groups, dtype=values.dtype, device=values.device)
    counts = torch.bincount(group_ids[valid], minlength=num_groups).to(torch.long)
    if not torch.any(valid):
        return output, counts

    selected_values = values[valid]
    selected_groups = group_ids[valid]
    value_order = torch.argsort(selected_values, stable=True)
    values_by_value = selected_values[value_order]
    groups_by_value = selected_groups[value_order]
    group_order = torch.argsort(groups_by_value, stable=True)
    sorted_values = values_by_value[group_order]
    offsets = torch.cumsum(counts, dim=0) - counts
    active = counts > 0
    fractional_rank = (counts.to(torch.float64) - 1.0).clamp_min(0.0) * quantile
    lower_rank = torch.floor(fractional_rank).to(torch.long)
    upper_rank = torch.ceil(fractional_rank).to(torch.long)
    lower = sorted_values[offsets[active] + lower_rank[active]]
    upper = sorted_values[offsets[active] + upper_rank[active]]
    fraction = fractional_rank[active] - lower_rank[active].to(torch.float64)
    output[active] = lower + (upper - lower) * fraction.to(values.dtype)
    return output, counts


def _weighted_group_mean(
    values: torch.Tensor,
    group_ids: torch.Tensor,
    valid: torch.Tensor,
    weights: torch.Tensor,
    num_groups: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    valid = valid & torch.isfinite(values) & torch.isfinite(weights) & (weights > 0.0)
    sums = torch.zeros(num_groups, dtype=values.dtype, device=values.device)
    weight_sums = torch.zeros_like(sums)
    if torch.any(valid):
        sums.scatter_add_(0, group_ids[valid], values[valid] * weights[valid])
        weight_sums.scatter_add_(0, group_ids[valid], weights[valid])
    means = torch.where(weight_sums > 0.0, sums / weight_sums.clamp_min(torch.finfo(values.dtype).tiny), 0.0)
    return means, weight_sums


def compute_motion_error(
    segment_error: torch.Tensor,
    segment_valid: torch.Tensor,
    segment_motion_ids: torch.Tensor,
    segment_observation_count: torch.Tensor,
    motion_outcome_ema: Mapping[str, torch.Tensor],
    motion_outcome_initialized: Mapping[str, torch.Tensor],
    motion_episode_count: torch.Tensor,
    *,
    min_motion_episodes: int,
    weights: Mapping[str, float],
) -> MotionErrorResult:
    """Combine observation-weighted segment error and motion outcomes."""

    segment_error = _require_vector("segment_error", segment_error).to(torch.float64)
    segment_valid = _require_vector("segment_valid", segment_valid, size=segment_error.numel()).to(
        device=segment_error.device, dtype=torch.bool
    )
    segment_motion_ids = _require_vector(
        "segment_motion_ids", segment_motion_ids, size=segment_error.numel()
    ).to(device=segment_error.device, dtype=torch.long)
    segment_counts = _require_vector(
        "segment_observation_count", segment_observation_count, size=segment_error.numel()
    ).to(device=segment_error.device, dtype=torch.float64)
    motion_episode_count = _require_vector("motion_episode_count", motion_episode_count).to(
        device=segment_error.device, dtype=torch.long
    )
    num_motions = motion_episode_count.numel()
    if min_motion_episodes < 1:
        raise ValueError("min_motion_episodes must be positive.")
    if segment_motion_ids.numel() and (
        torch.any(segment_motion_ids < 0) or torch.any(segment_motion_ids >= num_motions)
    ):
        raise ValueError("segment_motion_ids are outside the motion range.")

    segment_mean, segment_weight_sum = _weighted_group_mean(
        segment_error,
        segment_motion_ids,
        segment_valid,
        segment_counts.clamp_min(1.0),
        num_motions,
    )
    segment_p90, valid_segment_count = segmented_quantile(
        segment_error,
        segment_motion_ids,
        segment_valid,
        num_groups=num_motions,
        quantile=0.90,
    )
    numerator = torch.zeros(num_motions, dtype=torch.float64, device=segment_error.device)
    denominator = torch.zeros_like(numerator)
    contributions: dict[str, torch.Tensor] = {}

    for name, component in (("segment_mean", segment_mean), ("segment_p90", segment_p90)):
        weight = float(weights.get(name, 0.0))
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError(f"Motion error weight '{name}' must be finite and non-negative.")
        active = (valid_segment_count > 0) & (weight > 0.0)
        contribution = torch.where(active, component * weight, torch.zeros_like(component))
        contributions[name] = contribution
        numerator += contribution
        denominator += active.to(torch.float64) * weight

    for name in ("termination", "completion", "success"):
        if name not in motion_outcome_ema or name not in motion_outcome_initialized:
            raise ValueError(f"Missing motion outcome component '{name}'.")
        value = _require_vector(name, motion_outcome_ema[name], size=num_motions).to(
            device=segment_error.device, dtype=torch.float64
        )
        initialized = _require_vector(
            f"{name}_initialized", motion_outcome_initialized[name], size=num_motions
        ).to(device=segment_error.device, dtype=torch.bool)
        transformed = value if name == "termination" else 1.0 - value
        transformed = torch.clamp(transformed, 0.0, 1.0)
        weight = float(weights.get(name, 0.0))
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError(f"Motion error weight '{name}' must be finite and non-negative.")
        active = initialized & torch.isfinite(transformed) & (weight > 0.0)
        contribution = torch.where(active, transformed * weight, torch.zeros_like(transformed))
        contributions[name] = contribution
        numerator += contribution
        denominator += active.to(torch.float64) * weight

    error = torch.where(denominator > 0.0, numerator / denominator.clamp_min(torch.finfo(torch.float64).tiny), 0.0)
    error = torch.nan_to_num(error, nan=0.0, posinf=0.0, neginf=0.0)
    valid = (
        (motion_episode_count >= min_motion_episodes)
        & (segment_weight_sum > 0.0)
        & (denominator > 0.0)
    )
    return MotionErrorResult(
        error=error,
        valid=valid,
        segment_mean=segment_mean,
        segment_p90=segment_p90,
        contributions=contributions,
    )


def estimate_difficulty_bin_expectation(
    segment_error: torch.Tensor,
    segment_valid: torch.Tensor,
    difficulty_bin: torch.Tensor,
    *,
    num_bins: int,
    min_bin_valid_segments: int,
    sigma_floor: float,
    observation_weights: torch.Tensor | None = None,
) -> BinCalibrationResult:
    """Estimate current-policy error mean/std by frozen difficulty bin."""

    if num_bins < 1 or min_bin_valid_segments < 1:
        raise ValueError("num_bins and min_bin_valid_segments must be positive.")
    floor = _validate_positive_finite("sigma_floor", sigma_floor)
    error = _require_vector("segment_error", segment_error).to(torch.float64)
    valid = _require_vector("segment_valid", segment_valid, size=error.numel()).to(
        device=error.device, dtype=torch.bool
    )
    bins = _require_vector("difficulty_bin", difficulty_bin, size=error.numel()).to(
        device=error.device, dtype=torch.long
    )
    if bins.numel() and (torch.any(bins < 0) or torch.any(bins >= num_bins)):
        raise ValueError("difficulty_bin values are outside the configured range.")
    valid &= torch.isfinite(error)
    if observation_weights is None:
        weights = torch.ones_like(error)
    else:
        weights = _require_vector("observation_weights", observation_weights, size=error.numel()).to(
            device=error.device, dtype=torch.float64
        )
        valid &= torch.isfinite(weights) & (weights > 0.0)

    valid_counts = torch.bincount(bins[valid], minlength=num_bins).to(torch.long)
    weight_sums = torch.zeros(num_bins, dtype=torch.float64, device=error.device)
    weighted_sum = torch.zeros_like(weight_sums)
    weighted_square_sum = torch.zeros_like(weight_sums)
    if torch.any(valid):
        weight_sums.scatter_add_(0, bins[valid], weights[valid])
        weighted_sum.scatter_add_(0, bins[valid], weights[valid] * error[valid])
        weighted_square_sum.scatter_add_(0, bins[valid], weights[valid] * error[valid].square())
    direct_mean = torch.where(weight_sums > 0.0, weighted_sum / weight_sums.clamp_min(1.0e-300), 0.0)
    direct_variance = torch.where(
        weight_sums > 0.0,
        weighted_square_sum / weight_sums.clamp_min(1.0e-300) - direct_mean.square(),
        0.0,
    ).clamp_min(0.0)
    direct_sigma = direct_variance.sqrt().clamp_min(floor)

    total_weight = weights[valid].sum()
    if torch.any(valid):
        global_mean = (weights[valid] * error[valid]).sum() / total_weight
        global_variance = (
            (weights[valid] * error[valid].square()).sum() / total_weight - global_mean.square()
        ).clamp_min(0.0)
        global_sigma = global_variance.sqrt().clamp_min(floor)
    else:
        global_mean = torch.zeros((), dtype=torch.float64, device=error.device)
        global_sigma = torch.ones((), dtype=torch.float64, device=error.device)

    direct_reliable = valid_counts >= min_bin_valid_segments
    global_reliable = int(valid.sum().item()) >= min_bin_valid_segments
    fallback_mask = ~direct_reliable
    mean = torch.where(direct_reliable, direct_mean, global_mean.expand_as(direct_mean))
    sigma = torch.where(direct_reliable, direct_sigma, global_sigma.expand_as(direct_sigma)).clamp_min(floor)
    reliable = direct_reliable | global_reliable
    return BinCalibrationResult(
        mean=mean,
        sigma=sigma,
        valid_segment_count=valid_counts,
        fallback_mask=fallback_mask,
        reliable_mask=reliable,
        global_mean=global_mean,
        global_sigma=global_sigma,
    )


def compute_local_relative_gap(
    global_gap: torch.Tensor,
    global_valid: torch.Tensor,
    segment_motion_ids: torch.Tensor,
    *,
    num_motions: int,
    gap_clip: float,
) -> torch.Tensor:
    """Center valid gaps by the linear median within each motion."""

    clip = _validate_positive_finite("gap_clip", gap_clip)
    gap = _require_vector("global_gap", global_gap).to(torch.float64)
    valid = _require_vector("global_valid", global_valid, size=gap.numel()).to(
        device=gap.device, dtype=torch.bool
    )
    motion_ids = _require_vector(
        "segment_motion_ids", segment_motion_ids, size=gap.numel()
    ).to(device=gap.device, dtype=torch.long)
    median, count = segmented_quantile(
        gap,
        motion_ids,
        valid,
        num_groups=num_motions,
        quantile=0.50,
    )
    local = torch.where(
        valid,
        torch.clamp(gap - median[motion_ids], -clip, clip),
        torch.zeros_like(gap),
    )
    return torch.where(count[motion_ids] == 1, torch.zeros_like(local), local)


def compute_learning_gaps(
    segment_error: torch.Tensor,
    segment_valid: torch.Tensor,
    segment_motion_ids: torch.Tensor,
    segment_observation_count: torch.Tensor,
    difficulty_bin: torch.Tensor,
    calibration: BinCalibrationResult,
    motion_outcome_ema: Mapping[str, torch.Tensor],
    motion_outcome_initialized: Mapping[str, torch.Tensor],
    motion_episode_count: torch.Tensor,
    *,
    min_motion_episodes: int,
    gap_clip: float,
    motion_gap_weights: Mapping[str, float],
) -> GapResult:
    """Compute global, motion-level and within-motion relative learning gaps."""

    clip = _validate_positive_finite("gap_clip", gap_clip)
    error = _require_vector("segment_error", segment_error).to(torch.float64)
    valid = _require_vector("segment_valid", segment_valid, size=error.numel()).to(
        device=error.device, dtype=torch.bool
    )
    motion_ids = _require_vector("segment_motion_ids", segment_motion_ids, size=error.numel()).to(
        device=error.device, dtype=torch.long
    )
    observation_count = _require_vector(
        "segment_observation_count", segment_observation_count, size=error.numel()
    ).to(device=error.device, dtype=torch.float64)
    bins = _require_vector("difficulty_bin", difficulty_bin, size=error.numel()).to(
        device=error.device, dtype=torch.long
    )
    motion_episode_count = _require_vector("motion_episode_count", motion_episode_count).to(
        device=error.device, dtype=torch.long
    )
    num_motions = motion_episode_count.numel()
    reliable = calibration.reliable_mask[bins]
    global_valid = valid & reliable & torch.isfinite(error)
    raw_gap = (error - calibration.mean[bins]) / calibration.sigma[bins]
    raw_gap = torch.nan_to_num(raw_gap, nan=0.0, posinf=clip, neginf=-clip)
    clipped_mask = global_valid & (raw_gap.abs() > clip)
    global_gap = torch.where(global_valid, torch.clamp(raw_gap, -clip, clip), torch.zeros_like(raw_gap))

    positive = torch.clamp(global_gap, min=0.0)
    positive_mean, positive_weight_sum = _weighted_group_mean(
        positive,
        motion_ids,
        global_valid,
        observation_count.clamp_min(1.0),
        num_motions,
    )
    positive_p90, valid_gap_count = segmented_quantile(
        positive,
        motion_ids,
        global_valid,
        num_groups=num_motions,
        quantile=0.90,
    )

    numerator = torch.zeros(num_motions, dtype=torch.float64, device=error.device)
    denominator = torch.zeros_like(numerator)
    contributions: dict[str, torch.Tensor] = {}
    for name, component in (("positive_mean", positive_mean), ("positive_p90", positive_p90)):
        weight = float(motion_gap_weights.get(name, 0.0))
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError(f"Motion gap weight '{name}' must be finite and non-negative.")
        active = (valid_gap_count > 0) & (weight > 0.0)
        contribution = torch.where(active, component * weight, torch.zeros_like(component))
        contributions[name] = contribution
        numerator += contribution
        denominator += active.to(torch.float64) * weight

    for name in ("termination", "completion", "success"):
        value = _require_vector(name, motion_outcome_ema[name], size=num_motions).to(
            device=error.device, dtype=torch.float64
        )
        initialized = _require_vector(
            f"{name}_initialized", motion_outcome_initialized[name], size=num_motions
        ).to(device=error.device, dtype=torch.bool)
        transformed = value if name == "termination" else 1.0 - value
        transformed = torch.clamp(transformed, 0.0, 1.0)
        weight = float(motion_gap_weights.get(name, 0.0))
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError(f"Motion gap weight '{name}' must be finite and non-negative.")
        active = initialized & torch.isfinite(transformed) & (weight > 0.0)
        contribution = torch.where(active, transformed * weight, torch.zeros_like(transformed))
        contributions[name] = contribution
        numerator += contribution
        denominator += active.to(torch.float64) * weight

    motion_gap = torch.where(denominator > 0.0, numerator / denominator.clamp_min(1.0e-300), 0.0)
    motion_gap = torch.clamp(torch.nan_to_num(motion_gap, nan=0.0), 0.0, clip)
    motion_valid = (
        (motion_episode_count >= min_motion_episodes)
        & (positive_weight_sum > 0.0)
        & (denominator > 0.0)
    )

    local_gap = compute_local_relative_gap(
        global_gap,
        global_valid,
        motion_ids,
        num_motions=num_motions,
        gap_clip=clip,
    )
    return GapResult(
        global_gap=global_gap,
        global_valid=global_valid,
        motion_gap=motion_gap,
        motion_valid=motion_valid,
        local_gap=local_gap,
        motion_positive_mean=positive_mean,
        motion_positive_p90=positive_p90,
        clipped_mask=clipped_mask,
        contributions=contributions,
    )


def finite_distribution_summary(values: torch.Tensor, valid: torch.Tensor) -> dict[str, float]:
    """Return low-cardinality summaries without exporting full arrays."""

    tensor = _require_vector("values", values).to(torch.float64)
    mask = _require_vector("valid", valid, size=tensor.numel()).to(device=tensor.device, dtype=torch.bool)
    selected = tensor[mask & torch.isfinite(tensor)]
    if selected.numel() == 0:
        return {"mean": 0.0, "p50": 0.0, "p90": 0.0, "p99": 0.0}
    quantiles = torch.quantile(selected, torch.tensor([0.50, 0.90, 0.99], device=selected.device, dtype=selected.dtype))
    return {
        "mean": float(selected.mean().item()),
        "p50": float(quantiles[0].item()),
        "p90": float(quantiles[1].item()),
        "p99": float(quantiles[2].item()),
    }
