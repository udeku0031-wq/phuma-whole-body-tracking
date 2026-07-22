"""Deterministic adaptive sampling with masks, exploration floors and caps."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch


ADAPTIVE_SAMPLER_SCHEMA_VERSION = "wbt.adaptive_sampler.v1"


@dataclass(frozen=True)
class ProbabilityResult:
    probability: torch.Tensor
    fallback_used: bool
    fallback_reason: str


def _validate_probability_inputs(
    scores: torch.Tensor,
    score_valid: torch.Tensor,
    eligible: torch.Tensor,
    sample_count: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    scores = torch.as_tensor(scores, dtype=torch.float64)
    score_valid = torch.as_tensor(score_valid, dtype=torch.bool, device=scores.device)
    eligible = torch.as_tensor(eligible, dtype=torch.bool, device=scores.device)
    sample_count = torch.as_tensor(sample_count, dtype=torch.float64, device=scores.device)
    if scores.ndim != 1 or any(item.shape != scores.shape for item in (score_valid, eligible, sample_count)):
        raise ValueError("Probability inputs must be one-dimensional tensors with the same shape.")
    if not torch.any(eligible):
        raise ValueError("At least one eligible item is required.")
    if torch.any(sample_count < 0) or not torch.all(torch.isfinite(sample_count)):
        raise ValueError("sample_count must be finite and non-negative.")
    return scores, score_valid, eligible, sample_count


def uniform_probability(eligible: torch.Tensor) -> torch.Tensor:
    eligible = torch.as_tensor(eligible, dtype=torch.bool)
    if eligible.ndim != 1 or not torch.any(eligible):
        raise ValueError("eligible must be a one-dimensional mask with at least one true value.")
    probability = eligible.to(torch.float64)
    return probability / probability.sum()


def capped_simplex(probability: torch.Tensor, eligible: torch.Tensor, cap: float) -> torch.Tensor:
    """Apply an upper bound by proportional water-filling.

    Unlike clamp-and-renormalize, saturated entries stay at the cap while the
    remaining mass is redistributed among unsaturated entries.
    """

    probability = torch.as_tensor(probability, dtype=torch.float64)
    eligible = torch.as_tensor(eligible, dtype=torch.bool, device=probability.device)
    if probability.ndim != 1 or eligible.shape != probability.shape:
        raise ValueError("probability and eligible must be one-dimensional with the same shape.")
    if not math.isfinite(cap) or cap <= 0.0 or cap > 1.0:
        raise ValueError("cap must be finite and in (0, 1].")
    count = int(torch.count_nonzero(eligible).item())
    if count == 0:
        raise ValueError("At least one eligible item is required.")
    if cap * count < 1.0 - 1.0e-12:
        raise ValueError(f"Probability cap {cap:g} is infeasible for {count} eligible items.")
    if not torch.all(torch.isfinite(probability)) or torch.any(probability < 0.0):
        raise ValueError("probability must be finite and non-negative.")
    weights = torch.where(eligible, probability, torch.zeros_like(probability))
    if float(weights.sum().item()) <= 0.0:
        weights = eligible.to(torch.float64)
    weights = weights / weights.sum()
    result = torch.zeros_like(weights)
    active = eligible.clone()
    remaining = 1.0
    for _ in range(count + 1):
        if not torch.any(active):
            break
        active_weights = weights[active]
        active_count = int(active_weights.numel())
        if float(active_weights.sum().item()) > 0.0:
            candidate = active_weights / active_weights.sum() * remaining
        else:
            candidate = torch.full_like(active_weights, remaining / active_count)
        saturated = candidate > cap + 1.0e-12
        active_ids = torch.where(active)[0]
        if not torch.any(saturated):
            result[active_ids] = candidate
            active.zero_()
            break
        saturated_ids = active_ids[saturated]
        result[saturated_ids] = cap
        remaining -= cap * saturated_ids.numel()
        active[saturated_ids] = False
    result = torch.where(eligible, result, torch.zeros_like(result))
    error = abs(float(result.sum().item()) - 1.0)
    if error > 1.0e-10 or torch.any(result[eligible] > cap + 1.0e-10):
        raise RuntimeError("Capped-simplex water-filling failed numerical validation.")
    return result


def build_probability(
    scores: torch.Tensor,
    score_valid: torch.Tensor,
    eligible: torch.Tensor,
    sample_count: torch.Tensor,
    *,
    uniform_mix: float,
    temperature: float,
    under_sampling_weight: float,
    probability_cap: float,
    score_clip: float,
    force_uniform: bool = False,
) -> ProbabilityResult:
    """Build one masked adaptive distribution with a cold-start neutral prior."""

    scores, score_valid, eligible, sample_count = _validate_probability_inputs(
        scores, score_valid, eligible, sample_count
    )
    if not 0.0 <= uniform_mix <= 1.0:
        raise ValueError("uniform_mix must be in [0, 1].")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and greater than zero.")
    if not math.isfinite(under_sampling_weight) or under_sampling_weight < 0.0:
        raise ValueError("under_sampling_weight must be finite and non-negative.")
    if not math.isfinite(score_clip) or score_clip <= 0.0:
        raise ValueError("score_clip must be finite and greater than zero.")

    uniform = uniform_probability(eligible)
    effective_cap = float(probability_cap)
    if force_uniform:
        return ProbabilityResult(
            probability=capped_simplex(uniform, eligible, effective_cap),
            fallback_used=False,
            fallback_reason="",
        )

    reliable = eligible & score_valid & torch.isfinite(scores)
    if not torch.any(reliable):
        return ProbabilityResult(
            probability=capped_simplex(uniform, eligible, effective_cap),
            fallback_used=True,
            fallback_reason="no_reliable_score",
        )
    if torch.any(eligible & score_valid & ~torch.isfinite(scores)):
        return ProbabilityResult(
            probability=capped_simplex(uniform, eligible, effective_cap),
            fallback_used=True,
            fallback_reason="nonfinite_score",
        )

    neutral = torch.median(scores[reliable])
    resolved = torch.where(reliable, scores, neutral)
    resolved = torch.clamp(resolved, -score_clip, score_clip)
    exploration = under_sampling_weight / torch.sqrt(sample_count + 1.0)
    logits = (resolved + exploration) / temperature
    logits = torch.where(eligible, logits, torch.full_like(logits, -torch.inf))
    adaptive = torch.softmax(logits, dim=0)
    probability = uniform_mix * uniform + (1.0 - uniform_mix) * adaptive
    if not torch.all(torch.isfinite(probability)) or float(probability.sum().item()) <= 0.0:
        return ProbabilityResult(
            probability=capped_simplex(uniform, eligible, effective_cap),
            fallback_used=True,
            fallback_reason="invalid_probability",
        )
    probability = capped_simplex(probability, eligible, effective_cap)
    return ProbabilityResult(
        probability=probability,
        fallback_used=False,
        fallback_reason="",
    )


def grouped_probability(
    scores: torch.Tensor,
    score_valid: torch.Tensor,
    eligible: torch.Tensor,
    sample_count: torch.Tensor,
    group_ids: torch.Tensor,
    *,
    num_groups: int,
    uniform_mix: float,
    temperature: float,
    under_sampling_weight: float,
    probability_cap: float,
    score_clip: float,
    force_uniform: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Build independent conditional distributions for contiguous or ragged groups.

    Returns probabilities, non-empty group mask, fallback count and cap-relax
    count.  The water-filling loop is vectorized across groups; it never loops
    over environments or motions.
    """

    scores, score_valid, eligible, sample_count = _validate_probability_inputs(
        scores, score_valid, eligible, sample_count
    )
    group_ids = torch.as_tensor(group_ids, dtype=torch.long, device=scores.device)
    if group_ids.shape != scores.shape:
        raise ValueError("group_ids must match the score shape.")
    if group_ids.numel() and (torch.any(group_ids < 0) or torch.any(group_ids >= num_groups)):
        raise ValueError("group_ids are outside the configured group range.")
    if (
        not math.isfinite(uniform_mix)
        or not 0.0 <= uniform_mix <= 1.0
        or not math.isfinite(temperature)
        or temperature <= 0.0
        or not math.isfinite(under_sampling_weight)
        or under_sampling_weight < 0.0
    ):
        raise ValueError("Invalid grouped probability hyperparameters.")
    if (
        not math.isfinite(probability_cap)
        or not 0.0 < probability_cap <= 1.0
        or not math.isfinite(score_clip)
        or score_clip <= 0.0
    ):
        raise ValueError("Invalid grouped probability cap or score clip.")

    eligible_count = torch.zeros(num_groups, dtype=torch.long, device=scores.device)
    eligible_count.scatter_add_(0, group_ids, eligible.to(torch.long))
    nonempty = eligible_count > 0
    infeasible = nonempty & (
        eligible_count.to(torch.float64) * float(probability_cap) < 1.0 - 1.0e-12
    )
    if torch.any(infeasible):
        first_group = int(torch.where(infeasible)[0][0].item())
        first_count = int(eligible_count[first_group].item())
        raise ValueError(
            f"Conditional segment probability cap {probability_cap:g} is infeasible for "
            f"motion group {first_group} with {first_count} eligible segment(s)."
        )
    uniform = torch.where(
        eligible,
        1.0 / eligible_count[group_ids].clamp_min(1).to(torch.float64),
        torch.zeros_like(scores),
    )
    reliable = eligible & score_valid & torch.isfinite(scores)
    reliable_count = torch.zeros(num_groups, dtype=torch.long, device=scores.device)
    reliable_count.scatter_add_(0, group_ids, reliable.to(torch.long))
    fallback_group = nonempty & (reliable_count == 0)
    nonfinite_group = torch.zeros(num_groups, dtype=torch.bool, device=scores.device)
    bad = eligible & score_valid & ~torch.isfinite(scores)
    if torch.any(bad):
        bad_count = torch.zeros(num_groups, dtype=torch.long, device=scores.device)
        bad_count.scatter_add_(0, group_ids[bad], torch.ones_like(group_ids[bad]))
        nonfinite_group = bad_count > 0
    fallback_group |= nonfinite_group

    if force_uniform:
        fallback_group.zero_()
        probability = uniform
    else:
        reliable_scores = scores[reliable]
        neutral = torch.median(reliable_scores) if reliable_scores.numel() else torch.zeros((), device=scores.device)
        resolved = torch.where(reliable, scores, neutral)
        resolved = torch.clamp(resolved, -score_clip, score_clip)
        exploration = under_sampling_weight / torch.sqrt(sample_count + 1.0)
        logits = (resolved + exploration) / temperature
        logits = torch.where(eligible, logits, torch.full_like(logits, -torch.inf))
        group_max = torch.full((num_groups,), -torch.inf, dtype=torch.float64, device=scores.device)
        group_max.scatter_reduce_(0, group_ids, logits, reduce="amax", include_self=True)
        exp_logits = torch.where(eligible, torch.exp(logits - group_max[group_ids]), torch.zeros_like(logits))
        group_sum = torch.zeros(num_groups, dtype=torch.float64, device=scores.device)
        group_sum.scatter_add_(0, group_ids, exp_logits)
        adaptive = torch.where(
            eligible,
            exp_logits / group_sum[group_ids].clamp_min(1.0e-300),
            torch.zeros_like(exp_logits),
        )
        probability = uniform_mix * uniform + (1.0 - uniform_mix) * adaptive
        probability = torch.where(fallback_group[group_ids], uniform, probability)

    effective_cap_by_group = torch.full(
        (num_groups,), float(probability_cap), dtype=torch.float64, device=scores.device
    )
    result = torch.zeros_like(probability)
    active = eligible.clone()
    remaining = nonempty.to(torch.float64)
    max_group_size = int(eligible_count.max().item())
    for _ in range(max_group_size + 1):
        if not torch.any(active):
            break
        active_weight_sum = torch.zeros(num_groups, dtype=torch.float64, device=scores.device)
        active_count = torch.zeros(num_groups, dtype=torch.long, device=scores.device)
        active_weight_sum.scatter_add_(0, group_ids, torch.where(active, probability, torch.zeros_like(probability)))
        active_count.scatter_add_(0, group_ids, active.to(torch.long))
        candidate = torch.where(
            active,
            torch.where(
                active_weight_sum[group_ids] > 0.0,
                probability * remaining[group_ids] / active_weight_sum[group_ids].clamp_min(1.0e-300),
                remaining[group_ids] / active_count[group_ids].clamp_min(1).to(torch.float64),
            ),
            torch.zeros_like(probability),
        )
        saturated = active & (candidate > effective_cap_by_group[group_ids] + 1.0e-12)
        saturated_count = torch.zeros(num_groups, dtype=torch.long, device=scores.device)
        saturated_count.scatter_add_(0, group_ids, saturated.to(torch.long))
        groups_with_saturation = saturated_count > 0
        finish = active & ~groups_with_saturation[group_ids]
        result[finish] = candidate[finish]
        active[finish] = False
        if torch.any(saturated):
            result[saturated] = effective_cap_by_group[group_ids[saturated]]
            removed = torch.zeros(num_groups, dtype=torch.float64, device=scores.device)
            removed.scatter_add_(0, group_ids[saturated], result[saturated])
            remaining -= removed
            active[saturated] = False

    sums = torch.zeros(num_groups, dtype=torch.float64, device=scores.device)
    sums.scatter_add_(0, group_ids, result)
    if torch.any(torch.abs(sums[nonempty] - 1.0) > 1.0e-9):
        raise RuntimeError("Grouped probability water-filling failed to normalize a group.")
    if torch.any(result > effective_cap_by_group[group_ids] + 1.0e-9):
        raise RuntimeError("Grouped probability water-filling exceeded a cap.")
    return result, nonempty, int(torch.count_nonzero(fallback_group).item())


class HierarchicalAdaptiveSampler:
    """Low-frequency probability state and a dedicated sampling generator."""

    def __init__(
        self,
        segment_motion_ids: torch.Tensor,
        segment_start_frames: torch.Tensor,
        segment_end_frames: torch.Tensor,
        motion_lengths: torch.Tensor,
        *,
        motion_eligible_mask: torch.Tensor,
        segment_eligible_mask: torch.Tensor,
        motion_mode: str,
        segment_mode: str,
        warmup_iterations: int,
        probability_update_interval: int,
        uniform_mix: float,
        temperature: float,
        under_sampling_weight: float,
        motion_probability_cap: float,
        segment_probability_cap: float,
        score_clip: float,
        sampler_seed: int,
        config_hash: str,
        device: str | torch.device = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.segment_motion_ids = torch.as_tensor(segment_motion_ids, dtype=torch.long, device=self.device)
        self.segment_start_frames = torch.as_tensor(segment_start_frames, dtype=torch.long, device=self.device)
        self.segment_end_frames = torch.as_tensor(segment_end_frames, dtype=torch.long, device=self.device)
        self.motion_lengths = torch.as_tensor(motion_lengths, dtype=torch.long, device=self.device)
        self.num_segments = int(self.segment_motion_ids.numel())
        self.num_motions = int(self.motion_lengths.numel())
        if any(item.shape != self.segment_motion_ids.shape for item in (self.segment_start_frames, self.segment_end_frames)):
            raise ValueError("Segment layout tensors must have the same shape.")
        if self.segment_motion_ids.numel() and (
            torch.any(self.segment_motion_ids < 0) or torch.any(self.segment_motion_ids >= self.num_motions)
        ):
            raise ValueError("segment_motion_ids are outside the motion range.")
        self.motion_eligible_mask = torch.as_tensor(motion_eligible_mask, dtype=torch.bool, device=self.device)
        self.segment_eligible_mask = torch.as_tensor(segment_eligible_mask, dtype=torch.bool, device=self.device)
        if self.motion_eligible_mask.shape != self.motion_lengths.shape or self.segment_eligible_mask.shape != self.segment_motion_ids.shape:
            raise ValueError("Eligibility masks do not match the motion/segment layout.")
        if not torch.any(self.motion_eligible_mask) or not torch.any(self.segment_eligible_mask):
            raise ValueError("Adaptive sampling requires at least one eligible motion and segment.")
        if torch.any(self.segment_eligible_mask & ~self.motion_eligible_mask[self.segment_motion_ids]):
            raise ValueError("An eligible segment belongs to an ineligible motion.")
        legal_end = torch.minimum(self.segment_end_frames, self.motion_lengths[self.segment_motion_ids] - 1)
        self.segment_start_count = torch.clamp(legal_end - self.segment_start_frames, min=0)
        if torch.any(self.segment_eligible_mask & (self.segment_start_count <= 0)):
            raise ValueError("Every eligible segment must contain a legal legacy start frame.")

        self.motion_mode = str(motion_mode)
        self.segment_mode = str(segment_mode)
        self.warmup_iterations = int(warmup_iterations)
        self.probability_update_interval = int(probability_update_interval)
        self.uniform_mix = float(uniform_mix)
        self.temperature = float(temperature)
        self.under_sampling_weight = float(under_sampling_weight)
        self.motion_probability_cap = float(motion_probability_cap)
        self.segment_probability_cap = float(segment_probability_cap)
        self.score_clip = float(score_clip)
        self.sampler_seed = int(sampler_seed)
        self.config_hash = str(config_hash)
        if self.warmup_iterations < 0 or self.probability_update_interval < 1:
            raise ValueError("Warmup must be non-negative and update interval must be positive.")

        # Preserve the CUDA index (for example cuda:1); a generator on the
        # default CUDA device cannot drive draws on another CUDA device.
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(self.sampler_seed)
        self.motion_probability = uniform_probability(self.motion_eligible_mask)
        motion_count = int(torch.count_nonzero(self.motion_eligible_mask).item())
        if (
            self.segment_mode != "global_bin_raw_error"
            and self.motion_probability_cap * motion_count < 1.0 - 1.0e-12
        ):
            raise ValueError(
                "Configured motion probability cap is infeasible for the eligible motion set."
            )
        conditional_cap = (
            self.segment_probability_cap
            if self.segment_mode in {"raw_error", "relative_learning_gap"}
            else 1.0
        )
        self.segment_probability, self.nonempty_motion_mask, _ = grouped_probability(
            torch.zeros(self.num_segments, dtype=torch.float64, device=self.device),
            torch.ones(self.num_segments, dtype=torch.bool, device=self.device),
            self.segment_eligible_mask,
            torch.zeros(self.num_segments, dtype=torch.long, device=self.device),
            self.segment_motion_ids,
            num_groups=self.num_motions,
            uniform_mix=self.uniform_mix,
            temperature=self.temperature,
            under_sampling_weight=self.under_sampling_weight,
            probability_cap=conditional_cap,
            score_clip=self.score_clip,
            force_uniform=True,
        )
        self.global_segment_probability = uniform_probability(self.segment_eligible_mask)
        if self.segment_mode == "global_bin_raw_error":
            self.global_segment_probability = capped_simplex(
                self.global_segment_probability,
                self.segment_eligible_mask,
                self.segment_probability_cap,
            )
        self.current_iteration = -1
        self.last_probability_update_iteration = -1
        self.probability_update_count = 0
        self.fallback_count = 0
        self.last_fallback_reason = ""
        self._distribution_metrics: dict[str, float | int] = {}
        self._refresh_distribution_metrics()

    @property
    def warmup_active(self) -> bool:
        return self.current_iteration < self.warmup_iterations

    def should_update(self, iteration: int) -> bool:
        if iteration < self.warmup_iterations:
            return False
        if self.last_probability_update_iteration < self.warmup_iterations:
            return True
        return iteration - self.last_probability_update_iteration >= self.probability_update_interval

    def update_probabilities(
        self,
        iteration: int,
        *,
        motion_score: torch.Tensor,
        motion_score_valid: torch.Tensor,
        segment_score: torch.Tensor,
        segment_score_valid: torch.Tensor,
        motion_sample_count: torch.Tensor,
        segment_sample_count: torch.Tensor,
    ) -> bool:
        self.current_iteration = int(iteration)
        if self.current_iteration < self.warmup_iterations:
            return False
        if not self.should_update(self.current_iteration):
            return False

        if self.segment_mode != "global_bin_raw_error":
            motion_result = build_probability(
                motion_score,
                motion_score_valid,
                self.motion_eligible_mask,
                motion_sample_count,
                uniform_mix=self.uniform_mix,
                temperature=self.temperature,
                under_sampling_weight=self.under_sampling_weight,
                probability_cap=self.motion_probability_cap,
                score_clip=self.score_clip,
                force_uniform=self.motion_mode == "uniform",
            )
            self.motion_probability = motion_result.probability.to(device=self.device)
            if motion_result.fallback_used:
                self.fallback_count += 1
                self.last_fallback_reason = motion_result.fallback_reason
        if self.segment_mode == "global_bin_raw_error":
            global_result = build_probability(
                segment_score,
                segment_score_valid,
                self.segment_eligible_mask,
                segment_sample_count,
                uniform_mix=self.uniform_mix,
                temperature=self.temperature,
                under_sampling_weight=self.under_sampling_weight,
                probability_cap=self.segment_probability_cap,
                score_clip=self.score_clip,
            )
            self.global_segment_probability = global_result.probability.to(device=self.device)
            if global_result.fallback_used:
                self.fallback_count += 1
                self.last_fallback_reason = global_result.fallback_reason
        else:
            force_uniform = self.segment_mode == "uniform"
            segment_probability, nonempty, fallback_count = grouped_probability(
                segment_score,
                segment_score_valid,
                self.segment_eligible_mask,
                segment_sample_count,
                self.segment_motion_ids,
                num_groups=self.num_motions,
                uniform_mix=self.uniform_mix,
                temperature=self.temperature,
                under_sampling_weight=self.under_sampling_weight,
                probability_cap=1.0 if force_uniform else self.segment_probability_cap,
                score_clip=self.score_clip,
                force_uniform=force_uniform,
            )
            self.segment_probability = segment_probability
            self.nonempty_motion_mask = nonempty
            self.fallback_count += fallback_count
            if fallback_count:
                self.last_fallback_reason = "no_reliable_segment_score"

        self.last_probability_update_iteration = self.current_iteration
        self.probability_update_count += 1
        self._refresh_distribution_metrics()
        return True

    def _draw_uniform(self, count: int) -> torch.Tensor:
        return torch.rand(count, dtype=torch.float64, device=self.device, generator=self.generator)

    def _sample_segments_conditionally(self, motion_ids: torch.Tensor) -> torch.Tensor:
        group_sums = torch.zeros(self.num_motions, dtype=torch.float64, device=self.device)
        group_sums.scatter_add_(0, self.segment_motion_ids, self.segment_probability)
        group_offsets = torch.cumsum(group_sums, dim=0) - group_sums
        cumulative = torch.cumsum(self.segment_probability, dim=0)
        targets = group_offsets[motion_ids] + self._draw_uniform(motion_ids.numel()) * group_sums[motion_ids]
        segment_ids = torch.searchsorted(cumulative, targets, right=True)
        return torch.minimum(segment_ids, torch.full_like(segment_ids, self.num_segments - 1))

    def _start_frame_in_segment(self, segment_ids: torch.Tensor) -> torch.Tensor:
        counts = self.segment_start_count[segment_ids]
        offsets = (self._draw_uniform(segment_ids.numel()) * counts.to(torch.float64)).to(torch.long)
        return self.segment_start_frames[segment_ids] + torch.minimum(offsets, counts - 1)

    def sample(self, num_samples: int) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """Draw adaptive assignments without advancing any global torch RNG."""

        if num_samples < 0:
            raise ValueError("num_samples must be non-negative.")
        if num_samples == 0:
            empty = torch.empty(0, dtype=torch.long, device=self.device)
            return empty, empty, empty
        if self.segment_mode == "global_bin_raw_error":
            segment_ids = torch.multinomial(
                self.global_segment_probability, num_samples, replacement=True, generator=self.generator
            )
            motion_ids = self.segment_motion_ids[segment_ids]
            return motion_ids, segment_ids, self._start_frame_in_segment(segment_ids)

        motion_ids = torch.multinomial(
            self.motion_probability, num_samples, replacement=True, generator=self.generator
        )
        if self.segment_mode == "uniform":
            start_count = self.motion_lengths[motion_ids] - 1
            start_frames = (self._draw_uniform(num_samples) * start_count.to(torch.float64)).to(torch.long)
            return motion_ids, None, torch.minimum(start_frames, start_count - 1)
        segment_ids = self._sample_segments_conditionally(motion_ids)
        if torch.any(self.segment_motion_ids[segment_ids] != motion_ids):
            raise RuntimeError("Conditional segment sampling crossed a motion boundary.")
        return motion_ids, segment_ids, self._start_frame_in_segment(segment_ids)

    def _refresh_distribution_metrics(self) -> None:
        """Cache full-distribution diagnostics outside the reset hot path."""

        segment_distribution = (
            self.global_segment_probability
            if self.segment_mode == "global_bin_raw_error"
            else self.segment_probability
        )
        segment_nonzero = segment_distribution[segment_distribution > 0.0]
        if self.segment_mode == "global_bin_raw_error":
            motion_distribution = torch.zeros(
                self.num_motions, dtype=torch.float64, device=self.device
            )
            motion_distribution.scatter_add_(
                0, self.segment_motion_ids, self.global_segment_probability
            )
        else:
            motion_distribution = self.motion_probability
        motion_nonzero = motion_distribution[motion_distribution > 0.0]
        motion_entropy = -(motion_nonzero * torch.log(motion_nonzero)).sum()
        conditional_sums = torch.zeros(self.num_motions, dtype=torch.float64, device=self.device)
        if self.segment_mode != "global_bin_raw_error":
            conditional_sums.scatter_add_(0, self.segment_motion_ids, self.segment_probability)
            segment_sum_error = float(torch.max(torch.abs(conditional_sums[self.nonempty_motion_mask] - 1.0)).item())
            entropy_terms = torch.where(
                self.segment_probability > 0.0,
                -self.segment_probability * torch.log(self.segment_probability.clamp_min(1.0e-300)),
                torch.zeros_like(self.segment_probability),
            )
            entropy_by_motion = torch.zeros(
                self.num_motions, dtype=torch.float64, device=self.device
            )
            entropy_by_motion.scatter_add_(0, self.segment_motion_ids, entropy_terms)
            segment_entropy = entropy_by_motion[self.nonempty_motion_mask].mean()
        else:
            segment_sum_error = abs(float(self.global_segment_probability.sum().item()) - 1.0)
            segment_entropy = -(segment_nonzero * torch.log(segment_nonzero)).sum()
        self._distribution_metrics = {
            "motion_entropy": float(motion_entropy.item()),
            "segment_entropy": float(segment_entropy.item()),
            "max_motion_probability": float(motion_distribution.max().item()),
            "max_segment_probability": float(segment_distribution.max().item()),
            "min_nonzero_motion_probability": float(motion_nonzero.min().item()),
            "min_nonzero_segment_probability": float(segment_nonzero.min().item()),
            "effective_motion_count": int(torch.count_nonzero(self.motion_eligible_mask).item()),
            "effective_segment_count": int(torch.count_nonzero(self.segment_eligible_mask).item()),
            "probability_sum_error": max(
                abs(float(motion_distribution.sum().item()) - 1.0), segment_sum_error
            ),
        }

    def metrics(self) -> dict[str, float | int]:
        """Return cached distribution summaries plus current scalar counters."""

        return {
            "mode": {
                ("uniform", "uniform"): 0,
                ("raw_error", "uniform"): 2,
                ("uniform", "raw_error"): 3,
                ("raw_error", "raw_error"): 4,
                ("learning_gap", "relative_learning_gap"): 5,
                ("uniform", "global_bin_raw_error"): 7,
            }.get((self.motion_mode, self.segment_mode), -1),
            "warmup_active": int(self.warmup_active),
            "probability_update_count": self.probability_update_count,
            **self._distribution_metrics,
            "uniform_mix": self.uniform_mix,
            "fallback_count": self.fallback_count,
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": ADAPTIVE_SAMPLER_SCHEMA_VERSION,
            "config_hash": self.config_hash,
            "motion_mode": self.motion_mode,
            "segment_mode": self.segment_mode,
            "num_motions": self.num_motions,
            "num_segments": self.num_segments,
            "current_iteration": self.current_iteration,
            "last_probability_update_iteration": self.last_probability_update_iteration,
            "probability_update_count": self.probability_update_count,
            "fallback_count": self.fallback_count,
            "last_fallback_reason": self.last_fallback_reason,
            "motion_probability": self.motion_probability.detach().clone(),
            "segment_probability": self.segment_probability.detach().clone(),
            "global_segment_probability": self.global_segment_probability.detach().clone(),
            "generator_state": self.generator.get_state().detach().clone(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("schema_version") != ADAPTIVE_SAMPLER_SCHEMA_VERSION:
            raise ValueError("Unsupported adaptive sampler checkpoint schema.")
        identity = {
            "config_hash": self.config_hash,
            "motion_mode": self.motion_mode,
            "segment_mode": self.segment_mode,
            "num_motions": self.num_motions,
            "num_segments": self.num_segments,
        }
        for name, expected in identity.items():
            if state.get(name) != expected:
                raise ValueError(f"Checkpoint adaptive sampler field '{name}' does not match the current run.")
        for name in ("motion_probability", "segment_probability", "global_segment_probability"):
            target = getattr(self, name)
            saved = torch.as_tensor(state.get(name), dtype=torch.float64, device=self.device)
            if saved.shape != target.shape or not torch.all(torch.isfinite(saved)) or torch.any(saved < 0.0):
                raise ValueError(f"Checkpoint adaptive sampler field '{name}' is invalid.")
            target.copy_(saved)
        if abs(float(self.motion_probability.sum().item()) - 1.0) > 1.0e-9:
            raise ValueError("Checkpoint motion probability is not normalized.")
        if torch.any(self.motion_probability[~self.motion_eligible_mask] != 0.0):
            raise ValueError("Checkpoint assigns probability to an ineligible motion.")
        if torch.any(self.segment_probability[~self.segment_eligible_mask] != 0.0):
            raise ValueError("Checkpoint assigns conditional probability to an ineligible segment.")
        conditional_sums = torch.zeros(self.num_motions, dtype=torch.float64, device=self.device)
        conditional_sums.scatter_add_(0, self.segment_motion_ids, self.segment_probability)
        if torch.any(torch.abs(conditional_sums[self.nonempty_motion_mask] - 1.0) > 1.0e-9):
            raise ValueError("Checkpoint conditional segment probabilities are not normalized.")
        if torch.any(self.global_segment_probability[~self.segment_eligible_mask] != 0.0) or abs(
            float(self.global_segment_probability.sum().item()) - 1.0
        ) > 1.0e-9:
            raise ValueError("Checkpoint global segment probability is invalid.")

        motion_count = int(torch.count_nonzero(self.motion_eligible_mask).item())
        if self.segment_mode != "global_bin_raw_error":
            if self.motion_probability_cap * motion_count < 1.0 - 1.0e-12:
                raise ValueError("Configured motion probability cap is infeasible for the eligible set.")
            if torch.any(self.motion_probability > self.motion_probability_cap + 1.0e-9):
                raise ValueError("Checkpoint motion probability exceeds the configured cap.")
        eligible_per_motion = torch.zeros(
            self.num_motions, dtype=torch.long, device=self.device
        )
        eligible_per_motion.scatter_add_(
            0, self.segment_motion_ids, self.segment_eligible_mask.to(torch.long)
        )
        if self.segment_mode in {"raw_error", "relative_learning_gap"}:
            infeasible_segment_cap = (eligible_per_motion > 0) & (
                eligible_per_motion.to(torch.float64) * self.segment_probability_cap
                < 1.0 - 1.0e-12
            )
            if torch.any(infeasible_segment_cap):
                raise ValueError(
                    "Configured conditional segment probability cap is infeasible for an eligible motion."
                )
            if torch.any(
                self.segment_probability > self.segment_probability_cap + 1.0e-9
            ):
                raise ValueError("Checkpoint conditional segment probability exceeds the configured cap.")
        if self.segment_mode == "global_bin_raw_error":
            global_count = int(torch.count_nonzero(self.segment_eligible_mask).item())
            if self.segment_probability_cap * global_count < 1.0 - 1.0e-12:
                raise ValueError("Configured global segment probability cap is infeasible.")
            if torch.any(
                self.global_segment_probability > self.segment_probability_cap + 1.0e-9
            ):
                raise ValueError("Checkpoint global segment probability exceeds the configured cap.")

        current_iteration = int(state.get("current_iteration", -1))
        last_update = int(state.get("last_probability_update_iteration", -1))
        update_count = int(state.get("probability_update_count", 0))
        fallback_count = int(state.get("fallback_count", 0))
        if current_iteration < -1 or last_update < -1 or last_update > current_iteration:
            raise ValueError("Checkpoint adaptive sampler iteration counters are invalid.")
        if min(update_count, fallback_count) < 0:
            raise ValueError("Checkpoint adaptive sampler counters must be non-negative.")
        if (update_count == 0) != (last_update == -1):
            raise ValueError("Checkpoint probability update count and last update disagree.")
        self.current_iteration = current_iteration
        self.last_probability_update_iteration = last_update
        self.probability_update_count = update_count
        self.fallback_count = fallback_count
        self.last_fallback_reason = str(state.get("last_fallback_reason", ""))
        generator_state = torch.as_tensor(state.get("generator_state"), dtype=torch.uint8, device="cpu")
        self.generator.set_state(generator_state)
        self._refresh_distribution_metrics()
