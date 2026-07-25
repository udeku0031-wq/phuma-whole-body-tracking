"""Cluster-constrained hierarchical sampling for M7.

The diversity layer deliberately has no learning-gap input.  Its distribution
is a fixed target computed from the number of eligible motions in each
cluster.  Motion and segment probabilities reuse the M6 probability builders:

    P(c, m, s) = P(c) P(m | c) P(s | m).

All draws use one private :class:`torch.Generator` in cluster, motion, segment,
start-frame order.  No global PyTorch RNG state is consumed.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch

try:  # Package import used at runtime.
    from .adaptive_sampling import build_probability, grouped_probability
except ImportError:  # Direct-file import used by CPU-only unit tests.
    from adaptive_sampling import build_probability, grouped_probability


DIVERSITY_SAMPLER_SCHEMA_VERSION = "wbt.diversity_sampler.v1"


def build_cluster_target_probability(
    eligible_motion_count: torch.Tensor,
    *,
    minimum_budget_fraction_of_uniform: float,
    cluster_size_exponent: float,
) -> torch.Tensor:
    """Return the fixed diversity budget for each cluster.

    For the ``C`` clusters with at least one eligible motion, the minimum
    share is ``minimum_budget_fraction_of_uniform / C``.  The remaining mass
    is distributed in proportion to ``eligible_motion_count ** exponent``.
    Empty clusters receive exactly zero probability.
    """

    count = torch.as_tensor(eligible_motion_count)
    if count.ndim != 1 or count.numel() == 0:
        raise ValueError("eligible_motion_count must be a non-empty one-dimensional tensor.")
    if count.dtype == torch.bool:
        raise ValueError("eligible_motion_count must contain non-negative counts, not booleans.")
    count_float = count.to(dtype=torch.float64)
    if not torch.all(torch.isfinite(count_float)) or torch.any(count_float < 0.0):
        raise ValueError("eligible_motion_count must be finite and non-negative.")
    if not math.isfinite(minimum_budget_fraction_of_uniform) or not (
        0.0 <= minimum_budget_fraction_of_uniform <= 1.0
    ):
        raise ValueError("minimum_budget_fraction_of_uniform must be finite and in [0, 1].")
    if not math.isfinite(cluster_size_exponent) or cluster_size_exponent < 0.0:
        raise ValueError("cluster_size_exponent must be finite and non-negative.")

    eligible = count_float > 0.0
    eligible_cluster_count = int(torch.count_nonzero(eligible).item())
    if eligible_cluster_count == 0:
        raise ValueError("At least one cluster must contain an eligible motion.")

    minimum_share = (
        float(minimum_budget_fraction_of_uniform) / eligible_cluster_count
    )
    size_weight = torch.zeros_like(count_float)
    size_weight[eligible] = torch.pow(
        count_float[eligible], float(cluster_size_exponent)
    )
    weight_sum = size_weight.sum()
    if not torch.isfinite(weight_sum) or float(weight_sum.item()) <= 0.0:
        raise RuntimeError("Cluster size weights are not a valid probability distribution.")

    probability = torch.zeros_like(count_float)
    probability[eligible] = minimum_share + (
        1.0 - eligible_cluster_count * minimum_share
    ) * (size_weight[eligible] / weight_sum)
    # Remove the last few ulps from the analytic expression without ever
    # assigning mass to an empty cluster.
    probability[eligible] /= probability[eligible].sum()
    if (
        not torch.all(torch.isfinite(probability))
        or torch.any(probability < 0.0)
        or torch.any(probability[~eligible] != 0.0)
        or abs(float(probability.sum().item()) - 1.0) > 1.0e-12
        or torch.any(probability[eligible] < minimum_share - 1.0e-12)
    ):
        raise RuntimeError("Cluster target probability failed numerical validation.")
    return probability


# Public aliases make the formula easy to use without depending on one naming
# convention in configuration/loading code.
cluster_target_probability = build_cluster_target_probability
compute_cluster_target_probability = build_cluster_target_probability


def _as_1d(
    value: torch.Tensor,
    *,
    dtype: torch.dtype,
    device: torch.device,
    name: str,
) -> torch.Tensor:
    result = torch.as_tensor(value, dtype=dtype, device=device)
    if result.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    return result


def _entropy(probability: torch.Tensor) -> torch.Tensor:
    positive = probability > 0.0
    return -torch.sum(
        torch.where(
            positive,
            probability * torch.log(probability.clamp_min(1.0e-300)),
            torch.zeros_like(probability),
        )
    )


def _gini(nonnegative: torch.Tensor) -> float:
    values = torch.as_tensor(nonnegative, dtype=torch.float64)
    if values.numel() == 0 or float(values.sum().item()) == 0.0:
        return 0.0
    values = torch.sort(values).values
    n = values.numel()
    ranks = torch.arange(1, n + 1, dtype=torch.float64, device=values.device)
    result = (
        2.0 * torch.sum(ranks * values) / (n * values.sum())
        - (n + 1.0) / n
    )
    return float(result.item())


class DiversityConstrainedSampler:
    """M6 motion/segment adaptation with a fixed cluster budget in front.

    ``motion_probability`` is the global motion marginal ``P(c) P(m|c)``.
    The within-cluster distribution is available as
    ``motion_probability_conditional`` and ``segment_probability`` remains the
    M6 conditional distribution ``P(s|m)``.
    """

    def __init__(
        self,
        segment_motion_ids: torch.Tensor,
        segment_start_frames: torch.Tensor,
        segment_end_frames: torch.Tensor,
        motion_lengths: torch.Tensor,
        *,
        motion_cluster_ids: torch.Tensor,
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
        num_clusters: int | None = None,
        minimum_budget_fraction_of_uniform: float = 0.5,
        cluster_size_exponent: float = 0.5,
        budget_mode: str = "sqrt_size_with_floor",
        cluster_metadata_hash: str = "",
        cluster_profile_sha256: str = "",
        cluster_schema_version: str = "",
        device: str | torch.device = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.segment_motion_ids = _as_1d(
            segment_motion_ids,
            dtype=torch.long,
            device=self.device,
            name="segment_motion_ids",
        )
        self.segment_start_frames = _as_1d(
            segment_start_frames,
            dtype=torch.long,
            device=self.device,
            name="segment_start_frames",
        )
        self.segment_end_frames = _as_1d(
            segment_end_frames,
            dtype=torch.long,
            device=self.device,
            name="segment_end_frames",
        )
        self.motion_lengths = _as_1d(
            motion_lengths,
            dtype=torch.long,
            device=self.device,
            name="motion_lengths",
        )
        self.motion_cluster_ids = _as_1d(
            motion_cluster_ids,
            dtype=torch.long,
            device=self.device,
            name="motion_cluster_ids",
        )
        self.num_segments = int(self.segment_motion_ids.numel())
        self.num_motions = int(self.motion_lengths.numel())
        if self.num_motions == 0 or self.num_segments == 0:
            raise ValueError("Diversity sampling requires at least one motion and segment.")
        if any(
            item.shape != self.segment_motion_ids.shape
            for item in (self.segment_start_frames, self.segment_end_frames)
        ):
            raise ValueError("Segment layout tensors must have the same shape.")
        if self.motion_cluster_ids.shape != self.motion_lengths.shape:
            raise ValueError("motion_cluster_ids must contain one ID for every motion.")
        if torch.any(self.motion_lengths < 2):
            raise ValueError("Every motion must contain at least two frames.")
        if torch.any(self.segment_motion_ids < 0) or torch.any(
            self.segment_motion_ids >= self.num_motions
        ):
            raise ValueError("segment_motion_ids are outside the motion range.")
        if torch.any(self.motion_cluster_ids < 0):
            raise ValueError("motion_cluster_ids must be non-negative.")

        inferred_clusters = int(self.motion_cluster_ids.max().item()) + 1
        self.num_clusters = (
            inferred_clusters if num_clusters is None else int(num_clusters)
        )
        if self.num_clusters < inferred_clusters or self.num_clusters < 1:
            raise ValueError("num_clusters does not cover every motion cluster ID.")

        requested_motion_eligible = _as_1d(
            motion_eligible_mask,
            dtype=torch.bool,
            device=self.device,
            name="motion_eligible_mask",
        )
        self.segment_eligible_mask = _as_1d(
            segment_eligible_mask,
            dtype=torch.bool,
            device=self.device,
            name="segment_eligible_mask",
        )
        if requested_motion_eligible.shape != self.motion_lengths.shape:
            raise ValueError("motion_eligible_mask does not match the motion layout.")
        if self.segment_eligible_mask.shape != self.segment_motion_ids.shape:
            raise ValueError("segment_eligible_mask does not match the segment layout.")
        if torch.any(
            self.segment_eligible_mask
            & ~requested_motion_eligible[self.segment_motion_ids]
        ):
            raise ValueError("An eligible segment belongs to an ineligible motion.")

        eligible_segment_count = torch.zeros(
            self.num_motions, dtype=torch.long, device=self.device
        )
        eligible_segment_count.scatter_add_(
            0,
            self.segment_motion_ids,
            self.segment_eligible_mask.to(torch.long),
        )
        # Empty motions are runtime-excluded even if an upstream mask forgot to
        # do so.  This guarantees that every sampled motion has a legal segment.
        self.motion_eligible_mask = requested_motion_eligible & (
            eligible_segment_count > 0
        )
        self.empty_motion_mask = requested_motion_eligible & (
            eligible_segment_count == 0
        )
        if not torch.any(self.motion_eligible_mask) or not torch.any(
            self.segment_eligible_mask
        ):
            raise ValueError(
                "Diversity sampling requires at least one eligible motion and segment."
            )

        legal_end = torch.minimum(
            self.segment_end_frames,
            self.motion_lengths[self.segment_motion_ids] - 1,
        )
        self.segment_start_count = torch.clamp(
            legal_end - self.segment_start_frames, min=0
        )
        if torch.any(
            self.segment_eligible_mask & (self.segment_start_count <= 0)
        ):
            raise ValueError(
                "Every eligible segment must contain a legal legacy start frame."
            )

        self.motion_mode = str(motion_mode)
        self.segment_mode = str(segment_mode)
        if self.segment_mode == "global_bin_raw_error":
            raise ValueError(
                "M7 requires P(s|m); global-bin segment sampling is not compatible."
            )
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
        self.minimum_budget_fraction_of_uniform = float(
            minimum_budget_fraction_of_uniform
        )
        self.cluster_size_exponent = float(cluster_size_exponent)
        self.budget_mode = str(budget_mode)
        if self.budget_mode != "sqrt_size_with_floor":
            raise ValueError(
                "Only budget_mode='sqrt_size_with_floor' is implemented."
            )
        self.cluster_metadata_hash = str(cluster_metadata_hash)
        self.cluster_profile_sha256 = str(cluster_profile_sha256)
        self.cluster_schema_version = str(cluster_schema_version)
        if self.warmup_iterations < 0 or self.probability_update_interval < 1:
            raise ValueError(
                "Warmup must be non-negative and update interval must be positive."
            )
        if not math.isfinite(self.motion_probability_cap) or not (
            0.0 < self.motion_probability_cap <= 1.0
        ):
            raise ValueError("motion_probability_cap must be finite and in (0, 1].")
        if not math.isfinite(self.segment_probability_cap) or not (
            0.0 < self.segment_probability_cap <= 1.0
        ):
            raise ValueError("segment_probability_cap must be finite and in (0, 1].")

        self.eligible_motion_count_per_cluster = torch.zeros(
            self.num_clusters, dtype=torch.long, device=self.device
        )
        self.eligible_motion_count_per_cluster.scatter_add_(
            0,
            self.motion_cluster_ids,
            self.motion_eligible_mask.to(torch.long),
        )
        self.eligible_cluster_mask = (
            self.eligible_motion_count_per_cluster > 0
        )
        self.cluster_probability = build_cluster_target_probability(
            self.eligible_motion_count_per_cluster,
            minimum_budget_fraction_of_uniform=self.minimum_budget_fraction_of_uniform,
            cluster_size_exponent=self.cluster_size_exponent,
        )

        # A cap below 1/n is infeasible.  Relax only the affected cluster, not
        # every cluster, so large clusters retain the configured cap.
        reciprocal_size = torch.where(
            self.eligible_cluster_mask,
            1.0
            / self.eligible_motion_count_per_cluster.clamp_min(1).to(
                torch.float64
            ),
            torch.zeros(self.num_clusters, dtype=torch.float64, device=self.device),
        )
        self.effective_motion_cap_by_cluster = torch.where(
            self.eligible_cluster_mask,
            torch.maximum(
                torch.full_like(reciprocal_size, self.motion_probability_cap),
                reciprocal_size,
            ),
            torch.zeros_like(reciprocal_size),
        )
        self.conditional_motion_cap_relax_count = int(
            torch.count_nonzero(
                self.eligible_cluster_mask
                & (
                    self.effective_motion_cap_by_cluster
                    > self.motion_probability_cap + 1.0e-12
                )
            ).item()
        )

        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(self.sampler_seed)
        self.motion_probability_conditional = (
            self.motion_eligible_mask.to(torch.float64)
            / self.eligible_motion_count_per_cluster[
                self.motion_cluster_ids
            ].clamp_min(1).to(torch.float64)
        )
        self.motion_probability = (
            self.cluster_probability[self.motion_cluster_ids]
            * self.motion_probability_conditional
        )
        self.segment_probability, self.nonempty_motion_mask, _ = (
            grouped_probability(
                torch.zeros(
                    self.num_segments, dtype=torch.float64, device=self.device
                ),
                torch.ones(
                    self.num_segments, dtype=torch.bool, device=self.device
                ),
                self.segment_eligible_mask,
                torch.zeros(
                    self.num_segments, dtype=torch.long, device=self.device
                ),
                self.segment_motion_ids,
                num_groups=self.num_motions,
                uniform_mix=self.uniform_mix,
                temperature=self.temperature,
                under_sampling_weight=self.under_sampling_weight,
                probability_cap=1.0,
                score_clip=self.score_clip,
                force_uniform=True,
            )
        )

        # Composite keys make grouping deterministic even on torch versions
        # without stable argsort.  Within a group, item ID remains ascending.
        motion_ids = torch.arange(
            self.num_motions, dtype=torch.long, device=self.device
        )
        self._motion_group_order = torch.argsort(
            self.motion_cluster_ids * (self.num_motions + 1) + motion_ids
        )
        segment_ids = torch.arange(
            self.num_segments, dtype=torch.long, device=self.device
        )
        self._segment_group_order = torch.argsort(
            self.segment_motion_ids * (self.num_segments + 1) + segment_ids
        )

        self.current_iteration = -1
        self.last_probability_update_iteration = -1
        self.probability_update_count = 0
        self.fallback_count = 0
        self.cluster_fallback_count = 0
        self.last_fallback_reason = ""
        self.cluster_sample_count = torch.zeros(
            self.num_clusters, dtype=torch.long, device=self.device
        )
        self._distribution_metrics: dict[str, float | int] = {}
        self._refresh_distribution_metrics()

    @property
    def cluster_target_probability(self) -> torch.Tensor:
        return self.cluster_probability

    @property
    def cluster_eligible_mask(self) -> torch.Tensor:
        return self.eligible_cluster_mask

    @property
    def motion_probability_global(self) -> torch.Tensor:
        return self.motion_probability

    @property
    def segment_probability_conditional(self) -> torch.Tensor:
        return self.segment_probability

    @property
    def segment_probability_global(self) -> torch.Tensor:
        return (
            self.motion_probability[self.segment_motion_ids]
            * self.segment_probability
        )

    @property
    def observed_cluster_share(self) -> torch.Tensor:
        total = self.cluster_sample_count.sum()
        if int(total.item()) == 0:
            return torch.zeros(
                self.num_clusters, dtype=torch.float64, device=self.device
            )
        return self.cluster_sample_count.to(torch.float64) / total.to(torch.float64)

    @property
    def budget_deficit(self) -> torch.Tensor:
        return self.cluster_probability - self.observed_cluster_share

    @property
    def warmup_active(self) -> bool:
        return self.current_iteration < self.warmup_iterations

    def should_update(self, iteration: int) -> bool:
        iteration = int(iteration)
        if iteration < self.warmup_iterations:
            return False
        if self.last_probability_update_iteration < self.warmup_iterations:
            return True
        return (
            iteration - self.last_probability_update_iteration
            >= self.probability_update_interval
        )

    def _motion_inputs(
        self,
        score: torch.Tensor,
        valid: torch.Tensor,
        count: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        score = _as_1d(
            score, dtype=torch.float64, device=self.device, name="motion_score"
        )
        valid = _as_1d(
            valid,
            dtype=torch.bool,
            device=self.device,
            name="motion_score_valid",
        )
        count = _as_1d(
            count,
            dtype=torch.float64,
            device=self.device,
            name="motion_sample_count",
        )
        if any(item.shape != self.motion_lengths.shape for item in (score, valid, count)):
            raise ValueError("Motion update tensors do not match the motion layout.")
        return score, valid, count

    def _segment_inputs(
        self,
        score: torch.Tensor,
        valid: torch.Tensor,
        count: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        score = _as_1d(
            score, dtype=torch.float64, device=self.device, name="segment_score"
        )
        valid = _as_1d(
            valid,
            dtype=torch.bool,
            device=self.device,
            name="segment_score_valid",
        )
        count = _as_1d(
            count,
            dtype=torch.float64,
            device=self.device,
            name="segment_sample_count",
        )
        if any(
            item.shape != self.segment_motion_ids.shape
            for item in (score, valid, count)
        ):
            raise ValueError("Segment update tensors do not match the segment layout.")
        return score, valid, count

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
        """Refresh ``P(m|c)`` and ``P(s|m)``; ``P(c)`` stays fixed."""

        self.current_iteration = int(iteration)
        if self.current_iteration < self.warmup_iterations:
            return False
        if not self.should_update(self.current_iteration):
            return False

        motion_score, motion_score_valid, motion_sample_count = (
            self._motion_inputs(
                motion_score, motion_score_valid, motion_sample_count
            )
        )
        segment_score, segment_score_valid, segment_sample_count = (
            self._segment_inputs(
                segment_score, segment_score_valid, segment_sample_count
            )
        )

        conditional = torch.zeros(
            self.num_motions, dtype=torch.float64, device=self.device
        )
        motion_fallback_count = 0
        last_motion_fallback = ""
        for cluster_id in torch.where(self.eligible_cluster_mask)[0].tolist():
            in_cluster = (
                self.motion_cluster_ids == cluster_id
            ) & self.motion_eligible_mask
            result = build_probability(
                motion_score[in_cluster],
                motion_score_valid[in_cluster],
                torch.ones(
                    int(torch.count_nonzero(in_cluster).item()),
                    dtype=torch.bool,
                    device=self.device,
                ),
                motion_sample_count[in_cluster],
                uniform_mix=self.uniform_mix,
                temperature=self.temperature,
                under_sampling_weight=self.under_sampling_weight,
                probability_cap=float(
                    self.effective_motion_cap_by_cluster[cluster_id].item()
                ),
                score_clip=self.score_clip,
                force_uniform=self.motion_mode == "uniform",
            )
            conditional[in_cluster] = result.probability
            if result.fallback_used:
                motion_fallback_count += 1
                last_motion_fallback = result.fallback_reason
        self.motion_probability_conditional = conditional
        self.motion_probability = (
            self.cluster_probability[self.motion_cluster_ids] * conditional
        )

        force_uniform = self.segment_mode == "uniform"
        segment_probability, nonempty, segment_fallback_count = (
            grouped_probability(
                segment_score,
                segment_score_valid,
                self.segment_eligible_mask,
                segment_sample_count,
                self.segment_motion_ids,
                num_groups=self.num_motions,
                uniform_mix=self.uniform_mix,
                temperature=self.temperature,
                under_sampling_weight=self.under_sampling_weight,
                probability_cap=(
                    1.0 if force_uniform else self.segment_probability_cap
                ),
                score_clip=self.score_clip,
                force_uniform=force_uniform,
            )
        )
        self.segment_probability = segment_probability
        self.nonempty_motion_mask = nonempty
        self.fallback_count += motion_fallback_count + segment_fallback_count
        if segment_fallback_count:
            self.last_fallback_reason = "no_reliable_segment_score"
        elif motion_fallback_count:
            self.last_fallback_reason = last_motion_fallback

        self.last_probability_update_iteration = self.current_iteration
        self.probability_update_count += 1
        self._validate_probability_state()
        self._refresh_distribution_metrics()
        return True

    def _draw_uniform(self, count: int) -> torch.Tensor:
        return torch.rand(
            count,
            dtype=torch.float64,
            device=self.device,
            generator=self.generator,
        )

    def _sample_grouped(
        self,
        selected_group_ids: torch.Tensor,
        *,
        item_group_ids: torch.Tensor,
        item_probability: torch.Tensor,
        group_order: torch.Tensor,
        num_groups: int,
    ) -> torch.Tensor:
        """Sample conditionally without assuming groups are contiguous."""

        group_sum = torch.zeros(
            num_groups, dtype=torch.float64, device=self.device
        )
        group_sum.scatter_add_(0, item_group_ids, item_probability)
        selected_sum = group_sum[selected_group_ids]
        if torch.any(selected_sum <= 0.0):
            raise RuntimeError("A sampled group has no conditional probability mass.")

        ordered_probability = item_probability[group_order]
        cumulative = torch.cumsum(ordered_probability, dim=0)
        group_offset = torch.cumsum(group_sum, dim=0) - group_sum
        targets = (
            group_offset[selected_group_ids]
            + self._draw_uniform(selected_group_ids.numel()) * selected_sum
        )
        packed_ids = torch.searchsorted(cumulative, targets, right=True)
        packed_ids = torch.minimum(
            packed_ids,
            torch.full_like(packed_ids, group_order.numel() - 1),
        )
        item_ids = group_order[packed_ids]
        if torch.any(item_group_ids[item_ids] != selected_group_ids):
            raise RuntimeError("Conditional sampling crossed a group boundary.")
        return item_ids

    def _start_frame_in_segment(self, segment_ids: torch.Tensor) -> torch.Tensor:
        counts = self.segment_start_count[segment_ids]
        offsets = (
            self._draw_uniform(segment_ids.numel()) * counts.to(torch.float64)
        ).to(torch.long)
        return self.segment_start_frames[segment_ids] + torch.minimum(
            offsets, counts - 1
        )

    def sample(
        self, num_samples: int
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """Draw M7 assignments and update observed cluster counts."""

        num_samples = int(num_samples)
        if num_samples < 0:
            raise ValueError("num_samples must be non-negative.")
        if num_samples == 0:
            empty = torch.empty(0, dtype=torch.long, device=self.device)
            return empty, empty, empty

        cluster_ids = torch.multinomial(
            self.cluster_probability,
            num_samples,
            replacement=True,
            generator=self.generator,
        )
        motion_ids = self._sample_grouped(
            cluster_ids,
            item_group_ids=self.motion_cluster_ids,
            item_probability=self.motion_probability_conditional,
            group_order=self._motion_group_order,
            num_groups=self.num_clusters,
        )
        if torch.any(self.motion_cluster_ids[motion_ids] != cluster_ids):
            raise RuntimeError("Conditional motion sampling crossed a cluster boundary.")

        self.cluster_sample_count += torch.bincount(
            cluster_ids, minlength=self.num_clusters
        )
        segment_ids = self._sample_grouped(
            motion_ids,
            item_group_ids=self.segment_motion_ids,
            item_probability=self.segment_probability,
            group_order=self._segment_group_order,
            num_groups=self.num_motions,
        )
        if torch.any(self.segment_motion_ids[segment_ids] != motion_ids):
            raise RuntimeError("Conditional segment sampling crossed a motion boundary.")
        return motion_ids, segment_ids, self._start_frame_in_segment(segment_ids)

    def _validate_probability_state(self) -> None:
        tolerance = 1.0e-9
        if (
            not torch.all(torch.isfinite(self.cluster_probability))
            or torch.any(self.cluster_probability < 0.0)
            or abs(float(self.cluster_probability.sum().item()) - 1.0)
            > tolerance
            or torch.any(
                self.cluster_probability[~self.eligible_cluster_mask] != 0.0
            )
        ):
            raise ValueError("Cluster probability is invalid.")

        conditional_sum = torch.zeros(
            self.num_clusters, dtype=torch.float64, device=self.device
        )
        conditional_sum.scatter_add_(
            0,
            self.motion_cluster_ids,
            self.motion_probability_conditional,
        )
        if (
            not torch.all(torch.isfinite(self.motion_probability_conditional))
            or torch.any(self.motion_probability_conditional < 0.0)
            or torch.any(
                self.motion_probability_conditional[
                    ~self.motion_eligible_mask
                ]
                != 0.0
            )
            or torch.any(
                torch.abs(
                    conditional_sum[self.eligible_cluster_mask] - 1.0
                )
                > tolerance
            )
            or torch.any(
                self.motion_probability_conditional
                > self.effective_motion_cap_by_cluster[
                    self.motion_cluster_ids
                ]
                + tolerance
            )
        ):
            raise ValueError("Conditional motion probability is invalid.")
        expected_motion = (
            self.cluster_probability[self.motion_cluster_ids]
            * self.motion_probability_conditional
        )
        if (
            not torch.allclose(
                self.motion_probability,
                expected_motion,
                atol=tolerance,
                rtol=0.0,
            )
            or abs(float(self.motion_probability.sum().item()) - 1.0)
            > tolerance
        ):
            raise ValueError("Global motion probability does not factorize.")

        segment_sum = torch.zeros(
            self.num_motions, dtype=torch.float64, device=self.device
        )
        segment_sum.scatter_add_(
            0, self.segment_motion_ids, self.segment_probability
        )
        if (
            not torch.all(torch.isfinite(self.segment_probability))
            or torch.any(self.segment_probability < 0.0)
            or torch.any(
                self.segment_probability[~self.segment_eligible_mask] != 0.0
            )
            or torch.any(
                torch.abs(
                    segment_sum[self.nonempty_motion_mask] - 1.0
                )
                > tolerance
            )
        ):
            raise ValueError("Conditional segment probability is invalid.")
        if self.segment_mode != "uniform" and torch.any(
            self.segment_probability > self.segment_probability_cap + tolerance
        ):
            raise ValueError(
                "Conditional segment probability exceeds the configured cap."
            )

    def _refresh_distribution_metrics(self) -> None:
        cluster_entropy = _entropy(self.cluster_probability)
        conditional_entropy_terms = torch.where(
            self.motion_probability_conditional > 0.0,
            -self.motion_probability_conditional
            * torch.log(
                self.motion_probability_conditional.clamp_min(1.0e-300)
            ),
            torch.zeros_like(self.motion_probability_conditional),
        )
        entropy_by_cluster = torch.zeros(
            self.num_clusters, dtype=torch.float64, device=self.device
        )
        entropy_by_cluster.scatter_add_(
            0, self.motion_cluster_ids, conditional_entropy_terms
        )
        eligible_entropy = entropy_by_cluster[self.eligible_cluster_mask]

        segment_entropy_terms = torch.where(
            self.segment_probability > 0.0,
            -self.segment_probability
            * torch.log(self.segment_probability.clamp_min(1.0e-300)),
            torch.zeros_like(self.segment_probability),
        )
        segment_entropy_by_motion = torch.zeros(
            self.num_motions, dtype=torch.float64, device=self.device
        )
        segment_entropy_by_motion.scatter_add_(
            0, self.segment_motion_ids, segment_entropy_terms
        )
        eligible_segment_entropy = segment_entropy_by_motion[
            self.nonempty_motion_mask
        ]
        positive_motion = self.motion_probability[self.motion_probability > 0.0]
        positive_segment = self.segment_probability[
            self.segment_probability > 0.0
        ]
        eligible_sizes = self.eligible_motion_count_per_cluster[
            self.eligible_cluster_mask
        ]
        self._distribution_metrics = {
            "cluster_entropy": float(cluster_entropy.item()),
            "cluster_entropy_normalized": (
                float(cluster_entropy.item())
                / math.log(int(torch.count_nonzero(self.eligible_cluster_mask).item()))
                if int(torch.count_nonzero(self.eligible_cluster_mask).item()) > 1
                else 1.0
            ),
            "mean_conditional_motion_entropy": float(
                eligible_entropy.mean().item()
            ),
            "min_conditional_motion_entropy": float(
                eligible_entropy.min().item()
            ),
            "max_conditional_motion_probability": float(
                self.motion_probability_conditional.max().item()
            ),
            "motions_above_0_1_probability": int(
                torch.count_nonzero(
                    self.motion_probability_conditional > 0.1
                ).item()
            ),
            "motions_above_0_25_probability": int(
                torch.count_nonzero(
                    self.motion_probability_conditional > 0.25
                ).item()
            ),
            "motions_above_0_5_probability": int(
                torch.count_nonzero(
                    self.motion_probability_conditional > 0.5
                ).item()
            ),
            "motion_entropy": float(_entropy(self.motion_probability).item()),
            "segment_entropy": float(
                eligible_segment_entropy.mean().item()
            ),
            "max_motion_probability": float(
                self.motion_probability.max().item()
            ),
            "max_segment_probability": float(
                self.segment_probability.max().item()
            ),
            "min_nonzero_motion_probability": float(
                positive_motion.min().item()
            ),
            "min_nonzero_segment_probability": float(
                positive_segment.min().item()
            ),
            "effective_motion_count": int(
                torch.count_nonzero(self.motion_eligible_mask).item()
            ),
            "effective_segment_count": int(
                torch.count_nonzero(self.segment_eligible_mask).item()
            ),
            "eligible_cluster_count": int(
                torch.count_nonzero(self.eligible_cluster_mask).item()
            ),
            "empty_cluster_count": int(
                torch.count_nonzero(~self.eligible_cluster_mask).item()
            ),
            "min_cluster_size": int(eligible_sizes.min().item()),
            "max_cluster_size": int(eligible_sizes.max().item()),
            "cluster_size_gini": _gini(eligible_sizes),
            "cluster_probability_sum_error": abs(
                float(self.cluster_probability.sum().item()) - 1.0
            ),
            "conditional_motion_cap_relax_count": (
                self.conditional_motion_cap_relax_count
            ),
        }

    def metrics(self) -> dict[str, float | int]:
        observed = self.observed_cluster_share
        total = int(self.cluster_sample_count.sum().item())
        sampled_eligible = self.eligible_cluster_mask & (
            self.cluster_sample_count > 0
        )
        eligible_count = int(
            torch.count_nonzero(self.eligible_cluster_mask).item()
        )
        positive_observed = observed[observed > 0.0]
        if total:
            midpoint = 0.5 * (self.cluster_probability + observed)
            target_positive = self.cluster_probability > 0.0
            observed_positive = observed > 0.0
            js = 0.5 * torch.sum(
                torch.where(
                    target_positive,
                    self.cluster_probability
                    * torch.log(
                        self.cluster_probability.clamp_min(1.0e-300)
                        / midpoint.clamp_min(1.0e-300)
                    ),
                    torch.zeros_like(midpoint),
                )
            ) + 0.5 * torch.sum(
                torch.where(
                    observed_positive,
                    observed
                    * torch.log(
                        observed.clamp_min(1.0e-300)
                        / midpoint.clamp_min(1.0e-300)
                    ),
                    torch.zeros_like(midpoint),
                )
            )
            js_value = float(js.item())
            max_fraction = float(observed.max().item())
            min_fraction = float(positive_observed.min().item())
        else:
            js_value = 0.0
            max_fraction = 0.0
            min_fraction = 0.0

        result: dict[str, float | int] = {
            "mode": 7,
            "warmup_active": int(self.warmup_active),
            "probability_update_count": self.probability_update_count,
            **self._distribution_metrics,
            "cluster_coverage": (
                float(torch.count_nonzero(sampled_eligible).item())
                / eligible_count
            ),
            "max_cluster_sample_fraction": max_fraction,
            "min_nonzero_cluster_sample_fraction": min_fraction,
            "target_observed_l1": float(
                torch.sum(torch.abs(self.cluster_probability - observed)).item()
            ),
            "target_observed_js_divergence": js_value,
            "uniform_mix": self.uniform_mix,
            "fallback_count": self.fallback_count,
            "cluster_fallback_count": self.cluster_fallback_count,
            "cluster_sample_total": total,
        }
        deficit = self.cluster_probability - observed
        for cluster_id in range(self.num_clusters):
            result[f"cluster_{cluster_id}_target_share"] = float(
                self.cluster_probability[cluster_id].item()
            )
            result[f"cluster_{cluster_id}_observed_share"] = float(
                observed[cluster_id].item()
            )
            result[f"cluster_{cluster_id}_sample_count"] = int(
                self.cluster_sample_count[cluster_id].item()
            )
            result[f"cluster_{cluster_id}_budget_deficit"] = float(
                deficit[cluster_id].item()
            )
        return result

    def state_dict(self) -> dict[str, Any]:
        observed = self.observed_cluster_share
        return {
            "schema_version": DIVERSITY_SAMPLER_SCHEMA_VERSION,
            "config_hash": self.config_hash,
            "cluster_metadata_hash": self.cluster_metadata_hash,
            "cluster_profile_sha256": self.cluster_profile_sha256,
            "cluster_schema_version": self.cluster_schema_version,
            "budget_mode": self.budget_mode,
            "minimum_budget_fraction_of_uniform": (
                self.minimum_budget_fraction_of_uniform
            ),
            "cluster_size_exponent": self.cluster_size_exponent,
            "motion_mode": self.motion_mode,
            "segment_mode": self.segment_mode,
            "num_clusters": self.num_clusters,
            "num_motions": self.num_motions,
            "num_segments": self.num_segments,
            "motion_cluster_ids": self.motion_cluster_ids.detach().clone(),
            "motion_lengths": self.motion_lengths.detach().clone(),
            "segment_motion_ids": self.segment_motion_ids.detach().clone(),
            "segment_start_frames": self.segment_start_frames.detach().clone(),
            "segment_end_frames": self.segment_end_frames.detach().clone(),
            "motion_eligible_mask": self.motion_eligible_mask.detach().clone(),
            "segment_eligible_mask": self.segment_eligible_mask.detach().clone(),
            "eligible_cluster_mask": self.eligible_cluster_mask.detach().clone(),
            "eligible_motion_count_per_cluster": (
                self.eligible_motion_count_per_cluster.detach().clone()
            ),
            "effective_motion_cap_by_cluster": (
                self.effective_motion_cap_by_cluster.detach().clone()
            ),
            "cluster_target_probability": (
                self.cluster_probability.detach().clone()
            ),
            "cluster_probability": self.cluster_probability.detach().clone(),
            "motion_probability_conditional": (
                self.motion_probability_conditional.detach().clone()
            ),
            "motion_probability": self.motion_probability.detach().clone(),
            "segment_probability": self.segment_probability.detach().clone(),
            "cluster_sample_count": self.cluster_sample_count.detach().clone(),
            "observed_cluster_share": observed.detach().clone(),
            "budget_deficit": (
                self.cluster_probability - observed
            ).detach().clone(),
            "current_iteration": self.current_iteration,
            "last_probability_update_iteration": (
                self.last_probability_update_iteration
            ),
            "probability_update_count": self.probability_update_count,
            "fallback_count": self.fallback_count,
            "cluster_fallback_count": self.cluster_fallback_count,
            "last_fallback_reason": self.last_fallback_reason,
            "generator_state": self.generator.get_state().detach().clone(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("schema_version") != DIVERSITY_SAMPLER_SCHEMA_VERSION:
            raise ValueError("Unsupported diversity sampler checkpoint schema.")
        scalar_identity = {
            "config_hash": self.config_hash,
            "cluster_metadata_hash": self.cluster_metadata_hash,
            "cluster_profile_sha256": self.cluster_profile_sha256,
            "cluster_schema_version": self.cluster_schema_version,
            "budget_mode": self.budget_mode,
            "minimum_budget_fraction_of_uniform": (
                self.minimum_budget_fraction_of_uniform
            ),
            "cluster_size_exponent": self.cluster_size_exponent,
            "motion_mode": self.motion_mode,
            "segment_mode": self.segment_mode,
            "num_clusters": self.num_clusters,
            "num_motions": self.num_motions,
            "num_segments": self.num_segments,
        }
        for name, expected in scalar_identity.items():
            if state.get(name) != expected:
                raise ValueError(
                    f"Checkpoint diversity sampler field '{name}' "
                    "does not match the current run."
                )
        tensor_identity = {
            "motion_cluster_ids": self.motion_cluster_ids,
            "motion_lengths": self.motion_lengths,
            "segment_motion_ids": self.segment_motion_ids,
            "segment_start_frames": self.segment_start_frames,
            "segment_end_frames": self.segment_end_frames,
            "motion_eligible_mask": self.motion_eligible_mask,
            "segment_eligible_mask": self.segment_eligible_mask,
            "eligible_cluster_mask": self.eligible_cluster_mask,
            "eligible_motion_count_per_cluster": (
                self.eligible_motion_count_per_cluster
            ),
            "effective_motion_cap_by_cluster": (
                self.effective_motion_cap_by_cluster
            ),
        }
        for name, expected in tensor_identity.items():
            saved = torch.as_tensor(
                state.get(name), dtype=expected.dtype, device=self.device
            )
            if saved.shape != expected.shape or not torch.equal(saved, expected):
                raise ValueError(
                    f"Checkpoint diversity sampler field '{name}' "
                    "does not match the current run."
                )

        probability_fields = {
            "cluster_probability": self.cluster_probability,
            "motion_probability_conditional": (
                self.motion_probability_conditional
            ),
            "motion_probability": self.motion_probability,
            "segment_probability": self.segment_probability,
        }
        loaded_probability: dict[str, torch.Tensor] = {}
        for name, target in probability_fields.items():
            saved = torch.as_tensor(
                state.get(name), dtype=torch.float64, device=self.device
            )
            if (
                saved.shape != target.shape
                or not torch.all(torch.isfinite(saved))
                or torch.any(saved < 0.0)
            ):
                raise ValueError(
                    f"Checkpoint diversity sampler field '{name}' is invalid."
                )
            loaded_probability[name] = saved
        saved_target = torch.as_tensor(
            state.get("cluster_target_probability"),
            dtype=torch.float64,
            device=self.device,
        )
        if (
            saved_target.shape != self.cluster_probability.shape
            or not torch.allclose(
                saved_target,
                self.cluster_probability,
                atol=1.0e-12,
                rtol=0.0,
            )
            or not torch.allclose(
                loaded_probability["cluster_probability"],
                self.cluster_probability,
                atol=1.0e-12,
                rtol=0.0,
            )
        ):
            raise ValueError(
                "Checkpoint cluster target does not match the current eligible set."
            )

        self.motion_probability_conditional.copy_(
            loaded_probability["motion_probability_conditional"]
        )
        self.motion_probability.copy_(loaded_probability["motion_probability"])
        self.segment_probability.copy_(
            loaded_probability["segment_probability"]
        )
        self._validate_probability_state()

        saved_count = torch.as_tensor(
            state.get("cluster_sample_count"),
            dtype=torch.long,
            device=self.device,
        )
        if saved_count.shape != self.cluster_sample_count.shape or torch.any(
            saved_count < 0
        ):
            raise ValueError(
                "Checkpoint cluster_sample_count must be non-negative and correctly shaped."
            )
        observed = torch.as_tensor(
            state.get("observed_cluster_share"),
            dtype=torch.float64,
            device=self.device,
        )
        deficit = torch.as_tensor(
            state.get("budget_deficit"),
            dtype=torch.float64,
            device=self.device,
        )
        saved_total = saved_count.sum()
        expected_observed = (
            saved_count.to(torch.float64) / saved_total.to(torch.float64)
            if int(saved_total.item())
            else torch.zeros_like(self.cluster_probability)
        )
        if (
            observed.shape != expected_observed.shape
            or deficit.shape != expected_observed.shape
            or not torch.allclose(
                observed, expected_observed, atol=1.0e-12, rtol=0.0
            )
            or not torch.allclose(
                deficit,
                self.cluster_probability - expected_observed,
                atol=1.0e-12,
                rtol=0.0,
            )
        ):
            raise ValueError(
                "Checkpoint observed cluster budget fields are inconsistent."
            )

        current_iteration = int(state.get("current_iteration", -1))
        last_update = int(
            state.get("last_probability_update_iteration", -1)
        )
        update_count = int(state.get("probability_update_count", 0))
        fallback_count = int(state.get("fallback_count", 0))
        cluster_fallback_count = int(state.get("cluster_fallback_count", 0))
        if (
            current_iteration < -1
            or last_update < -1
            or last_update > current_iteration
        ):
            raise ValueError(
                "Checkpoint diversity sampler iteration counters are invalid."
            )
        if min(update_count, fallback_count, cluster_fallback_count) < 0:
            raise ValueError(
                "Checkpoint diversity sampler counters must be non-negative."
            )
        if (update_count == 0) != (last_update == -1):
            raise ValueError(
                "Checkpoint probability update count and last update disagree."
            )

        self.cluster_sample_count.copy_(saved_count)
        self.current_iteration = current_iteration
        self.last_probability_update_iteration = last_update
        self.probability_update_count = update_count
        self.fallback_count = fallback_count
        self.cluster_fallback_count = cluster_fallback_count
        self.last_fallback_reason = str(
            state.get("last_fallback_reason", "")
        )
        generator_state = torch.as_tensor(
            state.get("generator_state"), dtype=torch.uint8, device="cpu"
        )
        self.generator.set_state(generator_state)
        self._refresh_distribution_metrics()
