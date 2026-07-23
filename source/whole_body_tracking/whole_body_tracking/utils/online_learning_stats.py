"""Shared online Motion--Segment statistics for module three.

The implementation is intentionally independent of Isaac Sim.  All parallel
environments write into one set of global tensors.  Observations are first
reduced by ID into window sums/counts and every ID receives at most one EMA
update when :meth:`OnlineLearningStatistics.commit_window` is called.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch


ONLINE_STATS_SCHEMA_VERSION = "wbt.online_learning_stats.v1"
_TRACKING_COMPONENTS = ("body_error", "joint_error", "orientation_error")
_OUTCOME_COMPONENTS = ("termination", "completion", "success")
_ALL_COMPONENTS = _TRACKING_COMPONENTS + _OUTCOME_COMPONENTS


def _float_vector(values: Sequence[float] | torch.Tensor, *, device: torch.device) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float64, device=device)
    if tensor.ndim != 1:
        raise ValueError("Online statistics values must be one-dimensional.")
    return tensor


def _long_vector(values: Sequence[int] | torch.Tensor, *, device: torch.device) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.long, device=device)
    if tensor.ndim != 1:
        raise ValueError("Online statistics IDs must be one-dimensional.")
    return tensor


def segment_traversal_outcomes(
    observed_frames: Sequence[int] | torch.Tensor,
    required_remaining_frames: Sequence[int] | torch.Tensor,
    full_segment_frames: Sequence[int] | torch.Tensor,
    *,
    terminated: Sequence[bool] | torch.Tensor,
    natural_completion: Sequence[bool] | torch.Tensor,
    timed_out: Sequence[bool] | torch.Tensor,
    minimum_observed_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return local ``termination, completion, success`` outcomes.

    Completion uses the remaining traversal length, while eligibility for a
    completion/success observation uses the full segment length.  Therefore a
    reset that starts on the final one or two frames cannot create a spurious
    complete-success observation.  Administrative timeouts are censored rather
    than failures: they record termination=0, while completion and success are
    missing.  This prevents the fixed environment horizon from penalizing the
    segment that happened to be active at the cutoff.
    Missing outcomes are represented by NaN and ignored by the accumulator.
    """

    if not 0.0 <= minimum_observed_fraction <= 1.0:
        raise ValueError("minimum_observed_fraction must be in [0, 1].")
    observed = torch.as_tensor(observed_frames, dtype=torch.float64)
    remaining = torch.as_tensor(required_remaining_frames, dtype=torch.float64, device=observed.device)
    full = torch.as_tensor(full_segment_frames, dtype=torch.float64, device=observed.device)
    term = torch.as_tensor(terminated, dtype=torch.bool, device=observed.device)
    natural = torch.as_tensor(natural_completion, dtype=torch.bool, device=observed.device)
    timeout = torch.as_tensor(timed_out, dtype=torch.bool, device=observed.device)
    tensors = (remaining, full, term, natural, timeout)
    if observed.ndim != 1 or any(item.shape != observed.shape for item in tensors):
        raise ValueError("All traversal outcome inputs must be one-dimensional with the same shape.")
    if torch.any(observed < 0) or torch.any(remaining <= 0) or torch.any(full <= 0):
        raise ValueError("Traversal frame counts must be non-negative with positive denominators.")
    if torch.any(term & natural) or torch.any(term & timeout) or torch.any(natural & timeout):
        raise ValueError("A traversal can have only one closing reason.")

    progress = torch.clamp(observed / remaining, 0.0, 1.0)
    sufficiently_observed = observed / full >= minimum_observed_fraction
    nan = torch.full_like(progress, float("nan"))
    termination = term.to(torch.float64)
    completion = torch.where(sufficiently_observed, progress, nan)
    completion = torch.where(natural & sufficiently_observed, torch.ones_like(completion), completion)
    completion = torch.where(timeout, nan, completion)
    success = torch.where(natural, torch.ones_like(progress), torch.zeros_like(progress))
    success = torch.where(timeout | ~sufficiently_observed, nan, success)
    return termination, completion, success


def motion_episode_outcomes(
    observed_frames: Sequence[int] | torch.Tensor,
    expected_remaining_frames: Sequence[int] | torch.Tensor,
    *,
    terminated: Sequence[bool] | torch.Tensor,
    natural_completion: Sequence[bool] | torch.Tensor,
    timed_out: Sequence[bool] | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return motion termination, remaining-length completion and success.

    A generic framework timeout is an administrative truncation rather than a
    policy failure.  It contributes termination=0, but completion and success
    are censored so the arbitrary environment horizon cannot increase error.
    """

    observed = torch.as_tensor(observed_frames, dtype=torch.float64)
    expected = torch.as_tensor(expected_remaining_frames, dtype=torch.float64, device=observed.device)
    term = torch.as_tensor(terminated, dtype=torch.bool, device=observed.device)
    natural = torch.as_tensor(natural_completion, dtype=torch.bool, device=observed.device)
    timeout = torch.as_tensor(timed_out, dtype=torch.bool, device=observed.device)
    if observed.ndim != 1 or any(item.shape != observed.shape for item in (expected, term, natural, timeout)):
        raise ValueError("All motion outcome inputs must be one-dimensional with the same shape.")
    if torch.any(observed < 0) or torch.any(expected <= 0):
        raise ValueError("Motion frame counts must be non-negative with positive expected lengths.")
    if torch.any(term & natural) or torch.any(term & timeout) or torch.any(natural & timeout):
        raise ValueError("A motion episode can have only one closing reason.")

    completion = torch.clamp(observed / expected, 0.0, 1.0)
    completion = torch.where(natural, torch.ones_like(completion), completion)
    completion = torch.where(timeout, torch.full_like(completion, float("nan")), completion)
    termination = term.to(torch.float64)
    success = torch.where(natural, torch.ones_like(completion), torch.zeros_like(completion))
    success = torch.where(timeout, torch.full_like(success, float("nan")), success)
    return termination, completion, success


class OnlineLearningStatistics:
    """One shared set of window accumulators, cumulative counts and EMAs."""

    def __init__(
        self,
        num_motions: int,
        num_segments: int,
        *,
        ema_decay: float,
        device: str | torch.device = "cpu",
        config_hash: str = "",
    ) -> None:
        if num_motions < 1 or num_segments < 1:
            raise ValueError("num_motions and num_segments must be positive.")
        if not 0.0 <= ema_decay < 1.0:
            raise ValueError("ema_decay must be in [0, 1).")
        self.num_motions = int(num_motions)
        self.num_segments = int(num_segments)
        self.ema_decay = float(ema_decay)
        self.device = torch.device(device)
        self.config_hash = str(config_hash)

        self.segment_sample_count = torch.zeros(self.num_segments, dtype=torch.long, device=self.device)
        self.motion_sample_count = torch.zeros(self.num_motions, dtype=torch.long, device=self.device)
        self.segment_step_count = torch.zeros(self.num_segments, dtype=torch.long, device=self.device)
        self.motion_step_count = torch.zeros(self.num_motions, dtype=torch.long, device=self.device)
        self.segment_outcome_count = torch.zeros(self.num_segments, dtype=torch.long, device=self.device)
        self.motion_episode_count = torch.zeros(self.num_motions, dtype=torch.long, device=self.device)
        self.total_assignments = torch.zeros((), dtype=torch.long, device=self.device)
        self.total_step_observations = torch.zeros((), dtype=torch.long, device=self.device)
        self.total_segment_outcomes = torch.zeros((), dtype=torch.long, device=self.device)
        self.total_motion_episodes = torch.zeros((), dtype=torch.long, device=self.device)
        self.ema_update_count = torch.zeros((), dtype=torch.long, device=self.device)

        for level, size in (("segment", self.num_segments), ("motion", self.num_motions)):
            for component in _ALL_COMPONENTS:
                setattr(self, f"{level}_{component}_ema", torch.zeros(size, dtype=torch.float64, device=self.device))
                setattr(self, f"{level}_{component}_initialized", torch.zeros(size, dtype=torch.bool, device=self.device))
                setattr(self, f"_pending_{level}_{component}_sum", torch.zeros(size, dtype=torch.float64, device=self.device))
                setattr(self, f"_pending_{level}_{component}_count", torch.zeros(size, dtype=torch.long, device=self.device))

    def _validate_ids(self, ids: torch.Tensor, size: int, label: str) -> None:
        if ids.numel() and (torch.any(ids < 0) or torch.any(ids >= size)):
            raise ValueError(f"{label} must be in [0, {size}).")

    def _accumulate_component(
        self,
        *,
        level: str,
        component: str,
        ids: torch.Tensor,
        values: torch.Tensor,
        size: int,
    ) -> None:
        valid = torch.isfinite(values)
        if component in _OUTCOME_COMPONENTS and torch.any(valid & ((values < 0.0) | (values > 1.0))):
            raise ValueError(f"{component} outcomes must be in [0, 1] or NaN.")
        if not torch.any(valid):
            return
        valid_ids = ids[valid]
        valid_values = values[valid]
        sums = torch.bincount(valid_ids, weights=valid_values, minlength=size)
        counts = torch.bincount(valid_ids, minlength=size).to(torch.long)
        getattr(self, f"_pending_{level}_{component}_sum").add_(sums)
        getattr(self, f"_pending_{level}_{component}_count").add_(counts)

    def record_assignments(
        self,
        motion_ids: Sequence[int] | torch.Tensor,
        segment_ids: Sequence[int] | torch.Tensor,
    ) -> None:
        motion = _long_vector(motion_ids, device=self.device)
        segment = _long_vector(segment_ids, device=self.device)
        if motion.shape != segment.shape:
            raise ValueError("motion_ids and segment_ids must have the same shape.")
        self._validate_ids(motion, self.num_motions, "motion_ids")
        self._validate_ids(segment, self.num_segments, "segment_ids")
        if motion.numel() == 0:
            return
        self.motion_sample_count += torch.bincount(motion, minlength=self.num_motions)
        self.segment_sample_count += torch.bincount(segment, minlength=self.num_segments)
        self.total_assignments += motion.numel()

    def record_step_observations(
        self,
        motion_ids: Sequence[int] | torch.Tensor,
        segment_ids: Sequence[int] | torch.Tensor,
        *,
        body_error: Sequence[float] | torch.Tensor,
        joint_error: Sequence[float] | torch.Tensor,
        orientation_error: Sequence[float] | torch.Tensor,
    ) -> None:
        motion = _long_vector(motion_ids, device=self.device)
        segment = _long_vector(segment_ids, device=self.device)
        values = {
            "body_error": _float_vector(body_error, device=self.device),
            "joint_error": _float_vector(joint_error, device=self.device),
            "orientation_error": _float_vector(orientation_error, device=self.device),
        }
        if any(value.shape != motion.shape for value in (segment, *values.values())):
            raise ValueError("Step observation IDs and components must have the same shape.")
        self._validate_ids(motion, self.num_motions, "motion_ids")
        self._validate_ids(segment, self.num_segments, "segment_ids")
        if motion.numel() == 0:
            return
        self.segment_step_count += torch.bincount(segment, minlength=self.num_segments)
        self.motion_step_count += torch.bincount(motion, minlength=self.num_motions)
        self.total_step_observations += motion.numel()
        for component, component_values in values.items():
            self._accumulate_component(
                level="segment", component=component, ids=segment, values=component_values, size=self.num_segments
            )
            self._accumulate_component(
                level="motion", component=component, ids=motion, values=component_values, size=self.num_motions
            )

    def record_segment_outcomes(
        self,
        segment_ids: Sequence[int] | torch.Tensor,
        *,
        termination: Sequence[float] | torch.Tensor,
        completion: Sequence[float] | torch.Tensor,
        success: Sequence[float] | torch.Tensor,
    ) -> None:
        ids = _long_vector(segment_ids, device=self.device)
        values = {
            "termination": _float_vector(termination, device=self.device),
            "completion": _float_vector(completion, device=self.device),
            "success": _float_vector(success, device=self.device),
        }
        if any(value.shape != ids.shape for value in values.values()):
            raise ValueError("Segment outcome IDs and components must have the same shape.")
        self._validate_ids(ids, self.num_segments, "segment_ids")
        if ids.numel() == 0:
            return
        any_valid = torch.zeros_like(ids, dtype=torch.bool)
        for component, component_values in values.items():
            any_valid |= torch.isfinite(component_values)
            self._accumulate_component(
                level="segment", component=component, ids=ids, values=component_values, size=self.num_segments
            )
        valid_ids = ids[any_valid]
        self.segment_outcome_count += torch.bincount(valid_ids, minlength=self.num_segments)
        self.total_segment_outcomes += valid_ids.numel()

    def record_motion_outcomes(
        self,
        motion_ids: Sequence[int] | torch.Tensor,
        *,
        termination: Sequence[float] | torch.Tensor,
        completion: Sequence[float] | torch.Tensor,
        success: Sequence[float] | torch.Tensor,
    ) -> None:
        ids = _long_vector(motion_ids, device=self.device)
        values = {
            "termination": _float_vector(termination, device=self.device),
            "completion": _float_vector(completion, device=self.device),
            "success": _float_vector(success, device=self.device),
        }
        if any(value.shape != ids.shape for value in values.values()):
            raise ValueError("Motion outcome IDs and components must have the same shape.")
        self._validate_ids(ids, self.num_motions, "motion_ids")
        if ids.numel() == 0:
            return
        any_valid = torch.zeros_like(ids, dtype=torch.bool)
        for component, component_values in values.items():
            any_valid |= torch.isfinite(component_values)
            self._accumulate_component(
                level="motion", component=component, ids=ids, values=component_values, size=self.num_motions
            )
        valid_ids = ids[any_valid]
        self.motion_episode_count += torch.bincount(valid_ids, minlength=self.num_motions)
        self.total_motion_episodes += valid_ids.numel()

    def _commit_component(self, level: str, component: str) -> bool:
        pending_sum = getattr(self, f"_pending_{level}_{component}_sum")
        pending_count = getattr(self, f"_pending_{level}_{component}_count")
        active = pending_count > 0
        if not torch.any(active):
            return False
        ema = getattr(self, f"{level}_{component}_ema")
        initialized = getattr(self, f"{level}_{component}_initialized")
        batch_mean = pending_sum[active] / pending_count[active].to(torch.float64)
        active_initialized = initialized[active]
        ema[active] = torch.where(
            active_initialized,
            self.ema_decay * ema[active] + (1.0 - self.ema_decay) * batch_mean,
            batch_mean,
        )
        initialized[active] = True
        pending_sum.zero_()
        pending_count.zero_()
        return True

    def commit_window(self) -> bool:
        """Commit one aggregated window, updating every active ID exactly once."""

        changed = False
        for level in ("segment", "motion"):
            for component in _ALL_COMPONENTS:
                changed = self._commit_component(level, component) or changed
        if changed:
            self.ema_update_count += 1
        return changed

    def summary(self, *, min_segment_observations: int, min_motion_episodes: int) -> dict[str, float | int]:
        segment_valid = self.segment_step_count >= int(min_segment_observations)
        motion_valid = self.motion_episode_count >= int(min_motion_episodes)
        return {
            "segment_valid_ratio": float(segment_valid.to(torch.float64).mean().item()),
            "motion_valid_ratio": float(motion_valid.to(torch.float64).mean().item()),
            "segment_observation_count": int(self.total_step_observations.item()),
            "motion_episode_count": int(self.total_motion_episodes.item()),
            "ema_update_count": int(self.ema_update_count.item()),
            "cold_segment_count": int(torch.count_nonzero(~segment_valid).item()),
            "cold_motion_count": int(torch.count_nonzero(~motion_valid).item()),
        }

    def state_dict(self) -> dict[str, Any]:
        state: dict[str, Any] = {
            "schema_version": ONLINE_STATS_SCHEMA_VERSION,
            "num_motions": self.num_motions,
            "num_segments": self.num_segments,
            "ema_decay": self.ema_decay,
            "config_hash": self.config_hash,
        }
        scalar_or_count_names = (
            "segment_sample_count",
            "motion_sample_count",
            "segment_step_count",
            "motion_step_count",
            "segment_outcome_count",
            "motion_episode_count",
            "total_assignments",
            "total_step_observations",
            "total_segment_outcomes",
            "total_motion_episodes",
            "ema_update_count",
        )
        for name in scalar_or_count_names:
            state[name] = getattr(self, name).detach().clone()
        for level in ("segment", "motion"):
            for component in _ALL_COMPONENTS:
                for suffix in ("ema", "initialized"):
                    name = f"{level}_{component}_{suffix}"
                    state[name] = getattr(self, name).detach().clone()
                for suffix in ("sum", "count"):
                    name = f"_pending_{level}_{component}_{suffix}"
                    state[name] = getattr(self, name).detach().clone()
        return state

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("schema_version") != ONLINE_STATS_SCHEMA_VERSION:
            raise ValueError("Unsupported online-learning statistics schema version.")
        if int(state.get("num_motions", -1)) != self.num_motions or int(state.get("num_segments", -1)) != self.num_segments:
            raise ValueError("Checkpoint online statistics layout does not match the current segment index.")
        if float(state.get("ema_decay", -1.0)) != self.ema_decay:
            raise ValueError("Checkpoint online statistics EMA decay does not match the current configuration.")
        if str(state.get("config_hash", "")) != self.config_hash:
            raise ValueError("Checkpoint online statistics config hash does not match the current configuration.")

        current = self.state_dict()
        tensor_names = [name for name, value in current.items() if isinstance(value, torch.Tensor)]
        missing = [name for name in tensor_names if name not in state]
        if missing:
            raise ValueError(f"Online statistics state is missing fields: {missing}")
        for name in tensor_names:
            target = getattr(self, name)
            saved = torch.as_tensor(state[name], dtype=target.dtype, device=self.device)
            if saved.shape != target.shape:
                raise ValueError(f"Checkpoint field '{name}' has the wrong shape.")
            if target.dtype != torch.bool and torch.any(saved < 0) and ("count" in name or name.startswith("total_")):
                raise ValueError(f"Checkpoint count field '{name}' must be non-negative.")
            target.copy_(saved)

        total_invariants = (
            (self.motion_sample_count.sum(), self.total_assignments, "motion assignment"),
            (self.segment_sample_count.sum(), self.total_assignments, "segment assignment"),
            (self.motion_step_count.sum(), self.total_step_observations, "motion step"),
            (self.segment_step_count.sum(), self.total_step_observations, "segment step"),
            (self.segment_outcome_count.sum(), self.total_segment_outcomes, "segment outcome"),
            (self.motion_episode_count.sum(), self.total_motion_episodes, "motion episode"),
        )
        for observed, expected, label in total_invariants:
            if not torch.equal(observed, expected):
                raise ValueError(f"Checkpoint {label} counts disagree with their cumulative total.")

        for level in ("segment", "motion"):
            for component in _ALL_COMPONENTS:
                ema = getattr(self, f"{level}_{component}_ema")
                pending_sum = getattr(self, f"_pending_{level}_{component}_sum")
                pending_count = getattr(self, f"_pending_{level}_{component}_count")
                if not torch.all(torch.isfinite(ema)) or not torch.all(torch.isfinite(pending_sum)):
                    raise ValueError(
                        f"Checkpoint {level} {component} statistics must be finite."
                    )
                if torch.any(ema < 0.0) or torch.any(pending_sum < 0.0):
                    raise ValueError(
                        f"Checkpoint {level} {component} statistics must be non-negative."
                    )
                if component in _OUTCOME_COMPONENTS and (
                    torch.any(ema > 1.0)
                    or torch.any(pending_sum > pending_count.to(torch.float64) + 1.0e-12)
                ):
                    raise ValueError(
                        f"Checkpoint {level} {component} outcomes must remain in [0, 1]."
                    )
