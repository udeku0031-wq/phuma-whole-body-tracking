"""Module-three controller joining statistics, formulas and adaptive sampling."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from whole_body_tracking.utils.adaptive_sampling import HierarchicalAdaptiveSampler
from whole_body_tracking.utils.learning_gap import (
    BinCalibrationResult,
    GapResult,
    MotionErrorResult,
    SegmentErrorResult,
    compute_learning_gaps,
    compute_motion_error,
    compute_segment_error,
    estimate_difficulty_bin_expectation,
    finite_distribution_summary,
)
from whole_body_tracking.utils.online_learning_stats import (
    OnlineLearningStatistics,
    motion_episode_outcomes,
    segment_traversal_outcomes,
)


ONLINE_LEARNING_SCHEMA_VERSION = "wbt.online_learning.v1"


def canonical_config_hash(config: Mapping[str, object]) -> str:
    """Hash JSON-compatible module-three semantics, independent of key order."""

    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class OnlineLearningController:
    """Own one shared global state and small per-environment traversal cursors."""

    def __init__(
        self,
        *,
        num_envs: int,
        motion_lengths: torch.Tensor,
        segment_motion_ids: torch.Tensor,
        segment_start_frames: torch.Tensor,
        segment_end_frames: torch.Tensor,
        motion_mode: str,
        segment_mode: str,
        settings: Mapping[str, object],
        motion_eligible_mask: torch.Tensor,
        segment_eligible_mask: torch.Tensor,
        difficulty_bins: torch.Tensor | None,
        device: str | torch.device,
    ) -> None:
        self.device = torch.device(device)
        self.num_envs = int(num_envs)
        self.motion_lengths = torch.as_tensor(motion_lengths, dtype=torch.long, device=self.device)
        self.segment_motion_ids = torch.as_tensor(segment_motion_ids, dtype=torch.long, device=self.device)
        self.segment_start_frames = torch.as_tensor(segment_start_frames, dtype=torch.long, device=self.device)
        self.segment_end_frames = torch.as_tensor(segment_end_frames, dtype=torch.long, device=self.device)
        self.num_motions = int(self.motion_lengths.numel())
        self.num_segments = int(self.segment_motion_ids.numel())
        self.motion_mode = str(motion_mode)
        self.segment_mode = str(segment_mode)
        self.settings = dict(settings)
        self.config_hash = canonical_config_hash(
            {"motion_mode": self.motion_mode, "segment_mode": self.segment_mode, **self.settings}
        )
        self.difficulty_bins = (
            None
            if difficulty_bins is None
            else torch.as_tensor(difficulty_bins, dtype=torch.long, device=self.device)
        )
        if self.difficulty_bins is not None and self.difficulty_bins.shape != self.segment_motion_ids.shape:
            raise ValueError("difficulty_bins must contain one value per segment.")

        self.statistics = OnlineLearningStatistics(
            self.num_motions,
            self.num_segments,
            ema_decay=float(self.settings["ema_decay"]),
            device=self.device,
            config_hash=self.config_hash,
        )
        adaptive = self.motion_mode != "uniform" or self.segment_mode != "uniform"
        self.sampler: HierarchicalAdaptiveSampler | None = None
        if adaptive:
            self.sampler = HierarchicalAdaptiveSampler(
                self.segment_motion_ids,
                self.segment_start_frames,
                self.segment_end_frames,
                self.motion_lengths,
                motion_eligible_mask=motion_eligible_mask,
                segment_eligible_mask=segment_eligible_mask,
                motion_mode=self.motion_mode,
                segment_mode=self.segment_mode,
                warmup_iterations=int(self.settings["warmup_iterations"]),
                probability_update_interval=int(self.settings["probability_update_interval"]),
                uniform_mix=float(self.settings["uniform_mix"]),
                temperature=float(self.settings["temperature"]),
                under_sampling_weight=float(self.settings["under_sampling_weight"]),
                motion_probability_cap=float(self.settings["motion_probability_cap"]),
                segment_probability_cap=float(self.settings["segment_probability_cap"]),
                score_clip=float(self.settings["score_clip"]),
                sampler_seed=int(self.settings["sampler_seed"]),
                config_hash=self.config_hash,
                device=self.device,
            )

        self.current_iteration = -1
        self.completed_window_count = 0
        self.last_formula_update_iteration = -1
        self.segment_error_result: SegmentErrorResult | None = None
        self.motion_error_result: MotionErrorResult | None = None
        self.bin_calibration: BinCalibrationResult | None = None
        self.gap_result: GapResult | None = None

        self.active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.episode_motion_id = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.episode_start_frame = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.episode_expected_frames = torch.ones(self.num_envs, dtype=torch.long, device=self.device)
        self.episode_observed_frames = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.traversal_segment_id = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.traversal_start_frame = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.traversal_required_frames = torch.ones(self.num_envs, dtype=torch.long, device=self.device)
        self.traversal_full_frames = torch.ones(self.num_envs, dtype=torch.long, device=self.device)
        self.traversal_observed_frames = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    def _ids(self, env_ids: Sequence[int] | torch.Tensor) -> torch.Tensor:
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if ids.ndim != 1 or (ids.numel() and (torch.any(ids < 0) or torch.any(ids >= self.num_envs))):
            raise ValueError("env_ids are outside the environment range.")
        return ids

    def begin_assignments(
        self,
        env_ids: Sequence[int] | torch.Tensor,
        motion_ids: torch.Tensor,
        start_frames: torch.Tensor,
        segment_ids: torch.Tensor,
    ) -> None:
        env_ids = self._ids(env_ids)
        motion_ids = torch.as_tensor(motion_ids, dtype=torch.long, device=self.device)
        start_frames = torch.as_tensor(start_frames, dtype=torch.long, device=self.device)
        segment_ids = torch.as_tensor(segment_ids, dtype=torch.long, device=self.device)
        if any(item.shape != env_ids.shape for item in (motion_ids, start_frames, segment_ids)):
            raise ValueError("Assignment tensors must match env_ids.")
        if env_ids.numel() == 0:
            return
        self.statistics.record_assignments(motion_ids, segment_ids)
        self.active[env_ids] = True
        self.episode_motion_id[env_ids] = motion_ids
        self.episode_start_frame[env_ids] = start_frames
        self.episode_expected_frames[env_ids] = self.motion_lengths[motion_ids] - start_frames
        self.episode_observed_frames[env_ids] = 0
        self._begin_traversals(env_ids, segment_ids, start_frames)

    def _begin_traversals(
        self,
        env_ids: torch.Tensor,
        segment_ids: torch.Tensor,
        start_frames: torch.Tensor,
    ) -> None:
        self.traversal_segment_id[env_ids] = segment_ids
        self.traversal_start_frame[env_ids] = start_frames
        self.traversal_required_frames[env_ids] = self.segment_end_frames[segment_ids] - start_frames
        self.traversal_full_frames[env_ids] = (
            self.segment_end_frames[segment_ids] - self.segment_start_frames[segment_ids]
        )
        self.traversal_observed_frames[env_ids] = 0

    def account_for_unobserved_assignment_start(
        self, env_ids: Sequence[int] | torch.Tensor
    ) -> None:
        """Remove a reset-only reference frame that the policy never controlled."""

        env_ids = self._ids(env_ids)
        env_ids = env_ids[self.active[env_ids]]
        if env_ids.numel() == 0:
            return
        self.episode_expected_frames[env_ids] = torch.clamp(
            self.episode_expected_frames[env_ids] - 1, min=1
        )
        self.traversal_required_frames[env_ids] = torch.clamp(
            self.traversal_required_frames[env_ids] - 1, min=1
        )
        self.traversal_start_frame[env_ids] += 1

    def observe_steps(
        self,
        motion_ids: torch.Tensor,
        segment_ids: torch.Tensor,
        *,
        body_error: torch.Tensor,
        joint_error: torch.Tensor,
        orientation_error: torch.Tensor,
        env_ids: Sequence[int] | torch.Tensor | None = None,
    ) -> None:
        """Record the pre-reset terminal/current frame for every active env."""

        if env_ids is None:
            requested = None
            requested_active = None
            selected = self.active
        else:
            requested = self._ids(env_ids)
            requested_active = self.active[requested]
            selected = torch.zeros_like(self.active)
            selected[requested] = True
            selected &= self.active
        active_ids = torch.where(selected)[0]
        if active_ids.numel() == 0:
            return
        motion_ids = torch.as_tensor(motion_ids, dtype=torch.long, device=self.device)
        segment_ids = torch.as_tensor(segment_ids, dtype=torch.long, device=self.device)
        if motion_ids.shape != self.active.shape or segment_ids.shape != self.active.shape:
            raise ValueError("Current motion/segment IDs must contain one value per environment.")
        if torch.any(motion_ids[active_ids] != self.episode_motion_id[active_ids]):
            raise RuntimeError("Online motion attribution changed before the active assignment was closed.")
        if torch.any(segment_ids[active_ids] != self.traversal_segment_id[active_ids]):
            raise RuntimeError("Online segment attribution changed before traversal boundary handling.")

        def active_component(name: str, value: torch.Tensor) -> torch.Tensor:
            tensor = torch.as_tensor(value, device=self.device)
            if tensor.shape == self.active.shape:
                return tensor[active_ids]
            if requested is not None and tensor.shape == requested.shape:
                if requested_active is None:
                    raise RuntimeError("Requested online observation mask is unavailable.")
                return tensor[requested_active]
            raise ValueError(
                f"{name} must contain either one value per environment or one per requested env_id."
            )

        self.statistics.record_step_observations(
            motion_ids[active_ids],
            segment_ids[active_ids],
            body_error=active_component("body_error", body_error),
            joint_error=active_component("joint_error", joint_error),
            orientation_error=active_component("orientation_error", orientation_error),
        )
        self.episode_observed_frames[active_ids] += 1
        self.traversal_observed_frames[active_ids] += 1

    def _finish_traversals(
        self,
        env_ids: torch.Tensor,
        *,
        terminated: torch.Tensor,
        natural_completion: torch.Tensor,
        timed_out: torch.Tensor,
    ) -> None:
        termination, completion, success = segment_traversal_outcomes(
            self.traversal_observed_frames[env_ids],
            self.traversal_required_frames[env_ids],
            self.traversal_full_frames[env_ids],
            terminated=terminated,
            natural_completion=natural_completion,
            timed_out=timed_out,
            minimum_observed_fraction=float(self.settings["minimum_segment_observed_fraction"]),
        )
        self.statistics.record_segment_outcomes(
            self.traversal_segment_id[env_ids],
            termination=termination.to(self.device),
            completion=completion.to(self.device),
            success=success.to(self.device),
        )

    def cross_segment_boundaries(
        self,
        env_ids: Sequence[int] | torch.Tensor,
        new_segment_ids: torch.Tensor,
        new_start_frames: torch.Tensor,
    ) -> None:
        env_ids = self._ids(env_ids)
        if env_ids.numel() == 0:
            return
        if torch.any(~self.active[env_ids]):
            raise RuntimeError("Cannot cross a segment boundary without an active assignment.")
        zeros = torch.zeros(env_ids.numel(), dtype=torch.bool, device=self.device)
        self._finish_traversals(
            env_ids,
            terminated=zeros,
            natural_completion=torch.ones_like(zeros),
            timed_out=zeros,
        )
        self._begin_traversals(
            env_ids,
            torch.as_tensor(new_segment_ids, dtype=torch.long, device=self.device),
            torch.as_tensor(new_start_frames, dtype=torch.long, device=self.device),
        )

    def finish_assignments(
        self,
        env_ids: Sequence[int] | torch.Tensor,
        *,
        terminated: torch.Tensor,
        natural_completion: torch.Tensor,
        timed_out: torch.Tensor,
        segment_natural_completion: torch.Tensor | None = None,
    ) -> None:
        env_ids = self._ids(env_ids)
        if env_ids.numel() == 0:
            return
        active_mask = self.active[env_ids]
        if not torch.any(active_mask):
            return
        env_ids = env_ids[active_mask]
        terminated = torch.as_tensor(terminated, dtype=torch.bool, device=self.device)[active_mask]
        natural_completion = torch.as_tensor(natural_completion, dtype=torch.bool, device=self.device)[active_mask]
        timed_out = torch.as_tensor(timed_out, dtype=torch.bool, device=self.device)[active_mask]
        terminated = terminated.to(torch.bool)
        natural_completion &= ~terminated
        timed_out &= ~(terminated | natural_completion)
        unresolved = ~(terminated | natural_completion | timed_out)
        timed_out = timed_out | unresolved
        if segment_natural_completion is None:
            segment_natural = natural_completion
        else:
            segment_natural = torch.as_tensor(
                segment_natural_completion, dtype=torch.bool, device=self.device
            )[active_mask]
            segment_natural &= ~terminated
        segment_timed_out = timed_out & ~segment_natural
        self._finish_traversals(
            env_ids,
            terminated=terminated,
            natural_completion=segment_natural,
            timed_out=segment_timed_out,
        )
        motion_term, motion_completion, motion_success = motion_episode_outcomes(
            self.episode_observed_frames[env_ids],
            self.episode_expected_frames[env_ids],
            terminated=terminated,
            natural_completion=natural_completion,
            timed_out=timed_out,
        )
        self.statistics.record_motion_outcomes(
            self.episode_motion_id[env_ids],
            termination=motion_term.to(self.device),
            completion=motion_completion.to(self.device),
            success=motion_success.to(self.device),
        )
        self.active[env_ids] = False

    def sample(self, num_samples: int) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        if self.sampler is None:
            raise RuntimeError("Adaptive sampling was requested for a uniform-only controller.")
        return self.sampler.sample(num_samples)

    def _segment_components(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        stats = self.statistics
        values = {
            "body": stats.segment_body_error_ema,
            "joint": stats.segment_joint_error_ema,
            "orientation": stats.segment_orientation_error_ema,
            "termination": stats.segment_termination_ema,
            "completion": stats.segment_completion_ema,
            "success": stats.segment_success_ema,
        }
        initialized = {
            "body": stats.segment_body_error_initialized,
            "joint": stats.segment_joint_error_initialized,
            "orientation": stats.segment_orientation_error_initialized,
            "termination": stats.segment_termination_initialized,
            "completion": stats.segment_completion_initialized,
            "success": stats.segment_success_initialized,
        }
        return values, initialized

    def _motion_outcomes(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        stats = self.statistics
        values = {
            "termination": stats.motion_termination_ema,
            "completion": stats.motion_completion_ema,
            "success": stats.motion_success_ema,
        }
        initialized = {
            "termination": stats.motion_termination_initialized,
            "completion": stats.motion_completion_initialized,
            "success": stats.motion_success_initialized,
        }
        return values, initialized

    def on_iteration_end(self, iteration: int) -> bool:
        """Commit EMA every PPO iteration and refresh probability at low frequency."""

        resolved_iteration = int(iteration)
        if resolved_iteration < self.current_iteration:
            raise ValueError("Online-learning iteration cannot move backwards.")
        self.current_iteration = resolved_iteration
        sampling_iteration = self.completed_window_count
        self.completed_window_count += 1
        self.statistics.commit_window()
        if self.sampler is not None:
            self.sampler.current_iteration = sampling_iteration
        update_interval = int(self.settings["probability_update_interval"])
        formula_due = (
            self.segment_error_result is None
            or sampling_iteration - self.last_formula_update_iteration >= update_interval
            or (self.sampler is not None and self.sampler.should_update(sampling_iteration))
        )
        if not formula_due:
            return False

        segment_values, segment_initialized = self._segment_components()
        error_weights = self.settings["error_weights"]
        if not isinstance(error_weights, Mapping):
            raise TypeError("error_weights must be a mapping.")
        self.segment_error_result = compute_segment_error(
            segment_values,
            segment_initialized,
            self.statistics.segment_step_count,
            min_segment_observations=int(self.settings["min_segment_observations"]),
            body_position_scale_m=float(self.settings["body_position_scale_m"]),
            joint_position_scale_rad=float(self.settings["joint_position_scale_rad"]),
            orientation_scale_rad=float(self.settings["orientation_scale_rad"]),
            weights=error_weights,
            component_clip=float(self.settings["component_clip"]),
        )
        motion_values, motion_initialized = self._motion_outcomes()
        motion_error_weights = self.settings["motion_error_weights"]
        if not isinstance(motion_error_weights, Mapping):
            raise TypeError("motion_error_weights must be a mapping.")
        self.motion_error_result = compute_motion_error(
            self.segment_error_result.error,
            self.segment_error_result.valid,
            self.segment_motion_ids,
            self.statistics.segment_step_count,
            motion_values,
            motion_initialized,
            self.statistics.motion_episode_count,
            min_motion_episodes=int(self.settings["min_motion_episodes"]),
            weights=motion_error_weights,
        )

        if self.difficulty_bins is not None:
            bin_weights = (
                self.statistics.segment_step_count
                if bool(self.settings["bin_observation_weighted"])
                else None
            )
            self.bin_calibration = estimate_difficulty_bin_expectation(
                self.segment_error_result.error,
                self.segment_error_result.valid,
                self.difficulty_bins,
                num_bins=int(self.settings["num_difficulty_bins"]),
                min_bin_valid_segments=int(self.settings["min_bin_valid_segments"]),
                sigma_floor=float(self.settings["sigma_floor"]),
                observation_weights=bin_weights,
            )
            gap_weights = self.settings["motion_gap_weights"]
            if not isinstance(gap_weights, Mapping):
                raise TypeError("motion_gap_weights must be a mapping.")
            self.gap_result = compute_learning_gaps(
                self.segment_error_result.error,
                self.segment_error_result.valid,
                self.segment_motion_ids,
                self.statistics.segment_step_count,
                self.difficulty_bins,
                self.bin_calibration,
                motion_values,
                motion_initialized,
                self.statistics.motion_episode_count,
                min_motion_episodes=int(self.settings["min_motion_episodes"]),
                gap_clip=float(self.settings["gap_clip"]),
                motion_gap_weights=gap_weights,
            )
        self.last_formula_update_iteration = sampling_iteration

        if self.sampler is None:
            return False
        if self.motion_mode == "learning_gap":
            if self.gap_result is None:
                raise RuntimeError("learning_gap sampling requires difficulty-calibrated gap state.")
            motion_score = self.gap_result.motion_gap
            motion_valid = self.gap_result.motion_valid
        else:
            motion_score = self.motion_error_result.error
            motion_valid = self.motion_error_result.valid
        if self.segment_mode == "relative_learning_gap":
            if self.gap_result is None:
                raise RuntimeError("relative_learning_gap sampling requires difficulty-calibrated gap state.")
            segment_score = self.gap_result.local_gap
            segment_valid = self.gap_result.global_valid
        else:
            segment_score = self.segment_error_result.error
            segment_valid = self.segment_error_result.valid
        return self.sampler.update_probabilities(
            sampling_iteration,
            motion_score=motion_score,
            motion_score_valid=motion_valid,
            segment_score=segment_score,
            segment_score_valid=segment_valid,
            motion_sample_count=self.statistics.motion_sample_count,
            segment_sample_count=self.statistics.segment_sample_count,
        )

    def metrics(self) -> dict[str, float | int]:
        metrics: dict[str, float | int] = {
            f"online/{name}": value
            for name, value in self.statistics.summary(
                min_segment_observations=int(self.settings["min_segment_observations"]),
                min_motion_episodes=int(self.settings["min_motion_episodes"]),
            ).items()
        }
        if self.segment_error_result is not None:
            summary = finite_distribution_summary(
                self.segment_error_result.error, self.segment_error_result.valid
            )
            metrics.update({f"error/segment_{name}": value for name, value in summary.items()})
            valid = self.segment_error_result.valid
            for name, contribution in self.segment_error_result.contributions.items():
                metrics[f"error/{name}_component_mean"] = (
                    float(contribution[valid].mean().item()) if torch.any(valid) else 0.0
                )
        if self.motion_error_result is not None:
            summary = finite_distribution_summary(self.motion_error_result.error, self.motion_error_result.valid)
            metrics["error/motion_mean"] = summary["mean"]
            metrics["error/motion_p90"] = summary["p90"]
        if self.bin_calibration is not None:
            metrics["gap/bin_fallback_count"] = int(torch.count_nonzero(self.bin_calibration.fallback_mask).item())
            for index in range(self.bin_calibration.mean.numel()):
                metrics[f"gap/bin_{index}_mu"] = float(self.bin_calibration.mean[index].item())
                metrics[f"gap/bin_{index}_sigma"] = float(self.bin_calibration.sigma[index].item())
                metrics[f"gap/bin_{index}_valid_count"] = int(
                    self.bin_calibration.valid_segment_count[index].item()
                )
        if self.gap_result is not None:
            global_summary = finite_distribution_summary(self.gap_result.global_gap, self.gap_result.global_valid)
            motion_summary = finite_distribution_summary(self.gap_result.motion_gap, self.gap_result.motion_valid)
            local_summary = finite_distribution_summary(self.gap_result.local_gap, self.gap_result.global_valid)
            metrics.update(
                {
                    "gap/global_mean": global_summary["mean"],
                    "gap/global_p50": global_summary["p50"],
                    "gap/global_p90": global_summary["p90"],
                    "gap/global_positive_ratio": float(
                        torch.count_nonzero(self.gap_result.global_gap[self.gap_result.global_valid] > 0).item()
                    )
                    / max(int(torch.count_nonzero(self.gap_result.global_valid).item()), 1),
                    "gap/motion_mean": motion_summary["mean"],
                    "gap/motion_p90": motion_summary["p90"],
                    "gap/local_mean": local_summary["mean"],
                    "gap/local_p90": local_summary["p90"],
                    "gap/clipped_ratio": float(torch.count_nonzero(self.gap_result.clipped_mask).item())
                    / max(int(torch.count_nonzero(self.gap_result.global_valid).item()), 1),
                }
            )
        if self.sampler is not None:
            metrics.update({f"sampling/{name}": value for name, value in self.sampler.metrics().items()})
            if self.difficulty_bins is not None:
                if self.segment_mode == "global_bin_raw_error":
                    marginal = self.sampler.global_segment_probability
                else:
                    marginal = (
                        self.sampler.motion_probability[self.segment_motion_ids]
                        * self.sampler.segment_probability
                    )
                for index in range(int(self.settings["num_difficulty_bins"])):
                    metrics[f"sampling/difficulty_bin_budget_{index}"] = float(
                        marginal[self.difficulty_bins == index].sum().item()
                    )
        return metrics

    def state_dict(self) -> dict[str, Any]:
        state: dict[str, Any] = {
            "schema_version": ONLINE_LEARNING_SCHEMA_VERSION,
            "config_hash": self.config_hash,
            "current_iteration": self.current_iteration,
            "completed_window_count": self.completed_window_count,
            "last_formula_update_iteration": self.last_formula_update_iteration,
            "statistics": self.statistics.state_dict(),
            "sampler": self.sampler.state_dict() if self.sampler is not None else None,
        }
        cache_names = (
            "segment_error_result",
            "motion_error_result",
            "bin_calibration",
            "gap_result",
        )
        for name in cache_names:
            value = getattr(self, name)
            state[name] = None if value is None else {
                field: ({key: tensor.detach().clone() for key, tensor in item.items()} if isinstance(item, dict) else item.detach().clone())
                for field, item in value.__dict__.items()
            }
        return state

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("schema_version") != ONLINE_LEARNING_SCHEMA_VERSION:
            raise ValueError("Unsupported online-learning checkpoint schema.")
        if state.get("config_hash") != self.config_hash:
            raise ValueError("Checkpoint online-learning config identity does not match the current run.")
        statistics = state.get("statistics")
        if not isinstance(statistics, Mapping):
            raise ValueError("Checkpoint online-learning statistics state is missing.")
        self.statistics.load_state_dict(statistics)
        sampler_state = state.get("sampler")
        if self.sampler is None:
            if sampler_state is not None:
                raise ValueError("Checkpoint uses adaptive sampling but the current run does not.")
        else:
            if not isinstance(sampler_state, Mapping):
                raise ValueError("Adaptive resume requires sampler state and dedicated RNG state.")
            self.sampler.load_state_dict(sampler_state)
        current_iteration = int(state.get("current_iteration", -1))
        completed_window_count = int(state.get("completed_window_count", -1))
        last_formula_update = int(state.get("last_formula_update_iteration", -1))
        if (
            current_iteration < -1
            or completed_window_count < 0
            or last_formula_update < -1
            or last_formula_update >= completed_window_count
            or (current_iteration == -1) != (completed_window_count == 0)
        ):
            raise ValueError("Checkpoint online-learning formula cadence is invalid.")
        self.current_iteration = current_iteration
        self.completed_window_count = completed_window_count
        self.last_formula_update_iteration = last_formula_update

        def restored_fields(name: str) -> dict[str, Any] | None:
            saved = state.get(name)
            if saved is None:
                return None
            if not isinstance(saved, Mapping):
                raise ValueError(f"Checkpoint cached field '{name}' is invalid.")
            result: dict[str, Any] = {}
            for field, value in saved.items():
                if isinstance(value, Mapping):
                    result[field] = {
                        key: torch.as_tensor(tensor, device=self.device).clone()
                        for key, tensor in value.items()
                    }
                else:
                    result[field] = torch.as_tensor(value, device=self.device).clone()
            return result

        segment_fields = restored_fields("segment_error_result")
        motion_fields = restored_fields("motion_error_result")
        calibration_fields = restored_fields("bin_calibration")
        gap_fields = restored_fields("gap_result")
        self.segment_error_result = (
            None if segment_fields is None else SegmentErrorResult(**segment_fields)
        )
        self.motion_error_result = (
            None if motion_fields is None else MotionErrorResult(**motion_fields)
        )
        self.bin_calibration = (
            None if calibration_fields is None else BinCalibrationResult(**calibration_fields)
        )
        self.gap_result = None if gap_fields is None else GapResult(**gap_fields)
        if (self.segment_error_result is None) != (self.last_formula_update_iteration == -1):
            raise ValueError("Checkpoint formula cache and update cadence disagree.")
        expected_sampler_iteration = self.completed_window_count - 1
        if self.sampler is not None and self.sampler.current_iteration != expected_sampler_iteration:
            raise ValueError("Checkpoint controller and sampler iterations disagree.")
        if (
            self.sampler is not None
            and self.sampler.last_probability_update_iteration
            > self.last_formula_update_iteration
        ):
            raise ValueError("Checkpoint probability is newer than its formula cache.")
        self._validate_restored_caches()
        self.active.zero_()

    def _validate_restored_caches(self) -> None:
        """Validate derived checkpoint arrays before metrics or sampling can use them."""

        def vector(name: str, value: torch.Tensor, size: int, *, boolean: bool = False) -> None:
            if value.shape != (size,):
                raise ValueError(f"Checkpoint cache '{name}' has the wrong shape.")
            if boolean:
                if value.dtype != torch.bool:
                    raise ValueError(f"Checkpoint cache '{name}' must be boolean.")
            elif not torch.all(torch.isfinite(value)):
                raise ValueError(f"Checkpoint cache '{name}' must be finite.")

        if self.segment_error_result is None:
            if any(
                item is not None
                for item in (
                    self.motion_error_result,
                    self.bin_calibration,
                    self.gap_result,
                )
            ):
                raise ValueError("Checkpoint formula caches are only partially present.")
            return
        if self.motion_error_result is None:
            raise ValueError("Checkpoint motion-error cache is missing.")

        segment = self.segment_error_result
        vector("segment.error", segment.error, self.num_segments)
        vector("segment.valid", segment.valid, self.num_segments, boolean=True)
        vector("segment.active_weight", segment.active_weight, self.num_segments)
        if set(segment.contributions) != {
            "body",
            "joint",
            "orientation",
            "termination",
            "completion",
            "success",
        }:
            raise ValueError("Checkpoint segment component cache is incomplete.")
        for name, value in segment.contributions.items():
            vector(f"segment.contributions.{name}", value, self.num_segments)

        motion = self.motion_error_result
        vector("motion.error", motion.error, self.num_motions)
        vector("motion.valid", motion.valid, self.num_motions, boolean=True)
        vector("motion.segment_mean", motion.segment_mean, self.num_motions)
        vector("motion.segment_p90", motion.segment_p90, self.num_motions)
        if set(motion.contributions) != {
            "segment_mean",
            "segment_p90",
            "termination",
            "completion",
            "success",
        }:
            raise ValueError("Checkpoint motion component cache is incomplete.")
        for name, value in motion.contributions.items():
            vector(f"motion.contributions.{name}", value, self.num_motions)

        if self.difficulty_bins is None:
            if self.bin_calibration is not None or self.gap_result is not None:
                raise ValueError("Checkpoint contains learning-gap caches without difficulty metadata.")
            return
        if self.bin_calibration is None or self.gap_result is None:
            raise ValueError("Difficulty-enabled checkpoint is missing learning-gap caches.")

        calibration = self.bin_calibration
        num_bins = int(self.settings["num_difficulty_bins"])
        vector("bin.mean", calibration.mean, num_bins)
        vector("bin.sigma", calibration.sigma, num_bins)
        vector("bin.valid_segment_count", calibration.valid_segment_count, num_bins)
        vector("bin.fallback_mask", calibration.fallback_mask, num_bins, boolean=True)
        vector("bin.reliable_mask", calibration.reliable_mask, num_bins, boolean=True)
        if calibration.global_mean.shape != () or calibration.global_sigma.shape != ():
            raise ValueError("Checkpoint global bin calibration must be scalar.")
        if (
            not torch.isfinite(calibration.global_mean)
            or not torch.isfinite(calibration.global_sigma)
            or torch.any(calibration.sigma <= 0.0)
            or calibration.global_sigma <= 0.0
        ):
            raise ValueError("Checkpoint bin calibration is invalid.")

        gap = self.gap_result
        for name, value in (
            ("global_gap", gap.global_gap),
            ("local_gap", gap.local_gap),
            ("clipped_mask", gap.clipped_mask),
        ):
            vector(f"gap.{name}", value, self.num_segments, boolean=name == "clipped_mask")
        vector("gap.global_valid", gap.global_valid, self.num_segments, boolean=True)
        for name, value in (
            ("motion_gap", gap.motion_gap),
            ("motion_positive_mean", gap.motion_positive_mean),
            ("motion_positive_p90", gap.motion_positive_p90),
        ):
            vector(f"gap.{name}", value, self.num_motions)
        vector("gap.motion_valid", gap.motion_valid, self.num_motions, boolean=True)
        if set(gap.contributions) != {
            "positive_mean",
            "positive_p90",
            "termination",
            "completion",
            "success",
        }:
            raise ValueError("Checkpoint motion-gap component cache is incomplete.")
        for name, value in gap.contributions.items():
            vector(f"gap.contributions.{name}", value, self.num_motions)
