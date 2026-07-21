"""Shared segment indexing and sampling utilities.

This module deliberately depends only on PyTorch and the Python standard
library.  It can therefore be tested without launching Isaac Sim.
"""

from __future__ import annotations

import csv
import hashlib
import math
import os
import warnings
from collections.abc import Mapping, Sequence
from typing import Any

import torch


SAMPLING_STATE_VERSION = 1
_PROBABILITY_WARNING_LIMIT = 3
_probability_warning_counts: dict[str, int] = {}


def motion_pool_fingerprint(
    motion_files: Sequence[str], motion_lengths: Sequence[int], motion_fps: Sequence[float]
) -> str:
    """Return an order-sensitive fingerprint for a loaded motion pool."""

    if not (len(motion_files) == len(motion_lengths) == len(motion_fps)):
        raise ValueError("motion_files, motion_lengths, and motion_fps must have the same length.")
    if not motion_files:
        raise ValueError("motion_files must not be empty.")

    normalized_paths = [os.path.abspath(os.path.normpath(path)) for path in motion_files]
    common_root = os.path.commonpath([os.path.dirname(path) for path in normalized_paths])

    digest = hashlib.sha256()
    for path, length, fps in zip(normalized_paths, motion_lengths, motion_fps, strict=True):
        # Relative identities keep the fingerprint stable when an entire data
        # tree is relocated.  File size cheaply catches most in-place changes
        # without hashing every large NPZ during environment startup.
        digest.update(os.path.relpath(path, common_root).encode("utf-8"))
        digest.update(b"\0")
        file_size = os.path.getsize(path) if os.path.isfile(path) else -1
        digest.update(str(file_size).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(int(length)).encode("ascii"))
        digest.update(b"\0")
        digest.update(format(float(fps), ".17g").encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


class FixedLengthSegmentIndex:
    """Compact fixed-duration segment metadata and tensor ID mappings."""

    def __init__(
        self,
        motion_lengths: Sequence[int] | torch.Tensor,
        motion_fps: Sequence[float] | torch.Tensor,
        segment_length_seconds: float = 1.0,
        *,
        device: str | torch.device = "cpu",
    ) -> None:
        if not math.isfinite(segment_length_seconds) or segment_length_seconds <= 0.0:
            raise ValueError("segment_length_seconds must be finite and greater than zero.")

        self.device = torch.device(device)
        self.segment_length_seconds = float(segment_length_seconds)
        self.motion_lengths = torch.as_tensor(motion_lengths, dtype=torch.long, device=self.device)
        self.motion_fps = torch.as_tensor(motion_fps, dtype=torch.float64, device=self.device)

        if self.motion_lengths.ndim != 1 or self.motion_lengths.numel() == 0:
            raise ValueError("motion_lengths must be a non-empty one-dimensional sequence.")
        if self.motion_fps.ndim != 1 or self.motion_fps.shape != self.motion_lengths.shape:
            raise ValueError("motion_fps must be one-dimensional and match motion_lengths.")
        if torch.any(self.motion_lengths < 1):
            raise ValueError("Every motion must contain at least one frame.")
        if not torch.all(torch.isfinite(self.motion_fps)) or torch.any(self.motion_fps <= 0.0):
            raise ValueError("Every motion FPS must be finite and greater than zero.")

        # torch.round follows round-to-nearest-even, matching Python's round for
        # positive finite values.  Clamp preserves a segment for very low FPS.
        self.segment_frames = torch.round(self.motion_fps * self.segment_length_seconds).to(torch.long).clamp_min(1)
        self.motion_num_segments = torch.div(
            self.motion_lengths + self.segment_frames - 1,
            self.segment_frames,
            rounding_mode="floor",
        )
        self.motion_segment_offsets = torch.zeros(
            self.motion_lengths.numel() + 1, dtype=torch.long, device=self.device
        )
        self.motion_segment_offsets[1:] = torch.cumsum(self.motion_num_segments, dim=0)

        self.num_motions = int(self.motion_lengths.numel())
        self.num_segments = int(self.motion_segment_offsets[-1].item())

        global_ids = torch.arange(self.num_segments, dtype=torch.long, device=self.device)
        self.segment_motion_ids = torch.repeat_interleave(
            torch.arange(self.num_motions, dtype=torch.long, device=self.device), self.motion_num_segments
        )
        self.segment_local_ids = global_ids - self.motion_segment_offsets[self.segment_motion_ids]
        self.segment_start_frames = self.segment_local_ids * self.segment_frames[self.segment_motion_ids]
        self.segment_end_frames = torch.minimum(
            self.segment_start_frames + self.segment_frames[self.segment_motion_ids],
            self.motion_lengths[self.segment_motion_ids],
        )

        if torch.any(self.segment_start_frames >= self.segment_end_frames):
            raise RuntimeError("Fixed-length segment construction produced an empty segment.")

    def _long_tensor(self, values: Sequence[int] | torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(values, dtype=torch.long, device=self.device)

    @staticmethod
    def _require_same_shape(first: torch.Tensor, second: torch.Tensor, first_name: str, second_name: str) -> None:
        if first.shape != second.shape:
            raise ValueError(f"{first_name} and {second_name} must have the same shape.")

    def _validate_motion_ids(self, motion_ids: torch.Tensor) -> None:
        if motion_ids.numel() and (torch.any(motion_ids < 0) or torch.any(motion_ids >= self.num_motions)):
            raise ValueError(f"motion_ids must be in [0, {self.num_motions}).")

    def _validate_global_segment_ids(self, global_segment_ids: torch.Tensor) -> None:
        if global_segment_ids.numel() and (
            torch.any(global_segment_ids < 0) or torch.any(global_segment_ids >= self.num_segments)
        ):
            raise ValueError(f"global_segment_ids must be in [0, {self.num_segments}).")

    def motion_frame_to_segment(
        self, motion_ids: Sequence[int] | torch.Tensor, frame_ids: Sequence[int] | torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Map motion/frame pairs to local and global segment IDs."""

        motion_ids_tensor = self._long_tensor(motion_ids)
        frame_ids_tensor = self._long_tensor(frame_ids)
        self._require_same_shape(motion_ids_tensor, frame_ids_tensor, "motion_ids", "frame_ids")
        self._validate_motion_ids(motion_ids_tensor)
        if frame_ids_tensor.numel() and (
            torch.any(frame_ids_tensor < 0)
            or torch.any(frame_ids_tensor >= self.motion_lengths[motion_ids_tensor])
        ):
            raise ValueError("frame_ids must be within the selected motion lengths.")

        local_segment_ids = torch.div(
            frame_ids_tensor, self.segment_frames[motion_ids_tensor], rounding_mode="floor"
        )
        global_segment_ids = self.motion_segment_offsets[motion_ids_tensor] + local_segment_ids
        return local_segment_ids, global_segment_ids

    def motion_local_to_global(
        self, motion_ids: Sequence[int] | torch.Tensor, local_segment_ids: Sequence[int] | torch.Tensor
    ) -> torch.Tensor:
        """Map motion/local-segment pairs to global segment IDs."""

        motion_ids_tensor = self._long_tensor(motion_ids)
        local_ids_tensor = self._long_tensor(local_segment_ids)
        self._require_same_shape(motion_ids_tensor, local_ids_tensor, "motion_ids", "local_segment_ids")
        self._validate_motion_ids(motion_ids_tensor)
        if local_ids_tensor.numel() and (
            torch.any(local_ids_tensor < 0)
            or torch.any(local_ids_tensor >= self.motion_num_segments[motion_ids_tensor])
        ):
            raise ValueError("local_segment_ids must be within the selected motion segment counts.")
        return self.motion_segment_offsets[motion_ids_tensor] + local_ids_tensor

    def global_to_motion_local(
        self, global_segment_ids: Sequence[int] | torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Map global segment IDs back to motion and local segment IDs."""

        global_ids_tensor = self._long_tensor(global_segment_ids)
        self._validate_global_segment_ids(global_ids_tensor)
        return self.segment_motion_ids[global_ids_tensor], self.segment_local_ids[global_ids_tensor]

    def metadata(self, global_segment_ids: Sequence[int] | torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """Return compact metadata tensors for selected (or all) segments."""

        if global_segment_ids is None:
            global_ids_tensor = torch.arange(self.num_segments, dtype=torch.long, device=self.device)
        else:
            global_ids_tensor = self._long_tensor(global_segment_ids)
            self._validate_global_segment_ids(global_ids_tensor)
        motion_ids = self.segment_motion_ids[global_ids_tensor]
        start_frames = self.segment_start_frames[global_ids_tensor]
        end_frames = self.segment_end_frames[global_ids_tensor]
        return {
            "motion_id": motion_ids,
            "local_segment_id": self.segment_local_ids[global_ids_tensor],
            "global_segment_id": global_ids_tensor,
            "start_frame": start_frames,
            "end_frame_exclusive": end_frames,
            "num_frames": end_frames - start_frames,
            "motion_num_frames": self.motion_lengths[motion_ids],
            "fps": self.motion_fps[motion_ids],
        }

    def state_dict(self) -> dict[str, Any]:
        """Return the layout needed to validate checkpoint compatibility."""

        return {
            "version": SAMPLING_STATE_VERSION,
            "segment_length_seconds": self.segment_length_seconds,
            "motion_lengths": self.motion_lengths.detach().clone(),
            "motion_fps": self.motion_fps.detach().clone(),
            "segment_frames": self.segment_frames.detach().clone(),
            "motion_num_segments": self.motion_num_segments.detach().clone(),
            "motion_segment_offsets": self.motion_segment_offsets.detach().clone(),
            "num_segments": self.num_segments,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Validate that a saved layout describes this exact segment index."""

        required = {
            "version",
            "segment_length_seconds",
            "motion_lengths",
            "motion_fps",
            "segment_frames",
            "motion_num_segments",
            "motion_segment_offsets",
            "num_segments",
        }
        missing = required.difference(state)
        if missing:
            raise ValueError(f"Segment index state is missing fields: {sorted(missing)}")
        if int(state["version"]) != SAMPLING_STATE_VERSION:
            raise ValueError(
                f"Unsupported segment index state version {state['version']}; expected {SAMPLING_STATE_VERSION}."
            )
        if not math.isclose(
            float(state["segment_length_seconds"]), self.segment_length_seconds, rel_tol=0.0, abs_tol=1.0e-12
        ):
            raise ValueError("Checkpoint segment length does not match the current configuration.")
        if int(state["num_segments"]) != self.num_segments:
            raise ValueError("Checkpoint global segment count does not match the current motion pool.")

        for name in (
            "motion_lengths",
            "segment_frames",
            "motion_num_segments",
            "motion_segment_offsets",
        ):
            saved = torch.as_tensor(state[name], dtype=torch.long, device=self.device)
            current = getattr(self, name)
            if saved.shape != current.shape or not torch.equal(saved, current):
                raise ValueError(f"Checkpoint {name} does not match the current motion pool.")
        saved_fps = torch.as_tensor(state["motion_fps"], dtype=torch.float64, device=self.device)
        if saved_fps.shape != self.motion_fps.shape or not torch.allclose(
            saved_fps, self.motion_fps, rtol=0.0, atol=1.0e-12
        ):
            raise ValueError("Checkpoint motion_fps does not match the current motion pool.")


class QualityGatedStartIndex:
    """Compact deterministic mapping from uniforms to quality-eligible starts.

    The caller owns quality semantics and passes one boolean value per global
    segment.  For example, it may combine ``pass`` and ``borderline`` into the
    allowed mask.  This class only intersects that mask with the legacy start
    domain ``[0, motion_length - 1)`` and builds compact prefix counts; it does
    not materialize every frame and never draws random numbers.

    Every tensor stored here is derived from ``segment_index`` and
    ``segment_allowed_mask``.  There is deliberately no separate checkpoint
    state or mutable counter to persist.
    """

    _EMPTY_MOTION_POLICIES = frozenset({"error", "exclude"})

    def __init__(
        self,
        segment_index: FixedLengthSegmentIndex,
        segment_allowed_mask: Sequence[bool] | torch.Tensor,
        *,
        empty_motion_policy: str = "error",
    ) -> None:
        if empty_motion_policy not in self._EMPTY_MOTION_POLICIES:
            raise ValueError(
                "empty_motion_policy must be one of "
                f"{sorted(self._EMPTY_MOTION_POLICIES)}, got {empty_motion_policy!r}."
            )

        self.segment_index = segment_index
        self.device = segment_index.device
        self.empty_motion_policy = empty_motion_policy

        allowed_mask = torch.as_tensor(segment_allowed_mask, device=self.device)
        if allowed_mask.dtype != torch.bool:
            raise ValueError("segment_allowed_mask must contain boolean values.")
        if allowed_mask.ndim != 1 or allowed_mask.numel() != segment_index.num_segments:
            raise ValueError(
                "segment_allowed_mask must be one-dimensional with exactly "
                f"{segment_index.num_segments} values."
            )
        self.segment_allowed_mask = allowed_mask.detach().clone()

        segment_motion_ids = segment_index.segment_motion_ids
        # The legacy sampler maps [0, 1) to integer frames 0..T-2.  Intersect
        # each half-open segment range with that domain.  In particular, a
        # one-frame tail at frame T-1 contributes zero eligible starts.
        legacy_end_frames = torch.minimum(
            segment_index.segment_end_frames,
            segment_index.motion_lengths[segment_motion_ids] - 1,
        )
        ungated_start_counts = (
            legacy_end_frames - segment_index.segment_start_frames
        ).clamp_min(0)
        self.segment_eligible_start_counts = torch.where(
            self.segment_allowed_mask,
            ungated_start_counts,
            torch.zeros_like(ungated_start_counts),
        )
        self.eligible_segment_mask = self.segment_eligible_start_counts > 0

        self.motion_eligible_start_counts = torch.zeros(
            segment_index.num_motions, dtype=torch.long, device=self.device
        )
        self.motion_eligible_start_counts.scatter_add_(
            0, segment_motion_ids, self.segment_eligible_start_counts
        )
        self.motion_eligible_segment_counts = torch.zeros(
            segment_index.num_motions, dtype=torch.long, device=self.device
        )
        self.motion_eligible_segment_counts.scatter_add_(
            0, segment_motion_ids, self.eligible_segment_mask.to(torch.long)
        )

        self.eligible_motion_mask = self.motion_eligible_start_counts > 0
        self.eligible_motion_ids = torch.where(self.eligible_motion_mask)[0]
        self.empty_motion_ids = torch.where(~self.eligible_motion_mask)[0]
        if empty_motion_policy == "error" and self.empty_motion_ids.numel() > 0:
            empty_ids = self.empty_motion_ids.detach().cpu().tolist()
            raise ValueError(
                "Every motion must have at least one quality-eligible legacy start frame in [0, T-2]; "
                f"empty motion IDs: {empty_ids}."
            )

        # A global ordinal identifies one eligible start without rejection.
        # Repeated prefix values correspond to rejected or zero-length ranges;
        # searchsorted(..., right=True) skips them in one vectorized operation.
        self.motion_eligible_start_offsets = torch.zeros(
            segment_index.num_motions + 1, dtype=torch.long, device=self.device
        )
        self.motion_eligible_start_offsets[1:] = torch.cumsum(
            self.motion_eligible_start_counts, dim=0
        )
        self.segment_eligible_start_prefix_ends = torch.cumsum(
            self.segment_eligible_start_counts, dim=0
        )
        self.segment_eligible_start_prefix_starts = (
            self.segment_eligible_start_prefix_ends - self.segment_eligible_start_counts
        )

    def map_uniform_samples(
        self,
        motion_ids: Sequence[int] | torch.Tensor,
        uniform_samples: Sequence[float] | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Map caller-provided ``[0, 1)`` samples to eligible start frames.

        Returns ``(start_frames, local_segment_ids, global_segment_ids)`` with
        the same shape as the inputs.  The operation is deterministic and
        vectorized; callers remain responsible for drawing both motion IDs and
        uniform samples.
        """

        motion_ids_tensor = torch.as_tensor(motion_ids, dtype=torch.long, device=self.device)
        uniform_samples_tensor = torch.as_tensor(uniform_samples, device=self.device)
        if not uniform_samples_tensor.is_floating_point():
            raise ValueError("uniform_samples must be floating-point values in [0, 1).")
        FixedLengthSegmentIndex._require_same_shape(
            motion_ids_tensor, uniform_samples_tensor, "motion_ids", "uniform_samples"
        )
        self.segment_index._validate_motion_ids(motion_ids_tensor)
        if uniform_samples_tensor.numel() and (
            not torch.all(torch.isfinite(uniform_samples_tensor))
            or torch.any(uniform_samples_tensor < 0.0)
            or torch.any(uniform_samples_tensor >= 1.0)
        ):
            raise ValueError("uniform_samples must be finite values in [0, 1).")

        original_shape = motion_ids_tensor.shape
        motion_ids_flat = motion_ids_tensor.reshape(-1)
        uniform_samples_flat = uniform_samples_tensor.reshape(-1)
        if motion_ids_flat.numel() == 0:
            empty = torch.empty(original_shape, dtype=torch.long, device=self.device)
            return empty, empty.clone(), empty.clone()

        selected_motion_counts = self.motion_eligible_start_counts[motion_ids_flat]
        if torch.any(selected_motion_counts == 0):
            empty_selected = torch.unique(motion_ids_flat[selected_motion_counts == 0])
            empty_ids = empty_selected.detach().cpu().tolist()
            raise ValueError(
                "Cannot map a uniform sample for a motion with no quality-eligible legacy start frame; "
                f"motion IDs: {empty_ids}."
            )

        # Keeping the multiplication in the caller's floating-point dtype
        # reproduces the legacy `(phase * (length - 1).float()).long()` mapping
        # when every segment is allowed.  The clamp only guards extreme
        # floating-point rounding at very large frame counts.
        local_start_ordinals = (
            uniform_samples_flat * selected_motion_counts.to(uniform_samples_flat.dtype)
        ).to(torch.long)
        local_start_ordinals = torch.minimum(local_start_ordinals, selected_motion_counts - 1)
        global_start_ordinals = (
            self.motion_eligible_start_offsets[motion_ids_flat] + local_start_ordinals
        )

        global_segment_ids = torch.searchsorted(
            self.segment_eligible_start_prefix_ends,
            global_start_ordinals,
            right=True,
        )
        start_frames = (
            self.segment_index.segment_start_frames[global_segment_ids]
            + global_start_ordinals
            - self.segment_eligible_start_prefix_starts[global_segment_ids]
        )
        local_segment_ids = self.segment_index.segment_local_ids[global_segment_ids]
        return (
            start_frames.reshape(original_shape),
            local_segment_ids.reshape(original_shape),
            global_segment_ids.reshape(original_shape),
        )

    def summary(self) -> dict[str, float | int]:
        """Return low-cardinality eligibility counts derived from the mask."""

        num_legacy_starts = int((self.segment_index.motion_lengths - 1).sum().item())
        num_eligible_starts = int(self.motion_eligible_start_counts.sum().item())
        num_eligible_motions = int(torch.count_nonzero(self.eligible_motion_mask).item())
        num_empty_motions = int(self.empty_motion_ids.numel())
        return {
            "num_motions": self.segment_index.num_motions,
            "num_segments": self.segment_index.num_segments,
            "num_allowed_segments": int(torch.count_nonzero(self.segment_allowed_mask).item()),
            "num_eligible_segments": int(torch.count_nonzero(self.eligible_segment_mask).item()),
            "num_eligible_motions": num_eligible_motions,
            "num_empty_motions": num_empty_motions,
            "num_excluded_motions": num_empty_motions,
            "num_legacy_start_frames": num_legacy_starts,
            "num_eligible_start_frames": num_eligible_starts,
            "eligible_motion_fraction": num_eligible_motions / self.segment_index.num_motions,
            "eligible_start_fraction": (
                float(num_eligible_starts) / num_legacy_starts if num_legacy_starts else 0.0
            ),
        }

    def eligible_motion_mask_sha256(self) -> str:
        """Return a stable identity hash for the runtime eligible-motion mask."""

        mask_bytes = bytes(
            int(value)
            for value in self.eligible_motion_mask.to(dtype=torch.uint8).detach().cpu().tolist()
        )
        return hashlib.sha256(mask_bytes).hexdigest()

    def identity_state(self) -> dict[str, int | str]:
        """Return runtime gate identity fields for checkpoint compatibility."""

        summary = self.summary()
        return {
            "empty_motion_policy": self.empty_motion_policy,
            "manifest_motion_count": int(summary["num_motions"]),
            "effective_motion_count": int(summary["num_eligible_motions"]),
            "excluded_motion_count": int(summary["num_excluded_motions"]),
            "eligible_motion_mask_sha256": self.eligible_motion_mask_sha256(),
        }


class SamplingStatistics:
    """Global, vectorized assignment counters shared by all environments."""

    def __init__(self, segment_index: FixedLengthSegmentIndex, *, pool_fingerprint: str | None = None) -> None:
        self.segment_index = segment_index
        self.device = segment_index.device
        self.pool_fingerprint = pool_fingerprint
        self.motion_sample_count = torch.zeros(segment_index.num_motions, dtype=torch.long, device=self.device)
        self.segment_sample_count = torch.zeros(segment_index.num_segments, dtype=torch.long, device=self.device)
        self.total_assignments = torch.zeros((), dtype=torch.long, device=self.device)
        self.invalid_probability_fallback_count = torch.zeros((), dtype=torch.long, device=self.device)

    def record_assignments(
        self, motion_ids: Sequence[int] | torch.Tensor, start_frames: Sequence[int] | torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Record a batch chosen by the legacy sampler and return its segment IDs."""

        motion_ids_tensor = torch.as_tensor(motion_ids, dtype=torch.long, device=self.device)
        start_frames_tensor = torch.as_tensor(start_frames, dtype=torch.long, device=self.device)
        if motion_ids_tensor.shape != start_frames_tensor.shape:
            raise ValueError("motion_ids and start_frames must have the same shape.")
        original_shape = motion_ids_tensor.shape
        motion_ids_flat = motion_ids_tensor.reshape(-1)
        start_frames_flat = start_frames_tensor.reshape(-1)
        local_ids, global_ids = self.segment_index.motion_frame_to_segment(motion_ids_flat, start_frames_flat)
        if motion_ids_flat.numel() == 0:
            return local_ids.reshape(original_shape), global_ids.reshape(original_shape)

        self.motion_sample_count += torch.bincount(
            motion_ids_flat, minlength=self.segment_index.num_motions
        ).to(torch.long)
        self.segment_sample_count += torch.bincount(
            global_ids, minlength=self.segment_index.num_segments
        ).to(torch.long)
        self.total_assignments += motion_ids_flat.numel()
        return local_ids.reshape(original_shape), global_ids.reshape(original_shape)

    def record_probability_fallback(self, count: int = 1) -> None:
        """Increment the shared invalid-probability fallback counter."""

        if count < 0:
            raise ValueError("Probability fallback count must be non-negative.")
        self.invalid_probability_fallback_count += int(count)

    def get_motion_sample_counts(self) -> torch.Tensor:
        return self.motion_sample_count.detach().clone()

    def get_segment_sample_counts(self) -> torch.Tensor:
        return self.segment_sample_count.detach().clone()

    def reset_statistics(self) -> None:
        self.motion_sample_count.zero_()
        self.segment_sample_count.zero_()
        self.total_assignments.zero_()
        self.invalid_probability_fallback_count.zero_()

    def summary(self) -> dict[str, float | int]:
        """Compute low-cardinality logging metrics on demand."""

        total = int(self.total_assignments.item())
        motion_fraction = float(self.motion_sample_count.max().item()) / total if total else 0.0
        segment_fraction = float(self.segment_sample_count.max().item()) / total if total else 0.0
        return {
            "total_assignments": total,
            "motion_coverage": float(torch.count_nonzero(self.motion_sample_count).item())
            / self.segment_index.num_motions,
            "segment_coverage": float(torch.count_nonzero(self.segment_sample_count).item())
            / self.segment_index.num_segments,
            "max_motion_sample_fraction": motion_fraction,
            "max_segment_sample_fraction": segment_fraction,
            "mean_motion_sample_count": float(self.motion_sample_count.to(torch.float64).mean().item()),
            "mean_segment_sample_count": float(self.segment_sample_count.to(torch.float64).mean().item()),
            "invalid_probability_fallbacks": int(self.invalid_probability_fallback_count.item()),
            "num_motions": self.segment_index.num_motions,
            "num_segments": self.segment_index.num_segments,
            "segment_length_seconds": self.segment_index.segment_length_seconds,
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": SAMPLING_STATE_VERSION,
            "pool_fingerprint": self.pool_fingerprint,
            "segment_index": self.segment_index.state_dict(),
            "motion_sample_count": self.motion_sample_count.detach().clone(),
            "segment_sample_count": self.segment_sample_count.detach().clone(),
            "total_assignments": self.total_assignments.detach().clone(),
            "invalid_probability_fallback_count": self.invalid_probability_fallback_count.detach().clone(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        required = {
            "version",
            "pool_fingerprint",
            "segment_index",
            "motion_sample_count",
            "segment_sample_count",
            "total_assignments",
            "invalid_probability_fallback_count",
        }
        missing = required.difference(state)
        if missing:
            raise ValueError(f"Sampling statistics state is missing fields: {sorted(missing)}")
        if int(state["version"]) != SAMPLING_STATE_VERSION:
            raise ValueError(
                f"Unsupported sampling state version {state['version']}; expected {SAMPLING_STATE_VERSION}."
            )
        saved_fingerprint = state["pool_fingerprint"]
        if self.pool_fingerprint is not None and saved_fingerprint != self.pool_fingerprint:
            raise ValueError("Checkpoint motion pool fingerprint does not match the current ordered motion pool.")
        self.segment_index.load_state_dict(state["segment_index"])

        motion_counts = torch.as_tensor(state["motion_sample_count"], dtype=torch.long, device=self.device)
        segment_counts = torch.as_tensor(state["segment_sample_count"], dtype=torch.long, device=self.device)
        if motion_counts.shape != self.motion_sample_count.shape:
            raise ValueError("Checkpoint motion_sample_count has the wrong shape.")
        if segment_counts.shape != self.segment_sample_count.shape:
            raise ValueError("Checkpoint segment_sample_count has the wrong shape.")
        total = torch.as_tensor(state["total_assignments"], dtype=torch.long, device=self.device).reshape(())
        fallback_count = torch.as_tensor(
            state["invalid_probability_fallback_count"], dtype=torch.long, device=self.device
        ).reshape(())
        if (
            torch.any(motion_counts < 0)
            or torch.any(segment_counts < 0)
            or total.item() < 0
            or fallback_count.item() < 0
        ):
            raise ValueError("Checkpoint sampling counters must be non-negative.")
        total_value = int(total.item())
        if int(motion_counts.sum().item()) != total_value or int(segment_counts.sum().item()) != total_value:
            raise ValueError("Checkpoint assignment counters are inconsistent with total_assignments.")

        self.motion_sample_count.copy_(motion_counts)
        self.segment_sample_count.copy_(segment_counts)
        self.total_assignments.copy_(total)
        self.invalid_probability_fallback_count.copy_(fallback_count)


class AssignmentTraceRecorder:
    """Write a bounded, deterministic CSV trace of assignment-start samples.

    The recorder is intentionally an observer: callers pass already-sampled
    tensors to it, and it performs only shape checks, host copies, and file I/O.
    It must never draw random numbers or alter the tensors used by training.
    """

    HEADER = (
        "assignment_index",
        "env_id",
        "motion_id",
        "start_frame",
        "local_segment_id",
        "global_segment_id",
        "pool_fingerprint",
        "run_label",
    )

    def __init__(
        self,
        output_path: str,
        max_entries: int,
        *,
        pool_fingerprint: str | None = None,
        run_label: str = "",
    ) -> None:
        if not output_path:
            raise ValueError("Assignment trace output_path must not be empty.")
        if int(max_entries) < 1:
            raise ValueError("Assignment trace max_entries must be at least 1.")
        self.output_path = os.path.abspath(os.path.normpath(os.path.expanduser(output_path)))
        self.max_entries = int(max_entries)
        self.pool_fingerprint = pool_fingerprint or ""
        self.run_label = run_label
        self.recorded_entries = 0
        self.reset()

    def reset(self) -> None:
        directory = os.path.dirname(self.output_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(self.output_path, "w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(self.HEADER)
        self.recorded_entries = 0

    def record_assignments(
        self,
        env_ids: Sequence[int] | torch.Tensor,
        motion_ids: Sequence[int] | torch.Tensor,
        start_frames: Sequence[int] | torch.Tensor,
        local_segment_ids: Sequence[int] | torch.Tensor,
        global_segment_ids: Sequence[int] | torch.Tensor,
    ) -> int:
        if self.recorded_entries >= self.max_entries:
            return 0

        tensors = [
            torch.as_tensor(values, dtype=torch.long).reshape(-1).detach().cpu()
            for values in (env_ids, motion_ids, start_frames, local_segment_ids, global_segment_ids)
        ]
        num_items = tensors[0].numel()
        if any(tensor.numel() != num_items for tensor in tensors[1:]):
            raise ValueError("Assignment trace tensors must have the same number of elements.")
        if num_items == 0:
            return 0

        num_to_write = min(num_items, self.max_entries - self.recorded_entries)
        rows = zip(
            range(self.recorded_entries, self.recorded_entries + num_to_write),
            *(tensor[:num_to_write].tolist() for tensor in tensors),
            strict=True,
        )
        with open(self.output_path, "a", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            for assignment_index, env_id, motion_id, start_frame, local_id, global_id in rows:
                writer.writerow(
                    (
                        assignment_index,
                        env_id,
                        motion_id,
                        start_frame,
                        local_id,
                        global_id,
                        self.pool_fingerprint,
                        self.run_label,
                    )
                )
        self.recorded_entries += num_to_write
        return num_to_write

    def summary(self) -> dict[str, int | str]:
        return {
            "output_path": self.output_path,
            "max_entries": self.max_entries,
            "recorded_entries": self.recorded_entries,
            "pool_fingerprint": self.pool_fingerprint,
        }


def normalize_and_validate_probabilities(
    probabilities: torch.Tensor,
    *,
    epsilon: float = 1.0e-8,
    expected_size: int | None = None,
    fallback: str = "uniform",
    fallback_statistics: SamplingStatistics | None = None,
) -> tuple[torch.Tensor, bool]:
    """Normalize a probability vector, returning whether fallback was used.

    Passing ``fallback_statistics`` makes uniform fallback and the shared
    counter update atomic.  The returned boolean remains available for callers
    that manage a different counter.  The legacy uniform sampler intentionally
    does not call this function.
    """

    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be finite and greater than zero.")
    if probabilities.ndim != 1 or probabilities.numel() == 0:
        raise ValueError("probabilities must be a non-empty one-dimensional tensor.")
    if expected_size is not None and probabilities.numel() != expected_size:
        raise ValueError(f"Expected {expected_size} probabilities, got {probabilities.numel()}.")
    if not probabilities.is_floating_point():
        probabilities = probabilities.to(torch.float32)

    invalid_reason = None
    if not torch.all(torch.isfinite(probabilities)):
        invalid_reason = "contains NaN or infinity"
    elif torch.any(probabilities < 0):
        invalid_reason = "contains negative values"
    else:
        total = probabilities.sum()
        if not torch.isfinite(total) or total.item() <= epsilon:
            invalid_reason = f"has total mass <= epsilon ({epsilon:g})"

    if invalid_reason is None:
        normalized = probabilities / probabilities.sum()
        normalized_sum = normalized.sum()
        if not torch.all(torch.isfinite(normalized)) or not torch.isclose(
            normalized_sum,
            torch.ones((), dtype=normalized.dtype, device=normalized.device),
            rtol=1.0e-5,
            atol=max(epsilon, 1.0e-7),
        ):
            invalid_reason = "could not be normalized to finite unit mass"
        else:
            return normalized, False

    if fallback != "uniform":
        if fallback == "raise":
            raise ValueError(f"Invalid probability vector: {invalid_reason}.")
        raise ValueError(f"Unsupported probability fallback '{fallback}'.")

    assert invalid_reason is not None
    if fallback_statistics is not None:
        fallback_statistics.record_probability_fallback()
    warning_count = _probability_warning_counts.get(invalid_reason, 0)
    if warning_count < _PROBABILITY_WARNING_LIMIT:
        warnings.warn(
            f"Invalid probability vector ({invalid_reason}); using a uniform fallback.",
            RuntimeWarning,
            stacklevel=2,
        )
    _probability_warning_counts[invalid_reason] = warning_count + 1
    return torch.full_like(probabilities, 1.0 / probabilities.numel()), True
