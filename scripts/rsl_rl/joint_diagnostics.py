"""Joint-specific diagnostic helpers for deterministic WBT evaluation.

This module intentionally avoids Isaac imports.  It only consumes tensors and
metadata supplied by the evaluator, then writes opt-in diagnostic artifacts.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch


def infer_joint_group(joint_name: str) -> str:
    """Infer a coarse anatomical group from an actual joint name."""

    name = joint_name.lower()
    if "hip" in name:
        return "legs_hip"
    if "knee" in name:
        return "legs_knee"
    if "ankle" in name:
        return "legs_ankle"
    if "shoulder" in name:
        return "arms_shoulder"
    if "elbow" in name:
        return "arms_elbow"
    if "wrist" in name:
        return "arms_wrist"
    if "waist" in name or "torso" in name:
        return "waist"
    return "other"


def joint_mapping_rows(
    joint_names: Sequence[str],
    *,
    reference_joint_names: Sequence[str] | None = None,
    action_joint_names: Sequence[str] | None = None,
    evaluator_joint_names: Sequence[str] | None = None,
) -> list[dict[str, object]]:
    """Build the explicit joint-order mapping used by diagnostics."""

    names = [str(name) for name in joint_names]
    reference = [str(name) for name in (reference_joint_names or names)]
    action = [str(name) for name in (action_joint_names or names)]
    evaluator = [str(name) for name in (evaluator_joint_names or names)]
    expected = len(names)
    for label, values in (
        ("reference_joint_names", reference),
        ("action_joint_names", action),
        ("evaluator_joint_names", evaluator),
    ):
        if len(values) != expected:
            raise ValueError(f"{label} must contain {expected} values, got {len(values)}.")

    rows: list[dict[str, object]] = []
    for index, name in enumerate(names):
        rows.append(
            {
                "joint_index": index,
                "joint_name": name,
                "joint_group": infer_joint_group(name),
                "reference_joint_name": reference[index],
                "robot_joint_name": name,
                "action_joint_name": action[index],
                "evaluator_joint_name": evaluator[index],
                "reference_matches_robot": int(reference[index] == name),
                "action_matches_robot": int(action[index] == name),
                "evaluator_matches_robot": int(evaluator[index] == name),
            }
        )
    return rows


def mapping_hash(rows: Sequence[dict[str, object]]) -> str:
    text = "\n".join(
        f"{row['joint_index']},{row['joint_name']},{row['reference_joint_name']},"
        f"{row['robot_joint_name']},{row['action_joint_name']},{row['evaluator_joint_name']}"
        for row in rows
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_csv(path: str | Path, rows: Sequence[dict[str, object]], columns: Sequence[str]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_suffix(destination.suffix + ".tmp")
    with tmp_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp_path, destination)


def write_joint_mapping_csv(path: str | Path, rows: Sequence[dict[str, object]]) -> None:
    write_csv(
        path,
        rows,
        (
            "joint_index",
            "joint_name",
            "joint_group",
            "reference_joint_name",
            "robot_joint_name",
            "action_joint_name",
            "evaluator_joint_name",
            "reference_matches_robot",
            "action_matches_robot",
            "evaluator_matches_robot",
        ),
    )


def validate_joint_error_shape(joint_error: torch.Tensor, joint_count: int) -> None:
    if joint_error.ndim != 2 or joint_error.shape[1] != joint_count:
        raise ValueError(f"joint_error must have shape (N, {joint_count}), got {tuple(joint_error.shape)}.")


def aggregate_l2_max_abs_error(joint_error: torch.Tensor, aggregate_l2: torch.Tensor) -> float:
    validate_joint_error_shape(joint_error, joint_error.shape[1])
    vector_l2 = torch.linalg.vector_norm(joint_error.to(torch.float64), dim=-1)
    aggregate = torch.as_tensor(aggregate_l2, dtype=torch.float64, device=vector_l2.device)
    if aggregate.shape != vector_l2.shape:
        raise ValueError("aggregate_l2 must contain one value per joint-error row.")
    return float(torch.max(torch.abs(vector_l2 - aggregate)).item()) if vector_l2.numel() else 0.0


def _safe_ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    return numerator / denominator.clamp_min(torch.finfo(torch.float64).tiny)


def _dispersion_from_rms(rms: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = rms.mean(dim=1)
    maximum = rms.max(dim=1).values
    k = max(int(math.ceil(rms.shape[1] * 0.20)), 1)
    top_mean = torch.topk(rms, k=k, dim=1).values.mean(dim=1)
    std = rms.std(dim=1, unbiased=False)
    cv = torch.where(mean > 0.0, std / mean.clamp_min(torch.finfo(torch.float64).tiny), torch.zeros_like(mean))
    return mean, maximum, top_mean, cv


@dataclass(frozen=True)
class SegmentLayout:
    segment_motion_ids: torch.Tensor
    segment_local_ids: torch.Tensor
    segment_start_frames: torch.Tensor
    segment_end_frames: torch.Tensor

    @property
    def num_segments(self) -> int:
        return int(self.segment_motion_ids.numel())


class JointDiagnosticsAccumulator:
    """Accumulate per-motion and per-segment joint errors during evaluation."""

    def __init__(
        self,
        *,
        motion_paths: Sequence[str],
        categories: Sequence[str],
        source_groups: Sequence[str],
        motion_lengths: Sequence[int],
        joint_names: Sequence[str],
        device: str | torch.device,
        segment_layout: SegmentLayout | None = None,
    ) -> None:
        if not (len(motion_paths) == len(categories) == len(source_groups) == len(motion_lengths)):
            raise ValueError("Motion metadata arrays must have the same length.")
        self.motion_paths = [str(path) for path in motion_paths]
        self.categories = [str(item) for item in categories]
        self.source_groups = [str(item) for item in source_groups]
        self.motion_lengths = [int(item) for item in motion_lengths]
        self.joint_names = [str(name) for name in joint_names]
        self.joint_groups = [infer_joint_group(name) for name in self.joint_names]
        self.device = torch.device(device)
        self.segment_layout = segment_layout

        num_motions = len(self.motion_paths)
        joint_count = len(self.joint_names)
        self.motion_abs_sum = torch.zeros(num_motions, joint_count, dtype=torch.float64, device=self.device)
        self.motion_sq_sum = torch.zeros_like(self.motion_abs_sum)
        self.motion_count = torch.zeros(num_motions, dtype=torch.long, device=self.device)
        self.motion_l2_sum = torch.zeros(num_motions, dtype=torch.float64, device=self.device)
        self.motion_vector_l2_sum = torch.zeros(num_motions, dtype=torch.float64, device=self.device)

        if segment_layout is None:
            self.segment_abs_sum = None
            self.segment_sq_sum = None
            self.segment_count = None
            self.segment_l2_sum = None
        else:
            num_segments = segment_layout.num_segments
            self.segment_abs_sum = torch.zeros(num_segments, joint_count, dtype=torch.float64, device=self.device)
            self.segment_sq_sum = torch.zeros_like(self.segment_abs_sum)
            self.segment_count = torch.zeros(num_segments, dtype=torch.long, device=self.device)
            self.segment_l2_sum = torch.zeros(num_segments, dtype=torch.float64, device=self.device)

        self.max_l2_consistency_error = 0.0
        self.total_observations = 0

    @property
    def joint_count(self) -> int:
        return len(self.joint_names)

    def record_step(
        self,
        *,
        motion_ids: torch.Tensor,
        segment_ids: torch.Tensor | None,
        joint_error: torch.Tensor,
        aggregate_l2: torch.Tensor,
    ) -> None:
        motion = torch.as_tensor(motion_ids, dtype=torch.long, device=self.device)
        error = torch.as_tensor(joint_error, dtype=torch.float64, device=self.device)
        aggregate = torch.as_tensor(aggregate_l2, dtype=torch.float64, device=self.device)
        validate_joint_error_shape(error, self.joint_count)
        if motion.ndim != 1 or motion.shape[0] != error.shape[0]:
            raise ValueError("motion_ids must contain one value per joint-error row.")
        if torch.any(motion < 0) or torch.any(motion >= len(self.motion_paths)):
            raise ValueError("motion_ids are outside the diagnostic motion range.")

        vector_l2 = torch.linalg.vector_norm(error, dim=-1)
        if aggregate.shape != vector_l2.shape:
            raise ValueError("aggregate_l2 must contain one value per joint-error row.")
        if vector_l2.numel():
            self.max_l2_consistency_error = max(
                self.max_l2_consistency_error,
                float(torch.max(torch.abs(vector_l2 - aggregate)).item()),
            )

        abs_error = torch.abs(error)
        sq_error = torch.square(error)
        self.motion_abs_sum.index_add_(0, motion, abs_error)
        self.motion_sq_sum.index_add_(0, motion, sq_error)
        self.motion_count.index_add_(0, motion, torch.ones_like(motion))
        self.motion_l2_sum.index_add_(0, motion, aggregate)
        self.motion_vector_l2_sum.index_add_(0, motion, vector_l2)

        if segment_ids is not None and self.segment_layout is not None:
            segment = torch.as_tensor(segment_ids, dtype=torch.long, device=self.device)
            if segment.shape != motion.shape:
                raise ValueError("segment_ids must contain one value per joint-error row.")
            if torch.any(segment < 0) or torch.any(segment >= self.segment_layout.num_segments):
                raise ValueError("segment_ids are outside the diagnostic segment range.")
            assert self.segment_abs_sum is not None
            assert self.segment_sq_sum is not None
            assert self.segment_count is not None
            assert self.segment_l2_sum is not None
            self.segment_abs_sum.index_add_(0, segment, abs_error)
            self.segment_sq_sum.index_add_(0, segment, sq_error)
            self.segment_count.index_add_(0, segment, torch.ones_like(segment))
            self.segment_l2_sum.index_add_(0, segment, aggregate)

        self.total_observations += int(error.shape[0])

    def _per_motion_rows(self) -> list[dict[str, object]]:
        count = self.motion_count.to(torch.float64).unsqueeze(1)
        mean_abs = _safe_ratio(self.motion_abs_sum, count)
        rms = torch.sqrt(_safe_ratio(self.motion_sq_sum, count))
        mean_joint, max_joint, top_joint, cv = _dispersion_from_rms(rms)
        l2_mean = _safe_ratio(self.motion_l2_sum, self.motion_count.to(torch.float64))
        vector_l2_mean = _safe_ratio(self.motion_vector_l2_sum, self.motion_count.to(torch.float64))

        rows: list[dict[str, object]] = []
        for index in range(len(self.motion_paths)):
            row: dict[str, object] = {
                "motion_id": index,
                "motion_path": self.motion_paths[index],
                "category": self.categories[index],
                "source_group": self.source_groups[index],
                "num_frames": self.motion_lengths[index],
                "metric_count": int(self.motion_count[index].item()),
                "aggregate_l2_mean": f"{float(l2_mean[index].item()):.9f}",
                "vector_l2_mean": f"{float(vector_l2_mean[index].item()):.9f}",
                "mean_joint_rms_error": f"{float(mean_joint[index].item()):.9f}",
                "max_joint_rms_error": f"{float(max_joint[index].item()):.9f}",
                "top20pct_joint_rms_error": f"{float(top_joint[index].item()):.9f}",
                "joint_error_cv": f"{float(cv[index].item()):.9f}",
            }
            for joint_index, joint_name in enumerate(self.joint_names):
                prefix = f"joint_{joint_index:02d}"
                row[f"{prefix}_name"] = joint_name
                row[f"{prefix}_mean_abs_error"] = f"{float(mean_abs[index, joint_index].item()):.9f}"
                row[f"{prefix}_rms_error"] = f"{float(rms[index, joint_index].item()):.9f}"
            rows.append(row)
        return rows

    def _per_joint_rows(self) -> list[dict[str, object]]:
        total_count = torch.clamp(self.motion_count.sum().to(torch.float64), min=1.0)
        abs_sum = self.motion_abs_sum.sum(dim=0)
        sq_sum = self.motion_sq_sum.sum(dim=0)
        rows: list[dict[str, object]] = []
        for index, name in enumerate(self.joint_names):
            mean_abs = abs_sum[index] / total_count
            rms = torch.sqrt(sq_sum[index] / total_count)
            rows.append(
                {
                    "joint_index": index,
                    "joint_name": name,
                    "joint_group": self.joint_groups[index],
                    "mean_abs_error": f"{float(mean_abs.item()):.9f}",
                    "rms_error": f"{float(rms.item()):.9f}",
                    "abs_error_sum": f"{float(abs_sum[index].item()):.9f}",
                    "squared_error_sum": f"{float(sq_sum[index].item()):.9f}",
                    "sample_count": int(total_count.item()),
                }
            )
        return rows

    def _per_segment_rows(self) -> list[dict[str, object]]:
        if self.segment_layout is None:
            return []
        assert self.segment_abs_sum is not None
        assert self.segment_sq_sum is not None
        assert self.segment_count is not None
        assert self.segment_l2_sum is not None

        count = self.segment_count.to(torch.float64).unsqueeze(1)
        rms = torch.sqrt(_safe_ratio(self.segment_sq_sum, count))
        mean_joint, max_joint, top_joint, cv = _dispersion_from_rms(rms)
        l2_mean = _safe_ratio(self.segment_l2_sum, self.segment_count.to(torch.float64))
        layout = self.segment_layout

        rows: list[dict[str, object]] = []
        for segment_id in range(layout.num_segments):
            motion_id = int(layout.segment_motion_ids[segment_id].item())
            row: dict[str, object] = {
                "global_segment_id": segment_id,
                "motion_id": motion_id,
                "local_segment_id": int(layout.segment_local_ids[segment_id].item()),
                "start_frame": int(layout.segment_start_frames[segment_id].item()),
                "end_frame_exclusive": int(layout.segment_end_frames[segment_id].item()),
                "motion_path": self.motion_paths[motion_id],
                "category": self.categories[motion_id],
                "source_group": self.source_groups[motion_id],
                "metric_count": int(self.segment_count[segment_id].item()),
                "aggregate_l2_mean": f"{float(l2_mean[segment_id].item()):.9f}",
                "mean_joint_rms_error": f"{float(mean_joint[segment_id].item()):.9f}",
                "max_joint_rms_error": f"{float(max_joint[segment_id].item()):.9f}",
                "top20pct_joint_rms_error": f"{float(top_joint[segment_id].item()):.9f}",
                "joint_error_cv": f"{float(cv[segment_id].item()):.9f}",
            }
            for joint_index in range(self.joint_count):
                row[f"joint_{joint_index:02d}_rms_error"] = f"{float(rms[segment_id, joint_index].item()):.9f}"
            rows.append(row)
        return rows

    def write_outputs(
        self,
        output_dir: str | Path,
        *,
        mapping_rows: Sequence[dict[str, object]],
        evaluation_config: dict[str, object],
    ) -> dict[str, object]:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        write_joint_mapping_csv(out / "joint_mapping.csv", mapping_rows)

        per_motion_rows = self._per_motion_rows()
        per_joint_rows = self._per_joint_rows()
        per_segment_rows = self._per_segment_rows()

        motion_columns = list(per_motion_rows[0].keys()) if per_motion_rows else []
        joint_columns = (
            "joint_index",
            "joint_name",
            "joint_group",
            "mean_abs_error",
            "rms_error",
            "abs_error_sum",
            "squared_error_sum",
            "sample_count",
        )
        segment_columns = list(per_segment_rows[0].keys()) if per_segment_rows else []
        if motion_columns:
            write_csv(out / "per_motion_joint_diagnostics.csv", per_motion_rows, motion_columns)
        write_csv(out / "per_joint_summary.csv", per_joint_rows, joint_columns)
        if segment_columns:
            write_csv(out / "per_segment_joint_diagnostics.csv", per_segment_rows, segment_columns)

        summary = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "joint_count": self.joint_count,
            "num_motions": len(self.motion_paths),
            "num_segments": self.segment_layout.num_segments if self.segment_layout is not None else 0,
            "total_observations": self.total_observations,
            "max_l2_consistency_error": self.max_l2_consistency_error,
            "joint_mapping_hash": mapping_hash(mapping_rows),
            "evaluation_config": evaluation_config,
        }
        (out / "joint_diagnostic_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
        return summary
