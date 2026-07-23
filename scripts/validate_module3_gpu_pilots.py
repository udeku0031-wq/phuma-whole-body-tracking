#!/usr/bin/env python3
"""Validate module-three GPU pilot artifacts.

This script is deliberately read-only.  It can compare assignment traces,
explain M6 quality-gate segment counts, inspect local checkpoint sidecars, and
optionally query W&B histories for the low-cardinality module-three metrics.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "source" / "whole_body_tracking"
DEFAULT_QUALITY = (
    PROJECT_ROOT
    / "outputs"
    / "module1_quality_pilot_random100_seed42_v1"
    / "segment_quality_metadata.npz"
)
TRACE_COLUMNS = (
    "assignment_index",
    "env_id",
    "motion_id",
    "start_frame",
    "local_segment_id",
    "global_segment_id",
)
METHOD_MODE_CODE = {
    "M2": 2,
    "M3": 3,
    "M4": 4,
    "M5": 5,
    "M6": 5,
    "GLOBAL_BIN_RAW_ERROR": 7,
}


def _load_source_module(relative_path: str, module_name: str):
    path = SOURCE_ROOT / "whole_body_tracking" / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _read_trace(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        missing = set(TRACE_COLUMNS).difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} is missing required trace columns: {sorted(missing)}")
        return [row for row in reader]


def compare_assignment_traces(left: Path, right: Path) -> dict[str, int]:
    left_rows = _read_trace(left)
    right_rows = _read_trace(right)
    result = {
        "left_rows": len(left_rows),
        "right_rows": len(right_rows),
        "row_count_mismatch": int(len(left_rows) != len(right_rows)),
    }
    for column in TRACE_COLUMNS:
        result[f"{column}_mismatches"] = 0
    for left_row, right_row in zip(left_rows, right_rows, strict=False):
        for column in TRACE_COLUMNS:
            result[f"{column}_mismatches"] += int(left_row[column] != right_row[column])
    result["total_mismatches"] = sum(result[f"{column}_mismatches"] for column in TRACE_COLUMNS)
    return result


def explain_m6_quality_gate(
    metadata_path: Path,
    *,
    include_borderline: bool = False,
    reject_statuses: Sequence[str] = ("reject",),
) -> dict[str, Any]:
    quality_module = _load_source_module("utils/quality_metadata.py", "_wbt_pilot_quality_metadata")
    sampling_module = _load_source_module("utils/sampling.py", "_wbt_pilot_sampling")
    SegmentQualityMetadata = quality_module.SegmentQualityMetadata
    FixedLengthSegmentIndex = sampling_module.FixedLengthSegmentIndex
    QualityGatedStartIndex = sampling_module.QualityGatedStartIndex

    metadata = SegmentQualityMetadata.load(metadata_path)
    index = FixedLengthSegmentIndex(
        metadata.motion_lengths,
        metadata.motion_fps,
        metadata.segment_length_seconds,
        device="cpu",
    )
    status_masks = {
        "pass": metadata.pass_mask,
        "borderline": metadata.borderline_mask,
        "reject": metadata.reject_mask,
    }
    unknown = sorted(set(reject_statuses).difference(status_masks))
    if unknown:
        raise ValueError(f"Unknown quality status in reject set: {unknown}")
    rejected = np.zeros(metadata.num_segments, dtype=bool)
    for status in reject_statuses:
        rejected |= status_masks[status]
    allowed = ~rejected
    if not include_borderline:
        allowed &= ~metadata.borderline_mask

    gate = QualityGatedStartIndex(index, allowed, empty_motion_policy="exclude")
    segment_motion_ids = index.segment_motion_ids.detach().cpu().numpy()
    legal_end = np.minimum(
        index.segment_end_frames.detach().cpu().numpy(),
        metadata.motion_lengths[segment_motion_ids] - 1,
    )
    has_legal_assignment_start = legal_end > index.segment_start_frames.detach().cpu().numpy()
    effective = allowed & has_legal_assignment_start
    empty_motion_ids = gate.empty_motion_ids.detach().cpu().numpy()
    in_empty_motion = np.isin(segment_motion_ids, empty_motion_ids)

    breakdown = {
        "total_segments": metadata.num_segments,
        "total_motions": metadata.num_motions,
        "quality_pass_segments": int(np.count_nonzero(metadata.pass_mask)),
        "quality_borderline_segments": int(np.count_nonzero(metadata.borderline_mask)),
        "quality_reject_segments": int(np.count_nonzero(rejected)),
        "allowed_segments_after_quality": int(np.count_nonzero(allowed)),
        "disallowed_segments_after_quality": int(np.count_nonzero(~allowed)),
        "segments_without_legal_assignment_start": int(np.count_nonzero(allowed & ~has_legal_assignment_start)),
        "reject_segments_without_legal_assignment_start": int(np.count_nonzero(rejected & ~has_legal_assignment_start)),
        "effective_segments": int(np.count_nonzero(effective)),
        "excluded_segments_total": int(metadata.num_segments - np.count_nonzero(effective)),
        "effective_motions": int(gate.eligible_motion_ids.numel()),
        "empty_motions": int(gate.empty_motion_ids.numel()),
        "empty_motion_ids_first20": [int(item) for item in empty_motion_ids[:20]],
        "segments_in_empty_motions": int(np.count_nonzero(in_empty_motion)),
        "effective_segments_in_empty_motions": int(np.count_nonzero(effective & in_empty_motion)),
        "identity_equation_ok": int(
            int(np.count_nonzero(~allowed)) + int(np.count_nonzero(allowed & ~has_legal_assignment_start))
            == int(metadata.num_segments - np.count_nonzero(effective))
        ),
    }
    return breakdown


def _torch_load(path: Path) -> Mapping[str, Any]:
    import torch

    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError(f"Checkpoint is not a mapping: {path}")
    return payload


def _tensor_to_numpy(value: Any) -> np.ndarray:
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _scalar_int(value: Any) -> int:
    array = _tensor_to_numpy(value)
    return int(array.reshape(()).item())


def _walk_numeric_arrays(value: Any, prefix: str = "") -> list[tuple[str, np.ndarray]]:
    arrays: list[tuple[str, np.ndarray]] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            arrays.extend(_walk_numeric_arrays(item, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            arrays.extend(_walk_numeric_arrays(item, f"{prefix}[{index}]"))
    else:
        try:
            array = _tensor_to_numpy(value)
        except Exception:
            return arrays
        if np.issubdtype(array.dtype, np.number):
            arrays.append((prefix, array))
    return arrays


def _segment_motion_ids_from_index(segment_index_state: Mapping[str, Any]) -> np.ndarray:
    motion_num_segments = _tensor_to_numpy(segment_index_state["motion_num_segments"]).astype(np.int64)
    return np.repeat(np.arange(motion_num_segments.size, dtype=np.int64), motion_num_segments)


def checkpoint_summary(path: Path) -> dict[str, Any]:
    payload = _torch_load(path)
    infos = payload.get("infos", {})
    if not isinstance(infos, Mapping) or "sampling_state" not in infos:
        raise ValueError(f"Checkpoint has no infos.sampling_state sidecar: {path}")
    sampling_state = infos["sampling_state"]
    if not isinstance(sampling_state, Mapping):
        raise ValueError("infos.sampling_state is not a mapping.")

    bad_arrays = []
    for name, array in _walk_numeric_arrays(sampling_state, "sampling_state"):
        if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
            bad_arrays.append(name)

    summary: dict[str, Any] = {
        "checkpoint": str(path),
        "bad_numeric_arrays": bad_arrays,
        "has_segment_index": int(isinstance(sampling_state.get("segment_index"), Mapping)),
        "has_assignment_statistics": int(isinstance(sampling_state.get("statistics"), Mapping)),
        "has_online_learning": int(isinstance(sampling_state.get("online_learning"), Mapping)),
    }
    segment_index = sampling_state.get("segment_index")
    if isinstance(segment_index, Mapping):
        summary["num_segments"] = int(segment_index.get("num_segments", -1))
        summary["num_motions"] = int(_tensor_to_numpy(segment_index["motion_lengths"]).size)
    assignment_stats = sampling_state.get("statistics")
    if isinstance(assignment_stats, Mapping):
        summary["assignment_total"] = _scalar_int(assignment_stats["total_assignments"])
    online = sampling_state.get("online_learning")
    if isinstance(online, Mapping):
        summary["online_current_iteration"] = int(online.get("current_iteration", -1))
        summary["online_completed_window_count"] = int(online.get("completed_window_count", -1))
        summary["online_last_formula_update_iteration"] = int(online.get("last_formula_update_iteration", -1))
        online_stats = online.get("statistics")
        if isinstance(online_stats, Mapping):
            summary["online_total_assignments"] = _scalar_int(online_stats["total_assignments"])
            summary["online_total_step_observations"] = _scalar_int(online_stats["total_step_observations"])
            summary["online_total_motion_episodes"] = _scalar_int(online_stats["total_motion_episodes"])
            summary["online_ema_update_count"] = _scalar_int(online_stats["ema_update_count"])
        sampler = online.get("sampler")
        summary["has_adaptive_sampler"] = int(isinstance(sampler, Mapping))
        if isinstance(sampler, Mapping):
            motion_probability = _tensor_to_numpy(sampler["motion_probability"]).astype(np.float64)
            segment_probability = _tensor_to_numpy(sampler["segment_probability"]).astype(np.float64)
            global_segment_probability = _tensor_to_numpy(sampler["global_segment_probability"]).astype(np.float64)
            summary.update(
                {
                    "sampler_motion_mode": str(sampler.get("motion_mode")),
                    "sampler_segment_mode": str(sampler.get("segment_mode")),
                    "sampler_current_iteration": int(sampler.get("current_iteration", -1)),
                    "sampler_last_probability_update_iteration": int(
                        sampler.get("last_probability_update_iteration", -1)
                    ),
                    "sampler_probability_update_count": int(sampler.get("probability_update_count", -1)),
                    "sampler_fallback_count": int(sampler.get("fallback_count", -1)),
                    "motion_probability_sum": float(motion_probability.sum()),
                    "motion_probability_max": float(motion_probability.max()),
                    "motion_probability_min_nonzero": float(motion_probability[motion_probability > 0.0].min()),
                    "global_segment_probability_sum": float(global_segment_probability.sum()),
                    "global_segment_probability_max": float(global_segment_probability.max()),
                    "generator_state_bytes": int(_tensor_to_numpy(sampler["generator_state"]).size),
                }
            )
            if isinstance(segment_index, Mapping):
                segment_motion_ids = _segment_motion_ids_from_index(segment_index)
                conditional_sums = np.bincount(
                    segment_motion_ids,
                    weights=segment_probability,
                    minlength=motion_probability.size,
                )
                nonempty = conditional_sums > 0.0
                summary["conditional_segment_probability_sum_error"] = float(
                    np.max(np.abs(conditional_sums[nonempty] - 1.0)) if np.any(nonempty) else 0.0
                )
                summary["conditional_segment_probability_max"] = float(segment_probability.max())
                summary["multi_segment_max_conditional_probability"] = _multi_segment_max_probability(
                    segment_probability,
                    segment_motion_ids,
                    segment_index,
                )
                summary["multi_segment_motion_count_p_gt_0_5"] = _multi_segment_count_above(
                    segment_probability,
                    segment_motion_ids,
                    segment_index,
                    threshold=0.5,
                )
                summary["multi_segment_motion_count_p_gt_0_8"] = _multi_segment_count_above(
                    segment_probability,
                    segment_motion_ids,
                    segment_index,
                    threshold=0.8,
                )
    return summary


def _multi_segment_max_probability(
    segment_probability: np.ndarray,
    segment_motion_ids: np.ndarray,
    segment_index_state: Mapping[str, Any],
) -> float:
    motion_num_segments = _tensor_to_numpy(segment_index_state["motion_num_segments"]).astype(np.int64)
    if not np.any(motion_num_segments > 1):
        return 0.0
    values = [
        float(segment_probability[segment_motion_ids == motion_id].max())
        for motion_id in np.where(motion_num_segments > 1)[0]
    ]
    return max(values) if values else 0.0


def _multi_segment_count_above(
    segment_probability: np.ndarray,
    segment_motion_ids: np.ndarray,
    segment_index_state: Mapping[str, Any],
    *,
    threshold: float,
) -> int:
    motion_num_segments = _tensor_to_numpy(segment_index_state["motion_num_segments"]).astype(np.int64)
    count = 0
    for motion_id in np.where(motion_num_segments > 1)[0]:
        count += int(segment_probability[segment_motion_ids == motion_id].max() > threshold)
    return count


def compare_checkpoint_progress(before: Path, after: Path) -> dict[str, Any]:
    left = checkpoint_summary(before)
    right = checkpoint_summary(after)
    result = {
        "before": str(before),
        "after": str(after),
        "online_iteration_increased": int(
            right.get("online_current_iteration", -1) > left.get("online_current_iteration", -1)
        ),
        "step_observations_nondecreasing": int(
            right.get("online_total_step_observations", -1) >= left.get("online_total_step_observations", -1)
        ),
        "ema_updates_nondecreasing": int(
            right.get("online_ema_update_count", -1) >= left.get("online_ema_update_count", -1)
        ),
        "probability_updates_nondecreasing": int(
            right.get("sampler_probability_update_count", -1) >= left.get("sampler_probability_update_count", -1)
        ),
        "has_rng_state_before": int(left.get("generator_state_bytes", 0) > 0),
        "has_rng_state_after": int(right.get("generator_state_bytes", 0) > 0),
    }
    return result


def _parse_wandb_spec(spec: str) -> tuple[str, str]:
    if "=" not in spec:
        raise ValueError("W&B run spec must be METHOD=entity/project/run_id.")
    method, path = spec.split("=", 1)
    method = method.strip().upper()
    if method not in METHOD_MODE_CODE and method != "STATS":
        raise ValueError(f"Unknown method in W&B run spec: {method}")
    if path.count("/") != 2:
        raise ValueError("W&B path must have the form entity/project/run_id.")
    return method, path


def _wandb_required_keys(method: str) -> list[str]:
    keys = [
        "online/segment_observation_count",
        "online/ema_update_count",
        "sampling/probability_sum_error",
    ]
    if method == "STATS":
        return ["online/segment_observation_count", "online/ema_update_count", "sampling/total_assignments"]
    keys.extend(
        [
            "sampling/mode",
            "sampling/probability_update_count",
            "sampling/max_motion_probability",
            "sampling/max_segment_probability",
            "sampling/fallback_count",
        ]
    )
    if method in {"M2", "M4"}:
        keys.extend(["error/motion_mean", "error/motion_p90"])
    if method in {"M3", "M4", "M5", "M6"}:
        keys.extend(["error/segment_mean", "error/segment_p90"])
    if method in {"M5", "M6"}:
        keys.extend(
            [
                "gap/bin_0_mu",
                "gap/bin_0_sigma",
                "gap/bin_0_valid_count",
                "gap/global_p90",
                "gap/motion_p90",
                "gap/local_p90",
                "gap/clipped_ratio",
            ]
        )
    if method == "M6":
        keys.extend(
            [
                "quality/reject_start_assignment_count",
                "quality/excluded_motion_count",
                "quality/reject_rollout_exposure_ratio",
            ]
        )
    return keys


def validate_wandb_run(
    spec: str,
    *,
    motion_cap: float,
    segment_cap: float,
    probability_sum_tolerance: float,
    require_adaptive_update: bool,
) -> dict[str, Any]:
    method, path = _parse_wandb_spec(spec)
    import wandb

    api = wandb.Api()
    run = api.run(path)
    keys = _wandb_required_keys(method)
    history = run.history(keys=keys, pandas=True)
    result: dict[str, Any] = {"method": method, "run": path, "rows": int(len(history)), "missing_keys": []}
    failures: list[str] = []
    for key in keys:
        if key not in history:
            result["missing_keys"].append(key)
            continue
        values = history[key].dropna()
        if values.empty:
            failures.append(f"{key} has no non-null values")
            continue
        numeric = values.astype(float)
        if not np.isfinite(numeric.to_numpy()).all():
            failures.append(f"{key} contains NaN/Inf")
    if result["missing_keys"]:
        failures.append(f"missing required keys: {result['missing_keys']}")

    def column_max(key: str, default: float = 0.0) -> float:
        if key not in history:
            return default
        values = history[key].dropna()
        return float(values.astype(float).max()) if not values.empty else default

    if method != "STATS":
        if require_adaptive_update and column_max("sampling/probability_update_count") <= 0.0:
            failures.append("probability_update_count never became positive")
        if column_max("sampling/probability_sum_error") > probability_sum_tolerance:
            failures.append("probability_sum_error exceeded tolerance")
        if column_max("sampling/max_motion_probability") > motion_cap + 1.0e-9:
            failures.append("max_motion_probability exceeded configured cap")
        if column_max("sampling/max_segment_probability") > segment_cap + 1.0e-9:
            failures.append("max_segment_probability exceeded configured cap")
        if "sampling/mode" in history and not history["sampling/mode"].dropna().empty:
            observed_mode = int(round(float(history["sampling/mode"].dropna().iloc[-1])))
            expected_mode = METHOD_MODE_CODE[method]
            if observed_mode != expected_mode:
                failures.append(f"sampling/mode={observed_mode}, expected {expected_mode}")
    if method == "M6" and column_max("quality/reject_start_assignment_count") != 0.0:
        failures.append("M6 assigned at least one reject start")
    result["failures"] = failures
    result["ok"] = int(not failures)
    return result


def _print_payload(title: str, payload: Mapping[str, Any], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps({title: payload}, indent=2, sort_keys=True))
        return
    print(f"[{title}]")
    for key, value in payload.items():
        print(f"{key}={value}")


def _assert_expected(name: str, actual: int, expected: int | None, failures: list[str]) -> None:
    if expected is not None and actual != expected:
        failures.append(f"{name} expected {expected}, got {actual}")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-left", type=Path)
    parser.add_argument("--trace-right", type=Path)
    parser.add_argument("--quality-metadata", type=Path, default=DEFAULT_QUALITY)
    parser.add_argument("--explain-m6-quality", action="store_true")
    parser.add_argument(
        "--quality-include-borderline",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Explain M6 quality gate with borderline segments eligible. Default false for the mainline strict gate.",
    )
    parser.add_argument("--checkpoint", type=Path, action="append", default=[])
    parser.add_argument("--compare-checkpoints", type=Path, nargs=2, action="append", default=[])
    parser.add_argument(
        "--wandb-run",
        action="append",
        default=[],
        metavar="METHOD=ENTITY/PROJECT/RUN_ID",
        help="Validate one W&B run history, for example M5=my-ent/proj/runid.",
    )
    parser.add_argument("--motion-probability-cap", type=float, default=0.02)
    parser.add_argument("--segment-probability-cap", type=float, default=1.0)
    parser.add_argument("--probability-sum-tolerance", type=float, default=1.0e-6)
    parser.add_argument("--no-require-adaptive-update", action="store_true")
    parser.add_argument("--expect-total-segments", type=int)
    parser.add_argument("--expect-effective-segments", type=int)
    parser.add_argument("--expect-effective-motions", type=int)
    parser.add_argument("--expect-quality-reject-segments", type=int)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    failures: list[str] = []
    if args.trace_left or args.trace_right:
        if not args.trace_left or not args.trace_right:
            raise ValueError("Both --trace-left and --trace-right are required for trace comparison.")
        trace = compare_assignment_traces(args.trace_left, args.trace_right)
        _print_payload("trace", trace, as_json=args.json)
        if trace["row_count_mismatch"] or trace["total_mismatches"]:
            failures.append("assignment traces differ")
    if args.explain_m6_quality:
        quality = explain_m6_quality_gate(
            args.quality_metadata,
            include_borderline=args.quality_include_borderline,
        )
        _print_payload("m6_quality", quality, as_json=args.json)
        _assert_expected("total_segments", int(quality["total_segments"]), args.expect_total_segments, failures)
        _assert_expected(
            "effective_segments",
            int(quality["effective_segments"]),
            args.expect_effective_segments,
            failures,
        )
        _assert_expected(
            "effective_motions",
            int(quality["effective_motions"]),
            args.expect_effective_motions,
            failures,
        )
        _assert_expected(
            "quality_reject_segments",
            int(quality["quality_reject_segments"]),
            args.expect_quality_reject_segments,
            failures,
        )
        if not quality["identity_equation_ok"]:
            failures.append("M6 quality/start exclusion equation did not close")
    for checkpoint in args.checkpoint:
        summary = checkpoint_summary(checkpoint)
        _print_payload(f"checkpoint:{checkpoint.name}", summary, as_json=args.json)
        if summary["bad_numeric_arrays"]:
            failures.append(f"{checkpoint} contains NaN/Inf in sampling state")
        if summary.get("has_online_learning") and summary.get("has_adaptive_sampler"):
            if abs(float(summary.get("motion_probability_sum", 0.0)) - 1.0) > args.probability_sum_tolerance:
                failures.append(f"{checkpoint} motion probabilities are not normalized")
            if float(summary.get("motion_probability_max", 0.0)) > args.motion_probability_cap + 1.0e-9:
                failures.append(f"{checkpoint} motion probability cap exceeded")
            if (
                float(summary.get("conditional_segment_probability_max", 0.0))
                > args.segment_probability_cap + 1.0e-9
            ):
                failures.append(f"{checkpoint} segment probability cap exceeded")
            if (
                float(summary.get("conditional_segment_probability_sum_error", 0.0))
                > args.probability_sum_tolerance
            ):
                failures.append(f"{checkpoint} conditional segment probabilities are not normalized")
    for before, after in args.compare_checkpoints:
        progress = compare_checkpoint_progress(before, after)
        _print_payload(f"resume:{before.name}->{after.name}", progress, as_json=args.json)
        for key, value in progress.items():
            if key.endswith(("increased", "nondecreasing", "before", "after")):
                continue
            if key.startswith("has_rng_state") and not value:
                failures.append(f"resume check failed: {key}=0")
        if not progress["online_iteration_increased"]:
            failures.append("resume did not advance online iteration")
        if not progress["step_observations_nondecreasing"]:
            failures.append("resume step observations regressed")
        if not progress["ema_updates_nondecreasing"]:
            failures.append("resume EMA count regressed")
        if not progress["probability_updates_nondecreasing"]:
            failures.append("resume probability update count regressed")
    for spec in args.wandb_run:
        wandb_result = validate_wandb_run(
            spec,
            motion_cap=args.motion_probability_cap,
            segment_cap=args.segment_probability_cap,
            probability_sum_tolerance=args.probability_sum_tolerance,
            require_adaptive_update=not args.no_require_adaptive_update,
        )
        _print_payload(f"wandb:{wandb_result['method']}", wandb_result, as_json=args.json)
        failures.extend(str(item) for item in wandb_result["failures"])

    if failures:
        print("[FAIL]")
        for failure in failures:
            print(failure)
        return 1
    print("[OK] module-three pilot artifact checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
