"""Create a single W&B run by stitching history from multiple source runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any
from urllib.parse import urlparse


DEFAULT_METRIC_PREFIXES = (
    "Episode_Reward/",
    "Episode_Termination/",
    "Loss/",
    "Metrics/",
)
INTERNAL_KEYS = {"_step", "_timestamp", "_runtime"}


@dataclass(frozen=True)
class RunRef:
    entity: str
    project: str
    run_id: str

    @property
    def path(self) -> str:
        return f"{self.entity}/{self.project}/{self.run_id}"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stitch W&B scalar histories from resumed training runs.")
    parser.add_argument("--entity", required=True, help="Default W&B entity/team.")
    parser.add_argument("--project", required=True, help="Source W&B project.")
    parser.add_argument(
        "--source-runs",
        "--source_runs",
        required=True,
        help=(
            "Comma-separated source run ids, entity/project/run paths, or W&B run URLs. "
            "Order matters, e.g. phuma15 first and phuma16 second."
        ),
    )
    parser.add_argument("--output-project", "--output_project", default=None, help="Target W&B project. Defaults to --project.")
    parser.add_argument("--output-name", "--output_name", default="baseline_direct_random6000_stitched")
    parser.add_argument("--output-run-id", "--output_run_id", default=None, help="Stable W&B run id for the stitched run.")
    parser.add_argument("--group", default="direct_random6000_baseline")
    parser.add_argument(
        "--tags",
        default="baseline,direct_random6000,stitched",
        help="Comma-separated tags for the stitched W&B run.",
    )
    parser.add_argument(
        "--metric-prefixes",
        "--metric_prefixes",
        default=",".join(DEFAULT_METRIC_PREFIXES),
        help="Comma-separated metric prefixes to stitch. Use empty string with --metrics for explicit keys.",
    )
    parser.add_argument(
        "--metrics",
        default=None,
        help="Optional comma-separated exact metric keys. If omitted, numeric keys matching --metric-prefixes are used.",
    )
    parser.add_argument(
        "--step-mode",
        "--step_mode",
        choices=("auto", "preserve", "continue"),
        default="auto",
        help=(
            "preserve keeps source steps; continue appends every later run after the previous one; "
            "auto preserves if steps already continue and offsets only when they overlap/reset."
        ),
    )
    parser.add_argument("--step-key", "--step_key", default="_step", help="Source history key used as x-axis.")
    parser.add_argument("--page-size", "--page_size", type=int, default=10000)
    parser.add_argument("--local-output", "--local_output", default="results/wandb_stitched/baseline_direct_random6000_history.csv")
    parser.add_argument("--metadata-output", "--metadata_output", default="results/wandb_stitched/baseline_direct_random6000_metadata.json")
    parser.add_argument("--wandb-mode", "--wandb_mode", default="online", choices=("online", "offline", "disabled"))
    parser.add_argument("--resume", default="never", choices=("never", "allow", "must"))
    parser.add_argument("--dry-run", "--dry_run", action="store_true")
    return parser


def split_csv(text: str | None) -> list[str]:
    if text is None:
        return []
    return [item.strip() for item in text.split(",") if item.strip()]


def parse_run_ref(text: str, default_entity: str, default_project: str) -> RunRef:
    """Parse a run id, entity/project/run path, or W&B URL."""
    item = text.strip()
    parsed = urlparse(item)
    if parsed.scheme and parsed.netloc:
        parts = [part for part in parsed.path.split("/") if part]
        if "runs" in parts:
            index = parts.index("runs")
            if index >= 2 and index + 1 < len(parts):
                return RunRef(entity=parts[index - 2], project=parts[index - 1], run_id=parts[index + 1])
        raise ValueError(f"Could not parse W&B run URL: {text}")

    parts = [part for part in item.split("/") if part]
    if len(parts) == 3:
        return RunRef(entity=parts[0], project=parts[1], run_id=parts[2])
    if len(parts) == 1:
        return RunRef(entity=default_entity, project=default_project, run_id=parts[0])
    raise ValueError(f"Expected run id, entity/project/run, or W&B run URL: {text}")


def is_number(value: Any) -> bool:
    if isinstance(value, bool) or value is None:
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def metric_allowed(key: str, explicit_metrics: set[str] | None, prefixes: tuple[str, ...]) -> bool:
    if key in INTERNAL_KEYS or key.startswith("_wandb"):
        return False
    if explicit_metrics is not None:
        return key in explicit_metrics
    return any(key.startswith(prefix) for prefix in prefixes)


def infer_metrics(
    source_histories: list[list[dict[str, Any]]],
    explicit_metrics: set[str] | None,
    prefixes: tuple[str, ...],
) -> list[str]:
    metrics: set[str] = set()
    for history in source_histories:
        for row in history:
            for key, value in row.items():
                if metric_allowed(key, explicit_metrics, prefixes) and is_number(value):
                    metrics.add(key)
    if explicit_metrics is not None:
        missing = sorted(explicit_metrics.difference(metrics))
        if missing:
            print(f"[WARN]: Explicit metric(s) were not found with numeric values: {missing}")
    return sorted(metrics)


def row_step(row: dict[str, Any], step_key: str) -> int | None:
    value = row.get(step_key)
    if value is None and step_key != "_step":
        value = row.get("_step")
    if not is_number(value):
        return None
    return int(float(value))


def positive_step_delta(steps: list[int]) -> int:
    deltas = [right - left for left, right in zip(steps, steps[1:]) if right > left]
    if not deltas:
        return 1
    return max(1, int(round(median(deltas))))


def stitch_histories(
    source_refs: list[RunRef],
    source_histories: list[list[dict[str, Any]]],
    *,
    metrics: list[str],
    step_key: str,
    step_mode: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_step: dict[int, dict[str, Any]] = {}
    summaries: list[dict[str, Any]] = []
    previous_last: int | None = None

    for source_index, (run_ref, history) in enumerate(zip(source_refs, source_histories)):
        stepped_rows = [(row_step(row, step_key), row) for row in history]
        stepped_rows = [(step, row) for step, row in stepped_rows if step is not None]
        stepped_rows.sort(key=lambda item: item[0])
        metric_rows = [
            (step, row)
            for step, row in stepped_rows
            if any(metric in row and is_number(row[metric]) for metric in metrics)
        ]
        raw_steps = [int(step) for step, _ in metric_rows]
        if not raw_steps:
            summaries.append({"run": run_ref.path, "rows": 0, "logged_rows": 0, "offset": 0})
            continue

        delta = positive_step_delta(raw_steps)
        offset = 0
        if previous_last is not None:
            next_step = previous_last + delta
            if step_mode == "continue":
                offset = next_step - raw_steps[0]
            elif step_mode == "auto" and raw_steps[-1] <= previous_last:
                offset = next_step - raw_steps[0]

        logged_rows = 0
        for raw_step, row in metric_rows:
            payload = {
                metric: float(row[metric])
                for metric in metrics
                if metric in row and is_number(row[metric])
            }
            stitched_step = int(raw_step + offset)
            existing = by_step.setdefault(
                stitched_step,
                {
                    "_step": stitched_step,
                    "stitch/source_run": run_ref.path,
                    "stitch/source_run_index": source_index,
                    "stitch/source_step": raw_step,
                },
            )
            existing.update(payload)
            existing["stitch/source_run"] = run_ref.path
            existing["stitch/source_run_index"] = source_index
            existing["stitch/source_step"] = raw_step
            logged_rows += 1

        source_last = raw_steps[-1] + offset
        previous_last = source_last if previous_last is None else max(previous_last, source_last)
        summaries.append(
            {
                "run": run_ref.path,
                "rows": len(history),
                "numeric_history_rows": len(stepped_rows),
                "metric_history_rows": len(metric_rows),
                "logged_rows": logged_rows,
                "raw_first_step": raw_steps[0],
                "raw_last_step": raw_steps[-1],
                "offset": offset,
                "stitched_first_step": raw_steps[0] + offset,
                "stitched_last_step": raw_steps[-1] + offset,
            }
        )

    return [by_step[step] for step in sorted(by_step)], summaries


def write_csv(path: Path, rows: list[dict[str, Any]], metrics: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["_step", "stitch/source_run", "stitch/source_run_index", "stitch/source_step", *metrics]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fetch_histories(source_refs: list[RunRef], page_size: int) -> tuple[list[Any], list[list[dict[str, Any]]]]:
    import wandb

    api = wandb.Api()
    runs = []
    histories = []
    for run_ref in source_refs:
        print(f"[INFO]: Downloading W&B history: {run_ref.path}", flush=True)
        run = api.run(run_ref.path)
        history = list(run.scan_history(page_size=page_size))
        runs.append(run)
        histories.append(history)
        print(f"[INFO]:   rows={len(history)} name={run.name!r} state={run.state!r}", flush=True)
    return runs, histories


def upload_stitched_run(args: argparse.Namespace, rows: list[dict[str, Any]], metrics: list[str], metadata: dict[str, Any]) -> None:
    import wandb

    target_project = args.output_project or args.project
    tags = split_csv(args.tags)
    os.environ.setdefault("WANDB_DISABLE_CODE", "true")
    run = wandb.init(
        entity=args.entity,
        project=target_project,
        name=args.output_name,
        id=args.output_run_id,
        resume=args.resume,
        group=args.group,
        job_type="curve_stitch",
        tags=tags,
        mode=args.wandb_mode,
        config=metadata,
        save_code=False,
    )
    try:
        for metric in metrics:
            wandb.define_metric(metric)
        for row in rows:
            step = int(row["_step"])
            payload = {key: value for key, value in row.items() if key != "_step"}
            wandb.log(payload, step=step)
    finally:
        run.finish()


def main() -> int:
    parser = _parser()
    args = parser.parse_args()

    source_refs = [parse_run_ref(item, args.entity, args.project) for item in split_csv(args.source_runs)]
    if len(source_refs) < 2:
        parser.error("At least two --source-runs are expected for stitching.")

    explicit_metrics = set(split_csv(args.metrics)) if args.metrics else None
    metric_prefixes = tuple(split_csv(args.metric_prefixes))
    output_project = args.output_project or args.project

    runs, histories = fetch_histories(source_refs, args.page_size)
    metrics = infer_metrics(histories, explicit_metrics, metric_prefixes)
    if not metrics:
        raise SystemExit("No numeric metrics matched. Use --metrics or adjust --metric-prefixes.")

    rows, source_summaries = stitch_histories(
        source_refs,
        histories,
        metrics=metrics,
        step_key=args.step_key,
        step_mode=args.step_mode,
    )
    if not rows:
        raise SystemExit("No rows remained after stitching.")

    metadata = {
        "source_runs": [
            {
                "path": ref.path,
                "name": getattr(run, "name", ""),
                "state": getattr(run, "state", ""),
                "url": getattr(run, "url", ""),
            }
            for ref, run in zip(source_refs, runs)
        ],
        "source_summaries": source_summaries,
        "metrics": metrics,
        "step_key": args.step_key,
        "step_mode": args.step_mode,
        "output_project": output_project,
        "output_name": args.output_name,
    }

    local_output = Path(args.local_output)
    metadata_output = Path(args.metadata_output)
    write_csv(local_output, rows, metrics)
    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    metadata_output.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")

    print(f"[INFO]: matched_metrics={len(metrics)}")
    print(f"[INFO]: stitched_rows={len(rows)}")
    print(f"[INFO]: local_output={local_output}")
    print(f"[INFO]: metadata_output={metadata_output}")
    for summary in source_summaries:
        print(f"[INFO]: source_summary={summary}")

    if args.dry_run or args.wandb_mode == "disabled":
        print("[INFO]: Dry run / disabled mode; not uploading stitched W&B run.")
        return 0

    upload_stitched_run(args, rows, metrics, metadata)
    print(f"[INFO]: Uploaded stitched run to {args.entity}/{output_project}: {args.output_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
