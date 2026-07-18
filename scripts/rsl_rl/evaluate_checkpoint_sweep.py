"""Run deterministic validation sweeps over multiple checkpoints."""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
from pathlib import Path

import evaluation_utils as eval_utils


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a fixed validation manifest over multiple checkpoints.")
    parser.add_argument("--task", required=True, help="Isaac Lab task name.")
    parser.add_argument("--motion-file", "--motion_file", dest="motion_file", required=True, help="Validation manifest.")
    parser.add_argument(
        "--run-dir",
        "--run_dir",
        dest="run_dir",
        required=True,
        help="RSL-RL run directory, quoted glob, or comma-separated run directories.",
    )
    parser.add_argument(
        "--checkpoints",
        default="10000,20000,30000,final",
        help="Comma-separated targets, e.g. 10000,20000,30000,final.",
    )
    parser.add_argument("--checkpoint-pattern", default="model_*.pt", help="Glob pattern within --run-dir.")
    parser.add_argument("--num-envs", "--num_envs", dest="num_envs", type=int, default=16)
    parser.add_argument("--output-root", "--output_root", dest="output_root", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--disable-randomization", "--disable_randomization", dest="disable_randomization", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Resume individual checkpoint evaluations.")
    parser.add_argument("--wandb", action="store_true", help="Log comparison metrics to a separate W&B evaluation run.")
    parser.add_argument("--dry-run", "--dry_run", dest="dry_run", action="store_true")
    return parser


def _expand_run_dirs(run_dir_arg: str) -> list[Path]:
    run_dirs: list[Path] = []
    for item in [part.strip() for part in run_dir_arg.split(",") if part.strip()]:
        if any(char in item for char in "*?[]"):
            run_dirs.extend(Path(path) for path in sorted(glob.glob(item)))
        else:
            run_dirs.append(Path(item))

    unique: list[Path] = []
    seen: set[Path] = set()
    for path in run_dirs:
        resolved = path.resolve()
        if resolved not in seen:
            unique.append(path)
            seen.add(resolved)
    return unique


def _checkpoint_output_dir(output_root: Path, checkpoint: Path) -> Path:
    return output_root / checkpoint.stem


def _evaluate_command(args: argparse.Namespace, checkpoint: Path, output_dir: Path) -> list[str]:
    command = [
        sys.executable,
        "scripts/rsl_rl/evaluate.py",
        "--task",
        args.task,
        "--motion_file",
        args.motion_file,
        "--num_envs",
        str(args.num_envs),
        "--headless",
        "--load_run",
        checkpoint.parent.resolve().name,
        "--checkpoint",
        checkpoint.name,
        "--output_dir",
        str(output_dir),
        "--seed",
        str(args.seed),
    ]
    if args.deterministic:
        command.append("--deterministic")
    if args.disable_randomization:
        command.append("--disable_randomization")
    if args.resume:
        command.append("--resume")
    return command


def _read_summary(output_dir: Path) -> dict[str, object]:
    summary_path = output_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Expected summary after evaluation: {summary_path}")
    return json.loads(summary_path.read_text())


def _write_comparison(output_root: Path, rows: list[dict[str, object]], args: argparse.Namespace) -> dict[str, object]:
    output_root.mkdir(parents=True, exist_ok=True)
    eval_utils.write_csv(output_root / "checkpoint_comparison.csv", rows, eval_utils.COMPARISON_COLUMNS)
    eval_utils.atomic_write_text(
        output_root / "checkpoint_comparison.json",
        json.dumps(rows, indent=2, ensure_ascii=False) + "\n",
    )
    best = eval_utils.choose_best_checkpoint(rows)
    best_payload = {
        "selected_checkpoint": best["checkpoint"],
        "selected_load_run": best.get("load_run", ""),
        "selected_iteration": int(best["iteration"]),
        "selection_manifest": args.motion_file,
        "selection_metric": "macro_success_rate",
        "tie_break_rules": [
            "macro_success_rate higher is better",
            "if macro_success_rate differs by less than 0.002, compare micro_success_rate",
            "if still close, compare mean_completion_ratio",
            "if still close, choose lower mean_body_position_error_m",
            "if still close, choose the earlier checkpoint",
        ],
        "all_checkpoint_results": rows,
    }
    eval_utils.atomic_write_text(
        output_root / "best_checkpoint.json",
        json.dumps(best_payload, indent=2, ensure_ascii=False) + "\n",
    )
    return best_payload


def _maybe_log_wandb(rows: list[dict[str, object]], args: argparse.Namespace) -> None:
    if not args.wandb:
        return
    try:
        import wandb

        run = wandb.init(
            project="whole_body_tracking_phuma",
            job_type="evaluation",
            group="direct_random6000_seed42_env3072",
            name="direct_random6000_checkpoint_sweep",
            config={
                "motion_file": args.motion_file,
                "run_dir": args.run_dir,
                "seed": args.seed,
                "num_envs": args.num_envs,
            },
        )
        for row in rows:
            step = int(row["iteration"])
            wandb.log(
                {
                    "Validation/micro_success_rate": float(row["micro_success_rate"]),
                    "Validation/macro_success_rate": float(row["macro_success_rate"]),
                    "Validation/completion_ratio": float(row["mean_completion_ratio"]),
                    "Validation/body_position_error_m": float(row["mean_body_position_error_m"]),
                    "Validation/joint_position_error_l2_rad": float(row["mean_joint_position_error_l2_rad"]),
                    "Validation/joint_position_error_rms_rad": float(row["mean_joint_position_error_rms_rad"]),
                },
                step=step,
            )
        run.finish()
    except Exception as exc:
        print(f"[WARN]: W&B logging failed, local results are preserved: {exc}", flush=True)


def main() -> int:
    parser = _parser()
    args = parser.parse_args()

    motion_path = Path(args.motion_file)
    if eval_utils.is_final_test_manifest(motion_path):
        parser.error("Checkpoint selection cannot use the Test manifest.")

    run_dirs = _expand_run_dirs(args.run_dir)
    if not run_dirs:
        parser.error(f"No run directories matched: {args.run_dir}")
    available = []
    for run_dir in run_dirs:
        available.extend(eval_utils.list_checkpoints(run_dir, args.checkpoint_pattern))
    selected = eval_utils.select_checkpoints(available, args.checkpoints)
    output_root = Path(args.output_root)

    print("[INFO]: run_dir(s)=")
    for run_dir in run_dirs:
        print(f"  - {run_dir}")
    print(f"[INFO]: available_checkpoints={len(available)}")
    print("[INFO]: selected checkpoints:")
    for checkpoint in selected:
        print(f"  - {checkpoint.parent.name}/{checkpoint.name}")

    if args.dry_run:
        print("[DRY-RUN]: commands that would be executed:")
        for checkpoint in selected:
            output_dir = _checkpoint_output_dir(output_root, checkpoint)
            print(" ".join(_evaluate_command(args, checkpoint, output_dir)))
        return 0

    rows: list[dict[str, object]] = []
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("LD_LIBRARY_PATH", None)
    for checkpoint in selected:
        output_dir = _checkpoint_output_dir(output_root, checkpoint)
        command = _evaluate_command(args, checkpoint, output_dir)
        print(f"[INFO]: Evaluating {checkpoint.parent.name}/{checkpoint.name}")
        subprocess.run(command, check=True, env=env)
        summary = _read_summary(output_dir)
        rows.append(eval_utils.comparison_row_from_summary(summary))
        best_payload = _write_comparison(output_root, rows, args)
        print(f"[INFO]: Current best checkpoint: {best_payload['selected_checkpoint']}")

    best_payload = _write_comparison(output_root, rows, args)
    _maybe_log_wandb(rows, args)
    print(f"[INFO]: Best checkpoint: {best_payload['selected_checkpoint']}")
    print(f"[INFO]: Wrote {output_root / 'checkpoint_comparison.csv'}")
    print(f"[INFO]: Wrote {output_root / 'best_checkpoint.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
