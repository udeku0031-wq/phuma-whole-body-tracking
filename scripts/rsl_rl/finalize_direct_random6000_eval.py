"""Finish Direct Random-6000 validation selection and final test evaluation."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import evaluation_utils as eval_utils


DEFAULT_RUN_DIR = "logs/rsl_rl/g1_flat/2026-07-17_01-31-21_direct_random6000_seed42_env3072_resume18000"
DEFAULT_VALIDATION = "PHUMA_wbt_motions/manifests/splits_v1/validation.txt"
DEFAULT_TEST = "PHUMA_wbt_motions/manifests/splits_v1/test.txt"
DEFAULT_OUTPUT_ROOT = "results/direct_random6000"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run/resume full Validation for Direct Random-6000 candidate checkpoints, "
            "freeze the best Validation checkpoint, then run the final Test once."
        )
    )
    parser.add_argument("--task", default="Tracking-Flat-G1-v0")
    parser.add_argument("--run-dir", "--run_dir", dest="run_dir", default=DEFAULT_RUN_DIR)
    parser.add_argument(
        "--checkpoints",
        default="model_20000.pt,model_33999.pt",
        help="Comma-separated checkpoint filenames, numeric iterations, or final/latest.",
    )
    parser.add_argument("--validation-manifest", "--validation_manifest", dest="validation_manifest", default=DEFAULT_VALIDATION)
    parser.add_argument("--test-manifest", "--test_manifest", dest="test_manifest", default=DEFAULT_TEST)
    parser.add_argument("--output-root", "--output_root", dest="output_root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--num-envs", "--num_envs", dest="num_envs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress-interval", "--progress_interval", dest="progress_interval", type=int, default=50)
    parser.add_argument("--episode-length-s", "--episode_length_s", dest="episode_length_s", type=float, default=60.0)
    parser.add_argument("--device", default=None, help="Optional device forwarded to evaluate.py, e.g. cuda:0.")
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--disable-randomization", "--disable_randomization", dest="disable_randomization", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force-validation", "--force_validation", dest="force_validation", action="store_true")
    parser.add_argument("--force-test", "--force_test", dest="force_test", action="store_true")
    parser.add_argument("--skip-test", "--skip_test", dest="skip_test", action="store_true")
    parser.add_argument("--confirm-final-test", "--confirm_final_test", dest="confirm_final_test", action="store_true")
    parser.add_argument("--dry-run", "--dry_run", dest="dry_run", action="store_true")
    return parser


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _split_items(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _select_checkpoints(run_dir: Path, specs: str) -> list[Path]:
    available = eval_utils.list_checkpoints(run_dir)
    if not available:
        raise FileNotFoundError(f"No model_*.pt checkpoints were found in {run_dir}")

    selected: list[Path] = []
    seen: set[Path] = set()
    for spec in _split_items(specs):
        if spec.endswith(".pt"):
            checkpoint = run_dir / spec
            if not checkpoint.exists():
                raise FileNotFoundError(f"Requested checkpoint does not exist: {checkpoint}")
        else:
            checkpoint = eval_utils.select_checkpoints(available, [spec])[0]
        resolved = checkpoint.resolve()
        if resolved not in seen:
            selected.append(checkpoint)
            seen.add(resolved)

    if not selected:
        raise ValueError("No checkpoints were selected.")
    return selected


def _output_dir(output_root: Path, split: str, checkpoint: Path) -> Path:
    return output_root / split / checkpoint.stem


def _read_summary(output_dir: Path) -> dict[str, object] | None:
    summary_path = output_dir / "summary.json"
    if not summary_path.exists():
        return None
    return json.loads(summary_path.read_text())


def _summary_status(output_dir: Path, manifest: Path, checkpoint: Path) -> tuple[str, dict[str, object] | None]:
    summary = _read_summary(output_dir)
    if summary is None:
        return "missing", None
    if eval_utils.summary_is_complete(summary, expected_manifest=manifest, expected_checkpoint=checkpoint.name):
        return "complete", summary
    integrity = summary.get("manifest_integrity", {})
    observed = integrity.get("observed_count", summary.get("num_motions", "?")) if isinstance(integrity, dict) else "?"
    expected = integrity.get("expected_count", "?") if isinstance(integrity, dict) else "?"
    return f"incomplete ({observed}/{expected})", summary


def _evaluate_command(
    args: argparse.Namespace,
    *,
    manifest: Path,
    checkpoint: Path,
    output_dir: Path,
    confirm_final_test: bool = False,
) -> list[str]:
    command = [
        sys.executable,
        "scripts/rsl_rl/evaluate.py",
        "--task",
        args.task,
        "--motion_file",
        str(manifest),
        "--num_envs",
        str(args.num_envs),
        "--headless",
        "--load_run",
        checkpoint.parent.name,
        "--checkpoint",
        checkpoint.name,
        "--output_dir",
        str(output_dir),
        "--seed",
        str(args.seed),
        "--progress_interval",
        str(args.progress_interval),
        "--episode_length_s",
        str(args.episode_length_s),
    ]
    if args.device:
        command.extend(["--device", args.device])
    if args.deterministic:
        command.append("--deterministic")
    if args.disable_randomization:
        command.append("--disable_randomization")
    if args.resume:
        command.append("--resume")
    if confirm_final_test:
        command.append("--confirm_final_test")
    return command


def _display_command(command: list[str]) -> str:
    return "env -u PYTHONPATH -u LD_LIBRARY_PATH " + shlex.join(["python", *command[1:]])


def _run(command: list[str], project_root: Path) -> None:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("LD_LIBRARY_PATH", None)
    subprocess.run(command, cwd=project_root, env=env, check=True)


def _validation_rows(
    checkpoints: list[Path],
    output_root: Path,
    validation_manifest: Path,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for checkpoint in checkpoints:
        output_dir = _output_dir(output_root, "validation_full", checkpoint)
        summary = _read_summary(output_dir)
        if summary is None:
            raise RuntimeError(f"Validation summary is missing: {output_dir / 'summary.json'}")
        if not eval_utils.summary_is_complete(summary, expected_manifest=validation_manifest, expected_checkpoint=checkpoint.name):
            raise RuntimeError(f"Validation summary is incomplete: {output_dir / 'summary.json'}")
        rows.append(eval_utils.comparison_row_from_summary(summary))
    return rows


def _freeze_best(
    *,
    project_root: Path,
    output_root: Path,
    validation_manifest: Path,
    test_manifest: Path,
    checkpoints: list[Path],
    rows: list[dict[str, object]],
) -> dict[str, object]:
    output_root.mkdir(parents=True, exist_ok=True)
    validation_root = output_root / "validation_full"
    validation_root.mkdir(parents=True, exist_ok=True)
    eval_utils.write_csv(validation_root / "checkpoint_comparison.csv", rows, eval_utils.COMPARISON_COLUMNS)
    eval_utils.atomic_write_text(
        validation_root / "checkpoint_comparison.json",
        json.dumps(rows, indent=2, ensure_ascii=False) + "\n",
    )

    best = eval_utils.choose_best_checkpoint(rows)
    checkpoint_by_name = {checkpoint.name: checkpoint for checkpoint in checkpoints}
    best_checkpoint = checkpoint_by_name[str(best["checkpoint"])]
    payload = {
        "selected_checkpoint": best_checkpoint.name,
        "selected_load_run": best_checkpoint.parent.name,
        "selected_iteration": eval_utils.parse_checkpoint_iteration(best_checkpoint),
        "selected_checkpoint_path": str(best_checkpoint.resolve()),
        "selected_checkpoint_sha256": eval_utils.sha256_file(best_checkpoint),
        "selection_source": "full_validation",
        "selection_manifest": str(validation_manifest),
        "selection_manifest_sha256": eval_utils.sha256_file(validation_manifest),
        "test_manifest": str(test_manifest),
        "test_manifest_sha256": eval_utils.sha256_file(test_manifest),
        "test_not_used_for_selection": True,
        "selection_metric": "macro_success_rate",
        "tie_break_rules": [
            "macro_success_rate higher is better",
            "if macro_success_rate differs by less than 0.002, compare micro_success_rate",
            "if still close, compare mean_completion_ratio",
            "if still close, choose lower mean_body_position_error_m",
            "if still close, choose the earlier checkpoint",
        ],
        "candidate_checkpoints": [checkpoint.name for checkpoint in checkpoints],
        "all_checkpoint_results": rows,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": eval_utils.git_commit(project_root),
    }
    eval_utils.atomic_write_text(
        output_root / "frozen_best_checkpoint.json",
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
    )
    eval_utils.atomic_write_text(
        validation_root / "frozen_best_checkpoint.json",
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
    )
    return payload


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    project_root = _project_root()
    run_dir = (project_root / args.run_dir).resolve() if not Path(args.run_dir).is_absolute() else Path(args.run_dir)
    validation_manifest = (project_root / args.validation_manifest).resolve() if not Path(args.validation_manifest).is_absolute() else Path(args.validation_manifest)
    test_manifest = (project_root / args.test_manifest).resolve() if not Path(args.test_manifest).is_absolute() else Path(args.test_manifest)
    output_root = (project_root / args.output_root).resolve() if not Path(args.output_root).is_absolute() else Path(args.output_root)

    if not validation_manifest.exists():
        parser.error(f"Validation manifest does not exist: {validation_manifest}")
    if not test_manifest.exists():
        parser.error(f"Test manifest does not exist: {test_manifest}")
    if not run_dir.exists():
        parser.error(f"Run directory does not exist: {run_dir}")

    checkpoints = _select_checkpoints(run_dir, args.checkpoints)
    print(f"[INFO]: run_dir={run_dir}")
    print(f"[INFO]: validation_manifest={validation_manifest}")
    print(f"[INFO]: test_manifest={test_manifest}")
    print("[INFO]: candidate checkpoints:")
    for checkpoint in checkpoints:
        print(f"  - {checkpoint.name}")

    validation_commands: list[list[str]] = []
    for checkpoint in checkpoints:
        output_dir = _output_dir(output_root, "validation_full", checkpoint)
        status, _ = _summary_status(output_dir, validation_manifest, checkpoint)
        print(f"[INFO]: validation {checkpoint.name}: {status} -> {output_dir}")
        if args.force_validation or status != "complete":
            validation_commands.append(
                _evaluate_command(args, manifest=validation_manifest, checkpoint=checkpoint, output_dir=output_dir)
            )

    if args.dry_run:
        if validation_commands:
            print("[DRY-RUN]: validation commands:")
            for command in validation_commands:
                print(_display_command(command))
        else:
            rows = _validation_rows(checkpoints, output_root, validation_manifest)
            best = eval_utils.choose_best_checkpoint(rows)
            print(f"[DRY-RUN]: validation is complete; selected checkpoint would be {best['checkpoint']}")
            if not args.skip_test:
                best_checkpoint = {checkpoint.name: checkpoint for checkpoint in checkpoints}[str(best["checkpoint"])]
                test_output_dir = _output_dir(output_root, "test_final", best_checkpoint)
                status, _ = _summary_status(test_output_dir, test_manifest, best_checkpoint)
                print(f"[DRY-RUN]: test {best_checkpoint.name}: {status} -> {test_output_dir}")
                if args.force_test or status != "complete":
                    command = _evaluate_command(
                        args,
                        manifest=test_manifest,
                        checkpoint=best_checkpoint,
                        output_dir=test_output_dir,
                        confirm_final_test=True,
                    )
                    print("[DRY-RUN]: final test command:")
                    print(_display_command(command))
        return 0

    for command in validation_commands:
        print(f"[INFO]: running validation: {_display_command(command)}", flush=True)
        _run(command, project_root)

    rows = _validation_rows(checkpoints, output_root, validation_manifest)
    frozen = _freeze_best(
        project_root=project_root,
        output_root=output_root,
        validation_manifest=validation_manifest,
        test_manifest=test_manifest,
        checkpoints=checkpoints,
        rows=rows,
    )
    print(f"[INFO]: frozen best checkpoint: {frozen['selected_checkpoint']}")
    print(f"[INFO]: wrote {output_root / 'frozen_best_checkpoint.json'}")

    if args.skip_test:
        print("[INFO]: --skip-test supplied; final Test was not run.")
        return 0
    if not args.confirm_final_test:
        raise SystemExit("Final Test requires --confirm-final-test after the checkpoint is frozen.")

    best_checkpoint = {checkpoint.name: checkpoint for checkpoint in checkpoints}[str(frozen["selected_checkpoint"])]
    test_output_dir = _output_dir(output_root, "test_final", best_checkpoint)
    test_status, _ = _summary_status(test_output_dir, test_manifest, best_checkpoint)
    print(f"[INFO]: test {best_checkpoint.name}: {test_status} -> {test_output_dir}")
    if args.force_test or test_status != "complete":
        command = _evaluate_command(
            args,
            manifest=test_manifest,
            checkpoint=best_checkpoint,
            output_dir=test_output_dir,
            confirm_final_test=True,
        )
        print(f"[INFO]: running final test: {_display_command(command)}", flush=True)
        _run(command, project_root)

    test_status, _ = _summary_status(test_output_dir, test_manifest, best_checkpoint)
    if test_status != "complete":
        raise RuntimeError(f"Final Test did not complete cleanly: {test_status}")
    print(f"[INFO]: final Test complete: {test_output_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
