#!/usr/bin/env python3
"""Launch reproducible GPU pilots for module-three learning-gap sampling.

The script is intentionally a thin orchestrator around the real training
entrypoint.  It builds the same Hydra overrides used by the module-three docs,
adds short debug warmup settings when requested, records optional assignment
traces, and can run a checkpoint/resume pilot without manually hunting through
``logs/rsl_rl``.
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "outputs"
    / "module2_difficulty_pilot_random100_seed42_v1"
    / "normalized_manifest.txt"
)
DEFAULT_DIFFICULTY = (
    PROJECT_ROOT
    / "outputs"
    / "module2_difficulty_pilot_random100_seed42_v1"
    / "segment_difficulty_metadata.npz"
)
DEFAULT_QUALITY = (
    PROJECT_ROOT
    / "outputs"
    / "module1_quality_pilot_random100_seed42_v1"
    / "segment_quality_metadata.npz"
)
DEFAULT_PROJECT = "whole_body_tracking_module3_pilot"


@dataclass(frozen=True)
class PilotMethod:
    key: str
    method_name: str
    motion_mode: str
    segment_mode: str
    quality: bool = False
    difficulty: bool = False
    online_enabled: bool = True
    statistics_enabled: bool = True


METHODS: dict[str, PilotMethod] = {
    "m0_trace": PilotMethod(
        "m0_trace",
        "M0",
        "uniform",
        "uniform",
        online_enabled=False,
        statistics_enabled=False,
    ),
    "stats": PilotMethod("stats", "M0", "uniform", "uniform"),
    "m2": PilotMethod("m2", "M2", "raw_error", "uniform"),
    "m3": PilotMethod("m3", "M3", "uniform", "raw_error"),
    "m4": PilotMethod("m4", "M4", "raw_error", "raw_error"),
    "m5": PilotMethod("m5", "M5", "learning_gap", "relative_learning_gap", difficulty=True),
    "m6": PilotMethod(
        "m6",
        "M6",
        "learning_gap",
        "relative_learning_gap",
        quality=True,
        difficulty=True,
    ),
}

DEFAULT_METHODS = ("stats", "m2", "m3", "m4", "m5", "m6")
TRACE_METHODS = ("m0_trace", "stats")
RESUME_METHODS = ("m5",)


def _csv_items(value: str) -> tuple[str, ...]:
    items = tuple(item.strip().lower() for item in value.split(",") if item.strip())
    if not items:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated item.")
    unknown = sorted(set(items).difference(METHODS))
    if unknown:
        raise argparse.ArgumentTypeError(f"Unknown method(s): {unknown}. Known: {sorted(METHODS)}")
    return items


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


def _require_file(path: Path, label: str, *, skip: bool) -> None:
    if skip:
        return
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")


def _clean_env(args: argparse.Namespace) -> dict[str, str]:
    env = dict(os.environ)
    if not args.keep_pythonpath:
        env.pop("PYTHONPATH", None)
    if not args.keep_ld_library_path:
        env.pop("LD_LIBRARY_PATH", None)
    env["WBT_DISABLE_ONNX_ON_SAVE"] = "1"
    if args.wandb_mode != "inherit":
        env["WANDB_MODE"] = args.wandb_mode
    return env


def _cuda_check(args: argparse.Namespace) -> None:
    if args.dry_run or args.skip_cuda_check:
        return
    command = [
        str(args.python),
        "-c",
        "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))",
    ]
    result = subprocess.run(command, cwd=PROJECT_ROOT, env=_clean_env(args), check=False, text=True, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"CUDA check failed:\n{result.stdout}\n{result.stderr}")
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not lines or lines[0] != "True":
        raise RuntimeError("CUDA is not available to the selected Python interpreter.")
    print(f"[INFO] CUDA device: {lines[1] if len(lines) > 1 else 'unknown'}")


def _phase_defaults(args: argparse.Namespace, suite: str) -> tuple[int, int, int, int, int, int]:
    if suite == "smoke":
        return (
            args.smoke_iterations,
            args.formal_warmup_iterations,
            args.formal_probability_update_interval,
            args.formal_min_segment_observations,
            args.formal_min_motion_episodes,
            args.smoke_save_interval,
        )
    if suite == "debug":
        return (
            args.debug_iterations,
            args.debug_warmup_iterations,
            args.debug_probability_update_interval,
            args.debug_min_segment_observations,
            args.debug_min_motion_episodes,
            args.debug_save_interval,
        )
    if suite == "trace":
        return (
            args.trace_iterations,
            args.formal_warmup_iterations,
            args.formal_probability_update_interval,
            args.formal_min_segment_observations,
            args.formal_min_motion_episodes,
            args.trace_save_interval,
        )
    raise ValueError(f"Unsupported suite for phase defaults: {suite}")


def _run_name(method: PilotMethod, suite: str, iterations: int) -> str:
    token = "stats" if method.key == "stats" else method.key
    return f"module3_{token}_random100_seed42_{suite}{iterations}"


def _run_id(run_name: str) -> str:
    return run_name.replace("_", "-")


def _common_overrides(
    *,
    args: argparse.Namespace,
    method: PilotMethod,
    warmup_iterations: int,
    probability_update_interval: int,
    min_segment_observations: int,
    min_motion_episodes: int,
    save_interval: int,
    trace_path: Path | None,
) -> list[str]:
    overrides = [
        "env.commands.motion.research.segment.enabled=true",
        "env.commands.motion.research.segment.length_seconds=1.0",
        f"env.commands.motion.research.method_name={method.method_name}",
        f"env.commands.motion.research.motion_sampling.mode={method.motion_mode}",
        f"env.commands.motion.research.segment_sampling.mode={method.segment_mode}",
        f"env.commands.motion.research.quality_gate.enabled={_bool_text(method.quality)}",
        f"env.commands.motion.research.difficulty_calibration.enabled={_bool_text(method.difficulty)}",
        f"env.commands.motion.research.online_learning.enabled={_bool_text(method.online_enabled)}",
        f"env.commands.motion.research.online_learning.statistics_enabled={_bool_text(method.statistics_enabled)}",
        f"env.commands.motion.research.online_learning.warmup_iterations={warmup_iterations}",
        f"env.commands.motion.research.online_learning.probability_update_interval={probability_update_interval}",
        f"env.commands.motion.research.online_learning.min_segment_observations={min_segment_observations}",
        f"env.commands.motion.research.online_learning.min_motion_episodes={min_motion_episodes}",
        f"env.commands.motion.research.online_learning.sampler_seed={args.sampler_seed}",
        f"env.commands.motion.research.adaptive_sampling.uniform_mix={args.uniform_mix}",
        f"env.commands.motion.research.adaptive_sampling.temperature={args.temperature}",
        f"env.commands.motion.research.adaptive_sampling.under_sampling_weight={args.under_sampling_weight}",
        f"env.commands.motion.research.adaptive_sampling.motion_probability_cap={args.motion_probability_cap}",
        f"env.commands.motion.research.adaptive_sampling.segment_probability_cap={args.segment_probability_cap}",
        "env.commands.motion.research.sampling_statistics.enabled=true",
        "env.commands.motion.research.sampling_statistics.log_interval=1",
        "env.commands.motion.research.diversity_constraint.enabled=false",
        f"env.commands.motion.research.assignment_trace.enabled={_bool_text(trace_path is not None)}",
        f"agent.save_interval={save_interval}",
    ]
    if trace_path is not None:
        overrides.extend(
            [
                f"env.commands.motion.research.assignment_trace.output_path={trace_path}",
                f"env.commands.motion.research.assignment_trace.max_entries={args.trace_max_entries}",
            ]
        )
    if method.quality:
        overrides.extend(
            [
                f"env.commands.motion.research.quality_gate.metadata_path={args.quality_metadata}",
                f"env.commands.motion.research.quality_gate.include_borderline={_bool_text(args.quality_include_borderline)}",
                "env.commands.motion.research.quality_gate.empty_motion_policy=exclude",
                "env.commands.motion.research.quality_gate.strict_metadata_match=true",
            ]
        )
    if method.difficulty:
        overrides.extend(
            [
                f"env.commands.motion.research.difficulty_calibration.metadata_path={args.difficulty_metadata}",
                "env.commands.motion.research.difficulty_calibration.strict_metadata_match=true",
            ]
        )
    overrides.extend(args.extra_override)
    return overrides


def build_train_command(
    args: argparse.Namespace,
    method: PilotMethod,
    *,
    suite: str,
    iterations: int,
    warmup_iterations: int,
    probability_update_interval: int,
    min_segment_observations: int,
    min_motion_episodes: int,
    save_interval: int,
    run_name: str,
    wandb_run_name: str | None = None,
    wandb_run_id: str | None = None,
    wandb_resume: str = "never",
    trace_path: Path | None = None,
    resume_from: tuple[str, str] | None = None,
) -> list[str]:
    command = [
        str(args.python),
        "scripts/rsl_rl/train.py",
    ]
    if args.disable_fabric:
        command.append("--disable_fabric")
    command.extend(
        [
            "--task",
            args.task,
            "--motion_file",
            str(args.manifest),
            "--headless",
            "--logger",
            args.logger,
            "--log_project_name",
            args.wandb_project,
            "--num_envs",
            str(args.num_envs),
            "--seed",
            str(args.seed),
            "--max_iterations",
            str(iterations),
            "--run_name",
            run_name,
        ]
    )
    if args.device:
        command.extend(["--device", args.device])
    if args.logger == "wandb":
        command.extend(
            [
                "--wandb_run_name",
                wandb_run_name or run_name,
                "--wandb_run_id",
                wandb_run_id or _run_id(run_name),
                "--wandb_resume",
                wandb_resume,
            ]
        )
    if resume_from is not None:
        load_run, checkpoint = resume_from
        command.extend(["--resume", "True", "--load_run", load_run, "--checkpoint", checkpoint])
    command.extend(
        _common_overrides(
            args=args,
            method=method,
            warmup_iterations=warmup_iterations,
            probability_update_interval=probability_update_interval,
            min_segment_observations=min_segment_observations,
            min_motion_episodes=min_motion_episodes,
            save_interval=save_interval,
            trace_path=trace_path,
        )
    )
    return command


def _print_or_run(command: Sequence[str], args: argparse.Namespace, log_path: Path | None = None) -> None:
    printable = shlex.join(str(part) for part in command)
    print(printable)
    if args.dry_run:
        return
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Writing process log: {log_path}")
        with log_path.open("w", encoding="utf-8") as stream:
            subprocess.run(command, cwd=PROJECT_ROOT, env=_clean_env(args), check=True, stdout=stream, stderr=subprocess.STDOUT)
    else:
        subprocess.run(command, cwd=PROJECT_ROOT, env=_clean_env(args), check=True)


def _latest_run_dir(experiment_name: str, run_name_suffix: str) -> Path:
    root = PROJECT_ROOT / "logs" / "rsl_rl" / experiment_name
    candidates = sorted(root.glob(f"*_{run_name_suffix}"), key=lambda path: (path.stat().st_mtime, path.name))
    if not candidates:
        raise FileNotFoundError(f"No run directory matching '*_{run_name_suffix}' under {root}")
    return candidates[-1]


def _checkpoint_iteration(path: Path) -> int:
    match = re.fullmatch(r"model_(\d+)\.pt", path.name)
    return int(match.group(1)) if match else -1


def _latest_checkpoint(run_dir: Path) -> Path:
    checkpoints = [path for path in run_dir.glob("model_*.pt") if _checkpoint_iteration(path) >= 0]
    if not checkpoints:
        raise FileNotFoundError(f"No model_*.pt checkpoint found in {run_dir}")
    return sorted(checkpoints, key=lambda path: (_checkpoint_iteration(path), path.stat().st_mtime))[-1]


def _trace_path(args: argparse.Namespace, method: PilotMethod, suite: str, iterations: int) -> Path:
    return args.trace_dir / f"{_run_name(method, suite, iterations)}.csv"


def _run_trace_suite(args: argparse.Namespace) -> None:
    iterations, warmup, interval, min_seg, min_motion, save_interval = _phase_defaults(args, "trace")
    traces: dict[str, Path] = {}
    for key in TRACE_METHODS:
        method = METHODS[key]
        run_name = _run_name(method, "trace", iterations)
        trace_path = _trace_path(args, method, "trace", iterations)
        traces[key] = trace_path
        command = build_train_command(
            args,
            method,
            suite="trace",
            iterations=iterations,
            warmup_iterations=warmup,
            probability_update_interval=interval,
            min_segment_observations=min_seg,
            min_motion_episodes=min_motion,
            save_interval=save_interval,
            run_name=run_name,
            trace_path=trace_path,
        )
        _print_or_run(command, args, args.log_dir / f"{run_name}.log")
    compare = [
        str(args.python),
        "scripts/compare_assignment_traces.py",
        str(traces["m0_trace"]),
        str(traces["stats"]),
    ]
    _print_or_run(compare, args, args.log_dir / "module3_trace_compare.log")


def _run_standard_suite(args: argparse.Namespace, suite: str, methods: Iterable[str]) -> None:
    iterations, warmup, interval, min_seg, min_motion, save_interval = _phase_defaults(args, suite)
    for key in methods:
        method = METHODS[key]
        if key == "m0_trace":
            raise ValueError("m0_trace is only valid in the trace suite.")
        run_name = _run_name(method, suite, iterations)
        command = build_train_command(
            args,
            method,
            suite=suite,
            iterations=iterations,
            warmup_iterations=warmup,
            probability_update_interval=interval,
            min_segment_observations=min_seg,
            min_motion_episodes=min_motion,
            save_interval=save_interval,
            run_name=run_name,
        )
        _print_or_run(command, args, args.log_dir / f"{run_name}.log")


def _run_resume_suite(args: argparse.Namespace, methods: Iterable[str]) -> None:
    for key in methods:
        method = METHODS[key]
        if key not in {"m5", "m6"}:
            raise ValueError("Resume suite is intended for M5 or M6.")
        base_wandb_name = f"module3_{key}_random100_seed42_resume"
        stage1_run = f"{base_wandb_name}_stage1_{args.resume_initial_iterations}"
        stage1 = build_train_command(
            args,
            method,
            suite="resume",
            iterations=args.resume_initial_iterations,
            warmup_iterations=args.debug_warmup_iterations,
            probability_update_interval=args.debug_probability_update_interval,
            min_segment_observations=args.debug_min_segment_observations,
            min_motion_episodes=args.debug_min_motion_episodes,
            save_interval=args.resume_save_interval,
            run_name=stage1_run,
            wandb_run_name=stage1_run,
            wandb_run_id=_run_id(stage1_run),
            wandb_resume="never",
        )
        _print_or_run(stage1, args, args.log_dir / f"{stage1_run}.log")
        if args.dry_run:
            print(
                "[DRY-RUN] Resume stage 2 will use the latest model_*.pt from "
                f"logs/rsl_rl/{args.experiment_name}/*_{stage1_run} after stage 1 completes."
            )
            resume_from = (f".*_{stage1_run}$", "<latest_model_*.pt>")
        else:
            run_dir = _latest_run_dir(args.experiment_name, stage1_run)
            checkpoint = _latest_checkpoint(run_dir)
            print(f"[INFO] Resume source: {run_dir.name}/{checkpoint.name}")
            resume_from = (run_dir.name, checkpoint.name)
        stage2_run = f"{base_wandb_name}_stage2_{args.resume_extra_iterations}"
        stage2 = build_train_command(
            args,
            method,
            suite="resume",
            iterations=args.resume_extra_iterations,
            warmup_iterations=args.debug_warmup_iterations,
            probability_update_interval=args.debug_probability_update_interval,
            min_segment_observations=args.debug_min_segment_observations,
            min_motion_episodes=args.debug_min_motion_episodes,
            save_interval=args.resume_save_interval,
            run_name=stage2_run,
            wandb_run_name=stage2_run,
            wandb_run_id=_run_id(stage2_run),
            wandb_resume="never",
            resume_from=resume_from,
        )
        _print_or_run(stage2, args, args.log_dir / f"{stage2_run}.log")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("trace", "smoke", "debug", "resume", "all"), default="smoke")
    parser.add_argument("--methods", type=_csv_items, default=None, help="Comma-separated subset such as m2,m3,m4,m5,m6.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without launching Isaac Sim.")
    parser.add_argument("--skip-path-check", action="store_true", help="Do not require manifest/metadata files before launch.")
    parser.add_argument("--skip-cuda-check", action="store_true", help="Skip the torch CUDA availability probe.")
    parser.add_argument("--python", type=Path, default=Path(sys.executable), help="Python executable inside the Isaac/conda environment.")
    parser.add_argument("--task", default="Tracking-Flat-G1-v0")
    parser.add_argument("--device", default=None, help="Optional Isaac device override, for example cuda:0.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--difficulty-metadata", type=Path, default=DEFAULT_DIFFICULTY)
    parser.add_argument("--quality-metadata", type=Path, default=DEFAULT_QUALITY)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sampler-seed", type=int, default=42)
    parser.add_argument("--logger", choices=("wandb", "tensorboard"), default="wandb")
    parser.add_argument("--wandb-project", default=DEFAULT_PROJECT)
    parser.add_argument("--wandb-mode", choices=("inherit", "online", "offline", "disabled"), default="inherit")
    parser.add_argument(
        "--quality-include-borderline",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "When M6 enables the quality gate, include borderline segments as eligible starts. "
            "The module-three mainline default is false after manual module-one review."
        ),
    )
    parser.add_argument("--experiment-name", default="g1_flat")
    parser.add_argument("--log-dir", type=Path, default=Path("/tmp/module3_gpu_pilot_logs"))
    parser.add_argument("--trace-dir", type=Path, default=Path("/tmp/module3_gpu_pilot_traces"))
    parser.add_argument("--trace-max-entries", type=int, default=4096)
    parser.add_argument("--disable-fabric", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--keep-pythonpath", action="store_true")
    parser.add_argument("--keep-ld-library-path", action="store_true")
    parser.add_argument("--smoke-iterations", type=int, default=30)
    parser.add_argument("--trace-iterations", type=int, default=20)
    parser.add_argument("--debug-iterations", type=int, default=500)
    parser.add_argument("--resume-initial-iterations", type=int, default=150)
    parser.add_argument("--resume-extra-iterations", type=int, default=100)
    parser.add_argument("--smoke-save-interval", type=int, default=10)
    parser.add_argument("--trace-save-interval", type=int, default=10)
    parser.add_argument("--debug-save-interval", type=int, default=50)
    parser.add_argument("--resume-save-interval", type=int, default=50)
    parser.add_argument("--formal-warmup-iterations", type=int, default=1000)
    parser.add_argument("--formal-probability-update-interval", type=int, default=50)
    parser.add_argument("--formal-min-segment-observations", type=int, default=32)
    parser.add_argument("--formal-min-motion-episodes", type=int, default=8)
    parser.add_argument("--debug-warmup-iterations", type=int, default=50)
    parser.add_argument("--debug-probability-update-interval", type=int, default=10)
    parser.add_argument("--debug-min-segment-observations", type=int, default=4)
    parser.add_argument("--debug-min-motion-episodes", type=int, default=2)
    parser.add_argument("--uniform-mix", type=float, default=0.15)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--under-sampling-weight", type=float, default=0.25)
    parser.add_argument("--motion-probability-cap", type=float, default=0.02)
    parser.add_argument("--segment-probability-cap", type=float, default=1.0)
    parser.add_argument(
        "--extra-override",
        action="append",
        default=[],
        help="Additional Hydra override appended to every train command.",
    )
    return parser.parse_args(argv)


def _selected_methods(args: argparse.Namespace, default: Sequence[str]) -> tuple[str, ...]:
    return tuple(args.methods) if args.methods is not None else tuple(default)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    _require_file(args.manifest, "random100 manifest", skip=args.skip_path_check or args.dry_run)
    methods_for_check: Iterable[str]
    if args.suite == "trace":
        methods_for_check = TRACE_METHODS
    elif args.suite == "resume":
        methods_for_check = _selected_methods(args, RESUME_METHODS)
    else:
        methods_for_check = _selected_methods(args, DEFAULT_METHODS)
    if any(METHODS[key].difficulty for key in methods_for_check):
        _require_file(args.difficulty_metadata, "difficulty metadata", skip=args.skip_path_check or args.dry_run)
    if any(METHODS[key].quality for key in methods_for_check):
        _require_file(args.quality_metadata, "quality metadata", skip=args.skip_path_check or args.dry_run)

    _cuda_check(args)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    args.trace_dir.mkdir(parents=True, exist_ok=True)

    if args.suite == "trace":
        _run_trace_suite(args)
    elif args.suite == "smoke":
        _run_standard_suite(args, "smoke", _selected_methods(args, DEFAULT_METHODS))
    elif args.suite == "debug":
        _run_standard_suite(args, "debug", _selected_methods(args, DEFAULT_METHODS))
    elif args.suite == "resume":
        _run_resume_suite(args, _selected_methods(args, RESUME_METHODS))
    elif args.suite == "all":
        _run_trace_suite(args)
        _run_standard_suite(args, "smoke", _selected_methods(args, DEFAULT_METHODS))
        _run_standard_suite(args, "debug", _selected_methods(args, DEFAULT_METHODS))
        _run_resume_suite(args, _selected_methods(args, RESUME_METHODS))
    else:
        raise AssertionError(args.suite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
