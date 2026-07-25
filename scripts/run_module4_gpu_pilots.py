#!/usr/bin/env python3
"""Build or launch the short random100 GPU pilots for module four.

The script exposes only the three module-four debug contracts:

* Pilot A: diversity-only diagnostic (uniform motion/segment sampling);
* Pilot B: the M6 control with cluster diversity disabled;
* Pilot C: full M7 with quality, difficulty, and cluster diversity enabled.

There is deliberately no formal-training suite.  Use ``--dry-run`` to print
replayable commands without probing CUDA or launching Isaac Sim.
"""

from __future__ import annotations

import argparse
import os
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
DEFAULT_QUALITY_METADATA = (
    PROJECT_ROOT
    / "outputs"
    / "module1_quality_pilot_random100_seed42_v1"
    / "segment_quality_metadata.npz"
)
DEFAULT_DIFFICULTY_METADATA = (
    PROJECT_ROOT
    / "outputs"
    / "module2_difficulty_pilot_random100_seed42_v1"
    / "segment_difficulty_metadata.npz"
)
DEFAULT_CLUSTER_METADATA = (
    PROJECT_ROOT
    / "outputs"
    / "module4_clusters_random100_seed42_v1"
    / "motion_cluster_metadata.npz"
)
DEFAULT_PROJECT = "whole_body_tracking_module4_pilot"
MAX_PILOT_ITERATIONS = 2_000


@dataclass(frozen=True)
class PilotMethod:
    key: str
    label: str
    method_name: str
    motion_mode: str
    segment_mode: str
    quality: bool
    difficulty: bool
    diversity: bool


PILOTS: dict[str, PilotMethod] = {
    "diversity_only": PilotMethod(
        key="diversity_only",
        label="A",
        method_name="DIVERSITY_ONLY",
        motion_mode="uniform",
        segment_mode="uniform",
        quality=False,
        difficulty=False,
        diversity=True,
    ),
    "m6": PilotMethod(
        key="m6",
        label="B",
        method_name="M6",
        motion_mode="learning_gap",
        segment_mode="relative_learning_gap",
        quality=True,
        difficulty=True,
        diversity=False,
    ),
    "m7": PilotMethod(
        key="m7",
        label="C",
        method_name="M7",
        motion_mode="learning_gap",
        segment_mode="relative_learning_gap",
        quality=True,
        difficulty=True,
        diversity=True,
    ),
}
# Compatibility with the naming used by the module-three pilot helper.
METHODS = PILOTS
DEFAULT_PILOTS = ("diversity_only", "m6", "m7")


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


def _pilot_iterations(value: str) -> int:
    iterations = int(value)
    if not 1 <= iterations <= MAX_PILOT_ITERATIONS:
        raise argparse.ArgumentTypeError(
            f"Pilot iterations must be in [1, {MAX_PILOT_ITERATIONS}]; "
            "this helper intentionally does not launch formal training."
        )
    return iterations


def _csv_pilots(value: str) -> tuple[str, ...]:
    aliases = {
        "a": "diversity_only",
        "diversity-only": "diversity_only",
        "diversity_only": "diversity_only",
        "b": "m6",
        "m6": "m6",
        "c": "m7",
        "m7": "m7",
    }
    requested = tuple(item.strip().lower() for item in value.split(",") if item.strip())
    if not requested:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated pilot.")
    unknown = sorted(set(requested).difference(aliases))
    if unknown:
        raise argparse.ArgumentTypeError(
            f"Unknown pilot(s): {unknown}; use A/B/C or diversity_only/m6/m7."
        )
    normalized = tuple(aliases[item] for item in requested)
    return tuple(dict.fromkeys(normalized))


def _require_file(path: Path, label: str, *, skip: bool) -> None:
    if not skip and not path.is_file():
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
    result = subprocess.run(
        [
            str(args.python),
            "-c",
            "import torch; print(torch.cuda.is_available()); "
            "print(torch.cuda.get_device_name(0))",
        ],
        cwd=PROJECT_ROOT,
        env=_clean_env(args),
        check=False,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"CUDA check failed:\n{result.stdout}\n{result.stderr}")
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not lines or lines[0] != "True":
        raise RuntimeError("CUDA is not available to the selected Python interpreter.")
    print(f"[INFO] CUDA device: {lines[1] if len(lines) > 1 else 'unknown'}")


def _run_name(pilot: PilotMethod, iterations: int, *, resume: bool = False) -> str:
    phase = "resume" if resume else "debug"
    return f"module4_{pilot.key}_random100_seed42_{phase}{iterations}"


def _run_id(run_name: str) -> str:
    return run_name.replace("_", "-")


def _trace_path(args: argparse.Namespace, run_name: str) -> Path | None:
    if not args.trace:
        return None
    return args.trace_dir / f"{run_name}.csv"


def _research_overrides(
    args: argparse.Namespace,
    pilot: PilotMethod,
    *,
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
        f"env.commands.motion.research.method_name={pilot.method_name}",
        f"env.commands.motion.research.motion_sampling.mode={pilot.motion_mode}",
        f"env.commands.motion.research.segment_sampling.mode={pilot.segment_mode}",
        f"env.commands.motion.research.quality_gate.enabled={_bool_text(pilot.quality)}",
        (
            "env.commands.motion.research.difficulty_calibration.enabled="
            f"{_bool_text(pilot.difficulty)}"
        ),
        (
            "env.commands.motion.research.diversity_constraint.enabled="
            f"{_bool_text(pilot.diversity)}"
        ),
        "env.commands.motion.research.online_learning.enabled=true",
        "env.commands.motion.research.online_learning.statistics_enabled=true",
        (
            "env.commands.motion.research.online_learning.warmup_iterations="
            f"{warmup_iterations}"
        ),
        (
            "env.commands.motion.research.online_learning.probability_update_interval="
            f"{probability_update_interval}"
        ),
        (
            "env.commands.motion.research.online_learning.min_segment_observations="
            f"{min_segment_observations}"
        ),
        (
            "env.commands.motion.research.online_learning.min_motion_episodes="
            f"{min_motion_episodes}"
        ),
        (
            "env.commands.motion.research.online_learning.sampler_seed="
            f"{args.sampler_seed}"
        ),
        f"env.commands.motion.research.adaptive_sampling.uniform_mix={args.uniform_mix}",
        f"env.commands.motion.research.adaptive_sampling.temperature={args.temperature}",
        (
            "env.commands.motion.research.adaptive_sampling.under_sampling_weight="
            f"{args.under_sampling_weight}"
        ),
        (
            "env.commands.motion.research.adaptive_sampling.motion_probability_cap="
            f"{args.motion_probability_cap}"
        ),
        (
            "env.commands.motion.research.adaptive_sampling.segment_probability_cap="
            f"{args.segment_probability_cap}"
        ),
        "env.commands.motion.research.sampling_statistics.enabled=true",
        "env.commands.motion.research.sampling_statistics.log_interval=1",
        (
            "env.commands.motion.research.assignment_trace.enabled="
            f"{_bool_text(trace_path is not None)}"
        ),
        f"agent.save_interval={save_interval}",
    ]
    if trace_path is not None:
        overrides.extend(
            [
                f"env.commands.motion.research.assignment_trace.output_path={trace_path}",
                (
                    "env.commands.motion.research.assignment_trace.max_entries="
                    f"{args.trace_max_entries}"
                ),
            ]
        )
    if pilot.quality:
        overrides.extend(
            [
                f"env.commands.motion.research.quality_gate.metadata_path={args.quality_metadata}",
                (
                    "env.commands.motion.research.quality_gate.include_borderline="
                    f"{_bool_text(args.quality_include_borderline)}"
                ),
                "env.commands.motion.research.quality_gate.empty_motion_policy=exclude",
                "env.commands.motion.research.quality_gate.strict_metadata_match=true",
            ]
        )
    if pilot.difficulty:
        overrides.extend(
            [
                (
                    "env.commands.motion.research.difficulty_calibration.metadata_path="
                    f"{args.difficulty_metadata}"
                ),
                (
                    "env.commands.motion.research.difficulty_calibration."
                    "strict_metadata_match=true"
                ),
            ]
        )
    if pilot.diversity:
        overrides.extend(
            [
                (
                    "env.commands.motion.research.diversity_constraint.metadata_path="
                    f"{args.cluster_metadata}"
                ),
                (
                    "env.commands.motion.research.diversity_constraint."
                    "strict_metadata_match=true"
                ),
                (
                    "env.commands.motion.research.diversity_constraint."
                    f"expected_num_clusters={args.expected_num_clusters}"
                ),
                (
                    "env.commands.motion.research.diversity_constraint."
                    "budget_mode=sqrt_size_with_floor"
                ),
                (
                    "env.commands.motion.research.diversity_constraint."
                    "minimum_budget_fraction_of_uniform="
                    f"{args.minimum_budget_fraction_of_uniform}"
                ),
                (
                    "env.commands.motion.research.diversity_constraint."
                    f"cluster_size_exponent={args.cluster_size_exponent}"
                ),
                (
                    "env.commands.motion.research.diversity_constraint."
                    "diversity_during_warmup=true"
                ),
                (
                    "env.commands.motion.research.diversity_constraint."
                    "count_aware_correction=false"
                ),
            ]
        )
    return overrides


def build_train_command(
    args: argparse.Namespace,
    pilot: PilotMethod,
    *,
    iterations: int | None = None,
    warmup_iterations: int | None = None,
    probability_update_interval: int | None = None,
    min_segment_observations: int | None = None,
    min_motion_episodes: int | None = None,
    save_interval: int | None = None,
    run_name: str | None = None,
    trace_path: Path | None = None,
    resume_from: tuple[str, str] | None = None,
) -> list[str]:
    """Return one replayable training command without executing it."""

    iterations = args.iterations if iterations is None else iterations
    warmup_iterations = (
        args.warmup_iterations if warmup_iterations is None else warmup_iterations
    )
    probability_update_interval = (
        args.probability_update_interval
        if probability_update_interval is None
        else probability_update_interval
    )
    min_segment_observations = (
        args.min_segment_observations
        if min_segment_observations is None
        else min_segment_observations
    )
    min_motion_episodes = (
        args.min_motion_episodes
        if min_motion_episodes is None
        else min_motion_episodes
    )
    save_interval = args.save_interval if save_interval is None else save_interval
    run_name = run_name or _run_name(pilot, iterations, resume=resume_from is not None)
    if trace_path is None:
        trace_path = _trace_path(args, run_name)

    command = [str(args.python), "scripts/rsl_rl/train.py"]
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
                run_name,
                "--wandb_run_id",
                _run_id(run_name),
                "--wandb_resume",
                "allow" if resume_from is not None else "never",
            ]
        )
    if resume_from is not None:
        load_run, checkpoint = resume_from
        command.extend(
            [
                "--resume",
                "True",
                "--load_run",
                load_run,
                "--checkpoint",
                checkpoint,
            ]
        )
    command.extend(
        _research_overrides(
            args,
            pilot,
            warmup_iterations=warmup_iterations,
            probability_update_interval=probability_update_interval,
            min_segment_observations=min_segment_observations,
            min_motion_episodes=min_motion_episodes,
            save_interval=save_interval,
            trace_path=trace_path,
        )
    )
    return command


def _print_or_run(
    command: Sequence[str],
    args: argparse.Namespace,
    *,
    log_path: Path | None = None,
) -> None:
    print(shlex.join(str(part) for part in command))
    if args.dry_run:
        return
    if log_path is None:
        subprocess.run(command, cwd=PROJECT_ROOT, env=_clean_env(args), check=True)
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Writing process log: {log_path}")
    with log_path.open("w", encoding="utf-8") as stream:
        subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=_clean_env(args),
            check=True,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("debug", "resume"), default="debug")
    parser.add_argument(
        "--pilots",
        "--methods",
        dest="pilots",
        type=_csv_pilots,
        default=None,
        help="Comma-separated A/B/C or diversity_only/m6/m7 subset.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands only; skip path checks, CUDA checks, and training.",
    )
    parser.add_argument("--skip-path-check", action="store_true")
    parser.add_argument("--skip-cuda-check", action="store_true")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--task", default="Tracking-Flat-G1-v0")
    parser.add_argument("--device", default=None)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--quality-metadata",
        type=Path,
        default=DEFAULT_QUALITY_METADATA,
    )
    parser.add_argument(
        "--quality-include-borderline",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Include borderline segments in the runtime quality gate. The default "
            "matches the formal M7 contract and keeps 5998 random6000 motions eligible."
        ),
    )
    parser.add_argument(
        "--difficulty-metadata",
        type=Path,
        default=DEFAULT_DIFFICULTY_METADATA,
    )
    parser.add_argument(
        "--cluster-metadata",
        type=Path,
        default=DEFAULT_CLUSTER_METADATA,
    )
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sampler-seed", type=int, default=42)
    parser.add_argument("--logger", choices=("wandb", "tensorboard"), default="wandb")
    parser.add_argument("--wandb-project", default=DEFAULT_PROJECT)
    parser.add_argument(
        "--wandb-mode",
        choices=("inherit", "online", "offline", "disabled"),
        default="inherit",
    )
    parser.add_argument("--disable-fabric", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--keep-pythonpath", action="store_true")
    parser.add_argument("--keep-ld-library-path", action="store_true")
    parser.add_argument("--iterations", type=_pilot_iterations, default=500)
    parser.add_argument("--warmup-iterations", type=int, default=50)
    parser.add_argument("--probability-update-interval", type=int, default=10)
    parser.add_argument("--min-segment-observations", type=int, default=4)
    parser.add_argument("--min-motion-episodes", type=int, default=2)
    parser.add_argument("--save-interval", type=int, default=50)
    parser.add_argument("--uniform-mix", type=float, default=0.15)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--under-sampling-weight", type=float, default=0.25)
    parser.add_argument(
        "--motion-probability-cap",
        type=float,
        default=0.5,
        help=(
            "Conditional Motion cap for the random100 debug pilots. The 0.5 "
            "default leaves room for post-warmup adaptation inside small clusters; "
            "it does not change the formal training default."
        ),
    )
    parser.add_argument("--segment-probability-cap", type=float, default=1.0)
    parser.add_argument("--expected-num-clusters", type=int, default=8)
    parser.add_argument(
        "--minimum-budget-fraction-of-uniform",
        type=float,
        default=0.5,
    )
    parser.add_argument("--cluster-size-exponent", type=float, default=0.5)
    parser.add_argument(
        "--trace",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Record assignment traces for each pilot (enabled by default).",
    )
    parser.add_argument(
        "--trace-dir",
        type=Path,
        default=Path("/tmp/module4_gpu_pilot_traces"),
    )
    parser.add_argument("--trace-max-entries", type=int, default=4096)
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("/tmp/module4_gpu_pilot_logs"),
    )
    parser.add_argument(
        "--resume-run",
        default=None,
        help="Existing rsl_rl run directory name or load_run pattern for --suite resume.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint filename, for example model_500.pt, for --suite resume.",
    )
    parser.add_argument(
        "--resume-iterations",
        type=_pilot_iterations,
        default=100,
        help="Additional short-pilot iterations requested by the M7 resume command.",
    )
    return parser.parse_args(argv)


def _selected_pilots(args: argparse.Namespace) -> tuple[str, ...]:
    return tuple(args.pilots) if args.pilots is not None else DEFAULT_PILOTS


def _validate_positive_debug_settings(args: argparse.Namespace) -> None:
    values = {
        "num_envs": args.num_envs,
        "probability_update_interval": args.probability_update_interval,
        "min_segment_observations": args.min_segment_observations,
        "min_motion_episodes": args.min_motion_episodes,
        "save_interval": args.save_interval,
        "expected_num_clusters": args.expected_num_clusters,
        "trace_max_entries": args.trace_max_entries,
    }
    invalid = [name for name, value in values.items() if value < 1]
    if invalid:
        raise ValueError(f"Pilot settings must be positive: {', '.join(invalid)}")
    if args.warmup_iterations < 0:
        raise ValueError("warmup_iterations must be non-negative.")


def _check_inputs(args: argparse.Namespace, pilots: Iterable[PilotMethod]) -> None:
    skip = args.skip_path_check or args.dry_run
    methods = tuple(pilots)
    _require_file(args.manifest, "random100 manifest", skip=skip)
    if any(method.quality for method in methods):
        _require_file(args.quality_metadata, "quality metadata", skip=skip)
    if any(method.difficulty for method in methods):
        _require_file(args.difficulty_metadata, "difficulty metadata", skip=skip)
    if any(method.diversity for method in methods):
        _require_file(args.cluster_metadata, "cluster metadata", skip=skip)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    _validate_positive_debug_settings(args)

    if args.suite == "resume":
        if not args.resume_run or not args.checkpoint:
            raise ValueError(
                "--suite resume requires both --resume-run and --checkpoint."
            )
        selected = (PILOTS["m7"],)
    else:
        selected = tuple(PILOTS[key] for key in _selected_pilots(args))

    _check_inputs(args, selected)
    _cuda_check(args)
    if not args.dry_run:
        args.log_dir.mkdir(parents=True, exist_ok=True)
        if args.trace:
            args.trace_dir.mkdir(parents=True, exist_ok=True)

    for pilot in selected:
        if args.suite == "resume":
            iterations = args.resume_iterations
            resume_from = (args.resume_run, args.checkpoint)
        else:
            iterations = args.iterations
            resume_from = None
        run_name = _run_name(pilot, iterations, resume=resume_from is not None)
        command = build_train_command(
            args,
            pilot,
            iterations=iterations,
            run_name=run_name,
            trace_path=_trace_path(args, run_name),
            resume_from=resume_from,
        )
        print(f"[Pilot {pilot.label}] {pilot.method_name}")
        _print_or_run(
            command,
            args,
            log_path=args.log_dir / f"{run_name}.log",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
