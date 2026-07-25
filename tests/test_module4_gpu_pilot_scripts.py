from __future__ import annotations

import contextlib
import importlib.util
import io
import sys
import unittest
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = PROJECT_ROOT / "scripts" / "run_module4_gpu_pilots.py"


def _load_runner(name: str):
    spec = importlib.util.spec_from_file_location(name, RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _joined(command: list[str]) -> str:
    return "\n".join(str(item) for item in command)


class Module4GpuPilotScriptTest(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = _load_runner(f"module4_gpu_runner_test_{self._testMethodName}")
        self.args = self.runner._parse_args(
            [
                "--dry-run",
                "--manifest",
                "/tmp/random100.txt",
                "--quality-metadata",
                "/tmp/quality.npz",
                "--difficulty-metadata",
                "/tmp/difficulty.npz",
                "--cluster-metadata",
                "/tmp/clusters.npz",
                "--trace-dir",
                "/tmp/module4-traces",
            ]
        )

    def test_pilot_a_is_diversity_only_uniform_debug(self) -> None:
        command = self.runner.build_train_command(
            self.args,
            self.runner.PILOTS["diversity_only"],
        )
        joined = _joined(command)

        self.assertIn("--num_envs\n32", joined)
        self.assertIn("--seed\n42", joined)
        self.assertIn("--max_iterations\n500", joined)
        self.assertIn("env.commands.motion.research.method_name=DIVERSITY_ONLY", joined)
        self.assertIn(
            "env.commands.motion.research.motion_sampling.mode=uniform", joined
        )
        self.assertIn(
            "env.commands.motion.research.segment_sampling.mode=uniform", joined
        )
        self.assertIn(
            "env.commands.motion.research.quality_gate.enabled=false", joined
        )
        self.assertIn(
            "env.commands.motion.research.difficulty_calibration.enabled=false",
            joined,
        )
        self.assertIn(
            "env.commands.motion.research.diversity_constraint.enabled=true", joined
        )
        self.assertIn(
            "env.commands.motion.research.diversity_constraint.metadata_path="
            "/tmp/clusters.npz",
            joined,
        )
        self.assertIn(
            "env.commands.motion.research.online_learning.warmup_iterations=50",
            joined,
        )
        self.assertIn(
            "env.commands.motion.research.online_learning."
            "probability_update_interval=10",
            joined,
        )
        self.assertIn(
            "env.commands.motion.research.online_learning."
            "min_segment_observations=4",
            joined,
        )
        self.assertIn(
            "env.commands.motion.research.online_learning.min_motion_episodes=2",
            joined,
        )
        self.assertIn(
            "env.commands.motion.research.adaptive_sampling."
            "motion_probability_cap=0.5",
            joined,
        )
        self.assertIn(
            "env.commands.motion.research.assignment_trace.enabled=true", joined
        )

    def test_pilot_b_is_m6_with_diversity_off(self) -> None:
        command = self.runner.build_train_command(
            self.args,
            self.runner.PILOTS["m6"],
        )
        joined = _joined(command)

        self.assertIn("env.commands.motion.research.method_name=M6", joined)
        self.assertIn(
            "env.commands.motion.research.motion_sampling.mode=learning_gap", joined
        )
        self.assertIn(
            "env.commands.motion.research.segment_sampling.mode="
            "relative_learning_gap",
            joined,
        )
        self.assertIn(
            "env.commands.motion.research.quality_gate.enabled=true", joined
        )
        self.assertIn(
            "env.commands.motion.research.quality_gate.include_borderline=true",
            joined,
        )
        self.assertIn(
            "env.commands.motion.research.difficulty_calibration.enabled=true",
            joined,
        )
        self.assertIn(
            "env.commands.motion.research.diversity_constraint.enabled=false",
            joined,
        )
        self.assertNotIn(
            "env.commands.motion.research.diversity_constraint.metadata_path=",
            joined,
        )

    def test_pilot_c_m7_uses_all_strict_frozen_mappings(self) -> None:
        command = self.runner.build_train_command(
            self.args,
            self.runner.PILOTS["m7"],
        )
        joined = _joined(command)

        self.assertIn("env.commands.motion.research.method_name=M7", joined)
        self.assertIn(
            "env.commands.motion.research.quality_gate.metadata_path="
            "/tmp/quality.npz",
            joined,
        )
        self.assertIn(
            "env.commands.motion.research.quality_gate.strict_metadata_match=true",
            joined,
        )
        self.assertIn(
            "env.commands.motion.research.quality_gate.include_borderline=true",
            joined,
        )
        self.assertIn(
            "env.commands.motion.research.difficulty_calibration.metadata_path="
            "/tmp/difficulty.npz",
            joined,
        )
        self.assertIn(
            "env.commands.motion.research.difficulty_calibration."
            "strict_metadata_match=true",
            joined,
        )
        self.assertIn(
            "env.commands.motion.research.diversity_constraint.metadata_path="
            "/tmp/clusters.npz",
            joined,
        )
        self.assertIn(
            "env.commands.motion.research.diversity_constraint."
            "strict_metadata_match=true",
            joined,
        )
        self.assertIn(
            "env.commands.motion.research.diversity_constraint."
            "expected_num_clusters=8",
            joined,
        )

    def test_resume_suite_builds_strict_m7_resume_command(self) -> None:
        resume_args = self.runner._parse_args(
            [
                "--dry-run",
                "--suite",
                "resume",
                "--resume-run",
                "2026-07-24_10-00-00_module4_m7_debug",
                "--checkpoint",
                "model_500.pt",
                "--cluster-metadata",
                "/tmp/clusters.npz",
                "--quality-metadata",
                "/tmp/quality.npz",
                "--difficulty-metadata",
                "/tmp/difficulty.npz",
            ]
        )
        command = self.runner.build_train_command(
            resume_args,
            self.runner.PILOTS["m7"],
            iterations=resume_args.resume_iterations,
            resume_from=(resume_args.resume_run, resume_args.checkpoint),
        )
        joined = _joined(command)

        self.assertIn("--resume\nTrue", joined)
        self.assertIn(
            "--load_run\n2026-07-24_10-00-00_module4_m7_debug", joined
        )
        self.assertIn("--checkpoint\nmodel_500.pt", joined)
        self.assertIn("env.commands.motion.research.method_name=M7", joined)
        self.assertIn(
            "env.commands.motion.research.diversity_constraint."
            "strict_metadata_match=true",
            joined,
        )

    def test_dry_run_prints_all_three_commands_without_subprocess(self) -> None:
        output = io.StringIO()
        with (
            mock.patch.object(self.runner.subprocess, "run") as subprocess_run,
            contextlib.redirect_stdout(output),
        ):
            result = self.runner.main(["--dry-run"])

        rendered = output.getvalue()
        self.assertEqual(result, 0)
        subprocess_run.assert_not_called()
        self.assertEqual(rendered.count("scripts/rsl_rl/train.py"), 3)
        self.assertIn("[Pilot A] DIVERSITY_ONLY", rendered)
        self.assertIn("[Pilot B] M6", rendered)
        self.assertIn("[Pilot C] M7", rendered)


if __name__ == "__main__":
    unittest.main()
