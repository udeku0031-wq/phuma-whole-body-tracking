from __future__ import annotations

import csv
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = PROJECT_ROOT / "scripts" / "run_module3_gpu_pilots.py"
VALIDATOR_PATH = PROJECT_ROOT / "scripts" / "validate_module3_gpu_pilots.py"
SOURCE_ROOT = PROJECT_ROOT / "source" / "whole_body_tracking"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class Module3GpuPilotScriptTest(unittest.TestCase):
    def test_debug_m5_command_uses_short_warmup_and_difficulty_only(self) -> None:
        runner = _load_module(RUNNER_PATH, "module3_gpu_runner_test")
        args = runner._parse_args(
            [
                "--dry-run",
                "--suite",
                "debug",
                "--manifest",
                "/tmp/random100.txt",
                "--difficulty-metadata",
                "/tmp/difficulty.npz",
                "--quality-metadata",
                "/tmp/quality.npz",
            ]
        )
        method = runner.METHODS["m5"]
        command = runner.build_train_command(
            args,
            method,
            suite="debug",
            iterations=500,
            warmup_iterations=50,
            probability_update_interval=10,
            min_segment_observations=4,
            min_motion_episodes=2,
            save_interval=50,
            run_name="module3_m5_random100_seed42_debug500",
        )
        joined = "\n".join(str(item) for item in command)
        self.assertIn("env.commands.motion.research.motion_sampling.mode=learning_gap", joined)
        self.assertIn("env.commands.motion.research.segment_sampling.mode=relative_learning_gap", joined)
        self.assertIn("env.commands.motion.research.online_learning.warmup_iterations=50", joined)
        self.assertIn("env.commands.motion.research.online_learning.probability_update_interval=10", joined)
        self.assertIn("env.commands.motion.research.online_learning.min_segment_observations=4", joined)
        self.assertIn("env.commands.motion.research.online_learning.min_motion_episodes=2", joined)
        self.assertIn("env.commands.motion.research.difficulty_calibration.enabled=true", joined)
        self.assertIn("env.commands.motion.research.quality_gate.enabled=false", joined)

    def test_debug_m6_command_uses_strict_quality_gate_by_default(self) -> None:
        runner = _load_module(RUNNER_PATH, "module3_gpu_runner_m6_test")
        args = runner._parse_args(
            [
                "--dry-run",
                "--suite",
                "debug",
                "--manifest",
                "/tmp/random100.txt",
                "--difficulty-metadata",
                "/tmp/difficulty.npz",
                "--quality-metadata",
                "/tmp/quality.npz",
            ]
        )
        command = runner.build_train_command(
            args,
            runner.METHODS["m6"],
            suite="debug",
            iterations=500,
            warmup_iterations=50,
            probability_update_interval=10,
            min_segment_observations=4,
            min_motion_episodes=2,
            save_interval=50,
            run_name="module3_m6_random100_seed42_debug500",
        )
        joined = "\n".join(str(item) for item in command)
        self.assertIn("env.commands.motion.research.quality_gate.enabled=true", joined)
        self.assertIn("env.commands.motion.research.quality_gate.include_borderline=false", joined)
        self.assertIn("env.commands.motion.research.quality_gate.empty_motion_policy=exclude", joined)

    def test_trace_m0_command_keeps_online_learning_disabled(self) -> None:
        runner = _load_module(RUNNER_PATH, "module3_gpu_runner_trace_test")
        args = runner._parse_args(["--dry-run", "--suite", "trace", "--manifest", "/tmp/random100.txt"])
        method = runner.METHODS["m0_trace"]
        command = runner.build_train_command(
            args,
            method,
            suite="trace",
            iterations=20,
            warmup_iterations=1000,
            probability_update_interval=50,
            min_segment_observations=32,
            min_motion_episodes=8,
            save_interval=10,
            run_name="module3_m0_trace_random100_seed42_trace20",
            trace_path=Path("/tmp/m0_trace.csv"),
        )
        joined = "\n".join(str(item) for item in command)
        self.assertIn("env.commands.motion.research.online_learning.enabled=false", joined)
        self.assertIn("env.commands.motion.research.online_learning.statistics_enabled=false", joined)
        self.assertIn("env.commands.motion.research.assignment_trace.enabled=true", joined)

    def test_trace_comparison_reports_column_mismatches(self) -> None:
        validator = _load_module(VALIDATOR_PATH, "module3_gpu_validator_trace_test")
        with tempfile.TemporaryDirectory() as tmp:
            left = Path(tmp) / "left.csv"
            right = Path(tmp) / "right.csv"
            header = [
                "assignment_index",
                "env_id",
                "motion_id",
                "start_frame",
                "local_segment_id",
                "global_segment_id",
            ]
            rows = [
                dict(
                    assignment_index="0",
                    env_id="0",
                    motion_id="1",
                    start_frame="10",
                    local_segment_id="0",
                    global_segment_id="7",
                )
            ]
            for path, motion_id in ((left, "1"), (right, "2")):
                with path.open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(stream, fieldnames=header)
                    writer.writeheader()
                    row = dict(rows[0])
                    row["motion_id"] = motion_id
                    writer.writerow(row)
            result = validator.compare_assignment_traces(left, right)
            self.assertEqual(result["motion_id_mismatches"], 1)
            self.assertEqual(result["start_frame_mismatches"], 0)
            self.assertEqual(result["total_mismatches"], 1)

    def test_quality_breakdown_distinguishes_rejects_from_startless_segments(self) -> None:
        validator = _load_module(VALIDATOR_PATH, "module3_gpu_validator_quality_test")
        quality_module = _load_module(
            SOURCE_ROOT / "whole_body_tracking" / "utils" / "quality_metadata.py",
            "module3_gpu_quality_metadata_test",
        )

        with tempfile.TemporaryDirectory() as tmp:
            metadata_path = Path(tmp) / "quality.npz"
            payload = quality_module.metadata_npz_payload(
                segment_schema_version=1,
                segment_length_seconds=1.0,
                manifest_sha256="a" * 64,
                quality_config_sha256="b" * 64,
                pool_fingerprint="c" * 64,
                motion_keys=["a.npz", "b.npz"],
                motion_lengths=[51, 51],
                motion_fps=[50.0, 50.0],
                motion_segment_offsets=[0, 2, 4],
                global_segment_id=[0, 1, 2, 3],
                motion_id=[0, 0, 1, 1],
                local_segment_id=[0, 1, 0, 1],
                start_frame=[0, 50, 0, 50],
                end_frame_exclusive=[50, 51, 50, 51],
                quality_score=[0.0, 1.0, 0.0, 0.0],
                quality_status=[0, 2, 1, 0],
            )
            np.savez(metadata_path, **payload)
            breakdown = validator.explain_m6_quality_gate(metadata_path)
            self.assertEqual(breakdown["total_segments"], 4)
            self.assertEqual(breakdown["quality_reject_segments"], 1)
            self.assertEqual(breakdown["quality_borderline_segments"], 1)
            self.assertEqual(breakdown["disallowed_segments_after_quality"], 2)
            self.assertEqual(breakdown["segments_without_legal_assignment_start"], 1)
            self.assertEqual(breakdown["effective_segments"], 1)
            self.assertEqual(breakdown["identity_equation_ok"], 1)


if __name__ == "__main__":
    unittest.main()
