from __future__ import annotations

import csv
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BUILDER_PATH = PROJECT_ROOT / "scripts" / "build_segment_quality_metadata.py"
QUALITY_METADATA_PATH = (
    PROJECT_ROOT
    / "source"
    / "whole_body_tracking"
    / "whole_body_tracking"
    / "utils"
    / "quality_metadata.py"
)

_METADATA_SPEC = importlib.util.spec_from_file_location(
    "wbt_quality_metadata_builder_tests", QUALITY_METADATA_PATH
)
if _METADATA_SPEC is None or _METADATA_SPEC.loader is None:
    raise RuntimeError(f"Unable to load quality metadata utilities from {QUALITY_METADATA_PATH}")
quality_metadata = importlib.util.module_from_spec(_METADATA_SPEC)
sys.modules[_METADATA_SPEC.name] = quality_metadata
_METADATA_SPEC.loader.exec_module(quality_metadata)

METRIC_NAMES = (
    "nonfinite_values",
    "quaternion_norm",
    "joint_position_limits",
    "joint_velocity_limits",
    "joint_velocity_consistency",
    "body_velocity_consistency",
    "joint_acceleration_spike",
    "joint_jerk_spike",
    "root_linear_acceleration_spike",
    "root_angular_acceleration_spike",
    "root_position_continuity",
    "root_orientation_continuity",
    "body_position_continuity",
    "body_orientation_continuity",
    "joint_position_continuity",
    "ground_penetration",
    "foot_sliding",
)


class BuildSegmentQualityMetadataTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.motion_dir = self.root / "motions"
        self.motion_dir.mkdir()
        self.motion_keys = [f"motions/motion_{index}.npz" for index in range(3)]
        self.motion_lengths = [11, 8, 6]
        for index, num_frames in enumerate(self.motion_lengths):
            self._write_motion(self.root / self.motion_keys[index], num_frames=num_frames)

        self.manifest = self.root / "train_manifest.txt"
        self.manifest.write_text(
            "# ordered synthetic Train pool\n" + "".join(f"{key}\n" for key in self.motion_keys),
            encoding="utf-8",
        )
        self.dataset_metadata = self.root / "split_metadata.csv"
        self._write_split_metadata(self.dataset_metadata, ["train", "train", "train"])

        self.urdf = self.root / "synthetic.urdf"
        self.urdf.write_text(
            """<?xml version="1.0"?>
<robot name="synthetic">
  <link name="base"/>
  <link name="joint_link"/>
  <joint name="joint_a" type="revolute">
    <parent link="base"/>
    <child link="joint_link"/>
    <limit lower="-1.0" upper="1.0" effort="10.0" velocity="5.0"/>
  </joint>
</robot>
""",
            encoding="utf-8",
        )
        metric_config = {
            "unit": "synthetic_unit",
            "description": "Synthetic integration-test threshold.",
            "warning_threshold": 0.1,
            "reject_threshold": 1.0,
            "weight": 1.0,
            "hard_at_reject": True,
            # Only spike/continuity metrics consume these fields; providing
            # them uniformly keeps this synthetic config intentionally small.
            "absolute_floor": 1.0,
            "relative_multiplier": 8.0,
        }
        config = {
            "schema_version": 1,
            "provisional": True,
            "segment_length_seconds": 0.5,
            "robot": {"urdf_path": str(self.urdf)},
            "root": {"body_name": "pelvis"},
            "ground": {
                "z_m": 0.0,
                "foot_body_names": ["left_ankle_roll_link", "right_ankle_roll_link"],
                "sole_local_offsets_m": {
                    "left_ankle_roll_link": [0.0, 0.0, -0.037],
                    "right_ankle_roll_link": [0.0, 0.0, -0.037],
                },
                "contact_height_threshold_m": 0.06,
                "contact_vertical_speed_threshold_mps": 0.2,
            },
            "status": {
                "reject_score_threshold": 0.5,
                "pass_score_threshold": 0.9,
                "minimum_optional_metric_coverage": 0.0,
                "required_metrics": [],
                "optional_metric_coverage_profile": list(METRIC_NAMES),
                "reject_severity_count": 2,
                "borderline_on_warning": True,
            },
            "metrics": {name: dict(metric_config) for name in METRIC_NAMES},
        }
        config["metrics"]["foot_sliding"].update(
            {
                "persistent_reject_frame_ratio": 0.5,
                "persistent_reject_min_contact_samples": 2,
            }
        )
        self.quality_config = self.root / "quality_config.json"
        self.quality_config.write_text(
            json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    @staticmethod
    def _write_motion(path: Path, *, num_frames: int) -> None:
        fps = 10.0
        num_bodies = 3
        joint_pos = np.zeros((num_frames, 1), dtype=np.float64)
        joint_vel = np.zeros_like(joint_pos)
        body_pos_w = np.zeros((num_frames, num_bodies, 3), dtype=np.float64)
        body_pos_w[:, 0, 2] = 1.0
        body_pos_w[:, 1, :] = np.asarray([0.04, 0.10, 0.037])
        body_pos_w[:, 2, :] = np.asarray([0.04, -0.10, 0.037])
        body_quat_w = np.zeros((num_frames, num_bodies, 4), dtype=np.float64)
        body_quat_w[..., 0] = 1.0
        body_lin_vel_w = np.zeros((num_frames, num_bodies, 3), dtype=np.float64)
        body_ang_vel_w = np.zeros((num_frames, num_bodies, 3), dtype=np.float64)
        np.savez_compressed(
            path,
            fps=np.asarray(fps, dtype=np.float64),
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            body_pos_w=body_pos_w,
            body_quat_w=body_quat_w,
            body_lin_vel_w=body_lin_vel_w,
            body_ang_vel_w=body_ang_vel_w,
            joint_names=np.asarray(["joint_a"]),
            body_names=np.asarray(
                ["pelvis", "left_ankle_roll_link", "right_ankle_roll_link"]
            ),
            source_file=np.asarray("synthetic"),
            source_format=np.asarray("unit_test"),
        )

    def _write_split_metadata(self, path: Path, splits: list[str]) -> None:
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=("relative_path", "split", "category", "source_group"),
            )
            writer.writeheader()
            for index, (motion_key, split) in enumerate(zip(self.motion_keys, splits, strict=True)):
                writer.writerow(
                    {
                        "relative_path": motion_key,
                        "split": split,
                        "category": "synthetic",
                        "source_group": f"source_{index}",
                    }
                )

    def _run_builder(
        self,
        output_dir: Path,
        *,
        manifest: Path | None = None,
        dataset_metadata: Path | None = None,
        max_motions: int = 2,
        seed: int = 1234,
    ) -> subprocess.CompletedProcess[str]:
        command = [
            sys.executable,
            str(BUILDER_PATH),
            "--manifest",
            str(self.manifest if manifest is None else manifest),
            "--output-dir",
            str(output_dir),
            "--quality-config",
            str(self.quality_config),
            "--dataset-metadata",
            str(self.dataset_metadata if dataset_metadata is None else dataset_metadata),
            "--urdf-path",
            str(self.urdf),
            "--segment-length-seconds",
            "0.5",
            "--max-motions",
            str(max_motions),
            "--workers",
            "1",
            "--device",
            "cpu",
            "--seed",
            str(seed),
            "--strict",
        ]
        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = "0"
        return subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )

    def _run_successfully(self, output_dir: Path, *, seed: int = 1234) -> None:
        result = self._run_builder(output_dir, seed=seed)
        self.assertEqual(
            result.returncode,
            0,
            msg=f"builder stdout:\n{result.stdout}\nbuilder stderr:\n{result.stderr}",
        )

    def test_required_outputs_normalized_manifest_and_loadable_metadata(self) -> None:
        output_dir = self.root / "output"
        manifest_before = self.manifest.read_bytes()
        self._run_successfully(output_dir)
        self.assertEqual(self.manifest.read_bytes(), manifest_before)

        required_outputs = {
            "segment_quality_metadata.csv",
            "segment_quality_metadata.npz",
            "quality_summary.json",
            "quality_config_resolved.json",
            "quality_review_segments.csv",
            "empty_eligible_motions.csv",
            "normalized_manifest.txt",
        }
        for name in required_outputs:
            with self.subTest(output=name):
                path = output_dir / name
                self.assertTrue(path.is_file())
                self.assertGreater(path.stat().st_size, 0)

        normalized_manifest = output_dir / "normalized_manifest.txt"
        self.assertEqual(
            normalized_manifest.read_text(encoding="utf-8"),
            "motions/motion_0.npz\nmotions/motion_1.npz\n",
        )
        self.assertFalse((output_dir / "audited_manifest.txt").exists())

        with (output_dir / "segment_quality_metadata.csv").open(
            newline="", encoding="utf-8"
        ) as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 5)
        self.assertTrue(
            {
                "schema_version",
                "motion_key",
                "global_segment_id",
                "start_frame",
                "end_frame_exclusive",
                "quality_score",
                "quality_status",
                "optional_metric_coverage",
                "left_foot_min_height",
                "right_foot_min_height",
            }.issubset(rows[0])
        )
        self.assertEqual([int(row["global_segment_id"]) for row in rows], list(range(5)))
        self.assertEqual(
            [row["motion_key"] for row in rows],
            [self.motion_keys[0]] * 3 + [self.motion_keys[1]] * 2,
        )

        metadata_path = output_dir / "segment_quality_metadata.npz"
        metadata = quality_metadata.SegmentQualityMetadata.load(metadata_path)
        summary = json.loads((output_dir / "quality_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata.num_motions, 2)
        self.assertEqual(metadata.num_segments, 5)
        self.assertEqual(metadata.motion_keys.tolist(), self.motion_keys[:2])
        self.assertEqual(summary["motion_count"], 2)
        self.assertEqual(summary["segment_count"], 5)
        self.assertEqual(len(summary["per_motion"]), 2)
        self.assertEqual([item["motion_id"] for item in summary["per_motion"]], [0, 1])
        self.assertEqual(summary["max_motions"], 2)
        self.assertFalse(summary["normalized_manifest_preserves_input_pool"])
        self.assertTrue(summary["selected_pool_preserved_exactly"])
        self.assertEqual(summary["quality_filter_removed_motion_count"], 0)
        self.assertTrue(summary["metadata_match_ok"])
        self.assertIn("empty_eligible_motions", summary)
        self.assertIn("raw_p95", summary["per_metric"]["ground_penetration"])
        self.assertIn("reject_segments", summary)
        self.assertEqual(summary["quality_config_status"]["minimum_optional_metric_coverage"], 0.0)
        self.assertEqual(summary["ground_config"]["foot_body_names"], ["left_ankle_roll_link", "right_ankle_roll_link"])
        self.assertEqual(metadata.metadata_sha256, summary["metadata_npz_sha256"])
        self.assertEqual(metadata.metadata_sha256, quality_metadata.sha256_file(metadata_path))
        self.assertEqual(metadata.manifest_sha256, quality_metadata.sha256_file(normalized_manifest))
        self.assertEqual(metadata.manifest_sha256, summary["manifest_sha256"])
        self.assertEqual(metadata.quality_config_sha256, summary["quality_config_sha256"])
        self.assertTrue(
            metadata.validate_against(
                manifest_path=normalized_manifest,
                motion_keys=metadata.motion_keys.tolist(),
                motion_lengths=metadata.motion_lengths.tolist(),
                motion_fps=metadata.motion_fps.tolist(),
                motion_segment_offsets=metadata.motion_segment_offsets.tolist(),
                segment_start_frames=metadata.start_frame.tolist(),
                segment_end_frames=metadata.end_frame_exclusive.tolist(),
                segment_length_seconds=metadata.segment_length_seconds,
                segment_schema_version=metadata.segment_schema_version,
                pool_fingerprint=metadata.pool_fingerprint,
            )
        )

    def test_same_seed_repeats_semantic_outputs(self) -> None:
        first_output = self.root / "repeat_first"
        second_output = self.root / "repeat_second"
        self._run_successfully(first_output, seed=777)
        self._run_successfully(second_output, seed=777)

        for name in (
            "segment_quality_metadata.csv",
            "quality_config_resolved.json",
            "quality_review_segments.csv",
            "empty_eligible_motions.csv",
            "normalized_manifest.txt",
        ):
            with self.subTest(output=name):
                self.assertEqual(
                    (first_output / name).read_bytes(),
                    (second_output / name).read_bytes(),
                )

        with np.load(first_output / "segment_quality_metadata.npz", allow_pickle=False) as first:
            with np.load(second_output / "segment_quality_metadata.npz", allow_pickle=False) as second:
                self.assertEqual(first.files, second.files)
                for name in first.files:
                    with self.subTest(metadata_field=name):
                        np.testing.assert_array_equal(first[name], second[name])

        first_summary = json.loads(
            (first_output / "quality_summary.json").read_text(encoding="utf-8")
        )
        second_summary = json.loads(
            (second_output / "quality_summary.json").read_text(encoding="utf-8")
        )
        for summary in (first_summary, second_summary):
            summary.pop("generation_timestamp")
            summary.pop("effective_manifest")
            summary.pop("normalized_manifest")
            # np.savez_compressed writes a ZIP container whose timestamp may
            # differ even though every stored semantic array is identical.
            summary.pop("metadata_npz_sha256")
        self.assertEqual(first_summary, second_summary)

    def test_validation_and_test_manifest_names_are_refused(self) -> None:
        for split_name in ("validation", "test"):
            with self.subTest(split=split_name):
                manifest = self.root / f"{split_name}_manifest.txt"
                manifest.write_text(f"{self.motion_keys[0]}\n", encoding="utf-8")
                output_dir = self.root / f"refused_{split_name}"
                result = self._run_builder(output_dir, manifest=manifest, max_motions=1)

                self.assertNotEqual(result.returncode, 0)
                self.assertIn("Validation/Test manifests are refused", result.stderr)
                self.assertFalse((output_dir / "segment_quality_metadata.npz").exists())

    def test_split_metadata_refuses_hidden_validation_or_test_membership(self) -> None:
        for split_name in ("validation", "test"):
            with self.subTest(split=split_name):
                split_metadata = self.root / f"split_metadata_{split_name}.csv"
                self._write_split_metadata(split_metadata, ["train", split_name, "train"])
                output_dir = self.root / f"refused_membership_{split_name}"
                result = self._run_builder(
                    output_dir,
                    dataset_metadata=split_metadata,
                    max_motions=2,
                )

                self.assertNotEqual(result.returncode, 0)
                self.assertIn("according to split metadata", result.stderr)
                self.assertFalse((output_dir / "segment_quality_metadata.npz").exists())

    def test_strict_mode_requires_proven_train_membership(self) -> None:
        output_dir = self.root / "unknown_membership"
        result = self._run_builder(
            output_dir,
            dataset_metadata=self.root / "missing_split_metadata.csv",
            max_motions=1,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("requires split metadata proving every motion belongs to Train", result.stderr)
        self.assertFalse((output_dir / "segment_quality_metadata.npz").exists())


if __name__ == "__main__":
    unittest.main()
