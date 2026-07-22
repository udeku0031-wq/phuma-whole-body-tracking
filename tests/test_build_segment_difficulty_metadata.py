from __future__ import annotations

import csv
import hashlib
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
BUILDER_PATH = PROJECT_ROOT / "scripts" / "build_segment_difficulty_metadata.py"
DIFFICULTY_CONFIG_PATH = PROJECT_ROOT / "configs" / "difficulty" / "g1_segment_difficulty.yaml"
UTILS_DIR = (
    PROJECT_ROOT / "source" / "whole_body_tracking" / "whole_body_tracking" / "utils"
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


difficulty_metadata = _load_module(
    "wbt_difficulty_metadata_builder_tests", UTILS_DIR / "difficulty_metadata.py"
)
quality_metadata = _load_module(
    "wbt_quality_metadata_for_difficulty_builder_tests", UTILS_DIR / "quality_metadata.py"
)


OUTPUT_FILENAMES = {
    "segment_difficulty_metadata.csv",
    "segment_difficulty_metadata.npz",
    "motion_difficulty_metadata.csv",
    "motion_difficulty_metadata.npz",
    "difficulty_profile.json",
    "difficulty_summary.json",
    "difficulty_feature_statistics.csv",
    "difficulty_review_segments.csv",
    "difficulty_config_resolved.json",
    "normalized_manifest.txt",
}


class BuildSegmentDifficultyMetadataTest(unittest.TestCase):
    FPS = 20.0
    NUM_FRAMES = 201
    SEGMENT_LENGTH_SECONDS = 0.5

    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp_dir.name)
        cls.motion_dir = cls.root / "motions"
        cls.motion_dir.mkdir()
        cls.motion_keys = [f"motions/motion_{index}.npz" for index in range(3)]
        cls.motion_paths = [cls.root / key for key in cls.motion_keys]
        for index, path in enumerate(cls.motion_paths):
            cls._write_motion(path, motion_index=index)

        cls.manifest = cls.root / "synthetic_train_manifest.txt"
        cls.manifest.write_text(
            "# ordered synthetic WBT Train pool\n"
            + "".join(f"{motion_key}\n" for motion_key in cls.motion_keys),
            encoding="utf-8",
        )
        cls.manifest_before = cls.manifest.read_bytes()
        cls.baseline_output = cls.root / "baseline"
        cls._run_successfully(cls.baseline_output, seed=2026)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp_dir.cleanup()

    @classmethod
    def _write_motion(cls, path: Path, *, motion_index: int) -> None:
        fps = cls.FPS
        time = np.arange(cls.NUM_FRAMES, dtype=np.float64) / fps
        phase = 0.43 * motion_index
        motion_scale = 1.0 + 0.35 * motion_index

        joint_columns: list[np.ndarray] = []
        for joint_index in range(6):
            frequency = 0.22 + 0.055 * joint_index + 0.035 * motion_index
            amplitude = (0.22 + 0.025 * joint_index) * motion_scale
            angle = 2.0 * np.pi * frequency * time + phase + 0.015 * (joint_index + 1) * time**2
            joint_columns.append(
                amplitude * (np.sin(angle) + 0.18 * np.sin(0.47 * angle + joint_index))
            )
        joint_pos = np.stack(joint_columns, axis=1)
        joint_vel = np.gradient(joint_pos, 1.0 / fps, axis=0, edge_order=1)

        body_names = np.asarray(
            [
                "pelvis",
                "left_wrist_yaw_link",
                "right_wrist_yaw_link",
                "left_ankle_roll_link",
                "right_ankle_roll_link",
            ]
        )
        body_pos_w = np.zeros((cls.NUM_FRAMES, body_names.size, 3), dtype=np.float64)

        root_frequency = 0.16 + 0.025 * motion_index
        root_x = (
            (0.08 + 0.035 * motion_index) * time
            + 0.06 * motion_scale * np.sin(2.0 * np.pi * root_frequency * time + 0.018 * time**2)
        )
        root_y = 0.04 * motion_scale * np.sin(
            2.0 * np.pi * (root_frequency * 0.73) * time + phase + 0.012 * time**2
        )
        root_z = (
            0.78
            + (0.035 + 0.012 * motion_index)
            * np.sin(2.0 * np.pi * (0.19 + 0.035 * motion_index) * time + phase)
            + 0.012 * np.sin(0.13 * time**2 + 0.2 * motion_index)
        )
        body_pos_w[:, 0] = np.stack((root_x, root_y, root_z), axis=1)

        hand_angle = 2.0 * np.pi * (0.31 + 0.05 * motion_index) * time + phase + 0.01 * time**2
        body_pos_w[:, 1, 0] = root_x + 0.24 + 0.10 * motion_scale * np.sin(hand_angle)
        body_pos_w[:, 1, 1] = root_y + 0.22 + 0.04 * np.cos(0.8 * hand_angle)
        body_pos_w[:, 1, 2] = root_z + 0.12 + 0.08 * np.cos(1.17 * hand_angle)
        body_pos_w[:, 2, 0] = root_x + 0.24 + 0.09 * motion_scale * np.sin(hand_angle + np.pi)
        body_pos_w[:, 2, 1] = root_y - 0.22 - 0.04 * np.cos(0.82 * hand_angle)
        body_pos_w[:, 2, 2] = root_z + 0.12 + 0.07 * np.cos(1.11 * hand_angle + 0.5)

        gait_angle = (
            2.0 * np.pi * (0.58 + 0.09 * motion_index) * time
            + phase
            + 0.014 * (motion_index + 1) * time**2
        )
        left_lift = (0.10 + 0.018 * motion_index) * np.maximum(np.sin(gait_angle), 0.0)
        right_lift = (0.10 + 0.018 * motion_index) * np.maximum(-np.sin(gait_angle), 0.0)
        stride = (0.13 + 0.025 * motion_index) * np.sin(gait_angle)
        body_pos_w[:, 3, 0] = root_x - 0.05 + stride
        body_pos_w[:, 3, 1] = root_y + 0.10
        body_pos_w[:, 3, 2] = 0.037 + left_lift
        body_pos_w[:, 4, 0] = root_x - 0.05 - stride
        body_pos_w[:, 4, 1] = root_y - 0.10
        body_pos_w[:, 4, 2] = 0.037 + right_lift

        body_quat_w = np.zeros((cls.NUM_FRAMES, body_names.size, 4), dtype=np.float64)
        body_quat_w[..., 0] = 1.0
        yaw = (
            0.14 * motion_scale * np.sin(2.0 * np.pi * (0.14 + 0.02 * motion_index) * time + phase)
            + 0.018 * (motion_index + 1) * time
            + 0.025 * np.sin(0.09 * time**2)
        )
        body_quat_w[:, 0, 0] = np.cos(0.5 * yaw)
        body_quat_w[:, 0, 3] = np.sin(0.5 * yaw)

        body_lin_vel_w = np.gradient(body_pos_w, 1.0 / fps, axis=0, edge_order=1)
        body_ang_vel_w = np.zeros((cls.NUM_FRAMES, body_names.size, 3), dtype=np.float64)
        body_ang_vel_w[:, 0, 2] = np.gradient(yaw, 1.0 / fps, edge_order=1)
        np.savez_compressed(
            path,
            fps=np.asarray(fps, dtype=np.float64),
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            body_pos_w=body_pos_w,
            body_quat_w=body_quat_w,
            body_lin_vel_w=body_lin_vel_w,
            body_ang_vel_w=body_ang_vel_w,
            joint_names=np.asarray([f"joint_{index}" for index in range(joint_pos.shape[1])]),
            body_names=body_names,
            source_file=np.asarray(f"synthetic_source_{motion_index}"),
            source_format=np.asarray("unit_test_wbt"),
        )

    @classmethod
    def _run_builder(
        cls,
        output_dir: Path,
        *,
        mode: str = "fit_transform",
        profile: Path | None = None,
        quality_path: Path | None = None,
        dataset_path: Path | None = None,
        seed: int = 2026,
    ) -> subprocess.CompletedProcess[str]:
        command = [
            sys.executable,
            str(BUILDER_PATH),
            "--manifest",
            str(cls.manifest),
            "--output-dir",
            str(output_dir),
            "--difficulty-config",
            str(DIFFICULTY_CONFIG_PATH),
            "--segment-length-seconds",
            str(cls.SEGMENT_LENGTH_SECONDS),
            "--mode",
            mode,
            "--workers",
            "1",
            "--device",
            "cpu",
            "--seed",
            str(seed),
            "--strict",
        ]
        if profile is not None:
            command.extend(("--profile", str(profile)))
        if quality_path is not None:
            command.extend(("--quality-metadata", str(quality_path)))
        if dataset_path is not None:
            command.extend(("--dataset-metadata", str(dataset_path)))
        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = "0"
        return subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )

    @classmethod
    def _run_successfully(cls, output_dir: Path, **kwargs: object) -> None:
        result = cls._run_builder(output_dir, **kwargs)
        if result.returncode != 0:
            raise AssertionError(
                f"builder stdout:\n{result.stdout}\nbuilder stderr:\n{result.stderr}"
            )

    @staticmethod
    def _assert_npz_arrays_equal(first_path: Path, second_path: Path) -> None:
        with np.load(first_path, allow_pickle=False) as first:
            with np.load(second_path, allow_pickle=False) as second:
                if first.files != second.files:
                    raise AssertionError(f"NPZ fields differ: {first.files!r} != {second.files!r}")
                for name in first.files:
                    np.testing.assert_array_equal(first[name], second[name], err_msg=name)

    @classmethod
    def _motion_pool_fingerprint(cls) -> str:
        normalized_paths = [str(path.resolve()) for path in cls.motion_paths]
        common_root = os.path.commonpath([os.path.dirname(path) for path in normalized_paths])
        digest = hashlib.sha256()
        for path in normalized_paths:
            digest.update(os.path.relpath(path, common_root).encode("utf-8"))
            digest.update(b"\0")
            digest.update(str(os.path.getsize(path)).encode("ascii"))
            digest.update(b"\0")
            digest.update(str(cls.NUM_FRAMES).encode("ascii"))
            digest.update(b"\0")
            digest.update(format(cls.FPS, ".17g").encode("ascii"))
            digest.update(b"\n")
        return digest.hexdigest()

    @classmethod
    def _write_quality_metadata(cls, path: Path, *, variant: int) -> None:
        segment_frames = int(round(cls.FPS * cls.SEGMENT_LENGTH_SECONDS))
        segments_per_motion = (cls.NUM_FRAMES + segment_frames - 1) // segment_frames
        offsets = np.arange(4, dtype=np.int64) * segments_per_motion
        motion_ids = np.repeat(np.arange(3, dtype=np.int64), segments_per_motion)
        local_ids = np.tile(np.arange(segments_per_motion, dtype=np.int64), 3)
        starts = local_ids * segment_frames
        ends = np.minimum(starts + segment_frames, cls.NUM_FRAMES)
        num_segments = int(offsets[-1])
        if variant == 1:
            quality_scores = np.full(num_segments, 0.99, dtype=np.float64)
            quality_status = np.zeros(num_segments, dtype=np.int8)
        else:
            quality_scores = np.linspace(0.05, 0.95, num_segments, dtype=np.float64)
            quality_status = np.arange(num_segments, dtype=np.int8) % 3
        payload = quality_metadata.metadata_npz_payload(
            segment_schema_version=1,
            segment_length_seconds=cls.SEGMENT_LENGTH_SECONDS,
            manifest_sha256=difficulty_metadata.sha256_file(cls.manifest),
            quality_config_sha256=("a" if variant == 1 else "b") * 64,
            pool_fingerprint=cls._motion_pool_fingerprint(),
            motion_keys=cls.motion_keys,
            motion_lengths=[cls.NUM_FRAMES] * 3,
            motion_fps=[cls.FPS] * 3,
            motion_segment_offsets=offsets,
            global_segment_id=np.arange(num_segments),
            motion_id=motion_ids,
            local_segment_id=local_ids,
            start_frame=starts,
            end_frame_exclusive=ends,
            quality_score=quality_scores,
            quality_status=quality_status,
        )
        np.savez_compressed(path, **payload)

    def test_fit_transform_writes_all_outputs_and_loadable_ten_bin_metadata(self) -> None:
        output_dir = self.baseline_output
        self.assertEqual(self.manifest.read_bytes(), self.manifest_before)
        self.assertEqual({path.name for path in output_dir.iterdir()}, OUTPUT_FILENAMES)
        for name in OUTPUT_FILENAMES:
            with self.subTest(output=name):
                path = output_dir / name
                self.assertTrue(path.is_file())
                self.assertGreater(path.stat().st_size, 0)

        metadata = difficulty_metadata.SegmentDifficultyMetadata.load(
            output_dir / "segment_difficulty_metadata.npz"
        )
        self.assertEqual(metadata.num_motions, 3)
        self.assertEqual(metadata.num_segments, 63)
        self.assertEqual(metadata.num_bins, 10)
        self.assertEqual(metadata.motion_keys.tolist(), self.motion_keys)
        self.assertEqual(metadata.motion_segment_offsets.tolist(), [0, 21, 42, 63])
        self.assertEqual(metadata.global_segment_id.tolist(), list(range(63)))
        self.assertEqual(set(metadata.difficulty_bin.tolist()), set(range(10)))
        self.assertTrue(np.all((metadata.difficulty_score >= 0.0) & (metadata.difficulty_score <= 1.0)))
        self.assertEqual(
            metadata.profile_sha256,
            difficulty_metadata.sha256_file(output_dir / "difficulty_profile.json"),
        )
        self.assertEqual(metadata.manifest_sha256, difficulty_metadata.sha256_file(self.manifest))

        with (output_dir / "segment_difficulty_metadata.csv").open(
            newline="", encoding="utf-8"
        ) as stream:
            segment_rows = list(csv.DictReader(stream))
        self.assertEqual(len(segment_rows), metadata.num_segments)
        self.assertEqual([int(row["global_segment_id"]) for row in segment_rows], list(range(63)))
        self.assertTrue(
            {"difficulty_raw", "difficulty_score", "difficulty_bin", "top_contribution_1"}.issubset(
                segment_rows[0]
            )
        )

        with (output_dir / "motion_difficulty_metadata.csv").open(
            newline="", encoding="utf-8"
        ) as stream:
            motion_rows = list(csv.DictReader(stream))
        self.assertEqual(len(motion_rows), 3)
        with np.load(output_dir / "motion_difficulty_metadata.npz", allow_pickle=False) as motion_npz:
            self.assertEqual(motion_npz["motion_id"].tolist(), [0, 1, 2])
            self.assertEqual(motion_npz["motion_keys"].tolist(), self.motion_keys)
            self.assertEqual(motion_npz["segment_count"].tolist(), [21, 21, 21])

        profile = json.loads((output_dir / "difficulty_profile.json").read_text(encoding="utf-8"))
        summary = json.loads((output_dir / "difficulty_summary.json").read_text(encoding="utf-8"))
        resolved_config = json.loads(
            (output_dir / "difficulty_config_resolved.json").read_text(encoding="utf-8")
        )
        self.assertEqual(profile["num_bins"], 10)
        self.assertEqual(len(profile["difficulty_bin_edges"]), 9)
        self.assertEqual(summary["mode"], "fit_transform")
        self.assertTrue(summary["policy_independent"])
        self.assertFalse(summary["quality_affects_score"])
        self.assertEqual(summary["bin_counts"], np.bincount(metadata.difficulty_bin, minlength=10).tolist())
        self.assertTrue(all(count > 0 for count in summary["bin_counts"]))
        self.assertEqual(resolved_config["num_bins"], 10)
        self.assertEqual(resolved_config["segment_length_seconds"], self.SEGMENT_LENGTH_SECONDS)

        with (output_dir / "difficulty_feature_statistics.csv").open(
            newline="", encoding="utf-8"
        ) as stream:
            feature_rows = list(csv.DictReader(stream))
        with (output_dir / "difficulty_review_segments.csv").open(
            newline="", encoding="utf-8"
        ) as stream:
            review_rows = list(csv.DictReader(stream))
        self.assertEqual(len(feature_rows), metadata.num_features)
        self.assertGreater(len(review_rows), 0)
        self.assertEqual(
            (output_dir / "normalized_manifest.txt").read_text(encoding="utf-8"),
            "".join(f"{key}\n" for key in self.motion_keys),
        )

        split_metadata = self.root / "mixed_split_metadata.csv"
        split_metadata.write_text(
            "relative_path,split\n"
            + "".join(
                f"{key},{'validation' if index == 1 else 'train'}\n"
                for index, key in enumerate(self.motion_keys)
            ),
            encoding="utf-8",
        )
        leaked_output = self.root / "mixed_split_output"
        result = self._run_builder(leaked_output, dataset_path=split_metadata)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("non-Train", result.stderr)
        self.assertFalse((leaked_output / "difficulty_profile.json").exists())

    def test_same_seed_repeats_every_semantic_output(self) -> None:
        repeated_output = self.root / "repeated"
        self._run_successfully(repeated_output, seed=2026)

        for name in sorted(OUTPUT_FILENAMES - {
            "segment_difficulty_metadata.npz",
            "motion_difficulty_metadata.npz",
        }):
            with self.subTest(output=name):
                self.assertEqual(
                    (self.baseline_output / name).read_bytes(),
                    (repeated_output / name).read_bytes(),
                )
        self._assert_npz_arrays_equal(
            self.baseline_output / "segment_difficulty_metadata.npz",
            repeated_output / "segment_difficulty_metadata.npz",
        )
        self._assert_npz_arrays_equal(
            self.baseline_output / "motion_difficulty_metadata.npz",
            repeated_output / "motion_difficulty_metadata.npz",
        )

    def test_changing_matching_quality_labels_cannot_change_difficulty(self) -> None:
        first_quality = self.root / "quality_all_pass.npz"
        second_quality = self.root / "quality_mixed.npz"
        self._write_quality_metadata(first_quality, variant=1)
        self._write_quality_metadata(second_quality, variant=2)
        first_output = self.root / "quality_first"
        second_output = self.root / "quality_second"
        self._run_successfully(first_output, quality_path=first_quality, seed=17)
        self._run_successfully(second_output, quality_path=second_quality, seed=17)

        for name in (
            "segment_difficulty_metadata.csv",
            "motion_difficulty_metadata.csv",
            "difficulty_profile.json",
            "difficulty_feature_statistics.csv",
            "difficulty_config_resolved.json",
            "normalized_manifest.txt",
        ):
            with self.subTest(output=name):
                self.assertEqual((first_output / name).read_bytes(), (second_output / name).read_bytes())
        for name in ("segment_difficulty_metadata.npz", "motion_difficulty_metadata.npz"):
            self._assert_npz_arrays_equal(first_output / name, second_output / name)

        first_summary = json.loads((first_output / "difficulty_summary.json").read_text(encoding="utf-8"))
        second_summary = json.loads((second_output / "difficulty_summary.json").read_text(encoding="utf-8"))
        first_cross = first_summary.pop("quality_cross_statistics")
        second_cross = second_summary.pop("quality_cross_statistics")
        self.assertEqual(first_summary, second_summary)
        for cross in (first_cross, second_cross):
            self.assertTrue(cross["mapping_match_ok"])
            self.assertEqual(cross["segment_count_before"], 63)
            self.assertEqual(cross["segment_count_after"], 63)
            self.assertIn("did not change any score, bin, or row", cross["note"])

    def test_transform_reuses_frozen_profile_without_refitting(self) -> None:
        transformed_output = self.root / "transformed"
        profile_path = self.baseline_output / "difficulty_profile.json"
        self._run_successfully(
            transformed_output,
            mode="transform",
            profile=profile_path,
            seed=2026,
        )

        self.assertEqual(
            profile_path.read_bytes(),
            (transformed_output / "difficulty_profile.json").read_bytes(),
        )
        self._assert_npz_arrays_equal(
            self.baseline_output / "segment_difficulty_metadata.npz",
            transformed_output / "segment_difficulty_metadata.npz",
        )
        self._assert_npz_arrays_equal(
            self.baseline_output / "motion_difficulty_metadata.npz",
            transformed_output / "motion_difficulty_metadata.npz",
        )
        metadata = difficulty_metadata.SegmentDifficultyMetadata.load(
            transformed_output / "segment_difficulty_metadata.npz"
        )
        self.assertEqual(metadata.profile_sha256, difficulty_metadata.sha256_file(profile_path))
        summary = json.loads((transformed_output / "difficulty_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["mode"], "transform")
        self.assertEqual(summary["profile_sha256"], metadata.profile_sha256)

        incompatible_profile = self.root / "incompatible_segment_schema_profile.json"
        profile_payload = json.loads(profile_path.read_text(encoding="utf-8"))
        profile_payload["segment_schema_version"] = 999
        incompatible_profile.write_text(
            json.dumps(profile_payload, ensure_ascii=False, sort_keys=True), encoding="utf-8"
        )
        incompatible_output = self.root / "incompatible_segment_schema_output"
        result = self._run_builder(
            incompatible_output,
            mode="transform",
            profile=incompatible_profile,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("segment schema", result.stderr)
        self.assertFalse((incompatible_output / "difficulty_profile.json").exists())


if __name__ == "__main__":
    unittest.main()
