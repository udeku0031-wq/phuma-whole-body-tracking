from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    PROJECT_ROOT
    / "source"
    / "whole_body_tracking"
    / "whole_body_tracking"
    / "utils"
    / "difficulty_metadata.py"
)
SPEC = importlib.util.spec_from_file_location("wbt_difficulty_metadata_for_tests", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Unable to load difficulty metadata utilities from {MODULE_PATH}")
difficulty_metadata = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = difficulty_metadata
SPEC.loader.exec_module(difficulty_metadata)


class SegmentDifficultyMetadataTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.manifest = self.root / "train_motions.txt"
        self.manifest.write_text("motions/a.npz\nmotions/b.npz\n", encoding="utf-8")

        feature_values = np.asarray(
            [
                [0.0, 10.0, 1.0],
                [0.5, 11.0, np.nan],
                [1.0, 12.0, 1.0],
                [1.5, 13.0, np.nan],
                [2.0, 14.0, 1.0],
            ],
            dtype=np.float64,
        )
        feature_available = np.isfinite(feature_values)
        feature_z = np.where(feature_available, feature_values / 10.0, np.nan)
        self.payload = dict(
            difficulty_metadata.metadata_npz_payload(
                algorithm_schema_version="wbt.intrinsic_difficulty.test.v1",
                segment_schema_version=1,
                segment_length_seconds=1.0,
                manifest_sha256=difficulty_metadata.sha256_file(self.manifest),
                profile_sha256="b" * 64,
                difficulty_config_sha256="c" * 64,
                pool_fingerprint="d" * 64,
                num_bins=10,
                motion_keys=["motions/a.npz", "motions/b.npz"],
                motion_lengths=[51, 49],
                motion_fps=[50.0, 24.0],
                motion_segment_offsets=[0, 2, 5],
                global_segment_id=[0, 1, 2, 3, 4],
                motion_id=[0, 0, 1, 1, 1],
                local_segment_id=[0, 1, 0, 1, 2],
                start_frame=[0, 50, 0, 24, 48],
                end_frame_exclusive=[50, 51, 24, 48, 49],
                duration_seconds=[1.0, 0.02, 1.0, 1.0, 1.0 / 24.0],
                difficulty_raw=[-2.0, -0.5, 0.0, 0.5, 2.0],
                difficulty_score=[0.0, 0.25, 0.5, 0.75, 1.0],
                difficulty_bin=[0, 2, 5, 7, 9],
                feature_names=["speed", "acceleration", "contact"],
                feature_values=feature_values,
                feature_z=feature_z,
                feature_available_mask=feature_available,
                optional_feature_coverage=[1.0, 0.5, 1.0, 0.5, 1.0],
                near_constant_features=["contact"],
            )
        )
        self.metadata_path = self.root / "segment_difficulty_metadata.npz"
        np.savez_compressed(self.metadata_path, **self.payload)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _validation_arguments(self) -> dict[str, object]:
        return {
            "manifest_path": self.manifest,
            "motion_keys": ["motions/a.npz", "motions/b.npz"],
            "motion_lengths": [51, 49],
            "motion_fps": [50.0, 24.0],
            "motion_segment_offsets": [0, 2, 5],
            "segment_start_frames": [0, 50, 0, 24, 48],
            "segment_end_frames": [50, 51, 24, 48, 49],
            "segment_length_seconds": 1.0,
            "segment_schema_version": 1,
            "pool_fingerprint": "d" * 64,
            "segment_global_ids": [0, 1, 2, 3, 4],
            "segment_motion_ids": [0, 0, 1, 1, 1],
            "segment_local_ids": [0, 1, 0, 1, 2],
            "segment_duration_seconds": [1.0, 0.02, 1.0, 1.0, 1.0 / 24.0],
            "expected_profile_sha256": "b" * 64,
            "expected_difficulty_config_sha256": "c" * 64,
            "expected_num_bins": 10,
        }

    def test_payload_load_identity_metrics_and_exact_validation(self) -> None:
        metadata = difficulty_metadata.SegmentDifficultyMetadata.load(self.metadata_path)

        self.assertEqual(metadata.schema_version, "wbt.segment_difficulty.v1")
        self.assertEqual(metadata.num_motions, 2)
        self.assertEqual(metadata.num_segments, 5)
        self.assertEqual(metadata.num_features, 3)
        self.assertEqual(metadata.motion_keys.tolist(), ["motions/a.npz", "motions/b.npz"])
        self.assertEqual(metadata.near_constant_mask.tolist(), [False, False, True])
        self.assertEqual(metadata.available_feature_count.tolist(), [3, 2, 3, 2, 3])
        self.assertEqual(metadata.difficulty_bin.tolist(), [0, 2, 5, 7, 9])
        self.assertTrue(metadata.validate_against(**self._validation_arguments()))

        identity = metadata.identity_state()
        self.assertEqual(identity["metadata_path"], str(self.metadata_path.resolve()))
        self.assertEqual(identity["metadata_sha256"], difficulty_metadata.sha256_file(self.metadata_path))
        self.assertEqual(identity["profile_sha256"], "b" * 64)
        self.assertEqual(identity["difficulty_config_sha256"], "c" * 64)
        self.assertEqual(identity["manifest_sha256"], difficulty_metadata.sha256_file(self.manifest))
        self.assertEqual(identity["pool_fingerprint"], "d" * 64)
        self.assertEqual(identity["num_bins"], 10)

        metrics = metadata.difficulty_metrics()
        self.assertEqual(metrics["num_segments"], 5)
        self.assertEqual(metrics["num_motions"], 2)
        self.assertEqual(metrics["num_bins"], 10)
        self.assertEqual(metrics["near_constant_feature_count"], 1)
        self.assertEqual(metrics["metadata_match_ok"], 1)
        self.assertEqual(sum(int(metrics[f"bin_{index}_count"]) for index in range(10)), 5)

    def test_payload_rejects_inconsistent_available_feature_count(self) -> None:
        arguments = {
            "algorithm_schema_version": "test",
            "segment_schema_version": 1,
            "segment_length_seconds": 1.0,
            "manifest_sha256": "a" * 64,
            "profile_sha256": "b" * 64,
            "difficulty_config_sha256": "c" * 64,
            "pool_fingerprint": "d" * 64,
            "num_bins": 10,
            "motion_keys": ["a.npz"],
            "motion_lengths": [1],
            "motion_fps": [1.0],
            "motion_segment_offsets": [0, 1],
            "global_segment_id": [0],
            "motion_id": [0],
            "local_segment_id": [0],
            "start_frame": [0],
            "end_frame_exclusive": [1],
            "duration_seconds": [1.0],
            "difficulty_raw": [0.0],
            "difficulty_score": [0.5],
            "difficulty_bin": [5],
            "feature_names": ["feature"],
            "feature_values": [[1.0]],
            "feature_z": [[0.0]],
            "feature_available_mask": [[True]],
            "optional_feature_coverage": [1.0],
            "near_constant_features": [],
            "available_feature_count": [0],
        }
        with self.assertRaisesRegex(ValueError, "available_feature_count"):
            difficulty_metadata.metadata_npz_payload(**arguments)

    def test_loader_rejects_layout_hash_fps_bounds_and_bin_corruption(self) -> None:
        malformed_cases = {
            "missing field": ("feature_z", None, "missing fields"),
            "hash": ("profile_sha256", np.asarray("not-a-digest"), "SHA256"),
            "fps": ("motion_fps", np.asarray([50.0, 0.0]), "FPS"),
            "start bound": (
                "start_frame",
                np.asarray([0, 49, 0, 24, 48]),
                "start_frame",
            ),
            "end bound": (
                "end_frame_exclusive",
                np.asarray([50, 51, 24, 47, 49]),
                "end_frame_exclusive",
            ),
            "bin": ("difficulty_bin", np.asarray([0, 2, 5, 7, 10]), "difficulty_bin"),
        }
        for label, (field, replacement, message) in malformed_cases.items():
            with self.subTest(case=label):
                malformed = dict(self.payload)
                if replacement is None:
                    malformed.pop(field)
                else:
                    malformed[field] = replacement
                path = self.root / f"malformed_{label.replace(' ', '_')}.npz"
                np.savez_compressed(path, **malformed)
                with self.assertRaisesRegex(ValueError, message):
                    difficulty_metadata.SegmentDifficultyMetadata.load(path)

        # Keep the corrupted offsets, motion IDs, and local IDs internally
        # consistent so the loader must specifically reject their incompatible
        # Stage-0 fixed-length segmentation rather than a simpler shape error.
        incompatible_layout = dict(self.payload)
        incompatible_layout["motion_segment_offsets"] = np.asarray([0, 3, 5])
        incompatible_layout["motion_id"] = np.asarray([0, 0, 0, 1, 1])
        incompatible_layout["local_segment_id"] = np.asarray([0, 1, 2, 0, 1])
        path = self.root / "malformed_self_consistent_layout.npz"
        np.savez_compressed(path, **incompatible_layout)
        with self.assertRaisesRegex(ValueError, "Stage 0 segment layout"):
            difficulty_metadata.SegmentDifficultyMetadata.load(path)

    def test_validate_against_reports_every_identity_and_layout_mismatch(self) -> None:
        metadata = difficulty_metadata.SegmentDifficultyMetadata.load(self.metadata_path)
        other_manifest = self.root / "other_train_motions.txt"
        other_manifest.write_text(self.manifest.read_text(encoding="utf-8") + "# changed\n", encoding="utf-8")
        common = self._validation_arguments()
        variants = {
            "manifest SHA256": {"manifest_path": other_manifest},
            "manifest motion order": {"motion_keys": ["motions/b.npz", "motions/a.npz"]},
            "motion frame counts": {"motion_lengths": [52, 49]},
            "motion FPS": {"motion_fps": [50.0, 25.0]},
            "segment offsets/global count": {"motion_segment_offsets": [0, 1, 5]},
            "segment start frames": {"segment_start_frames": [0, 49, 0, 24, 48]},
            "segment end frames": {"segment_end_frames": [50, 51, 24, 47, 49]},
            "segment length": {"segment_length_seconds": 0.5},
            "segment schema version": {"segment_schema_version": 2},
            "ordered motion pool fingerprint": {"pool_fingerprint": "e" * 64},
            "global segment IDs": {"segment_global_ids": [0, 1, 3, 2, 4]},
            "segment motion IDs": {"segment_motion_ids": [0, 1, 0, 1, 1]},
            "local segment IDs": {"segment_local_ids": [0, 0, 1, 1, 2]},
            "segment durations": {"segment_duration_seconds": [1.0, 0.03, 1.0, 1.0, 1.0 / 24.0]},
            "difficulty profile SHA256": {"expected_profile_sha256": "f" * 64},
            "difficulty config SHA256": {"expected_difficulty_config_sha256": "f" * 64},
            "difficulty bin count": {"expected_num_bins": 9},
        }
        for message, override in variants.items():
            arguments = common | override
            with self.subTest(mismatch=message):
                with self.assertRaisesRegex(ValueError, message):
                    metadata.validate_against(**arguments)
                self.assertFalse(metadata.validate_against(**arguments, strict=False))

    def test_manifest_reader_preserves_order_and_ignores_comments(self) -> None:
        self.manifest.write_text("# header\n\nmotions/b.npz\n motions/a.npz \n", encoding="utf-8")
        self.assertEqual(
            difficulty_metadata.canonical_manifest_entries(self.manifest),
            ["motions/b.npz", "motions/a.npz"],
        )


if __name__ == "__main__":
    unittest.main()
