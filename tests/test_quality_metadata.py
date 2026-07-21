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
    / "quality_metadata.py"
)
SPEC = importlib.util.spec_from_file_location("wbt_quality_metadata_for_tests", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Unable to load quality metadata utilities from {MODULE_PATH}")
quality_metadata = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = quality_metadata
SPEC.loader.exec_module(quality_metadata)


class SegmentQualityMetadataTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.manifest = self.root / "motions.txt"
        self.manifest.write_text("motions/a.npz\nmotions/b.npz\n", encoding="utf-8")
        self.payload = dict(
            quality_metadata.metadata_npz_payload(
                segment_schema_version=1,
                segment_length_seconds=1.0,
                manifest_sha256=quality_metadata.sha256_file(self.manifest),
                quality_config_sha256="b" * 64,
                pool_fingerprint="c" * 64,
                motion_keys=["motions/a.npz", "motions/b.npz"],
                motion_lengths=[51, 49],
                motion_fps=[50.0, 24.0],
                motion_segment_offsets=[0, 2, 5],
                global_segment_id=[0, 1, 2, 3, 4],
                motion_id=[0, 0, 1, 1, 1],
                local_segment_id=[0, 1, 0, 1, 2],
                start_frame=[0, 50, 0, 24, 48],
                end_frame_exclusive=[50, 51, 24, 48, 49],
                quality_score=[0.99, 0.75, 0.20, 0.95, 0.70],
                quality_status=[0, 1, 2, 0, 1],
            )
        )
        self.metadata_path = self.root / "segment_quality_metadata.npz"
        np.savez_compressed(self.metadata_path, **self.payload)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_load_and_validate_exact_layout(self) -> None:
        metadata = quality_metadata.SegmentQualityMetadata.load(self.metadata_path)

        self.assertEqual(metadata.num_motions, 2)
        self.assertEqual(metadata.num_segments, 5)
        self.assertEqual(metadata.quality_status.tolist(), [0, 1, 2, 0, 1])
        self.assertEqual(metadata.accepted_mask(include_borderline=True).tolist(), [True, True, False, True, True])
        self.assertEqual(metadata.accepted_mask(include_borderline=False).tolist(), [True, False, False, True, False])
        self.assertTrue(
            metadata.validate_against(
                manifest_path=self.manifest,
                motion_keys=["motions/a.npz", "motions/b.npz"],
                motion_lengths=[51, 49],
                motion_fps=[50.0, 24.0],
                motion_segment_offsets=[0, 2, 5],
                segment_start_frames=[0, 50, 0, 24, 48],
                segment_end_frames=[50, 51, 24, 48, 49],
                segment_length_seconds=1.0,
                segment_schema_version=1,
                pool_fingerprint="c" * 64,
            )
        )
        metrics = metadata.quality_metrics()
        self.assertEqual(metrics["num_pass_segments"], 2)
        self.assertEqual(metrics["num_borderline_segments"], 2)
        self.assertEqual(metrics["num_reject_segments"], 1)

    def test_manifest_order_hash_frame_fps_and_segment_mismatches_fail(self) -> None:
        metadata = quality_metadata.SegmentQualityMetadata.load(self.metadata_path)
        common = dict(
            manifest_path=self.manifest,
            motion_keys=["motions/a.npz", "motions/b.npz"],
            motion_lengths=[51, 49],
            motion_fps=[50.0, 24.0],
            motion_segment_offsets=[0, 2, 5],
            segment_start_frames=[0, 50, 0, 24, 48],
            segment_end_frames=[50, 51, 24, 48, 49],
            segment_length_seconds=1.0,
            segment_schema_version=1,
            pool_fingerprint="c" * 64,
        )
        variants = {
            "manifest SHA256": {"manifest_path": self.root / "other.txt"},
            "manifest motion order": {"motion_keys": ["motions/b.npz", "motions/a.npz"]},
            "motion frame counts": {"motion_lengths": [52, 49]},
            "motion FPS": {"motion_fps": [50.0, 30.0]},
            "segment offsets": {"motion_segment_offsets": [0, 1, 5]},
            "segment start frames": {"segment_start_frames": [0, 49, 0, 24, 48]},
            "segment length": {"segment_length_seconds": 2.0},
            "segment schema": {"segment_schema_version": 2},
            "fingerprint": {"pool_fingerprint": "d" * 64},
        }
        (self.root / "other.txt").write_text(self.manifest.read_text(encoding="utf-8") + "#changed\n")
        for label, override in variants.items():
            with self.subTest(label=label), self.assertRaisesRegex(ValueError, "does not match"):
                metadata.validate_against(**(common | override))

    def test_malformed_status_masks_are_rejected(self) -> None:
        malformed = dict(self.payload)
        malformed["pass_mask"] = np.ones(5, dtype=bool)
        np.savez_compressed(self.metadata_path, **malformed)

        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            quality_metadata.SegmentQualityMetadata.load(self.metadata_path)

    def test_out_of_range_score_and_non_contiguous_ids_are_rejected(self) -> None:
        for field, value, message in (
            ("quality_score", np.asarray([0.9, 0.7, -0.1, 0.8, 0.6]), "scores"),
            ("global_segment_id", np.asarray([0, 1, 3, 2, 4]), "contiguous"),
        ):
            with self.subTest(field=field):
                malformed = dict(self.payload)
                malformed[field] = value
                np.savez_compressed(self.metadata_path, **malformed)
                with self.assertRaisesRegex(ValueError, message):
                    quality_metadata.SegmentQualityMetadata.load(self.metadata_path)

    def test_manifest_reader_uses_order_and_ignores_comments(self) -> None:
        self.manifest.write_text("# header\n\nmotions/b.npz\n motions/a.npz \n", encoding="utf-8")
        self.assertEqual(
            quality_metadata.canonical_manifest_entries(self.manifest),
            ["motions/b.npz", "motions/a.npz"],
        )


if __name__ == "__main__":
    unittest.main()
