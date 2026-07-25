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
    / "cluster_metadata.py"
)
SPEC = importlib.util.spec_from_file_location("wbt_cluster_metadata_for_tests", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Unable to load cluster metadata utilities from {MODULE_PATH}")
cluster_metadata = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = cluster_metadata
SPEC.loader.exec_module(cluster_metadata)


class MotionClusterMetadataTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.manifest = self.root / "train_motions.txt"
        self.manifest.write_text(
            "motions/a.npz\nmotions/b.npz\nmotions/c.npz\nmotions/d.npz\n",
            encoding="utf-8",
        )
        self.payload = dict(
            cluster_metadata.metadata_npz_payload(
                algorithm_schema_version=(
                    cluster_metadata.MOTION_CLUSTERING_ALGORITHM_SCHEMA_VERSION
                ),
                manifest_sha256=cluster_metadata.sha256_file(self.manifest),
                pool_fingerprint="a" * 64,
                difficulty_metadata_sha256="b" * 64,
                difficulty_profile_sha256="c" * 64,
                cluster_profile_sha256="d" * 64,
                cluster_config_sha256="e" * 64,
                motion_keys=[
                    "motions/a.npz",
                    "motions/b.npz",
                    "motions/c.npz",
                    "motions/d.npz",
                ],
                motion_lengths=[51, 49, 100, 25],
                motion_fps=[50.0, 24.0, 50.0, 25.0],
                motion_segment_offsets=[0, 2, 5, 7, 8],
                motion_id=[0, 1, 2, 3],
                cluster_id=[0, 1, 0, 2],
                cluster_sizes=[2, 1, 1],
                centroids=[
                    [-0.75, -0.25],
                    [0.5, 0.25],
                    [1.0, 1.5],
                ],
                feature_names=["root_speed__mean", "flight_ratio__mean"],
                motion_feature_matrix=[
                    [0.5, 0.0],
                    [1.0, 0.1],
                    [0.6, 0.0],
                    [2.0, 0.5],
                ],
                standardized_feature_matrix=[
                    [-1.0, -0.5],
                    [0.5, 0.25],
                    [-0.5, 0.0],
                    [1.0, 1.5],
                ],
            )
        )
        self.metadata_path = self.root / "motion_cluster_metadata.npz"
        np.savez_compressed(self.metadata_path, **self.payload)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _validation_arguments(self) -> dict[str, object]:
        return {
            "manifest_path": self.manifest,
            "motion_keys": [
                "motions/a.npz",
                "motions/b.npz",
                "motions/c.npz",
                "motions/d.npz",
            ],
            "motion_lengths": [51, 49, 100, 25],
            "motion_fps": [50.0, 24.0, 50.0, 25.0],
            "motion_segment_offsets": [0, 2, 5, 7, 8],
            "pool_fingerprint": "a" * 64,
            "difficulty_metadata_sha256": "b" * 64,
            "difficulty_profile_sha256": "c" * 64,
            "expected_num_clusters": 3,
        }

    def test_payload_load_identity_and_exact_validation(self) -> None:
        metadata = cluster_metadata.MotionClusterMetadata.load(self.metadata_path)

        self.assertEqual(metadata.schema_version, "wbt.motion_cluster.v1")
        self.assertEqual(metadata.num_motions, 4)
        self.assertEqual(metadata.num_segments, 8)
        self.assertEqual(metadata.num_clusters, 3)
        self.assertEqual(metadata.num_features, 2)
        self.assertEqual(metadata.motion_ids.tolist(), [0, 1, 2, 3])
        self.assertEqual(metadata.cluster_ids.tolist(), [0, 1, 0, 2])
        self.assertEqual(metadata.cluster_sizes.tolist(), [2, 1, 1])
        self.assertTrue(metadata.validate_against(**self._validation_arguments()))

        identity = metadata.identity_state()
        self.assertEqual(identity["metadata_path"], str(self.metadata_path.resolve()))
        self.assertEqual(
            identity["metadata_sha256"],
            cluster_metadata.sha256_file(self.metadata_path),
        )
        self.assertEqual(identity["cluster_profile_sha256"], "d" * 64)
        self.assertEqual(identity["cluster_config_sha256"], "e" * 64)
        self.assertEqual(identity["manifest_sha256"], cluster_metadata.sha256_file(self.manifest))
        self.assertEqual(identity["pool_fingerprint"], "a" * 64)
        self.assertEqual(identity["difficulty_metadata_sha256"], "b" * 64)
        self.assertEqual(identity["difficulty_profile_sha256"], "c" * 64)
        self.assertEqual(identity["num_clusters"], 3)

    def test_loader_rejects_corrupt_structure_hash_ids_sizes_and_centroids(self) -> None:
        malformed_cases = {
            "missing field": ("feature_names", None, "missing fields"),
            "hash": (
                "difficulty_metadata_sha256",
                np.asarray("not-a-digest"),
                "SHA256",
            ),
            "duplicate motion ID": (
                "motion_id",
                np.asarray([0, 1, 1, 3]),
                "exactly one unique ID",
            ),
            "cluster range": (
                "cluster_id",
                np.asarray([0, 1, 0, 3]),
                "cluster_id",
            ),
            "cluster sizes": (
                "cluster_sizes",
                np.asarray([1, 2, 1]),
                "cluster_sizes",
            ),
            "centroid finite": (
                "centroids",
                np.asarray(
                    [[-0.75, -0.25], [0.5, np.inf], [1.0, 1.5]],
                    dtype=np.float64,
                ),
                "centroids.*finite",
            ),
            "feature matrix shape": (
                "motion_feature_matrix",
                np.ones((4, 3), dtype=np.float64),
                "motion_feature_matrix",
            ),
            "available standardized finite": (
                "standardized_feature_matrix",
                np.asarray(
                    [
                        [np.nan, -0.5],
                        [0.5, 0.25],
                        [-0.5, 0.0],
                        [1.0, 1.5],
                    ]
                ),
                "available standardized.*finite",
            ),
            "algorithm schema": (
                "algorithm_schema_version",
                np.asarray("unknown.algorithm.v999"),
                "Unsupported motion-clustering algorithm schema",
            ),
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
                    cluster_metadata.MotionClusterMetadata.load(path)

    def test_empty_profile_cluster_is_valid_but_assignment_must_be_in_range(self) -> None:
        payload = dict(self.payload)
        payload["cluster_id"] = np.asarray([0, 1, 0, 1])
        payload["cluster_sizes"] = np.asarray([2, 2, 0])
        path = self.root / "empty_cluster.npz"
        np.savez_compressed(path, **payload)

        metadata = cluster_metadata.MotionClusterMetadata.load(path)
        self.assertEqual(metadata.cluster_sizes.tolist(), [2, 2, 0])
        self.assertEqual(metadata.num_clusters, 3)

    def test_strict_false_relaxes_only_provenance(self) -> None:
        metadata = cluster_metadata.MotionClusterMetadata.load(self.metadata_path)
        changed_manifest = self.root / "changed_comments.txt"
        changed_manifest.write_text(
            "# provenance changed\n" + self.manifest.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        arguments = self._validation_arguments() | {
            "manifest_path": changed_manifest,
            "pool_fingerprint": "f" * 64,
            "difficulty_metadata_sha256": "1" * 64,
            "difficulty_profile_sha256": "2" * 64,
        }

        with self.assertRaisesRegex(ValueError, "provenance.*manifest SHA256"):
            metadata.validate_against(**arguments)
        self.assertFalse(metadata.validate_against(**arguments, strict=False))

    def test_layout_mismatch_always_raises_even_when_not_strict(self) -> None:
        metadata = cluster_metadata.MotionClusterMetadata.load(self.metadata_path)
        common = self._validation_arguments()
        variants = {
            "manifest motion order": {
                "motion_keys": [
                    "motions/b.npz",
                    "motions/a.npz",
                    "motions/c.npz",
                    "motions/d.npz",
                ]
            },
            "motion frame counts": {"motion_lengths": [52, 49, 100, 25]},
            "motion FPS": {"motion_fps": [50.0, 25.0, 50.0, 25.0]},
            "segment offsets/global count": {
                "motion_segment_offsets": [0, 2, 4, 7, 8]
            },
            "cluster count": {"expected_num_clusters": 4},
        }
        for message, override in variants.items():
            with self.subTest(mismatch=message):
                with self.assertRaisesRegex(ValueError, message):
                    metadata.validate_against(
                        **(common | override),
                        strict=False,
                    )

    def test_save_round_trip_and_refuses_accidental_overwrite(self) -> None:
        destination = self.root / "nested" / "clusters.npz"
        saved = cluster_metadata.save_metadata(destination, self.payload)
        self.assertEqual(saved.cluster_ids.tolist(), [0, 1, 0, 2])
        self.assertTrue(saved.validate_against(**self._validation_arguments()))

        with self.assertRaises(FileExistsError):
            cluster_metadata.save_metadata(destination, self.payload)

        overwritten = cluster_metadata.save_metadata(
            destination,
            self.payload,
            overwrite=True,
        )
        copied = overwritten.save(self.root / "copied.npz")
        self.assertEqual(copied.identity_state()["cluster_profile_sha256"], "d" * 64)

    def test_payload_builder_rejects_non_integer_ids_and_non_finite_features(self) -> None:
        common = {
            key: value
            for key, value in self.payload.items()
            if key
            not in {
                "schema_version",
                "manifest_motion_count",
                "num_clusters",
            }
        }
        with self.assertRaisesRegex(ValueError, "motion_id.*integers"):
            cluster_metadata.metadata_npz_payload(
                **(common | {"motion_id": [0.0, 1.0, 2.0, 3.0]})
            )
        matrix = np.asarray(common["motion_feature_matrix"], dtype=np.float64).copy()
        matrix[0, 0] = np.inf
        with self.assertRaisesRegex(ValueError, "motion_feature_matrix.*finite"):
            cluster_metadata.metadata_npz_payload(
                **(common | {"motion_feature_matrix": matrix})
            )

    def test_missing_optional_features_preserve_nan_and_availability(self) -> None:
        common = {
            key: value
            for key, value in self.payload.items()
            if key
            not in {
                "schema_version",
                "manifest_motion_count",
                "num_clusters",
                "feature_available_mask",
            }
        }
        raw = np.asarray(common["motion_feature_matrix"], dtype=np.float64).copy()
        standardized = np.asarray(
            common["standardized_feature_matrix"], dtype=np.float64
        ).copy()
        raw[1, 1] = np.nan
        standardized[1, 1] = np.nan
        payload = cluster_metadata.metadata_npz_payload(
            **(
                common
                | {
                    "motion_feature_matrix": raw,
                    "standardized_feature_matrix": standardized,
                }
            )
        )
        path = self.root / "optional_nan.npz"
        np.savez_compressed(path, **payload)

        metadata = cluster_metadata.MotionClusterMetadata.load(path)
        self.assertFalse(metadata.feature_available_mask[1, 1])
        self.assertTrue(np.isnan(metadata.motion_feature_matrix[1, 1]))
        self.assertTrue(np.isnan(metadata.standardized_feature_matrix[1, 1]))


if __name__ == "__main__":
    unittest.main()
