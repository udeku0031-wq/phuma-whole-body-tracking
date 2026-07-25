from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BUILDER_PATH = PROJECT_ROOT / "scripts" / "build_motion_cluster_metadata.py"
UTILS_DIR = (
    PROJECT_ROOT
    / "source"
    / "whole_body_tracking"
    / "whole_body_tracking"
    / "utils"
)
sys.path.insert(0, str(UTILS_DIR))

import cluster_metadata  # noqa: E402
import difficulty  # noqa: E402
import difficulty_metadata  # noqa: E402


OUTPUT_FILENAMES = {
    "motion_cluster_metadata.csv",
    "motion_cluster_metadata.npz",
    "cluster_profile.json",
    "cluster_summary.json",
    "cluster_feature_statistics.csv",
    "cluster_centroids.csv",
    "cluster_review_motions.csv",
    "cluster_config_resolved.json",
    "normalized_manifest.txt",
}


def _motion_pool_fingerprint(
    motion_files: list[str], motion_lengths: list[int], motion_fps: list[float]
) -> str:
    normalized = [
        os.path.abspath(os.path.normpath(path)) for path in motion_files
    ]
    common_root = os.path.commonpath(
        [os.path.dirname(path) for path in normalized]
    )
    digest = hashlib.sha256()
    for path, length, fps in zip(
        normalized, motion_lengths, motion_fps, strict=True
    ):
        digest.update(os.path.relpath(path, common_root).encode())
        digest.update(b"\0")
        digest.update(str(os.path.getsize(path)).encode())
        digest.update(b"\0")
        digest.update(str(length).encode())
        digest.update(b"\0")
        digest.update(format(fps, ".17g").encode())
        digest.update(b"\n")
    return digest.hexdigest()


class BuildMotionClusterMetadataTest(unittest.TestCase):
    NUM_MOTIONS = 12
    SEGMENTS_PER_MOTION = 2
    FPS = 10.0
    MOTION_LENGTH = 20

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.motion_dir = self.root / "motions"
        self.motion_dir.mkdir()
        self.motion_keys = [
            f"motions/motion_{index:02d}.npz"
            for index in range(self.NUM_MOTIONS)
        ]
        self.motion_paths = [self.root / key for key in self.motion_keys]
        for index, path in enumerate(self.motion_paths):
            # Pool fingerprints only inspect path, size, length, and FPS.  The
            # cluster builder must never open these as trajectory archives.
            path.write_bytes(f"not-a-trajectory-{index}\n".encode())
        self.manifest = self.root / "synthetic_train_manifest.txt"
        self.manifest.write_text(
            "".join(f"{key}\n" for key in self.motion_keys),
            encoding="utf-8",
        )
        self.cluster_config = self.root / "cluster_config.json"
        self.cluster_config.write_text(
            json.dumps(self._cluster_config(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self.difficulty_profile = self.root / "difficulty_profile.json"
        self.difficulty_metadata = self.root / "segment_difficulty_metadata.npz"
        self._write_difficulty_artifacts()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    @staticmethod
    def _cluster_config() -> dict[str, object]:
        feature_list = [
            "root_linear_speed_p95",
            "root_angular_speed_p95",
            "joint_speed_p95",
            "flight_ratio",
        ]
        return {
            "schema_version": 1,
            "algorithm_schema_version": "wbt.motion_clustering.numpy_kmeans.v1",
            "provisional": True,
            "num_clusters": 3,
            "random_seed": 17,
            "feature_list": feature_list,
            "feature_aggregations": {
                name: ["duration_weighted_mean"] for name in feature_list
            },
            "feature_units": {
                "root_linear_speed_p95": "m/s",
                "root_angular_speed_p95": "rad/s",
                "joint_speed_p95": "rad/s",
                "flight_ratio": "frame_fraction",
            },
            "required_features": feature_list[:3],
            "optional_features": ["flight_ratio"],
            "minimum_optional_feature_coverage": 0.75,
            "robust_scale_epsilon": 1.0e-6,
            "near_constant_scale_threshold": 1.0e-5,
            "zero_mad_fallback_quantiles": [0.05, 0.95],
            "zero_mad_fallback_scale_divisor": 3.289707253902945,
            "robust_clip": 5.0,
            "use_pca": False,
            "pca_components": None,
            "explained_variance_target": None,
            "kmeans_n_init": 8,
            "kmeans_max_iterations": 100,
            "kmeans_tolerance": 1.0e-8,
            "canonicalize_labels": True,
            "minimum_cluster_size_warning": 1,
            "cluster_budget_mode": "sqrt_size_with_floor",
            "minimum_budget_fraction_of_uniform": 0.5,
            "cluster_size_exponent": 0.5,
        }

    def _write_difficulty_artifacts(self) -> None:
        num_segments = self.NUM_MOTIONS * self.SEGMENTS_PER_MOTION
        num_features = len(difficulty.FEATURE_NAMES)
        values = np.empty((num_segments, num_features), dtype=np.float64)
        for motion_id in range(self.NUM_MOTIONS):
            group = motion_id // 4
            group_center = (-4.0, 0.5, 5.0)[group]
            for local_segment in range(self.SEGMENTS_PER_MOTION):
                row = motion_id * self.SEGMENTS_PER_MOTION + local_segment
                feature_index = np.arange(num_features, dtype=np.float64)
                values[row] = (
                    2.0
                    + 0.035 * feature_index
                    + 0.43 * np.sin(0.19 * feature_index + 0.37 * motion_id)
                    + group_center
                    * (0.15 + 0.01 * (feature_index % 5))
                    + 0.025 * local_segment
                )
        available = np.ones_like(values, dtype=bool)
        optional_index = difficulty.FEATURE_NAMES.index("flight_ratio")
        first_motion = slice(0, self.SEGMENTS_PER_MOTION)
        values[first_motion, optional_index] = np.nan
        available[first_motion, optional_index] = False

        lengths = [self.MOTION_LENGTH] * self.NUM_MOTIONS
        fps = [self.FPS] * self.NUM_MOTIONS
        pool_fingerprint = _motion_pool_fingerprint(
            [str(path) for path in self.motion_paths], lengths, fps
        )
        manifest_sha256 = difficulty_metadata.sha256_file(self.manifest)
        config = difficulty.load_difficulty_config()
        profile = difficulty.fit_difficulty_profile(
            values,
            available,
            config,
            training_manifest_sha256=manifest_sha256,
            training_pool_fingerprint=pool_fingerprint,
            segment_schema_version=1,
            config_sha256=difficulty.canonical_json_sha256(config),
            git_commit="synthetic-test",
        )
        self.difficulty_profile.write_text(
            json.dumps(
                profile.to_dict(),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        transformed = difficulty.transform_difficulty_features(
            values, available, profile
        )
        offsets = np.arange(
            0,
            num_segments + 1,
            self.SEGMENTS_PER_MOTION,
            dtype=np.int64,
        )
        global_ids = np.arange(num_segments, dtype=np.int64)
        motion_ids = np.repeat(
            np.arange(self.NUM_MOTIONS, dtype=np.int64),
            self.SEGMENTS_PER_MOTION,
        )
        local_ids = np.tile(
            np.arange(self.SEGMENTS_PER_MOTION, dtype=np.int64),
            self.NUM_MOTIONS,
        )
        starts = local_ids * 10
        ends = starts + 10
        optional_names = set(profile.optional_feature_profile)
        optional_columns = np.asarray(
            [
                index
                for index, name in enumerate(difficulty.FEATURE_NAMES)
                if name in optional_names
            ],
            dtype=np.int64,
        )
        optional_coverage = np.mean(
            available[:, optional_columns], axis=1
        )
        payload = difficulty_metadata.metadata_npz_payload(
            algorithm_schema_version=profile.algorithm_schema_version,
            segment_schema_version=1,
            segment_length_seconds=1.0,
            manifest_sha256=manifest_sha256,
            profile_sha256=difficulty_metadata.sha256_file(
                self.difficulty_profile
            ),
            difficulty_config_sha256=profile.config_sha256,
            pool_fingerprint=pool_fingerprint,
            num_bins=profile.num_bins,
            motion_keys=self.motion_keys,
            motion_lengths=lengths,
            motion_fps=fps,
            motion_segment_offsets=offsets,
            global_segment_id=global_ids,
            motion_id=motion_ids,
            local_segment_id=local_ids,
            start_frame=starts,
            end_frame_exclusive=ends,
            duration_seconds=np.ones(num_segments, dtype=np.float64),
            difficulty_raw=transformed.difficulty_raw,
            difficulty_score=transformed.difficulty_score,
            difficulty_bin=transformed.difficulty_bin,
            feature_names=difficulty.FEATURE_NAMES,
            feature_values=values,
            feature_z=transformed.feature_z,
            feature_available_mask=available,
            optional_feature_coverage=optional_coverage,
            near_constant_features=profile.near_constant_features,
        )
        np.savez_compressed(self.difficulty_metadata, **payload)
        # Prove the synthetic fixture obeys the same strict loader as real data.
        difficulty_metadata.SegmentDifficultyMetadata.load(
            self.difficulty_metadata
        )

    def _run(
        self,
        output_dir: Path,
        *,
        mode: str = "fit_transform",
        profile: Path | None = None,
        manifest: Path | None = None,
        metadata: Path | None = None,
        cluster_config: Path | None = None,
        diagnose: bool = False,
        expect_success: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        command = [
            sys.executable,
            str(BUILDER_PATH),
            "--manifest",
            str(manifest or self.manifest),
            "--difficulty-metadata",
            str(metadata or self.difficulty_metadata),
            "--difficulty-profile",
            str(self.difficulty_profile),
            "--cluster-config",
            str(cluster_config or self.cluster_config),
            "--output-dir",
            str(output_dir),
            "--mode",
            mode,
            "--seed",
            "2026",
            "--strict",
        ]
        if profile is not None:
            command.extend(("--profile", str(profile)))
        if diagnose:
            command.extend(("--diagnose-k", "2", "3", "4"))
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        if expect_success and result.returncode != 0:
            self.fail(
                f"Builder failed with {result.returncode}:\n"
                f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )
        if not expect_success and result.returncode == 0:
            self.fail("Builder unexpectedly succeeded.")
        return result

    def test_fit_outputs_are_deterministic_and_transform_is_frozen(self) -> None:
        first_output = self.root / "fit_first"
        second_output = self.root / "fit_second"
        self._run(first_output, diagnose=True)
        self._run(second_output, diagnose=True)

        self.assertEqual(
            {path.name for path in first_output.iterdir()}, OUTPUT_FILENAMES
        )
        first = cluster_metadata.MotionClusterMetadata.load(
            first_output / "motion_cluster_metadata.npz"
        )
        second = cluster_metadata.MotionClusterMetadata.load(
            second_output / "motion_cluster_metadata.npz"
        )
        np.testing.assert_array_equal(first.cluster_id, second.cluster_id)
        np.testing.assert_array_equal(first.centroids, second.centroids)
        np.testing.assert_array_equal(
            first.feature_available_mask, second.feature_available_mask
        )
        self.assertFalse(first.feature_available_mask[0, -1])
        self.assertEqual(
            (first_output / "cluster_profile.json").read_bytes(),
            (second_output / "cluster_profile.json").read_bytes(),
        )

        summary = json.loads(
            (first_output / "cluster_summary.json").read_text(encoding="utf-8")
        )
        self.assertEqual(summary["motion_count"], self.NUM_MOTIONS)
        self.assertEqual(sum(summary["cluster_sizes"]), self.NUM_MOTIONS)
        self.assertEqual(
            [item["num_clusters"] for item in summary["diagnose_k"]["results"]],
            [2, 3, 4],
        )
        for declaration in (
            "source_category_was_not_used_for_fitting",
            "quality_was_not_used_for_fitting",
            "difficulty_score_bin_was_not_used_for_fitting",
            "policy_statistics_were_not_used_for_fitting",
        ):
            self.assertIs(summary[declaration], True)

        transform_output = self.root / "transform"
        source_profile = first_output / "cluster_profile.json"
        self._run(
            transform_output,
            mode="transform",
            profile=source_profile,
        )
        transformed = cluster_metadata.MotionClusterMetadata.load(
            transform_output / "motion_cluster_metadata.npz"
        )
        np.testing.assert_array_equal(first.cluster_id, transformed.cluster_id)
        np.testing.assert_array_equal(first.centroids, transformed.centroids)
        self.assertEqual(
            source_profile.read_bytes(),
            (transform_output / "cluster_profile.json").read_bytes(),
        )

    def test_fit_rejects_manifest_identity_change_before_writing_artifacts(self) -> None:
        changed_manifest = self.root / "changed_train_manifest.txt"
        changed_manifest.write_text(
            "# changed identity\n" + self.manifest.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        output = self.root / "identity_failure"
        result = self._run(
            output,
            manifest=changed_manifest,
            expect_success=False,
        )
        self.assertIn("manifest SHA256", result.stderr)
        self.assertFalse(
            any((output / filename).exists() for filename in OUTPUT_FILENAMES)
        )

    def test_fit_rejects_feature_unit_mismatch_with_module_two_profile(self) -> None:
        mismatched_config = self.root / "mismatched_units.json"
        config = self._cluster_config()
        config["feature_units"]["root_linear_speed_p95"] = "rad/s"
        mismatched_config.write_text(
            json.dumps(config, sort_keys=True),
            encoding="utf-8",
        )
        output = self.root / "unit_failure"
        result = self._run(
            output,
            cluster_config=mismatched_config,
            expect_success=False,
        )
        self.assertIn(
            "Cluster feature units do not match the frozen module-two Profile",
            result.stderr,
        )
        self.assertFalse(
            any((output / filename).exists() for filename in OUTPUT_FILENAMES)
        )

    def test_difficulty_scores_and_bins_do_not_change_cluster_assignment(self) -> None:
        baseline_output = self.root / "baseline"
        self._run(baseline_output)
        baseline = cluster_metadata.MotionClusterMetadata.load(
            baseline_output / "motion_cluster_metadata.npz"
        )

        changed_metadata = self.root / "diagnostic_scores_changed.npz"
        with np.load(self.difficulty_metadata, allow_pickle=False) as archive:
            payload = {name: np.asarray(archive[name]) for name in archive.files}
        payload["difficulty_score"] = np.linspace(
            0.0, 1.0, payload["difficulty_score"].size, dtype=np.float32
        )
        payload["difficulty_raw"] = np.linspace(
            -3.0, 3.0, payload["difficulty_raw"].size, dtype=np.float32
        )
        payload["difficulty_bin"] = (
            np.arange(payload["difficulty_bin"].size)
            % int(payload["num_bins"])
        ).astype(np.int16)
        np.savez_compressed(changed_metadata, **payload)
        difficulty_metadata.SegmentDifficultyMetadata.load(changed_metadata)

        changed_output = self.root / "changed_diagnostics"
        self._run(changed_output, metadata=changed_metadata)
        changed = cluster_metadata.MotionClusterMetadata.load(
            changed_output / "motion_cluster_metadata.npz"
        )
        np.testing.assert_array_equal(
            baseline.cluster_id, changed.cluster_id
        )
        np.testing.assert_array_equal(baseline.centroids, changed.centroids)


if __name__ == "__main__":
    unittest.main()
