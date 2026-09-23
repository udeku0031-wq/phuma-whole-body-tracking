from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT
    / "source"
    / "whole_body_tracking"
    / "whole_body_tracking"
    / "utils"
    / "motion_clustering.py"
)
SPEC = importlib.util.spec_from_file_location(
    "wbt_motion_clustering_for_tests", MODULE_PATH
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Unable to load motion clustering utilities from {MODULE_PATH}")
motion_clustering = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = motion_clustering
SPEC.loader.exec_module(motion_clustering)


def _config(
    *,
    num_clusters: int = 3,
    use_pca: bool = False,
    minimum_optional_coverage: float = 0.75,
    aggregations: dict[str, list[str]] | None = None,
) -> dict[str, object]:
    definitions = aggregations or {
        "feature_x": ["duration_weighted_mean"],
        "feature_y": ["duration_weighted_mean"],
        "constant": ["duration_weighted_mean"],
    }
    feature_list = list(definitions)
    required = [name for name in feature_list if name != "constant"]
    optional = [name for name in feature_list if name == "constant"]
    return {
        "schema_version": 1,
        "algorithm_schema_version": "wbt.motion_clustering.numpy_kmeans.v1",
        "provisional": True,
        "num_clusters": num_clusters,
        "random_seed": 42,
        "feature_list": feature_list,
        "feature_aggregations": definitions,
        "feature_units": {name: "test_unit" for name in feature_list},
        "required_features": required,
        "optional_features": optional,
        "minimum_optional_feature_coverage": minimum_optional_coverage,
        "robust_scale_epsilon": 1.0e-6,
        "near_constant_scale_threshold": 1.0e-5,
        "zero_mad_fallback_quantiles": [0.05, 0.95],
        "zero_mad_fallback_scale_divisor": 3.289707253902945,
        "robust_clip": 5.0,
        "use_pca": use_pca,
        "pca_components": 2 if use_pca else None,
        "explained_variance_target": None,
        "kmeans_n_init": 10,
        "kmeans_max_iterations": 100,
        "kmeans_tolerance": 1.0e-8,
        "canonicalize_labels": True,
        "minimum_cluster_size_warning": 1,
        "cluster_budget_mode": "sqrt_size_with_floor",
        "minimum_budget_fraction_of_uniform": 0.5,
        "cluster_size_exponent": 0.5,
    }


def _metadata(
    motion_rows: np.ndarray,
    *,
    segment_noise: float = 0.05,
    unavailable: set[tuple[int, int, int]] | None = None,
) -> SimpleNamespace:
    """Build two module-two-like segments for every supplied motion row."""

    rows = np.asarray(motion_rows, dtype=np.float64)
    num_motions, num_features = rows.shape
    values = np.repeat(rows, 2, axis=0)
    direction = np.asarray([-1.0, 1.0] * num_motions)[:, None]
    values = values + direction * segment_noise
    available = np.ones_like(values, dtype=bool)
    for motion, local_segment, feature in unavailable or set():
        index = 2 * motion + local_segment
        available[index, feature] = False
        values[index, feature] = np.nan
    offsets = np.arange(0, 2 * num_motions + 1, 2, dtype=np.int64)
    return SimpleNamespace(
        feature_names=np.asarray(
            ["feature_x", "feature_y", "constant"][:num_features]
        ),
        feature_values=values,
        feature_available_mask=available,
        duration_seconds=np.tile(np.asarray([0.75, 0.25]), num_motions),
        motion_segment_offsets=offsets,
        motion_id=np.repeat(np.arange(num_motions, dtype=np.int64), 2),
        motion_keys=np.asarray([f"motion_{index}" for index in range(num_motions)]),
        source_category=np.asarray(["ignored"] * num_motions),
        quality_status=np.asarray(["ignored"] * num_motions),
        policy_error=np.full(num_motions, 123.0),
        difficulty_score=np.full(num_motions, 0.99),
        difficulty_bin=np.full(num_motions, 9),
    )


class MotionFeatureAggregationTest(unittest.TestCase):
    def test_duration_weighted_mean_p90_and_std(self) -> None:
        config = _config(
            num_clusters=2,
            aggregations={
                "feature_x": [
                    "duration_weighted_mean",
                    "p90",
                    "duration_weighted_std",
                ]
            },
        )
        config["required_features"] = ["feature_x"]
        config["optional_features"] = []
        metadata = SimpleNamespace(
            feature_names=np.asarray(["feature_x"]),
            feature_values=np.asarray([[0.0], [10.0]]),
            feature_available_mask=np.ones((2, 1), dtype=bool),
            duration_seconds=np.asarray([9.0, 1.0]),
            motion_segment_offsets=np.asarray([0, 2], dtype=np.int64),
            motion_id=np.asarray([0, 0], dtype=np.int64),
            motion_keys=np.asarray(["one"]),
        )
        result = motion_clustering.aggregate_motion_features(metadata, config)
        np.testing.assert_allclose(result.values[0], [1.0, 9.0, 3.0])
        self.assertEqual(
            result.feature_names,
            (
                "feature_x__duration_weighted_mean",
                "feature_x__p90",
                "feature_x__duration_weighted_std",
            ),
        )
        self.assertEqual(result.motion_duration_seconds.tolist(), [10.0])

    def test_partial_required_feature_is_missing_not_zero_filled(self) -> None:
        rows = np.asarray([[-1.0, 0.0, 7.0], [1.0, 0.0, 7.0]])
        metadata = _metadata(rows, unavailable={(0, 1, 0)})
        config = _config(num_clusters=2)
        features = motion_clustering.aggregate_motion_features(metadata, config)
        self.assertFalse(features.available_mask[0, 0])
        self.assertTrue(np.isnan(features.values[0, 0]))
        with self.assertRaisesRegex(ValueError, "Required motion clustering"):
            motion_clustering.fit_motion_clustering(features, config)

    def test_non_finite_available_value_and_order_mismatch_fail(self) -> None:
        metadata = _metadata(np.asarray([[-1.0, 0.0, 7.0], [1.0, 0.0, 7.0]]))
        metadata.feature_values[0, 0] = np.inf
        with self.assertRaisesRegex(ValueError, "NaN or Inf"):
            motion_clustering.aggregate_motion_features(
                metadata, _config(num_clusters=2)
            )

        metadata = _metadata(np.asarray([[-1.0, 0.0, 7.0], [1.0, 0.0, 7.0]]))
        metadata.motion_id[[0, 2]] = metadata.motion_id[[2, 0]]
        with self.assertRaisesRegex(ValueError, "motion_id/order"):
            motion_clustering.aggregate_motion_features(
                metadata, _config(num_clusters=2)
            )

    def test_optional_coverage_threshold_and_incomplete_disable(self) -> None:
        rows = np.asarray(
            [
                [-2.0, 0.0, 7.0],
                [-1.0, 0.0, 7.0],
                [1.0, 0.0, 7.0],
                [2.0, 0.0, 7.0],
            ]
        )
        missing = {(3, 0, 2)}
        metadata = _metadata(rows, unavailable=missing)
        with self.assertRaisesRegex(ValueError, "coverage is below"):
            motion_clustering.fit_transform_motion_clustering(
                metadata,
                _config(num_clusters=2, minimum_optional_coverage=0.8),
            )

        result = motion_clustering.fit_transform_motion_clustering(
            metadata,
            _config(num_clusters=2, minimum_optional_coverage=0.75),
        )
        constant_index = result.profile.feature_names.index(
            "constant__duration_weighted_mean"
        )
        self.assertFalse(result.profile.active_feature_mask[constant_index])
        self.assertTrue(
            any("coverage" in warning for warning in result.profile.warnings)
        )


class MotionClusteringFitTransformTest(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(7)
        centers = np.asarray([[-5.0, -4.0], [0.0, 5.0], [6.0, -2.0]])
        rows = np.vstack(
            [
                np.column_stack(
                    (
                        rng.normal(center[0], 0.15, size=8),
                        rng.normal(center[1], 0.15, size=8),
                        np.full(8, 3.25),
                    )
                )
                for center in centers
            ]
        )
        self.rows = rows
        self.metadata = _metadata(rows)

    def test_three_clusters_recovered_and_repeat_is_bitwise_deterministic(self) -> None:
        config = _config()
        first = motion_clustering.fit_transform_motion_clustering(
            self.metadata, config
        )
        second = motion_clustering.fit_transform_motion_clustering(
            self.metadata, config
        )
        np.testing.assert_array_equal(first.cluster_id, second.cluster_id)
        np.testing.assert_array_equal(
            first.profile.centroids, second.profile.centroids
        )
        self.assertEqual(first.profile.sha256, second.profile.sha256)
        self.assertEqual(set(first.cluster_id.tolist()), {0, 1, 2})
        for group in range(3):
            labels = first.cluster_id[group * 8 : (group + 1) * 8]
            self.assertEqual(np.unique(labels).size, 1)
        self.assertTrue(np.isfinite(first.profile.centroids).all())
        self.assertTrue(np.all(first.profile.active_feature_mask[:2]))
        self.assertFalse(first.profile.active_feature_mask[2])

    def test_irrelevant_labels_never_enter_clustering(self) -> None:
        config = _config()
        baseline = motion_clustering.fit_transform_motion_clustering(
            self.metadata, config
        )
        changed = _metadata(self.rows)
        changed.source_category = np.asarray(["dance", "fitness"] * 12)
        changed.quality_status = np.asarray(["Reject", "Pass"] * 12)
        changed.policy_error = np.arange(24, dtype=np.float64)
        changed.difficulty_score = np.linspace(0.0, 1.0, 24)
        changed.difficulty_bin = np.arange(24) % 10
        result = motion_clustering.fit_transform_motion_clustering(
            changed, config
        )
        np.testing.assert_array_equal(result.cluster_id, baseline.cluster_id)
        self.assertEqual(result.profile.sha256, baseline.profile.sha256)

    def test_profile_json_roundtrip_and_frozen_transform(self) -> None:
        config = _config(use_pca=True)
        fitted = motion_clustering.fit_transform_motion_clustering(
            self.metadata,
            config,
            training_manifest_sha256="a" * 64,
            training_pool_fingerprint="b" * 64,
            difficulty_metadata_sha256="c" * 64,
            difficulty_profile_sha256="d" * 64,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cluster_profile.json"
            fitted.profile.save_json(path)
            loaded = motion_clustering.MotionClusteringProfile.load_json(path)
            self.assertEqual(loaded.to_dict(), fitted.profile.to_dict())
            self.assertEqual(loaded.sha256, fitted.profile.sha256)
            # Saved JSON must itself be canonical-hashable and contain no NaN.
            json.loads(path.read_text(encoding="utf-8"))

        transformed = motion_clustering.transform_motion_clustering(
            self.metadata, loaded
        )
        np.testing.assert_array_equal(
            transformed.cluster_id, fitted.cluster_id
        )
        np.testing.assert_allclose(
            transformed.standardized_features,
            fitted.standardized_features,
            equal_nan=True,
        )
        np.testing.assert_allclose(
            transformed.clustering_features, fitted.clustering_features
        )
        np.testing.assert_allclose(
            transformed.profile.feature_centers,
            fitted.profile.feature_centers,
        )
        self.assertEqual(transformed.profile.sha256, fitted.profile.sha256)

    def test_pca_component_sign_and_canonical_centroid_order(self) -> None:
        fitted = motion_clustering.fit_transform_motion_clustering(
            self.metadata, _config(use_pca=True)
        )
        for component in fitted.profile.pca_components:
            pivot = int(np.argmax(np.abs(component)))
            self.assertGreater(component[pivot], 0.0)
        centroid_rows = [
            tuple(float(item) for item in row)
            for row in fitted.profile.centroids
        ]
        self.assertEqual(centroid_rows, sorted(centroid_rows))
        self.assertEqual(
            sorted(fitted.profile.canonical_label_remap.tolist()),
            [0, 1, 2],
        )

    def test_frozen_profile_rejects_missing_identity_and_noncanonical_labels(self) -> None:
        fitted = motion_clustering.fit_transform_motion_clustering(
            self.metadata,
            _config(),
            training_manifest_sha256="a" * 64,
            training_pool_fingerprint="b" * 64,
            difficulty_metadata_sha256="c" * 64,
            difficulty_profile_sha256="d" * 64,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cluster_profile.json"
            payload = fitted.profile.to_dict()
            payload["training_manifest_sha256"] = ""
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Frozen Profile identity"):
                motion_clustering.load_motion_clustering_profile(path)

            payload = fitted.profile.to_dict()
            payload["centroids"][0], payload["centroids"][1] = (
                payload["centroids"][1],
                payload["centroids"][0],
            )
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "canonical lexicographic"):
                motion_clustering.load_motion_clustering_profile(path)

    def test_direct_kmeans_is_deterministic_and_canonical(self) -> None:
        values = np.asarray(
            [
                [-10.0, -10.0],
                [-9.8, -10.2],
                [0.0, 8.0],
                [0.1, 8.2],
                [12.0, -1.0],
                [12.2, -0.9],
            ]
        )
        first = motion_clustering.deterministic_kmeans(
            values, num_clusters=3, random_seed=11, n_init=8
        )
        second = motion_clustering.deterministic_kmeans(
            values, num_clusters=3, random_seed=11, n_init=8
        )
        np.testing.assert_array_equal(first.labels, second.labels)
        np.testing.assert_array_equal(first.centroids, second.centroids)
        rows = [tuple(row) for row in first.centroids.tolist()]
        self.assertEqual(rows, sorted(rows))
        self.assertTrue(np.isfinite(first.inertia))
        self.assertTrue(first.converged)


class DefaultMotionClusteringConfigTest(unittest.TestCase):
    def test_default_config_has_expected_robot_features_and_settings(self) -> None:
        config = motion_clustering.load_motion_clustering_config()
        self.assertEqual(config["num_clusters"], 8)
        self.assertEqual(config["random_seed"], 42)
        self.assertEqual(len(config["feature_list"]), 17)
        self.assertFalse(config["use_pca"])
        forbidden = {
            "dance",
            "fitness",
            "humanml",
            "quality_score",
            "difficulty_score",
            "difficulty_bin",
            "policy_error",
        }
        self.assertTrue(forbidden.isdisjoint(config["feature_list"]))


if __name__ == "__main__":
    unittest.main()
