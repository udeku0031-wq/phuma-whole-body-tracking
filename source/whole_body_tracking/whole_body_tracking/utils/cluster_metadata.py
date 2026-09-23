"""Frozen motion-cluster metadata used by the diversity sampling layer.

The offline clustering pipeline writes one compact ``.npz`` containing only
motion-level features, cluster assignments, and immutable provenance.  This
module is deliberately NumPy-only so the metadata can be checked in CPU unit
tests and before Isaac Sim is imported by a training process.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Mapping, Sequence

import numpy as np


CLUSTER_METADATA_SCHEMA_VERSION: Final[str] = "wbt.motion_cluster.v1"
"""Schema version for the motion-to-cluster mapping archive."""

MOTION_CLUSTERING_ALGORITHM_SCHEMA_VERSION: Final[str] = (
    "wbt.motion_clustering.numpy_kmeans.v1"
)

# A descriptive alias makes imports unambiguous next to segment metadata.
MOTION_CLUSTER_METADATA_SCHEMA_VERSION: Final[str] = CLUSTER_METADATA_SCHEMA_VERSION

_REQUIRED_ARRAYS: Final[tuple[str, ...]] = (
    "schema_version",
    "algorithm_schema_version",
    "manifest_sha256",
    "manifest_motion_count",
    "pool_fingerprint",
    "difficulty_metadata_sha256",
    "difficulty_profile_sha256",
    "cluster_profile_sha256",
    "cluster_config_sha256",
    "num_clusters",
    "motion_keys",
    "motion_lengths",
    "motion_fps",
    "motion_segment_offsets",
    "motion_id",
    "cluster_id",
    "cluster_sizes",
    "centroids",
    "feature_names",
    "motion_feature_matrix",
    "feature_available_mask",
    "standardized_feature_matrix",
)

_SHA256_FIELDS: Final[tuple[str, ...]] = (
    "manifest_sha256",
    "pool_fingerprint",
    "difficulty_metadata_sha256",
    "difficulty_profile_sha256",
    "cluster_profile_sha256",
    "cluster_config_sha256",
)


def sha256_file(path: str | os.PathLike[str]) -> str:
    """Return the lowercase SHA256 digest of *path* without loading it at once."""

    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scalar_text(value: np.ndarray, name: str) -> str:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"Cluster metadata field '{name}' must be scalar.")
    if array.dtype.kind not in "SU":
        raise ValueError(f"Cluster metadata field '{name}' must be text.")
    item = array.reshape(()).item()
    result = item.decode("utf-8") if isinstance(item, bytes) else str(item)
    if not result:
        raise ValueError(f"Cluster metadata field '{name}' must not be empty.")
    return result


def _scalar_int(value: np.ndarray, name: str) -> int:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"Cluster metadata field '{name}' must be scalar.")
    if array.dtype.kind not in "iu":
        raise ValueError(f"Cluster metadata field '{name}' must be an integer.")
    return int(array.reshape(()).item())


def _text_array(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in "SU":
        raise ValueError(f"Cluster metadata field '{name}' must contain text.")
    return array.astype(str, copy=False)


def _integer_array(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in "iu":
        raise ValueError(f"Cluster metadata field '{name}' must contain integers.")
    return array.astype(np.int64, copy=False)


def _numeric_array(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in "fiu":
        raise ValueError(f"Cluster metadata field '{name}' must be numeric.")
    return array.astype(np.float64, copy=False)


def _boolean_array(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind != "b":
        raise ValueError(f"Cluster metadata field '{name}' must contain booleans.")
    return array.astype(bool, copy=False)


def _validate_sha256(name: str, digest: str) -> None:
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest.lower()):
        raise ValueError(f"Cluster metadata field '{name}' is not a SHA256 hex digest.")


def _validate_payload(values: Mapping[str, np.ndarray]) -> dict[str, object]:
    """Validate archive arrays and return normalized scalar/array values."""

    missing = sorted(set(_REQUIRED_ARRAYS).difference(values))
    if missing:
        raise ValueError(f"Cluster metadata is missing fields: {missing}")

    schema_version = _scalar_text(values["schema_version"], "schema_version")
    if schema_version != CLUSTER_METADATA_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported cluster metadata schema '{schema_version}'; "
            f"expected '{CLUSTER_METADATA_SCHEMA_VERSION}'."
        )
    algorithm_schema_version = _scalar_text(
        values["algorithm_schema_version"], "algorithm_schema_version"
    )
    if (
        algorithm_schema_version
        != MOTION_CLUSTERING_ALGORITHM_SCHEMA_VERSION
    ):
        raise ValueError(
            "Unsupported motion-clustering algorithm schema "
            f"'{algorithm_schema_version}'; expected "
            f"'{MOTION_CLUSTERING_ALGORITHM_SCHEMA_VERSION}'."
        )

    digests = {name: _scalar_text(values[name], name) for name in _SHA256_FIELDS}
    for name, digest in digests.items():
        _validate_sha256(name, digest)

    motion_keys = _text_array(values["motion_keys"], "motion_keys")
    motion_lengths = _integer_array(values["motion_lengths"], "motion_lengths")
    motion_fps = _numeric_array(values["motion_fps"], "motion_fps")
    motion_segment_offsets = _integer_array(
        values["motion_segment_offsets"], "motion_segment_offsets"
    )
    motion_id = _integer_array(values["motion_id"], "motion_id")
    cluster_id = _integer_array(values["cluster_id"], "cluster_id")

    one_dimensional_motion_arrays = {
        "motion_keys": motion_keys,
        "motion_lengths": motion_lengths,
        "motion_fps": motion_fps,
        "motion_id": motion_id,
        "cluster_id": cluster_id,
    }
    if any(array.ndim != 1 for array in one_dimensional_motion_arrays.values()):
        raise ValueError("Cluster metadata motion arrays must be one-dimensional.")
    num_motions = int(motion_keys.size)
    if num_motions == 0:
        raise ValueError("Cluster metadata must describe at least one motion.")
    for name, array in one_dimensional_motion_arrays.items():
        if array.size != num_motions:
            raise ValueError(
                f"Cluster metadata field '{name}' must have shape ({num_motions},)."
            )
    if np.any(np.char.str_len(motion_keys) == 0):
        raise ValueError("Cluster metadata motion_keys must not contain empty strings.")
    if np.unique(motion_keys).size != num_motions:
        raise ValueError("Cluster metadata motion_keys must be unique.")

    manifest_motion_count = _scalar_int(
        values["manifest_motion_count"], "manifest_motion_count"
    )
    if manifest_motion_count != num_motions:
        raise ValueError(
            "Cluster metadata manifest_motion_count does not match the motion arrays."
        )
    if np.any(motion_lengths < 1):
        raise ValueError("Cluster metadata motion_lengths must be positive.")
    if not np.isfinite(motion_fps).all() or np.any(motion_fps <= 0.0):
        raise ValueError("Cluster metadata motion_fps must contain finite positive values.")
    if (
        motion_segment_offsets.shape != (num_motions + 1,)
        or motion_segment_offsets[0] != 0
        or np.any(np.diff(motion_segment_offsets) <= 0)
    ):
        raise ValueError("Cluster metadata motion_segment_offsets is invalid.")
    expected_motion_id = np.arange(num_motions, dtype=np.int64)
    if not np.array_equal(motion_id, expected_motion_id):
        raise ValueError(
            "Cluster metadata motion_id must assign each ordered motion exactly one "
            "unique ID in 0..N-1."
        )

    num_clusters = _scalar_int(values["num_clusters"], "num_clusters")
    if num_clusters < 1:
        raise ValueError("Cluster metadata num_clusters must be positive.")
    cluster_sizes = _integer_array(values["cluster_sizes"], "cluster_sizes")
    if cluster_sizes.shape != (num_clusters,):
        raise ValueError(
            f"Cluster metadata cluster_sizes must have shape ({num_clusters},)."
        )
    if np.any(cluster_id < 0) or np.any(cluster_id >= num_clusters):
        raise ValueError(
            f"Cluster metadata cluster_id must be in 0..{num_clusters - 1}."
        )
    expected_cluster_sizes = np.bincount(cluster_id, minlength=num_clusters).astype(
        np.int64, copy=False
    )
    if not np.array_equal(cluster_sizes, expected_cluster_sizes):
        raise ValueError(
            "Cluster metadata cluster_sizes does not match the motion cluster assignments."
        )

    feature_names = _text_array(values["feature_names"], "feature_names")
    if feature_names.ndim != 1 or feature_names.size == 0:
        raise ValueError(
            "Cluster metadata feature_names must be a non-empty one-dimensional array."
        )
    if (
        np.any(np.char.str_len(feature_names) == 0)
        or np.unique(feature_names).size != feature_names.size
    ):
        raise ValueError("Cluster metadata feature_names must be non-empty and unique.")
    num_features = int(feature_names.size)

    centroids = _numeric_array(values["centroids"], "centroids")
    motion_feature_matrix = _numeric_array(
        values["motion_feature_matrix"], "motion_feature_matrix"
    )
    feature_available_mask = _boolean_array(
        values["feature_available_mask"], "feature_available_mask"
    )
    standardized_feature_matrix = _numeric_array(
        values["standardized_feature_matrix"], "standardized_feature_matrix"
    )
    expected_motion_feature_shape = (num_motions, num_features)
    if motion_feature_matrix.shape != expected_motion_feature_shape:
        raise ValueError(
            "Cluster metadata motion_feature_matrix must have shape "
            f"{expected_motion_feature_shape}."
        )
    if standardized_feature_matrix.shape != expected_motion_feature_shape:
        raise ValueError(
            "Cluster metadata standardized_feature_matrix must have shape "
            f"{expected_motion_feature_shape}."
        )
    if feature_available_mask.shape != expected_motion_feature_shape:
        raise ValueError(
            "Cluster metadata feature_available_mask must have shape "
            f"{expected_motion_feature_shape}."
        )
    if centroids.ndim != 2 or centroids.shape[0] != num_clusters or centroids.shape[1] == 0:
        raise ValueError(
            "Cluster metadata centroids must be a non-empty matrix with one row "
            "per cluster."
        )
    if not np.isfinite(motion_feature_matrix[feature_available_mask]).all():
        raise ValueError(
            "Cluster metadata available motion_feature_matrix values must be finite."
        )
    if not np.isnan(motion_feature_matrix[~feature_available_mask]).all():
        raise ValueError(
            "Cluster metadata unavailable motion_feature_matrix values must be NaN."
        )
    # Optional or near-constant columns can be inactive in the distance space,
    # so their standardized entries may remain NaN.  Infinity is never a valid
    # missing-value representation.
    if np.isinf(standardized_feature_matrix).any():
        raise ValueError(
            "Cluster metadata standardized_feature_matrix must not contain infinity."
        )
    if not np.isfinite(
        standardized_feature_matrix[feature_available_mask]
    ).all():
        raise ValueError(
            "Cluster metadata available standardized feature values must be finite."
        )
    if not np.isnan(standardized_feature_matrix[~feature_available_mask]).all():
        raise ValueError(
            "Cluster metadata unavailable standardized_feature_matrix values must be NaN."
        )
    if not np.isfinite(centroids).all():
        raise ValueError("Cluster metadata centroids must contain only finite values.")

    return {
        "schema_version": schema_version,
        "algorithm_schema_version": algorithm_schema_version,
        "manifest_sha256": digests["manifest_sha256"],
        "manifest_motion_count": manifest_motion_count,
        "pool_fingerprint": digests["pool_fingerprint"],
        "difficulty_metadata_sha256": digests["difficulty_metadata_sha256"],
        "difficulty_profile_sha256": digests["difficulty_profile_sha256"],
        "cluster_profile_sha256": digests["cluster_profile_sha256"],
        "cluster_config_sha256": digests["cluster_config_sha256"],
        "num_clusters": num_clusters,
        "motion_keys": motion_keys.copy(),
        "motion_lengths": motion_lengths.copy(),
        "motion_fps": motion_fps.copy(),
        "motion_segment_offsets": motion_segment_offsets.copy(),
        "motion_id": motion_id.copy(),
        "cluster_id": cluster_id.copy(),
        "cluster_sizes": cluster_sizes.copy(),
        "centroids": centroids.copy(),
        "feature_names": feature_names.copy(),
        "motion_feature_matrix": motion_feature_matrix.copy(),
        "feature_available_mask": feature_available_mask.copy(),
        "standardized_feature_matrix": standardized_feature_matrix.copy(),
    }


@dataclass(frozen=True)
class MotionClusterMetadata:
    """Structurally validated motion-to-cluster metadata from one frozen NPZ."""

    path: str
    metadata_sha256: str
    schema_version: str
    algorithm_schema_version: str
    manifest_sha256: str
    manifest_motion_count: int
    pool_fingerprint: str
    difficulty_metadata_sha256: str
    difficulty_profile_sha256: str
    cluster_profile_sha256: str
    cluster_config_sha256: str
    num_clusters: int
    motion_keys: np.ndarray
    motion_lengths: np.ndarray
    motion_fps: np.ndarray
    motion_segment_offsets: np.ndarray
    motion_id: np.ndarray
    cluster_id: np.ndarray
    cluster_sizes: np.ndarray
    centroids: np.ndarray
    feature_names: np.ndarray
    motion_feature_matrix: np.ndarray
    feature_available_mask: np.ndarray
    standardized_feature_matrix: np.ndarray

    @property
    def num_motions(self) -> int:
        return int(self.motion_id.size)

    @property
    def num_features(self) -> int:
        return int(self.feature_names.size)

    @property
    def num_segments(self) -> int:
        return int(self.motion_segment_offsets[-1])

    @property
    def motion_ids(self) -> np.ndarray:
        """Compatibility alias for call sites that use plural array names."""

        return self.motion_id

    @property
    def cluster_ids(self) -> np.ndarray:
        """Compatibility alias used by the runtime diversity sampler."""

        return self.cluster_id

    @property
    def profile_sha256(self) -> str:
        """Return the frozen clustering-profile identity."""

        return self.cluster_profile_sha256

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> "MotionClusterMetadata":
        """Load and strictly validate a frozen motion-cluster archive."""

        resolved_path = str(Path(path).resolve())
        if not os.path.isfile(resolved_path):
            raise FileNotFoundError(
                f"Cluster metadata file does not exist: {resolved_path}"
            )
        metadata_digest = sha256_file(resolved_path)
        try:
            archive = np.load(resolved_path, allow_pickle=False)
        except Exception as exc:
            raise ValueError(
                f"Unable to load cluster metadata '{resolved_path}': {exc}"
            ) from exc
        try:
            missing = sorted(set(_REQUIRED_ARRAYS).difference(archive.files))
            if missing:
                raise ValueError(f"Cluster metadata is missing fields: {missing}")
            values = {name: np.asarray(archive[name]) for name in _REQUIRED_ARRAYS}
        finally:
            archive.close()
        validated = _validate_payload(values)
        return cls(
            path=resolved_path,
            metadata_sha256=metadata_digest,
            **validated,
        )

    def validate_against(
        self,
        *,
        manifest_path: str | os.PathLike[str],
        motion_keys: Sequence[str],
        motion_lengths: Sequence[int],
        motion_fps: Sequence[float],
        motion_segment_offsets: Sequence[int],
        pool_fingerprint: str,
        difficulty_metadata_sha256: str,
        difficulty_profile_sha256: str,
        expected_num_clusters: int | None = None,
        strict: bool = True,
    ) -> bool:
        """Validate the exact Stage-0 layout and upstream difficulty identity.

        ``strict=False`` relaxes provenance only (file/config hashes).  Layout
        mismatches always raise because accepting a differently ordered motion
        pool would silently attach clusters to the wrong trajectories.
        """

        layout_mismatches: list[str] = []
        normalized_keys = np.asarray(
            [Path(key).as_posix() for key in motion_keys], dtype=str
        )
        current_lengths = np.asarray(motion_lengths)
        current_fps = np.asarray(motion_fps)
        current_offsets = np.asarray(motion_segment_offsets)

        if not np.array_equal(normalized_keys, self.motion_keys):
            layout_mismatches.append("manifest motion order")
        if (
            current_lengths.dtype.kind not in "iu"
            or not np.array_equal(current_lengths.astype(np.int64, copy=False), self.motion_lengths)
        ):
            layout_mismatches.append("motion frame counts")
        if current_fps.dtype.kind not in "fiu":
            layout_mismatches.append("motion FPS")
        else:
            numeric_fps = current_fps.astype(np.float64, copy=False)
            if numeric_fps.shape != self.motion_fps.shape or not np.allclose(
                numeric_fps, self.motion_fps, rtol=0.0, atol=1.0e-12
            ):
                layout_mismatches.append("motion FPS")
        if (
            current_offsets.dtype.kind not in "iu"
            or not np.array_equal(
                current_offsets.astype(np.int64, copy=False),
                self.motion_segment_offsets,
            )
        ):
            layout_mismatches.append("segment offsets/global count")
        if (
            expected_num_clusters is not None
            and int(expected_num_clusters) != self.num_clusters
        ):
            layout_mismatches.append("cluster count")
        if layout_mismatches:
            raise ValueError(
                "Cluster metadata layout does not match Stage 0: "
                + ", ".join(layout_mismatches)
                + "."
            )

        provenance_mismatches: list[str] = []
        if sha256_file(manifest_path) != self.manifest_sha256:
            provenance_mismatches.append("manifest SHA256")
        if pool_fingerprint != self.pool_fingerprint:
            provenance_mismatches.append("ordered motion pool fingerprint")
        if difficulty_metadata_sha256 != self.difficulty_metadata_sha256:
            provenance_mismatches.append("difficulty metadata SHA256")
        if difficulty_profile_sha256 != self.difficulty_profile_sha256:
            provenance_mismatches.append("difficulty profile SHA256")
        if provenance_mismatches and strict:
            raise ValueError(
                "Cluster metadata provenance does not match training data: "
                + ", ".join(provenance_mismatches)
                + "."
            )
        return not provenance_mismatches

    def identity_state(self) -> dict[str, object]:
        """Return immutable identities needed for runtime/checkpoint compatibility."""

        return {
            "schema_version": self.schema_version,
            "algorithm_schema_version": self.algorithm_schema_version,
            "metadata_path": self.path,
            "metadata_sha256": self.metadata_sha256,
            "cluster_metadata_sha256": self.metadata_sha256,
            "cluster_profile_sha256": self.cluster_profile_sha256,
            "cluster_config_sha256": self.cluster_config_sha256,
            "manifest_sha256": self.manifest_sha256,
            "pool_fingerprint": self.pool_fingerprint,
            "difficulty_metadata_sha256": self.difficulty_metadata_sha256,
            "difficulty_profile_sha256": self.difficulty_profile_sha256,
            "manifest_motion_count": self.manifest_motion_count,
            "num_clusters": self.num_clusters,
            "num_features": self.num_features,
        }

    def npz_payload(self) -> Mapping[str, np.ndarray]:
        """Return a validated archive payload equivalent to this object."""

        return metadata_npz_payload(
            algorithm_schema_version=self.algorithm_schema_version,
            manifest_sha256=self.manifest_sha256,
            pool_fingerprint=self.pool_fingerprint,
            difficulty_metadata_sha256=self.difficulty_metadata_sha256,
            difficulty_profile_sha256=self.difficulty_profile_sha256,
            cluster_profile_sha256=self.cluster_profile_sha256,
            cluster_config_sha256=self.cluster_config_sha256,
            motion_keys=self.motion_keys,
            motion_lengths=self.motion_lengths,
            motion_fps=self.motion_fps,
            motion_segment_offsets=self.motion_segment_offsets,
            motion_id=self.motion_id,
            cluster_id=self.cluster_id,
            cluster_sizes=self.cluster_sizes,
            centroids=self.centroids,
            feature_names=self.feature_names,
            motion_feature_matrix=self.motion_feature_matrix,
            feature_available_mask=self.feature_available_mask,
            standardized_feature_matrix=self.standardized_feature_matrix,
            num_clusters=self.num_clusters,
        )

    def save(
        self, path: str | os.PathLike[str], *, overwrite: bool = False
    ) -> "MotionClusterMetadata":
        """Save this metadata to a new archive and load the verified result."""

        return save_metadata(path, self.npz_payload(), overwrite=overwrite)


def metadata_npz_payload(
    *,
    algorithm_schema_version: str,
    manifest_sha256: str,
    pool_fingerprint: str,
    difficulty_metadata_sha256: str,
    difficulty_profile_sha256: str,
    cluster_profile_sha256: str,
    cluster_config_sha256: str,
    motion_keys: Sequence[str],
    motion_lengths: Sequence[int],
    motion_fps: Sequence[float],
    motion_segment_offsets: Sequence[int],
    motion_id: Sequence[int],
    cluster_id: Sequence[int],
    cluster_sizes: Sequence[int],
    centroids: Sequence[Sequence[float]] | np.ndarray,
    feature_names: Sequence[str],
    motion_feature_matrix: Sequence[Sequence[float]] | np.ndarray,
    standardized_feature_matrix: Sequence[Sequence[float]] | np.ndarray,
    feature_available_mask: Sequence[Sequence[bool]] | np.ndarray | None = None,
    num_clusters: int | None = None,
) -> Mapping[str, np.ndarray]:
    """Build and strictly validate the trajectory-free cluster NPZ payload."""

    keys_input = np.asarray(motion_keys)
    names_input = np.asarray(feature_names)
    if keys_input.dtype.kind not in "SU":
        raise ValueError("motion_keys must contain text.")
    if names_input.dtype.kind not in "SU":
        raise ValueError("feature_names must contain text.")
    canonical_keys = np.asarray(
        [Path(str(key)).as_posix() for key in keys_input.tolist()], dtype=str
    )

    cluster_sizes_array = np.asarray(cluster_sizes)
    if cluster_sizes_array.dtype.kind not in "iu":
        raise ValueError("cluster_sizes must contain integers.")
    inferred_num_clusters = int(cluster_sizes_array.size)
    stored_num_clusters = (
        inferred_num_clusters if num_clusters is None else int(num_clusters)
    )
    if stored_num_clusters != inferred_num_clusters:
        raise ValueError("num_clusters does not match cluster_sizes.")

    feature_values = np.asarray(motion_feature_matrix)
    if feature_available_mask is None:
        # NaN is the sole missing-value representation.  Infinity remains
        # marked available here so structural validation rejects it.
        availability = ~np.isnan(feature_values)
    else:
        availability = np.asarray(feature_available_mask)

    raw_payload: dict[str, np.ndarray] = {
        "schema_version": np.asarray(CLUSTER_METADATA_SCHEMA_VERSION),
        "algorithm_schema_version": np.asarray(str(algorithm_schema_version)),
        "manifest_sha256": np.asarray(manifest_sha256),
        "manifest_motion_count": np.asarray(canonical_keys.size, dtype=np.int64),
        "pool_fingerprint": np.asarray(pool_fingerprint),
        "difficulty_metadata_sha256": np.asarray(difficulty_metadata_sha256),
        "difficulty_profile_sha256": np.asarray(difficulty_profile_sha256),
        "cluster_profile_sha256": np.asarray(cluster_profile_sha256),
        "cluster_config_sha256": np.asarray(cluster_config_sha256),
        "num_clusters": np.asarray(stored_num_clusters, dtype=np.int64),
        "motion_keys": canonical_keys,
        "motion_lengths": np.asarray(motion_lengths),
        "motion_fps": np.asarray(motion_fps),
        "motion_segment_offsets": np.asarray(motion_segment_offsets),
        "motion_id": np.asarray(motion_id),
        "cluster_id": np.asarray(cluster_id),
        "cluster_sizes": cluster_sizes_array,
        "centroids": np.asarray(centroids),
        "feature_names": names_input.astype(str, copy=False),
        "motion_feature_matrix": feature_values,
        "feature_available_mask": availability,
        "standardized_feature_matrix": np.asarray(standardized_feature_matrix),
    }
    validated = _validate_payload(raw_payload)
    return {
        "schema_version": np.asarray(validated["schema_version"]),
        "algorithm_schema_version": np.asarray(
            validated["algorithm_schema_version"]
        ),
        "manifest_sha256": np.asarray(validated["manifest_sha256"]),
        "manifest_motion_count": np.asarray(
            validated["manifest_motion_count"], dtype=np.int64
        ),
        "pool_fingerprint": np.asarray(validated["pool_fingerprint"]),
        "difficulty_metadata_sha256": np.asarray(
            validated["difficulty_metadata_sha256"]
        ),
        "difficulty_profile_sha256": np.asarray(
            validated["difficulty_profile_sha256"]
        ),
        "cluster_profile_sha256": np.asarray(
            validated["cluster_profile_sha256"]
        ),
        "cluster_config_sha256": np.asarray(
            validated["cluster_config_sha256"]
        ),
        "num_clusters": np.asarray(validated["num_clusters"], dtype=np.int64),
        "motion_keys": np.asarray(validated["motion_keys"], dtype=str),
        "motion_lengths": np.asarray(validated["motion_lengths"], dtype=np.int64),
        "motion_fps": np.asarray(validated["motion_fps"], dtype=np.float64),
        "motion_segment_offsets": np.asarray(
            validated["motion_segment_offsets"], dtype=np.int64
        ),
        "motion_id": np.asarray(validated["motion_id"], dtype=np.int64),
        "cluster_id": np.asarray(validated["cluster_id"], dtype=np.int64),
        "cluster_sizes": np.asarray(validated["cluster_sizes"], dtype=np.int64),
        "centroids": np.asarray(validated["centroids"], dtype=np.float64),
        "feature_names": np.asarray(validated["feature_names"], dtype=str),
        "motion_feature_matrix": np.asarray(
            validated["motion_feature_matrix"], dtype=np.float64
        ),
        "feature_available_mask": np.asarray(
            validated["feature_available_mask"], dtype=bool
        ),
        "standardized_feature_matrix": np.asarray(
            validated["standardized_feature_matrix"], dtype=np.float64
        ),
    }


def save_metadata(
    path: str | os.PathLike[str],
    payload: Mapping[str, np.ndarray],
    *,
    overwrite: bool = False,
) -> MotionClusterMetadata:
    """Atomically save a validated cluster payload and return its loaded object."""

    required_values = {
        name: np.asarray(payload[name])
        for name in _REQUIRED_ARRAYS
        if name in payload
    }
    _validate_payload(required_values)

    destination = Path(path).resolve()
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Cluster metadata file already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            np.savez_compressed(stream, **required_values)
        os.replace(temporary_path, destination)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return MotionClusterMetadata.load(destination)


# Explicit aliases keep offline scripts readable without multiplying semantics.
save_cluster_metadata = save_metadata
load_cluster_metadata = MotionClusterMetadata.load
