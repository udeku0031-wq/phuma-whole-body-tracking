from __future__ import annotations

import subprocess
import sys
import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_phuma_splits.py"
SPEC = importlib.util.spec_from_file_location("build_phuma_splits", SCRIPT)
splits = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = splits
SPEC.loader.exec_module(splits)


def make_npz(path: Path, source_file: str | None = None, fps: float = 50.0, frames: int = 4) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "fps": np.asarray([fps], dtype=np.float32),
        "joint_pos": np.zeros((frames, 29), dtype=np.float32),
        "joint_vel": np.zeros((frames, 29), dtype=np.float32),
        "body_pos_w": np.zeros((frames, 30, 3), dtype=np.float32),
        "body_quat_w": np.zeros((frames, 30, 4), dtype=np.float32),
        "body_lin_vel_w": np.zeros((frames, 30, 3), dtype=np.float32),
        "body_ang_vel_w": np.zeros((frames, 30, 3), dtype=np.float32),
        "source_format": np.asarray("PHUMA_G1"),
    }
    if frames > 0:
        data["body_quat_w"][:, :, 0] = 1.0
    if source_file is not None:
        data["source_file"] = np.asarray(source_file)
    np.savez(path, **data)


def create_dataset(project: Path, groups_per_category: int = 8) -> tuple[Path, Path]:
    data_root = project / "PHUMA_wbt_motions" / "g1_all"
    for category in ("dance", "fitness"):
        for group_idx in range(groups_per_category):
            chunk_count = 2 if group_idx % 2 == 0 else 1
            for chunk_idx in range(chunk_count):
                name = f"motion_{group_idx:03d}_chunk_{chunk_idx:04d}"
                make_npz(
                    data_root / category / f"{name}.npz",
                    source_file=f"PHUMA/data/g1/{category}/{name}.npy",
                    frames=4 + chunk_idx,
                )
    return project, data_root


def test_normalize_source_group_strips_only_explicit_slice_suffixes(tmp_path: Path) -> None:
    data_root = tmp_path / "g1_all"
    npz_path = data_root / "dance" / "dummy.npz"

    assert splits.normalize_source_group(
        "PHUMA/data/g1/dance/subset_0000/Apink_Mr_Chu_chunk_0004.npy", npz_path, data_root
    ) == ("dance/subset_0000/Apink_Mr_Chu", False)
    assert splits.normalize_source_group("PHUMA/data/g1/dance/foo_clip_001.npy", npz_path, data_root)[0] == "dance/foo"
    assert (
        splits.normalize_source_group("PHUMA/data/g1/dance/foo_segment_12.npy", npz_path, data_root)[0]
        == "dance/foo"
    )
    assert splits.normalize_source_group("PHUMA/data/g1/dance/foo_part_3.npy", npz_path, data_root)[0] == "dance/foo"
    assert (
        splits.normalize_source_group("PHUMA/data/g1/humanml/motion_1234.npy", npz_path, data_root)[0]
        == "humanml/motion_1234"
    )


def test_parent_directories_keep_same_stem_separate(tmp_path: Path) -> None:
    data_root = tmp_path / "g1_all"
    npz_path = data_root / "dance" / "dummy.npz"
    group_a = splits.normalize_source_group("PHUMA/data/g1/dance/subset_a/same_chunk_0000.npy", npz_path, data_root)[0]
    group_b = splits.normalize_source_group("PHUMA/data/g1/dance/subset_b/same_chunk_0000.npy", npz_path, data_root)[0]

    assert group_a == "dance/subset_a/same"
    assert group_b == "dance/subset_b/same"
    assert group_a != group_b


def test_fallback_source_group_is_recorded_and_grouped(tmp_path: Path) -> None:
    data_root = tmp_path / "g1_all"
    npz_path = data_root / "dance" / "fallback_chunk_0001.npz"
    group, used_fallback = splits.normalize_source_group("", npz_path, data_root)

    assert group == "dance/fallback"
    assert used_fallback is True


def test_grouped_split_is_leak_free_and_reproducible(tmp_path: Path) -> None:
    project, data_root = create_dataset(tmp_path)
    records = splits.scan_metadata(project, data_root, workers=1)

    assignment_a, _ = splits.split_groups_by_category(records, (0.8, 0.1, 0.1), seed=42)
    assignment_b, _ = splits.split_groups_by_category(records, (0.8, 0.1, 0.1), seed=42)
    assignment_c, _ = splits.split_groups_by_category(records, (0.8, 0.1, 0.1), seed=43)

    assert assignment_a == assignment_b
    assert assignment_a != assignment_c

    splits.assign_splits(records, assignment_a)
    checks = splits.run_integrity_checks(records, project, data_root)
    assert checks["ok"] is True

    group_to_split = {}
    for record in records:
        group_to_split.setdefault(record.source_group, record.split)
        assert group_to_split[record.source_group] == record.split

    manifests = splits.manifest_paths_by_split(records, project, data_root, "relative")
    for lines in manifests.values():
        assert len(lines) == len(set(lines))


def test_invalid_fps_and_empty_joint_pos_are_detected(tmp_path: Path) -> None:
    project = tmp_path
    data_root = project / "PHUMA_wbt_motions" / "g1_all"
    make_npz(data_root / "dance" / "bad_fps.npz", "PHUMA/data/g1/dance/bad_fps.npy", fps=0.0)
    make_npz(data_root / "dance" / "empty.npz", "PHUMA/data/g1/dance/empty.npy", frames=0)

    records = splits.scan_metadata(project, data_root, workers=1)
    reasons = {Path(record.path).name: record.invalid_reason for record in records}

    assert "invalid_fps" in reasons["bad_fps.npz"]
    assert "empty_joint_pos" in reasons["empty.npz"]


def test_manifest_path_resolution_supports_relative_and_absolute(tmp_path: Path) -> None:
    project = tmp_path
    data_root = project / "PHUMA_wbt_motions" / "g1_all"
    motion = data_root / "dance" / "a.npz"
    make_npz(motion, "PHUMA/data/g1/dance/a.npy")
    manifest_dir = project / "PHUMA_wbt_motions" / "manifests" / "splits_v1"
    manifest_dir.mkdir(parents=True)

    assert splits.resolve_manifest_entry("PHUMA_wbt_motions/g1_all/dance/a.npz", manifest_dir, project, data_root) == motion
    assert splits.resolve_manifest_entry("../g1_all/dance/a.npz", manifest_dir, project, data_root) == motion
    assert splits.resolve_manifest_entry(motion.as_posix(), manifest_dir, project, data_root) == motion


def test_cli_refuses_overwrite_without_force_and_validate_only_finds_leak(tmp_path: Path) -> None:
    project, data_root = create_dataset(tmp_path, groups_per_category=5)
    output_dir = project / "PHUMA_wbt_motions" / "manifests" / "splits_v1"
    cmd = [
        sys.executable,
        str(SCRIPT),
        "--project-root",
        str(project),
        "--data-root",
        str(data_root),
        "--output-dir",
        str(output_dir),
        "--split-mode",
        "grouped-random",
        "--seed",
        "42",
        "--strict",
    ]

    first = subprocess.run(cmd, check=False, capture_output=True, text=True)
    assert first.returncode == 0, first.stderr + first.stdout

    second = subprocess.run(cmd, check=False, capture_output=True, text=True)
    assert second.returncode != 0
    assert "already exists" in second.stderr

    train_line = (output_dir / "train_pool.txt").read_text().splitlines()[0]
    with (output_dir / "test.txt").open("a") as f:
        f.write(f"{train_line}\n")

    validate = subprocess.run(
        cmd + ["--validate-only"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert validate.returncode != 0
    assert "split validation failed" in validate.stderr
