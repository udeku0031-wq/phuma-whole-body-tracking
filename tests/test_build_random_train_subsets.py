from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_random_train_subsets.py"


def make_project(tmp_path: Path, train_count: int = 14) -> Path:
    project = tmp_path
    data_root = project / "PHUMA_wbt_motions" / "g1_all"
    split_dir = project / "PHUMA_wbt_motions" / "manifests" / "splits_v1"
    split_dir.mkdir(parents=True)

    rows = []
    manifests = {"train": [], "validation": [], "test": []}

    def add_motion(split: str, index: int, category: str, source_group: str) -> None:
        rel = f"PHUMA_wbt_motions/g1_all/{category}/{split}_{index:03d}.npz"
        path = project / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fake npz; metadata supplies stats")
        manifests[split].append(rel)
        rows.append(
            {
                "path": path.as_posix(),
                "relative_path": rel,
                "category": category,
                "source_file": "",
                "source_format": "PHUMA_G1",
                "source_group": source_group,
                "num_frames": str(100 + index),
                "fps": "50",
                "split": split,
                "used_fallback": "false",
                "valid": "true",
                "invalid_reason": "",
            }
        )

    for i in range(train_count):
        category = "dance" if i % 3 == 0 else "fitness"
        add_motion("train", i, category, f"train_group_{i // 2:03d}")
    for i in range(2):
        add_motion("validation", i, "dance", f"validation_group_{i:03d}")
        add_motion("test", i, "fitness", f"test_group_{i:03d}")

    (split_dir / "train_pool.txt").write_text("".join(f"{item}\n" for item in manifests["train"]))
    (split_dir / "validation.txt").write_text("".join(f"{item}\n" for item in manifests["validation"]))
    (split_dir / "test.txt").write_text("".join(f"{item}\n" for item in manifests["test"]))
    (split_dir / "split_config.json").write_text('{"split_version":"splits_v1"}\n')
    with (split_dir / "metadata.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return project


def run_builder(
    project: Path,
    output_dir: Path,
    *,
    seed: int = 42,
    sizes: str = "3,6,12",
    force: bool = False,
    validate_only: bool = False,
) -> subprocess.CompletedProcess[str]:
    cmd = [
        sys.executable,
        str(SCRIPT),
        "--project-root",
        str(project),
        "--output-dir",
        str(output_dir),
        "--seed",
        str(seed),
        "--sizes",
        sizes,
        "--strict",
    ]
    if force:
        cmd.append("--force")
    if validate_only:
        cmd.append("--validate-only")
    return subprocess.run(cmd, check=False, capture_output=True, text=True)


def read_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def test_random_subsets_are_reproducible_nested_and_leak_free(tmp_path: Path) -> None:
    project = make_project(tmp_path / "project")
    out_a = project / "PHUMA_wbt_motions" / "manifests" / "experiments" / "random_seed42_a"
    out_b = project / "PHUMA_wbt_motions" / "manifests" / "experiments" / "random_seed42_b"
    out_c = project / "PHUMA_wbt_motions" / "manifests" / "experiments" / "random_seed43"

    first = run_builder(project, out_a, seed=42)
    assert first.returncode == 0, first.stderr + first.stdout
    second = run_builder(project, out_b, seed=42)
    assert second.returncode == 0, second.stderr + second.stdout
    third = run_builder(project, out_c, seed=43)
    assert third.returncode == 0, third.stderr + third.stdout

    assert (out_a / "random_order_seed42.txt").read_text() == (out_b / "random_order_seed42.txt").read_text()
    assert (out_a / "random3_seed42.txt").read_text() == (out_b / "random3_seed42.txt").read_text()
    assert (out_a / "sampling_metadata.csv").read_text() == (out_b / "sampling_metadata.csv").read_text()
    assert (out_a / "random_order_seed42.txt").read_text() != (out_c / "random_order_seed43.txt").read_text()

    train_pool = set(read_lines(project / "PHUMA_wbt_motions" / "manifests" / "splits_v1" / "train_pool.txt"))
    random3 = read_lines(out_a / "random3_seed42.txt")
    random6 = read_lines(out_a / "random6_seed42.txt")
    random12 = read_lines(out_a / "random12_seed42.txt")

    assert len(random3) == 3
    assert len(random6) == 6
    assert len(random12) == 12
    assert len(random12) == len(set(random12))
    assert set(random3) < set(random6)
    assert set(random6) < set(random12)
    assert set(random12).issubset(train_pool)

    report = json.loads((out_a / "sampling_report.json").read_text())
    assert report["integrity_checks"]["validation_file_intersections"]["random12"] == 0
    assert report["integrity_checks"]["test_file_intersections"]["random12"] == 0
    assert report["integrity_checks"]["validation_source_group_intersections"]["random12"] == 0
    assert report["integrity_checks"]["test_source_group_intersections"]["random12"] == 0

    with (out_a / "sampling_metadata.csv").open(newline="") as f:
        random3_categories = [
            row["category"] for row in csv.DictReader(f) if row["in_random3"].lower() == "true"
        ]
    expected_counts = {category: random3_categories.count(category) for category in set(random3_categories)}
    assert report["subsets"]["random3"]["category_file_counts"] == expected_counts


def test_cli_refuses_overwrite_without_force(tmp_path: Path) -> None:
    project = make_project(tmp_path / "project")
    out = project / "PHUMA_wbt_motions" / "manifests" / "experiments" / "random_seed42"

    first = run_builder(project, out)
    assert first.returncode == 0, first.stderr + first.stdout
    second = run_builder(project, out)
    assert second.returncode != 0
    assert "already exists" in second.stderr


def test_validate_only_finds_duplicate_paths(tmp_path: Path) -> None:
    project = make_project(tmp_path / "project")
    out = project / "PHUMA_wbt_motions" / "manifests" / "experiments" / "random_seed42"
    assert run_builder(project, out).returncode == 0

    lines = read_lines(out / "random3_seed42.txt")
    lines[1] = lines[0]
    (out / "random3_seed42.txt").write_text("".join(f"{line}\n" for line in lines))

    validate = run_builder(project, out, validate_only=True)
    assert validate.returncode != 0
    assert "duplicate" in validate.stderr or "nested check failed" in validate.stderr


def test_validate_only_finds_non_train_pool_paths(tmp_path: Path) -> None:
    project = make_project(tmp_path / "project")
    out = project / "PHUMA_wbt_motions" / "manifests" / "experiments" / "random_seed42"
    assert run_builder(project, out).returncode == 0

    extra = project / "PHUMA_wbt_motions" / "g1_all" / "dance" / "outside_pool.npz"
    extra.write_bytes(b"fake")
    lines = read_lines(out / "random3_seed42.txt")
    lines[2] = "PHUMA_wbt_motions/g1_all/dance/outside_pool.npz"
    (out / "random3_seed42.txt").write_text("".join(f"{line}\n" for line in lines))

    validate = run_builder(project, out, validate_only=True)
    assert validate.returncode != 0
    assert "subset_outside_train_pool_count" in validate.stderr


def test_validate_only_finds_broken_nested_prefix(tmp_path: Path) -> None:
    project = make_project(tmp_path / "project")
    out = project / "PHUMA_wbt_motions" / "manifests" / "experiments" / "random_seed42"
    assert run_builder(project, out).returncode == 0

    order = read_lines(out / "random_order_seed42.txt")
    lines = read_lines(out / "random6_seed42.txt")
    lines[0] = order[10]
    (out / "random6_seed42.txt").write_text("".join(f"{line}\n" for line in lines))

    validate = run_builder(project, out, validate_only=True)
    assert validate.returncode != 0
    assert "nested check failed" in validate.stderr


def test_sizes_bounds_and_unsorted_input(tmp_path: Path) -> None:
    project = make_project(tmp_path / "project")
    out_bad = project / "PHUMA_wbt_motions" / "manifests" / "experiments" / "too_large"
    bad = run_builder(project, out_bad, sizes="3,20")
    assert bad.returncode != 0
    assert "exceeds Train Pool size" in bad.stderr

    out_unsorted = project / "PHUMA_wbt_motions" / "manifests" / "experiments" / "unsorted"
    ok = run_builder(project, out_unsorted, sizes="6,3,12")
    assert ok.returncode == 0, ok.stderr + ok.stdout
    assert (out_unsorted / "random3_seed42.txt").exists()
    assert (out_unsorted / "random6_seed42.txt").exists()
    assert (out_unsorted / "random12_seed42.txt").exists()
