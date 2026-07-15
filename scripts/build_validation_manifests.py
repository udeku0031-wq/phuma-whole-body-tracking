#!/usr/bin/env python3
"""Build fixed PHUMA validation subsets for model evaluation.

The probe and smoke subsets select at most one chunk from each source_group
so repeated chunks from the same original sequence do not receive extra weight.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class Record:
    relative_path: str
    category: str
    source_group: str
    num_frames: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build fixed validation manifests for PHUMA WBT evaluation.")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--split-dir", type=Path, default=Path("PHUMA_wbt_motions/manifests/splits_v1"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-count", type=int, default=20)
    parser.add_argument("--probe-count", type=int, default=500)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def bool_from_csv(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def stable_shuffle(items: list[Record], seed: int, namespace: str) -> list[Record]:
    digest = hashlib.sha256(f"{seed}:{namespace}".encode("utf-8")).hexdigest()
    rng = random.Random(int(digest[:16], 16))
    shuffled = list(items)
    rng.shuffle(shuffled)
    return shuffled


def read_validation_records(metadata_path: Path) -> list[Record]:
    records: list[Record] = []
    with metadata_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("split") != "validation":
                continue
            if row.get("valid") and not bool_from_csv(row["valid"]):
                continue
            try:
                num_frames = int(row.get("num_frames", "0"))
            except ValueError:
                num_frames = 0
            records.append(
                Record(
                    relative_path=row["relative_path"],
                    category=row["category"],
                    source_group=row["source_group"],
                    num_frames=num_frames,
                )
            )
    if not records:
        raise RuntimeError(f"No validation records found in {metadata_path}")
    return records


def choose_one_chunk_per_group(records: list[Record], seed: int) -> list[Record]:
    grouped: dict[str, list[Record]] = defaultdict(list)
    for record in records:
        grouped[record.source_group].append(record)

    selected: list[Record] = []
    for source_group, group_records in grouped.items():
        # Prefer longer chunks for evaluation, then make remaining ties stable.
        max_frames = max(record.num_frames for record in group_records)
        candidates = [record for record in group_records if record.num_frames == max_frames]
        selected.append(stable_shuffle(candidates, seed, f"chunk:{source_group}")[0])
    return selected


def allocate_by_category(records: list[Record], target_count: int) -> dict[str, int]:
    category_counts = Counter(record.category for record in records)
    categories = sorted(category_counts)
    if target_count <= 0:
        return {category: 0 for category in categories}
    if target_count > len(records):
        raise ValueError(f"Requested {target_count} records, but only {len(records)} source groups are available.")

    allocations = {category: 0 for category in categories}
    remaining = target_count

    if target_count >= len(categories):
        for category in categories:
            allocations[category] = 1
        remaining -= len(categories)

    desired = {category: target_count * (category_counts[category] / len(records)) for category in categories}
    while remaining > 0:
        candidates = [
            category
            for category in categories
            if allocations[category] < category_counts[category]
        ]
        if not candidates:
            break
        category = max(
            candidates,
            key=lambda name: (
                desired[name] - allocations[name],
                category_counts[name] - allocations[name],
                name,
            ),
        )
        allocations[category] += 1
        remaining -= 1

    return allocations


def stratified_sample(records: list[Record], target_count: int, seed: int, namespace: str) -> list[Record]:
    one_per_group = choose_one_chunk_per_group(records, seed)
    allocations = allocate_by_category(one_per_group, target_count)

    by_category: dict[str, list[Record]] = defaultdict(list)
    for record in one_per_group:
        by_category[record.category].append(record)

    selected: list[Record] = []
    for category in sorted(by_category):
        shuffled = stable_shuffle(by_category[category], seed, f"{namespace}:{category}")
        selected.extend(shuffled[: allocations[category]])

    return stable_shuffle(selected, seed, f"{namespace}:final")


def write_manifest(path: Path, records: list[Record], force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"{path} already exists. Use --force to overwrite.")
    path.write_text("\n".join(record.relative_path for record in records) + "\n")


def summarize(records: list[Record]) -> dict[str, object]:
    return {
        "num_chunks": len(records),
        "num_source_groups": len({record.source_group for record in records}),
        "categories": dict(sorted(Counter(record.category for record in records).items())),
        "min_frames": min(record.num_frames for record in records) if records else 0,
        "max_frames": max(record.num_frames for record in records) if records else 0,
    }


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    split_dir = args.split_dir if args.split_dir.is_absolute() else project_root / args.split_dir
    metadata_path = split_dir / "metadata.csv"

    records = read_validation_records(metadata_path)
    smoke = stratified_sample(records, args.smoke_count, args.seed, "smoke")
    probe = stratified_sample(records, args.probe_count, args.seed, "probe")
    full = sorted(records, key=lambda record: record.relative_path)

    outputs = {
        f"validation_smoke{args.smoke_count}_seed{args.seed}.txt": smoke,
        f"validation_probe{args.probe_count}_seed{args.seed}.txt": probe,
        "validation_full.txt": full,
    }
    for name, subset in outputs.items():
        write_manifest(split_dir / name, subset, args.force)

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "metadata_path": metadata_path.relative_to(project_root).as_posix(),
        "validation": summarize(records),
        f"smoke{args.smoke_count}": summarize(smoke),
        f"probe{args.probe_count}": summarize(probe),
        "full": summarize(full),
        "notes": [
            "smoke and probe contain at most one chunk per source_group",
            "full contains every validation chunk",
        ],
    }
    report_path = split_dir / f"validation_subsets_seed{args.seed}.json"
    if report_path.exists() and not args.force:
        raise FileExistsError(f"{report_path} already exists. Use --force to overwrite.")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    print(f"[INFO] Wrote validation subsets to {split_dir.relative_to(project_root)}")
    for name, subset in outputs.items():
        summary = summarize(subset)
        print(
            f"[INFO] {name}: {summary['num_chunks']} chunk(s), "
            f"{summary['num_source_groups']} source_group(s)"
        )


if __name__ == "__main__":
    main()
