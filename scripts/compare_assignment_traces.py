#!/usr/bin/env python3
"""Compare bounded assignment-start traces from two smoke runs.

The trace is meant for RNG-equivalence checks.  It compares the sampled
assignment identity columns only: assignment_index, env_id, motion_id,
start_frame, local_segment_id, and global_segment_id.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


DEFAULT_COLUMNS = (
    "assignment_index",
    "env_id",
    "motion_id",
    "start_frame",
    "local_segment_id",
    "global_segment_id",
)


def _read_trace(path: Path, columns: tuple[str, ...]) -> list[tuple[str, ...]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        missing = set(columns).difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
        return [tuple(row[column] for column in columns) for row in reader]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path, help="First assignment_trace.csv")
    parser.add_argument("right", type=Path, help="Second assignment_trace.csv")
    parser.add_argument(
        "--columns",
        nargs="+",
        default=DEFAULT_COLUMNS,
        help="CSV columns to compare. Defaults to assignment identity columns.",
    )
    parser.add_argument(
        "--max-diffs",
        type=int,
        default=10,
        help="Maximum mismatch rows to print before exiting.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    columns = tuple(args.columns)
    left = _read_trace(args.left, columns)
    right = _read_trace(args.right, columns)

    if len(left) != len(right):
        print(f"TRACE_MISMATCH row_count left={len(left)} right={len(right)}", file=sys.stderr)
        return 1

    diff_count = 0
    for row_id, (left_row, right_row) in enumerate(zip(left, right, strict=True)):
        if left_row == right_row:
            continue
        if diff_count < args.max_diffs:
            print(
                f"TRACE_MISMATCH row={row_id} left={left_row} right={right_row}",
                file=sys.stderr,
            )
        diff_count += 1

    if diff_count:
        print(f"TRACE_MISMATCH total_diffs={diff_count}", file=sys.stderr)
        return 1

    print(f"TRACE_MATCH rows={len(left)} columns={','.join(columns)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
