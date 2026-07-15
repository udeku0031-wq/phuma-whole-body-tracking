# PHUMA WBT Fixed Split v1

This directory contains the fixed PHUMA split used by the local whole_body_tracking + PHUMA project.

## Why group-aware splitting is required

PHUMA motions are often chunked from one original sequence, for example:

```text
Apink_Mr_Chu_chunk_0000.npz
Apink_Mr_Chu_chunk_0001.npz
Apink_Mr_Chu_chunk_0002.npz
```

Adjacent chunks are highly similar. Splitting each `.npz` independently would leak nearly identical motion into train,
validation, and test. This split therefore assigns all chunks from the same `source_group` to exactly one split.

## Source group

`source_group` is derived from the converted `.npz` field `source_file` when available. Paths are normalized to POSIX
format, PHUMA `data/g1` prefixes are removed, file extensions are removed, and only explicit slice suffixes such as
`_chunk_0000`, `_clip_0000`, `_segment_0000`, and `_part_0000` are stripped. Plain trailing numbers are kept.

## Split mode

Actual mode used: `grouped-random`.

`official` uses PHUMA official train/test files when they can be mapped reliably. `grouped-random` uses a fixed seed and
category-stratified source-group splitting. `auto` tries official first and falls back to grouped-random.

## Files

- `metadata.csv`: one row per converted `.npz`, including category, source_group, split, validity, frame count, and fps.
- `train_pool.txt`: training pool manifest.
- `validation.txt`: validation manifest for checkpoint/model-selection experiments.
- `test.txt`: final test manifest. Do not tune on this file.
- `*_source_groups.txt`: source-group lists for each split.
- `split_report.json`: counts, category distribution, and leakage checks.
- `split_config.json`: reproducibility metadata.
- `checksums.sha256`: checksums for the split manifests/config.
- `invalid_files.txt`: invalid or corrupted files excluded from all splits.

## Counts

```text
train      files=60858 source_groups=28902
validation files=7636 source_groups=3615
test       files=7592 source_groups=3610
```

## Regenerate

```bash
python scripts/build_phuma_splits.py \
  --project-root . \
  --data-root PHUMA_wbt_motions/g1_all \
  --output-dir PHUMA_wbt_motions/manifests/splits_v1 \
  --split-mode auto \
  --train-ratio 0.8 \
  --val-ratio 0.1 \
  --test-ratio 0.1 \
  --seed 42 \
  --path-mode relative \
  --strict \
  --force
```

## Validate

```bash
python scripts/build_phuma_splits.py \
  --project-root . \
  --data-root PHUMA_wbt_motions/g1_all \
  --output-dir PHUMA_wbt_motions/manifests/splits_v1 \
  --validate-only \
  --strict
```

The default refuses to overwrite an existing fixed split. Use `--force` only when intentionally replacing the split.

Future Random-6000 subsets, quality filtering, curriculum learning, direct mixed training, validation, and final testing
should all be derived from this fixed split. Existing models trained before this split may have already seen data that is
now in `test.txt`, so they should not be reported as formal held-out test results.
