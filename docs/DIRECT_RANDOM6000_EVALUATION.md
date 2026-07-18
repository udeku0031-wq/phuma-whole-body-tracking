# Direct Random-6000 Evaluation

This document defines the fixed-split evaluation workflow for the Direct Mixed
Training baseline trained on `random6000_seed42.txt`.

## Scope

The evaluation workflow does not modify PPO, rewards, motion sampling, random
start-frame logic, or the fixed `splits_v1` train/validation/test manifests.
Checkpoint selection is done only on Validation Probe 500. Test is reserved for
the frozen final checkpoint.

## Fixed Splits

- Train pool: `PHUMA_wbt_motions/manifests/splits_v1/train_pool.txt`
- Validation: `PHUMA_wbt_motions/manifests/splits_v1/validation.txt`
- Test: `PHUMA_wbt_motions/manifests/splits_v1/test.txt`
- Checkpoint-selection probe: `PHUMA_wbt_motions/manifests/splits_v1/validation_probe500_seed42.txt`
- Smoke probe: `PHUMA_wbt_motions/manifests/splits_v1/validation_smoke20_seed42.txt`

The probe exists as a frozen subset. It must not be regenerated when comparing
checkpoints.

## Why Probe 500 First

Full validation is large enough that checking every saved checkpoint would be
slow. The 500-motion validation probe gives a fixed, repeatable selection set.
After the best checkpoint is chosen, full Validation is run once for the frozen
checkpoint.

Test is not used for checkpoint selection. Using Test to pick a checkpoint would
leak information from the final held-out set and invalidate the baseline.

## Deterministic Motion Evaluation

`scripts/rsl_rl/evaluate.py` assigns exact motion ids with
`MotionCommand.set_eval_motion_state(...)`. Each manifest motion is evaluated
once, starting from frame 0. The evaluator records early termination as failure
and natural completion at the final motion frame as success.

Use `--disable_randomization` to disable reset pose/velocity/joint noise,
observation corruption, pushes, and common domain-randomization events. If a
randomization block cannot be found, the evaluator records a warning in
`evaluation_config.json`.

## Outputs

Each evaluation output directory contains:

- `per_motion.csv`: one row per manifest motion
- `category_summary.csv`: category-level aggregate
- `source_group_summary.csv`: source-group aggregate across chunks
- `summary.json`: global metrics and manifest/checkpoint integrity
- `evaluation_config.json`: exact evaluation configuration
- `failures.txt`: failed motion paths and reasons

Chunk-level results are the rows in `per_motion.csv`. Source-group results group
multiple chunks from the same original source clip.

## Metrics

- `success`: 1 only if the motion reaches its final frame without early
  termination.
- `completion_ratio`: `completed_frames / num_frames`, clamped to `[0, 1]`.
- `body_position_error_m`: mean over evaluation steps of
  `MotionCommand.metrics["error_body_pos"]`. That metric is the mean Euclidean
  distance over configured tracked bodies after the command's yaw/root alignment
  in `_update_relative_body_targets`.
- `joint_position_error_l2_rad`: mean over evaluation steps of the L2 norm over
  all robot joints. This preserves the old 29-joint L2-style metric.
- `joint_position_error_rms_rad`: `joint_position_error_l2_rad / sqrt(29)` for
  the current G1 policy, or `/ sqrt(joint_count)` as recorded in
  `evaluation_config.json`.

The older four-stage model's 84.8% result is useful as an engineering reference,
but this Direct Random-6000 model is the fixed-split baseline for this control
experiment.

## Checkpoint Sweep

Dry run:

```bash
env -u PYTHONPATH -u LD_LIBRARY_PATH \
python scripts/rsl_rl/evaluate_checkpoint_sweep.py \
  --task Tracking-Flat-G1-v0 \
  --motion-file PHUMA_wbt_motions/manifests/splits_v1/validation_probe500_seed42.txt \
  --run-dir 'logs/rsl_rl/g1_flat/*direct_random6000_seed42_env3072*' \
  --checkpoints 10000,20000,30000,final \
  --num-envs 16 \
  --output-root results/direct_random6000/checkpoint_sweep \
  --seed 42 \
  --deterministic \
  --disable-randomization \
  --dry-run
```

Formal run: remove `--dry-run`.

The sweep writes `checkpoint_comparison.csv`, `checkpoint_comparison.json`, and
`best_checkpoint.json`. `best_checkpoint.json` records both
`selected_checkpoint` and `selected_load_run`; use both for full Validation and
final Test. The default tie-break rules are:

1. higher `macro_success_rate`;
2. if within `0.002`, higher `micro_success_rate`;
3. if still close, higher `mean_completion_ratio`;
4. if still close, lower `mean_body_position_error_m`;
5. if still close, earlier checkpoint.

## Full Validation

After `best_checkpoint.json` is frozen, run full Validation with the selected
checkpoint:

```bash
env -u PYTHONPATH -u LD_LIBRARY_PATH \
python scripts/rsl_rl/evaluate.py \
  --task Tracking-Flat-G1-v0 \
  --motion_file PHUMA_wbt_motions/manifests/splits_v1/validation.txt \
  --num_envs 16 \
  --headless \
  --load_run 2026-07-17_01-31-21_direct_random6000_seed42_env3072_resume18000 \
  --checkpoint <BEST_CHECKPOINT> \
  --output_dir results/direct_random6000/validation_full/<BEST_CHECKPOINT_STEM> \
  --seed 42 \
  --deterministic \
  --disable_randomization \
  --resume
```

`--resume` reads an existing `per_motion.csv`, skips completed rows, and checks
that every manifest motion appears exactly once at the end.

## Final Test

Only run Test after full Validation is complete and the checkpoint is frozen.
The evaluator refuses `test.txt` unless `--confirm_final_test` is supplied.

```bash
env -u PYTHONPATH -u LD_LIBRARY_PATH \
python scripts/rsl_rl/evaluate.py \
  --task Tracking-Flat-G1-v0 \
  --motion_file PHUMA_wbt_motions/manifests/splits_v1/test.txt \
  --num_envs 16 \
  --headless \
  --load_run 2026-07-17_01-31-21_direct_random6000_seed42_env3072_resume18000 \
  --checkpoint <BEST_CHECKPOINT> \
  --output_dir results/direct_random6000/test_final/<BEST_CHECKPOINT_STEM> \
  --seed 42 \
  --deterministic \
  --disable_randomization \
  --resume \
  --confirm_final_test
```

Do not change checkpoint after seeing Test results.

## Full Validation Candidate Freeze

For the final Direct Random-6000 comparison requested after training, compare
the two candidate checkpoints on full Validation before freezing:

- `model_20000.pt`
- `model_33999.pt`

Use the orchestration helper below. It resumes incomplete Validation outputs,
writes `validation_full/checkpoint_comparison.csv`,
writes `results/direct_random6000/frozen_best_checkpoint.json`, and then runs
Test only for the frozen checkpoint.

Dry run:

```bash
env -u PYTHONPATH -u LD_LIBRARY_PATH \
python scripts/rsl_rl/finalize_direct_random6000_eval.py \
  --task Tracking-Flat-G1-v0 \
  --run-dir logs/rsl_rl/g1_flat/2026-07-17_01-31-21_direct_random6000_seed42_env3072_resume18000 \
  --checkpoints model_20000.pt,model_33999.pt \
  --num-envs 16 \
  --seed 42 \
  --deterministic \
  --disable-randomization \
  --dry-run
```

Formal run:

```bash
env -u PYTHONPATH -u LD_LIBRARY_PATH \
python scripts/rsl_rl/finalize_direct_random6000_eval.py \
  --task Tracking-Flat-G1-v0 \
  --run-dir logs/rsl_rl/g1_flat/2026-07-17_01-31-21_direct_random6000_seed42_env3072_resume18000 \
  --checkpoints model_20000.pt,model_33999.pt \
  --num-envs 16 \
  --seed 42 \
  --deterministic \
  --disable-randomization \
  --confirm-final-test
```

The helper refuses to run Test unless `--confirm-final-test` is supplied. Test
results are never used to switch checkpoint.

## W&B

Local CSV/JSON files are the primary record. The sweep can optionally log
Validation metrics to a separate W&B evaluation run with:

```bash
--wandb
```

Metrics are logged as `Validation/micro_success_rate`,
`Validation/macro_success_rate`, `Validation/completion_ratio`,
`Validation/body_position_error_m`,
`Validation/joint_position_error_l2_rad`, and
`Validation/joint_position_error_rms_rad`, using checkpoint iteration as the
step.

## Manual Playback

After evaluation, inspect both successful and failed samples from
`per_motion.csv`/`failures.txt` with `scripts/rsl_rl/play.py`, using
`--skip_export` to avoid large ONNX exports.
