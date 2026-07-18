# W&B Baseline Curve Stitching

This workflow creates one clean synthetic W&B run for the Direct Mixed Training
baseline by replaying scalar history from the two interrupted baseline runs.

It does not modify training checkpoints or original W&B runs.

## Recommended Dashboard Project

Use a separate clean project for baseline comparison:

```text
whole_body_tracking_phuma_baseline
```

The stitched baseline run is uploaded there first. Future comparison trainings
can use the same project with `--log_project_name whole_body_tracking_phuma_baseline`.

## Stitch phuma15 + phuma16

Dry run:

```bash
cd ~/whole_body_tracking_new
conda activate hybrid_robot

env -u PYTHONPATH -u LD_LIBRARY_PATH \
python scripts/wandb_stitch_runs.py \
  --entity longxianli222-northeastern-university \
  --project whole_body_tracking_phuma \
  --source-runs h7iunh9y,bfrcob1u \
  --output-project whole_body_tracking_phuma_baseline \
  --output-name baseline_direct_random6000_model33999_stitched \
  --output-run-id baseline-direct-random6000-model33999-v1 \
  --step-mode auto \
  --dry-run
```

Formal upload:

```bash
env -u PYTHONPATH -u LD_LIBRARY_PATH \
python scripts/wandb_stitch_runs.py \
  --entity longxianli222-northeastern-university \
  --project whole_body_tracking_phuma \
  --source-runs h7iunh9y,bfrcob1u \
  --output-project whole_body_tracking_phuma_baseline \
  --output-name baseline_direct_random6000_model33999_stitched \
  --output-run-id baseline-direct-random6000-model33999-v1 \
  --step-mode auto
```

If the run ids are uncertain, paste full W&B URLs in `--source-runs` instead of
short ids.

`--step-mode auto` preserves a resumed run whose metric steps already continue
past the previous run, even if the two runs overlap slightly around the resume
boundary. It only offsets a later run when its metric steps fully reset behind
the previous run.

## Future Training

To show future experiments in the same clean W&B project:

```bash
env -u PYTHONPATH -u LD_LIBRARY_PATH \
python scripts/rsl_rl/train.py \
  --task Tracking-Flat-G1-v0 \
  --motion_file <YOUR_TRAIN_MANIFEST> \
  --headless \
  --logger wandb \
  --log_project_name whole_body_tracking_phuma_baseline \
  --run_name <NEW_EXPERIMENT_NAME> \
  --num_envs <N> \
  --max_iterations <ITERATIONS>
```

For resumed trainings, keep logging to `whole_body_tracking_phuma_baseline` and
use a descriptive `--run_name`. The stitched baseline will remain as the fixed
reference run in that clean project.

## Outputs

The script also writes local copies:

```text
results/wandb_stitched/baseline_direct_random6000_history.csv
results/wandb_stitched/baseline_direct_random6000_metadata.json
```

These files are useful for checking which source run contributed each stitched
step.
