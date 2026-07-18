# Random Seed 42 Direct Mixed Training Baseline

This directory contains fixed nested random training subsets derived only from
`PHUMA_wbt_motions/manifests/splits_v1/train_pool.txt`.

Sampling method: `uniform_random_file_sampling`.

Important exclusions:

- No category balancing
- No quality filtering
- No source-group deduplication
- No difficulty weighting

The full random order is stored in `random_order_seed42.txt`. Future larger
or smaller random subsets for this experiment should be prefixes of that file.

## Subsets

- Random-3000: files=3000 source_groups=2776 frames=514356 duration_sec=10287.12
- Random-6000: files=6000 source_groups=5337 frames=1027280 duration_sec=20545.6
- Random-12000: files=12000 source_groups=9932 frames=2059820 duration_sec=41196.4

Nested checks and leakage checks are recorded in `sampling_report.json`.

## Direct Mixed Training Command

Run this manually from a fresh policy initialization. Do not add `--resume`,
`--load_run`, or `--checkpoint`.

```bash
cd ~/whole_body_tracking_new
conda activate hybrid_robot

env -u PYTHONPATH -u LD_LIBRARY_PATH \
python scripts/rsl_rl/train.py \
  --task Tracking-Flat-G1-v0 \
  --motion_file PHUMA_wbt_motions/manifests/experiments/random_seed42/random6000_seed42.txt \
  --headless \
  --logger wandb \
  --log_project_name whole_body_tracking_phuma \
  --run_name direct_random6000_seed42_env3072 \
  --num_envs 3072 \
  --max_iterations 34000
```

This command loads the complete Random-6000 motion library from iteration
1. Motion id sampling and start-frame sampling remain the current project logic
implemented in `MotionCommand`.

## W&B Config

`scripts/rsl_rl/train.py` records direct-mixed metadata when the manifest sits
next to `sampling_config.json`, including sampling method, split version,
manifest name, train size, train-pool size, sampling seed, training seed,
`num_envs`, `max_iterations`, `resume=false`, and `curriculum=false`.

## Checkpoints

The G1 PPO config currently sets `save_interval = 500` in
`source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/agents/rsl_rl_ppo_cfg.py`.
That is already more frequent than every 2000 iterations, so the 34000-iteration
run should keep periodic checkpoints such as `model_10000.pt`, `model_20000.pt`,
and `model_30000.pt`, plus the final checkpoint produced by RSL-RL at the end of
learning. If you want exactly every 2000 iterations, change `save_interval` from
`500` to `2000` in that config before launching.

With `num_envs=3072` and the current G1 PPO setting `num_steps_per_env=24`, each
iteration collects `3072 * 24 = 73728` environment steps. Keep `num_envs=3072`
for later curriculum comparisons.
