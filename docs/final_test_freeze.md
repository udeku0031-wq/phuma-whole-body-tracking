# Final Test Freeze Protocol

Created before accessing any final Test results.

All method and checkpoint choices were frozen before accessing the Test results.

## Git Identity

- Project: `/home/l/whole_body_tracking_new`
- Branch at freeze: `feat/joint-gap-probe`
- Code baseline commit: `833438a9a759bff301fc8050ab9937bd40977c10`
- Final Test protocol tag: `paper-final-test-v1`

## Final Proposed Method

Proposed method name:

- Diversity-Budgeted Hierarchical Error Sampling
- Also referred to as Diversity-Constrained Hierarchical Raw Error Sampling

Method contract:

- Method id: `D-only`
- Quality Gate: OFF
- Cluster Diversity: ON
- Motion sampling: Raw Error
- Segment sampling: Raw Error
- Difficulty calibration: OFF
- Generic Learning Gap: OFF
- Joint Gap: OFF
- PPO, reward, environment, raw error definition, cluster budget, randomization, and evaluation implementation are unchanged from the frozen formal experiments.

Sampling factorization:

```text
P(c, m, s) = P_div(c) P_raw(m | c) P_raw(s | m)
```

## Frozen Models

No checkpoint may be changed after this freeze.

| Role | Method | Checkpoint | Run directory | SHA256 |
|---|---|---:|---|---|
| Proposed | D-only | `model_33500.pt` | `logs/rsl_rl/g1_flat/2026-08-23_15-31-13_formal_v1_ablation_Donly_n6000_trainseed42` | `91adfe6bc3b1e37430fa107a2de9fba979d2be49112bd6cf1d45c05388ef218c` |
| Global Raw baseline | GlobalRaw | `model_33500.pt` | `logs/rsl_rl/g1_flat/2026-08-27_15-36-10_formal_GlobalRaw_n6000_trainseed42` | `b363e46fd8e9afc40c65f296d1bc8cb5c65a471290e8c379647645a6393444c9` |
| Hierarchical Raw baseline | M4 | `model_33999.pt` | `logs/rsl_rl/g1_flat/2026-07-27_22-54-05_formal_v1_M4_n6000_trainseed42` | `9ab10ab6a57c750d22a36f3215abb7d6be5cf6e66858125a098c9af0cbe73da8` |
| Strong competing variant | M7-Raw | `model_33999.pt` | `logs/rsl_rl/g1_flat/2026-08-05_15-31-20_formal_v1_ablation_M7Raw_n6000_trainseed42` | `b7d7b685bd4ee9baa3aeb08a426b3452717b45c7167c7fa53a5746dafe430c02` |

## Data Manifests

| Split | Path | Motions | SHA256 |
|---|---|---:|---|
| Training | `PHUMA_wbt_motions/manifests/experiments/random_seed42/random6000_seed42.txt` | 6000 | `51f592792f412c5d5e31caf3621162a3b2ef356e05f98dbce43153d08a59e82d` |
| Validation full | `PHUMA_wbt_motions/manifests/splits_v1/validation_full.txt` | 7636 | `fb70e8b444b20f828226a2dea782d6cf24c2db3900a11e8c01f567f24b0bd42d` |
| Validation probe500 | `PHUMA_wbt_motions/manifests/splits_v1/validation_probe500_seed42.txt` | 500 | `1e44947a99334b104d826776cf1f04a313ad96a08e43de1c21e35569076c8a4d` |
| Test | `PHUMA_wbt_motions/manifests/splits_v1/test.txt` | 7592 | `febffa69e70ed3df966a3d2092b7c0956142babe7de53dc30dc46a8668de570b` |

Manifest independence audit:

- Training intersect Validation full: 0
- Training intersect Validation probe500: 0
- Training intersect Test: 0
- Validation full intersect Test: 0
- Validation probe500 intersect Test: 0
- Validation probe500 is a subset of Validation full and was not used for final Test.

## Validation Selection Rule

Checkpoints were selected using Validation only.

Selection rule:

1. Higher macro success rate.
2. If macro differs by at most `0.002`, higher micro success rate.
3. If still close, higher mean completion ratio.
4. If still close, lower mean body position error.
5. If still close, earlier checkpoint.

Joint L2 does not participate in checkpoint selection.

The Test split was not used for method selection, checkpoint selection, hyperparameter tuning, sampling design, or retraining decisions.

## Final Test Protocol

Each frozen method is evaluated once on the same Test manifest.

- Test manifest: `PHUMA_wbt_motions/manifests/splits_v1/test.txt`
- Evaluation seed: `42`
- Number of environments: `3072`
- Device: `cuda:0`
- Policy execution: deterministic
- Randomization: disabled
- Episode horizon: `60.0` seconds
- Fabric: disabled
- Checkpoint sweep: forbidden
- Test result driven model changes: forbidden

Success definition:

```text
success = 1 only when the evaluator reaches the final motion frame before early termination
completion_ratio = completed_frames / num_frames, clamped to [0, 1]
```

Metrics to report:

- Micro Success
- Macro Success
- Completion
- Body Error
- Joint L2
- Failures
- Success count
- Per-category success, completion, and joint error
- Paired per-motion comparisons
- Deterministic paired bootstrap with seed 42 and 10000 resamples

## Post-Test Rule

After Test is accessed, the proposed method, checkpoints, cluster budget, raw error definition, quality setting, gap setting, and model set must not be changed based on Test outcomes.

If an evaluation implementation bug is discovered, the bug must be recorded explicitly and every frozen Test method must be rerun under the same corrected evaluator. Otherwise, no second-round method selection is allowed.
