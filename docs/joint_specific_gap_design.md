# Joint-Specific Difficulty-Calibrated Learning-Gap Reweighting

## 1. Motivation

This document freezes the STEP 4 design contract for the internal method
`M7-JGap`:

```text
Joint-Specific Difficulty-Calibrated Learning-Gap Reweighting
```

The current frozen baseline is `M7-Raw`:

```text
Quality Gate
  -> Cluster Diversity
  -> Motion Raw Error
  -> Segment Raw Error
```

`M7-JGap` keeps that structure intact. Joint-specific gap is allowed only as a
bounded secondary correction to the segment raw-error priority. It must not
change the PPO algorithm, rewards, environment dynamics, quality gate,
difficulty metadata, cluster-diversity budget, raw-error definition, motion
sampling, or cluster sampling.

The design goal is not to prefer wrist, elbow, upper body, or any manually named
joint group. The method must apply one uniform mathematical rule to all 29 G1
tracking DOFs and let the online statistics decide which joints are
under-learned on each segment.

## 2. STEP 3 Evidence

STEP 3 compared the frozen `M7-Raw` checkpoint with the A-full checkpoint:

```text
M7-Raw model_33999.pt
Micro       = 0.884887
Macro       = 0.893026
Completion  = 0.934052
Joint L2    = 0.808584
Failures    = 879

A-full model_25000.pt
Micro       = 0.874804
Macro       = 0.886414
Completion  = 0.928185
Joint L2    = 0.794810
Failures    = 956
```

The joint diagnostics reconstructed evaluator Joint L2 with max consistency
error below `4e-7`, so the Joint L2 improvement is a real tracking effect.
A-full improved mostly in distal arm joints:

```text
arms_wrist RMS delta = -11.72%
arms_elbow RMS delta = -7.50%
waist RMS delta      = +10.11%
```

This supports a joint-specific research direction, but those anatomical results
are evidence only. They are not sampling rules.

## 3. Problem Definition

For cluster `c`, motion `m`, and segment `s`, the joint probability remains
hierarchical:

```text
P(c,m,s) = P_div(c) * P_raw(m | c) * P_joint(s | m)
```

The contract is:

```text
P_new(c)      = P_M7Raw(c)
P_new(m | c)  = P_M7Raw(m | c)
P_new(s | m)  may change only through corrected segment priority
```

The segment path is:

```text
existing M7-Raw segment raw priority
  -> Joint Gap correction
  -> corrected segment priority
  -> existing grouped normalization
  -> existing probability cap / water filling
  -> existing uniform mixture
  -> existing fallback
  -> sampling probability
```

The implementation must not multiply an already normalized probability by a gap
term, and must not blend a separate Joint-Gap probability distribution with the
Raw distribution.

## 4. Per-Joint Error

At each observed training step, the current pipeline has:

```text
q_ref.shape   = (..., 29)
q_robot.shape = (..., 29)
```

Existing evaluator Joint L2 is:

```text
|| q_ref - q_robot ||_2
```

Existing online scalar joint RMS is:

```text
|| q_ref - q_robot ||_2 / sqrt(29)
```

`M7-JGap` keeps the same joint semantics and does not introduce angle
wrap-around or a new representation. STEP 3 recorded
`command.joint_pos - command.robot_joint_pos` and reconstructed evaluator Joint
L2 to numerical precision, so the current reference/error pipeline is the source
of truth. If a future representation needs wrap-around, that change must happen
upstream and consistently affect both evaluator and training metrics; it is not
part of JGap.

The online joint-specific observation is:

```text
e[t,j] = abs(q_ref[t,j] - q_robot[t,j])
```

Unit: radians.

Alternatives considered:

```text
absolute error:
  chosen. It is directly related to Joint L2, stable, interpretable in rad,
  and does not amplify rare extreme steps before difficulty normalization.

squared error:
  rejected for the first JGap contract. It emphasizes outliers and would make
  noisy single-joint spikes more likely to dominate the segment correction.

per-joint RMS accumulator:
  reasonable but redundant for a one-dimensional joint stream. RMS is useful
  when aggregating several joints; for one joint, EMA(abs(error)) is simpler
  and less outlier-sensitive.
```

## 5. Per-Segment Per-Joint Statistics

For global segment `g` and joint `j`, maintain:

```text
E[g,j] = EMA over training-step observations of e[t,j]
```

Expected formal random6000 shape:

```text
num_segments = 21575
num_joints   = 29
E.shape      = (21575, 29)
```

The implementation should keep the existing scalar online joint statistic for
M7-Raw raw error. To avoid ambiguity with the existing scalar field
`statistics.segment_joint_error_ema`, the new matrix should live in a dedicated
JGap namespace:

```text
joint_gap.segment_joint_error_ema         shape (num_segments, 29)
joint_gap.segment_joint_error_initialized shape (num_segments,)
joint_gap.segment_joint_pending_sum       shape (num_segments, 29)
joint_gap.segment_joint_pending_count     shape (num_segments,)
```

`pending_count` is shared across joints because each valid observation contains
all 29 DOFs. If any row contains non-finite joint values, drop that row for
JGap rather than introducing per-joint counts in v1. A later schema may migrate
to `(num_segments, num_joints)` counts only if missing-joint observations become
real.

## 6. EMA Rule And Update Cadence

JGap reuses the existing online-learning cadence:

```text
EMA decay                   = 0.95
probability update interval = 50 PPO iterations
min segment observations    = 32
warmup iterations           = existing M7-Raw value
```

Within one PPO window, observations are first reduced by segment ID:

```text
x_bar[g,j] = pending_sum[g,j] / pending_count[g]
```

Then, at iteration end:

```text
E[g,j] = x_bar[g,j]                         first valid window
E[g,j] = 0.95 * E[g,j] + 0.05 * x_bar[g,j] later valid windows
```

Every active segment receives at most one EMA update per window, matching the
existing scalar statistics semantics.

## 7. Difficulty Calibration

Module 2 provides only segment-level intrinsic difficulty:

```text
difficulty_bin[g] in {0, ..., 9}
```

Therefore this method is difficulty-calibrated per-joint tracking error
conditioned on segment intrinsic difficulty. It is not joint-specific intrinsic
difficulty estimation.

For bin `b` and joint `j`:

```text
mu[b,j]    = expectation of E[g,j] for valid segments with difficulty_bin[g] = b
sigma[b,j] = scale of E[g,j] for valid segments with difficulty_bin[g] = b
```

Frozen shape:

```text
mu.shape    = (10, 29)
sigma.shape = (10, 29)
```

The estimator should vectorize the existing generic function
`estimate_difficulty_bin_expectation`:

```text
mu[b,j] = weighted_mean(E[g,j] | bin[g]=b)
sigma[b,j] = max(weighted_population_std(E[g,j] | bin[g]=b), sigma_floor)
sigma_floor = existing 0.10
min_bin_valid_segments = existing 32
```

No new median/MAD estimator is introduced in STEP 4. Robustness comes from the
same finite mask, min-count fallback, sigma floor, and later gap clipping used
by generic Gap.

If a `(bin, joint)` cell has fewer than 32 valid segments, fall back to the
joint-specific global mean/std over all valid Train segments for that joint. If
the global joint statistic is also insufficient, that joint contribution is
invalid and contributes zero gap. Unknown statistics must never produce a high
priority.

The implementation must report:

```text
joint_gap/bin_fallback_count
joint_gap/sigma_floor_clamp_count
joint_gap/bin_{b}_valid_count_min / mean
```

Full `(10,29)` matrices are saved locally/checkpointed, not expanded into many
W&B scalar keys by default.

## 8. Motion-Local Centering

Difficulty calibration removes global bin/joint scale effects, but motions can
still differ in overall tracking difficulty. Therefore JGap keeps motion-local
centering.

For motion `m`, segment set `S_m`, and joint `j`:

```text
G_global[g,j] = (E[g,j] - mu[difficulty_bin[g],j])
                / (sigma[difficulty_bin[g],j] + delta)

center[m,j] = Median_{k in S_m and valid} G_global[k,j]

G_local[g,j] = G_global[g,j] - center[m(g),j]
```

`G_global` and `G_local` are clipped with the existing `gap_clip = 5.0`.
If a motion has fewer than two valid segments for a joint, `G_local[:,j]` for
that motion contributes zero.

## 9. Positive Gap

Only positive under-learning deficit can increase priority:

```text
G_pos[g,j] = max(0, G_local[g,j])
```

Negative gap must not lower segment raw priority. This is a correction on top
of M7-Raw, not a replacement for raw error.

## 10. Top-K Aggregation

Do not average over all 29 joints; it would dilute localized deficits. Do not
use max over joints; one noisy joint could dominate.

Freeze:

```text
top_k_fraction = 0.20
num_joints     = 29
K              = ceil(29 * 0.20) = 6
aggregation    = mean of top 6 positive joint gaps
```

Formula:

```text
TopK(g)    = largest 6 values of G_pos[g,:]
G_joint[g] = mean(TopK(g))
```

Invalid or insufficient joint entries are treated as zero before `topk`, so they
cannot create boost. `K=6` is frozen structurally before training. It is not a
Validation-swept hyperparameter.

## 11. Raw Gate

JGap must not revive segments that M7-Raw currently considers low-priority.

Let `w_raw[g]` be the existing M7-Raw segment raw score before grouped
normalization, cap, water filling, and uniform mixture. For each motion:

```text
raw_median[m] = Median_{k in S_m and raw_valid} w_raw[k]

H[g] = 1 if raw_valid[g]
          and w_raw[g] >= raw_median[m(g)]
       else 0
```

If a raw median cannot be computed, `H[g]=0`. This gate uses the existing raw
segment priority only. It does not define a new raw joint threshold.

## 12. Bounded Correction

Define:

```text
C[g] = H[g] * tanh(G_joint[g])
```

Then:

```text
0 <= C[g] < 1
```

Final corrected segment priority:

```text
w_new[g] = w_raw[g] * (1 + lambda_joint * C[g])
```

Therefore:

```text
w_raw[g] <= w_new[g] < (1 + lambda_joint) * w_raw[g]
```

`lambda_joint` is symbolic in STEP 4. It must be non-negative and finite, but
its value is not frozen here. STEP 7 will choose one value through offline
probability replay based on sampling perturbation magnitude, not through a
Validation sweep.

The contract requires:

```text
lambda_joint = 0  =>  w_new = w_raw
```

The STEP 5 implementation should bypass correction arithmetic entirely when
`lambda_joint=0` so the probability builder receives the same raw tensor and
does not consume any additional RNG.

## 13. Probability Contract

Only `w_new` replaces the segment score passed to the existing grouped
probability builder. All downstream mechanics remain unchanged:

```text
score clipping
under-sampling bonus
softmax
uniform mix
conditional cap
water filling
fallback
```

Cluster diversity and motion raw-error probabilities are computed exactly as in
M7-Raw. JGap must not affect:

```text
cluster probability
motion probability
motion validity
cluster target-observed budget correction
```

## 14. Tensor Shapes

For formal random6000:

```text
num_segments = 21575
num_motions  = 6000
num_joints   = 29
num_bins     = 10
```

Core runtime tensors:

```text
joint_gap.segment_joint_error_ema         (21575, 29)
joint_gap.segment_joint_error_initialized (21575,)
joint_gap.segment_joint_pending_sum       (21575, 29)
joint_gap.segment_joint_pending_count     (21575,)

joint_gap.bin_mu                          (10, 29)
joint_gap.bin_sigma                       (10, 29)
joint_gap.bin_valid_count                 (10, 29) or (10,) if shared-count only
joint_gap.bin_fallback_mask               (10, 29)
joint_gap.sigma_floor_clamped             (10, 29)

joint_gap.global_gap                      (21575, 29)
joint_gap.local_gap                       (21575, 29)
joint_gap.positive_gap                    (21575, 29)
joint_gap.segment_gap_score               (21575,)
joint_gap.raw_gate                        (21575,)
joint_gap.correction                      (21575,)
joint_gap.corrected_segment_priority      (21575,)
```

`global_gap`, `local_gap`, and `positive_gap` are formula caches and can be
recomputed deterministically. They should be saved only if the existing
online-learning cache pattern stores comparable formula outputs.

## 15. Fallback Behavior

Fallbacks are conservative:

```text
segment observation count < 32:
  C[g] = 0
  w_new[g] = w_raw[g]

segment joint EMA not initialized:
  C[g] = 0
  w_new[g] = w_raw[g]

insufficient (bin, joint) and insufficient global joint fallback:
  that joint contribution = 0

motion has fewer than two valid local segments for a joint:
  that joint contribution = 0

raw_valid[g] is false or raw median missing:
  H[g] = 0

any non-finite intermediate:
  replace contribution with 0 and log fallback/clamp counters
```

The method must never turn unknown statistics into high sampling priority.

## 16. Checkpoint State

The new checkpoint state must be namespaced so old scalar raw-error state remains
available:

```text
online_learning.joint_gap = {
  schema_version,
  config_hash,
  enabled,
  num_segments,
  num_joints,
  num_bins,
  top_k,
  top_k_fraction,
  positive_only,
  local_center,
  raw_gate,
  lambda_joint,
  ema_decay,
  probability_update_interval,
  min_segment_observations,
  min_bin_valid_segments,
  difficulty_metadata_identity,

  segment_joint_error_ema,
  segment_joint_error_initialized,
  segment_joint_pending_sum,
  segment_joint_pending_count,

  bin_mu,
  bin_sigma,
  bin_valid_count,
  bin_fallback_mask,
  sigma_floor_clamped,

  global_gap,
  local_gap,
  segment_gap_score,
  raw_gate_mask,
  correction,
}
```

Resume must restore the pending window and formula caches consistently. If
`bin_mu/bin_sigma` are deterministically rebuildable from restored EMA/counts,
they may still be checkpointed as cached formula state to match the existing
`bin_calibration` / `gap_result` pattern. On load, the implementation must
validate shape, dtype, finite values, config hash, and difficulty metadata
identity.

Old `M7-Raw` checkpoints do not contain per-joint history. A formal JGap run
must start from scratch. Do not fabricate per-joint state from old scalar
checkpoints.

## 17. RNG Contract

JGap computation is deterministic tensor math and introduces no random draws.

Required degeneracy behavior:

```text
lambda_joint = 0:
  no extra RNG consumption
  corrected segment priority equals raw priority
  assignment trace should match M7-Raw as closely as existing floating behavior allows

JGap enabled but uninitialized:
  C = 0
  no additional assignment RNG consumption
  initial sampling follows the M7-Raw path
```

All actual sampling remains inside the existing adaptive sampler generator.

## 18. W&B Diagnostics

Do not upload `(21575,29)` matrices to W&B. Log scalar summaries:

```text
joint_gap/mean
joint_gap/p90
joint_gap/max
joint_gap/correction_mean
joint_gap/correction_p90
joint_gap/active_segment_fraction
joint_gap/raw_gate_pass_fraction
joint_gap/top1_probability_mass
joint_gap/top5_probability_mass
joint_gap/effective_segment_count
joint_gap/bin_fallback_count
joint_gap/sigma_floor_clamp_count
```

For joint-level summaries, log either 29 compact keys or anatomical-group
aggregates:

```text
joint_gap/per_joint_mean_ema/*
joint_gap/per_joint_positive_gap/*
joint_gap/group_mean_ema/*
joint_gap/group_positive_gap/*
```

The full per-joint arrays should be saved locally as `npz/csv` diagnostics,
not streamed as W&B time-series matrices.

## 19. Evaluator Metrics

Formal validation for M7-JGap must keep the original frozen metrics:

```text
Micro
Macro
Completion
Body Error
Joint L2
Failures
```

It should additionally report:

```text
per-joint RMS
joint-group RMS
```

Checkpoint selection rule does not change:

```text
1. Macro first
2. if Macro difference <= 0.002, compare Micro
3. then Completion
4. then Body Error
5. then earlier checkpoint
```

Joint L2 is reported as a diagnostic outcome, not a selection tie-breaker.

## 20. Tests Required For STEP 5

The implementation should add CPU tests before any training:

```text
per-joint accumulator:
  shape validation, shared pending_count, finite-row filtering, EMA first/update

vectorized bin calibration:
  (10,29) mu/sigma, sigma floor count, sparse bin fallback, global fallback

motion-local centering:
  per-joint median, single-segment motion returns zero, translation invariance

Top-K aggregation:
  K=6, invalid joints zero, all-zero gap returns zero, max does not dominate

Raw Gate:
  within-motion median gate, missing raw median disables correction

bounded correction:
  non-negative, finite, w_new >= w_raw, upper bound, lambda=0 exact degeneracy

probability isolation:
  cluster probability unchanged, motion probability unchanged, only P(s|m) changes

fallback:
  cold segment, sparse bin, NaN/Inf, no high priority from unknown state

checkpoint/resume:
  state schema, config identity, metadata identity, pending window, formula cache

RNG:
  lambda=0 and uninitialized JGap do not consume sampler RNG beyond M7-Raw path

contract:
  no hard-coded joint names, no wrist/elbow whitelist, no waist exclusion
```

No training, Validation, or Test is part of STEP 4.

## 21. Frozen And Tunable Parameters

Frozen for STEP 5:

```text
method internal name              = M7-JGap
num_joints                        = 29
difficulty bins                   = existing 10
per-joint statistic               = EMA(abs(q_ref - q_robot))
EMA decay                         = 0.95
probability update interval       = 50
min segment observations          = 32
min bin valid segments            = 32
sigma_floor                       = existing 0.10
gap_clip                          = existing 5.0
positive_only                     = true
local_center                      = motion median per joint
aggregation                       = Top-K positive mean
top_k_fraction                    = 0.20
top_k                             = 6
raw gate                          = within-motion median raw priority
transform                         = tanh
motion sampling                   = M7-Raw raw error
cluster sampling                  = M7-Raw diversity
checkpoint selection              = frozen Macro-first rule
```

Not frozen in STEP 4:

```text
lambda_joint
```

`lambda_joint` may only be frozen later by offline probability replay based on
sampling perturbation magnitude. It must not be selected by Validation sweep.

## 22. Memory And Performance Estimate

For `21575 x 29`:

```text
elements = 625675
float32  = 2,502,700 bytes = 2.39 MiB per matrix
float64  = 5,005,400 bytes = 4.77 MiB per matrix
```

Recommended v1 state uses float32 for the new per-joint matrices:

```text
EMA                 2.39 MiB
pending_sum         2.39 MiB
G_global cache      2.39 MiB
G_local cache       2.39 MiB
G_pos/topk temp     2.39 MiB
initialized mask    0.02 MiB if shared per segment
pending_count       0.16 MiB if int64 per segment
bin mu/sigma        <0.01 MiB
```

Conservative additional GPU memory is about `12-15 MiB`, or about `25 MiB` if
the implementation keeps all large matrices in float64 to match existing online
statistics. This is small relative to 3072-environment training memory.

The update should stay GPU-native. Segment window reduction can use
`scatter_add_` / `bincount` with a `(num_observations, 29)` value matrix.
Difficulty-bin statistics should avoid CPU round-trips by flattening
`(bin, joint)` into group IDs:

```text
group_id = difficulty_bin[g] * num_joints + j
num_groups = 10 * 29
```

Quantiles for motion-local medians may reuse the existing segmented quantile
approach per joint or a vectorized grouped implementation. Any CPU fallback
would need explicit profiling before formal training.

## 23. Expected STEP 5 Implementation Files

Expected files for implementation, after this design is reviewed:

```text
source/whole_body_tracking/whole_body_tracking/utils/joint_gap.py
source/whole_body_tracking/whole_body_tracking/utils/online_learning_stats.py
source/whole_body_tracking/whole_body_tracking/utils/online_learning.py
source/whole_body_tracking/whole_body_tracking/utils/adaptive_sampling.py
source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py
scripts/rsl_rl/train.py
tests/test_joint_gap.py
tests/test_online_learning_controller.py
tests/test_online_learning_runtime_integration.py
tests/test_adaptive_sampling.py
```

`scripts/rsl_rl/evaluate.py` and STEP 3 diagnostics can be reused for reporting
per-joint validation metrics.

## 24. Final Formula

For every segment `g` and joint `j`:

```text
e[t,j] = abs(q_ref[t,j] - q_robot[t,j])

E[g,j] = EMA(e[t,j])

G_global[g,j] =
  clip((E[g,j] - mu[difficulty_bin[g],j])
       / (sigma[difficulty_bin[g],j] + delta),
       -5, 5)

G_local[g,j] =
  clip(G_global[g,j] - Median_{k in S_m(g)} G_global[k,j],
       -5, 5)

G_pos[g,j] = max(0, G_local[g,j])

G_joint[g] = mean(largest_6(G_pos[g,:]))

H[g] = 1[w_raw[g] >= Median_{k in S_m(g)} w_raw[k]]

C[g] = H[g] * tanh(G_joint[g])

w_new[g] = w_raw[g] * (1 + lambda_joint * C[g])
```

`lambda_joint=0` strictly degenerates to M7-Raw segment priority.
