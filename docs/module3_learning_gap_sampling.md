# 模块三：Motion–Segment 在线误差与学习缺口采样

## 1. 目标与边界

模块三维护一套由所有并行环境共享的 Motion/Segment 在线统计，并支持 M2–M6
及独立的 `GLOBAL_BIN_RAW_ERROR` 基线。它复用 Stage-0 的
`FixedLengthSegmentIndex`、模块一质量标签和模块二难度区间，不重新生成或修改任何
映射。

三类信息严格解耦：

- quality 只生成 assignment-start eligibility mask；
- difficulty 只提供冻结的 10-bin ID，用来估计当前策略的同难度期望误差；
- tracking error 和 traversal/episode outcome 只来自当前训练策略，用来计算 raw error
  与 learning gap。

质量状态、难度分数都不会进入 `E_segment`。在线误差也不会反写模块二的固有难度。
M7 cluster 层、动作聚类、Test 评测和正式训练不属于本模块。

`configs/online_learning/g1_module3.yaml` 是 provisional v1 参数基线和模式速查表，
不是可直接合并到 Hydra 的 config group；实际运行配置由 `ResearchExperimentCfg`
提供，并通过下文的 Hydra overrides 选择模式。

## 2. 代码结构与调用链

- `utils/online_learning_stats.py`：共享计数、窗口聚合、EMA、segment/motion outcome；
- `utils/learning_gap.py`：`E_segment`、`E_motion`、bin calibration 和三层 gap；
- `utils/adaptive_sampling.py`：mask、探索、uniform mix、water-filling cap 和专用 RNG；
- `utils/online_learning.py`：统计、公式、采样器和 checkpoint 的控制器；
- `tasks/tracking/mdp/commands.py`：提取已有 tracking 向量、处理生命周期并请求 assignment；
- `tasks/tracking/online_learning_env.py`：在 Isaac 重置前保留 terminal frame，并在首次
  external reset 后初始化在线 reference target；
- `utils/my_on_policy_runner.py`：PPO iteration 边界、W&B 和 checkpoint/resume。

运行时链路为：

```text
simulation/control step
  -> 当前 reference motion/segment 的 tracking error
  -> 全局 window sum/count accumulator
  -> segment boundary / reset 前结算 traversal 与 motion outcome
PPO iteration end
  -> 每个 ID 至多一次 EMA 更新
低频 score 边界
  -> E_segment / E_motion
  -> difficulty-bin mu/sigma（M5/M6）
  -> G_global / G_motion / G_local（M5/M6）
低频 probability 边界
  -> masked probability + uniform floor + cap
下一次 assignment
  -> 专用 sampler generator 抽样
```

模块三关闭时，环境 hook 立即委托基类，uniform/uniform 直接调用原 legacy sampler；
不会创建 adaptive generator，也不会新增 legacy sampler 的随机调用。

## 3. 模式定义

| 方法 | Motion mode | Segment mode | Quality | Difficulty | 在线统计 |
|---|---|---|---:|---:|---:|
| M0 | `uniform` | `uniform` | off | off | off |
| M1 | `uniform` | `uniform` | on | off | off |
| M2 | `raw_error` | `uniform` | off | off | on |
| M3 | `uniform` | `raw_error` | off | off | on |
| M4 | `raw_error` | `raw_error` | off | off | on |
| M5 | `learning_gap` | `relative_learning_gap` | off | on | on |
| M6 | `learning_gap` | `relative_learning_gap` | on | on | on |
| GLOBAL_BIN_RAW_ERROR | ignored/global | `global_bin_raw_error` | configurable | off | on |

M2 选中 motion 后，仍在该 motion 的全部 legacy 合法 start frame 中均匀抽样。
M3/M4/M5/M6 先抽 motion，再抽该 motion 内的 segment，最后在 segment 的合法 start
frame 中均匀抽样。全局 bin 模式直接跨 motion 抽 segment，再反查 motion。

未知模式和非法组合会抛出 `NotImplementedError` 或 `ValueError`，不会静默回退。

## 4. 共享统计、批量更新和归因

所有环境写入同一套大小由当前 manifest 动态确定的 tensor。三类 count 相互独立：

- `*_sample_count`：assignment 起点被选中的次数；
- `segment_step_count` / `motion_step_count`：真实 reference-frame observation 次数；
- `segment_outcome_count` / `motion_episode_count`：traversal/episode outcome 次数。

每个 control step 的 body、joint、orientation error 按该时刻的
`current_motion_id/current_global_segment_id` 归因，而不是按 assignment 起点归因。
跨 segment 边界时先结算旧 segment，再创建新 traversal；跨段不会伪造新的 assignment
count。常规内部 reset 会先创建 reference assignment，随后 command update 才推进一帧；
这个仅用于初始化且未产生 tracking observation 的起始帧会从 traversal/motion 的预期剩余
分母中扣除。显式 `env.reset(env_ids=...)` 会先删失旧 cursor，完成一次 reset 和 target
刷新后只计算一次 observation，避免伪 terminal 和重复 observation。

一个 PPO window 内先用 `torch.bincount` 按 ID 聚合 sum/count，再在 iteration end 对每个
ID 更新一次：

```text
x_bar(i,t) = window_sum(i,t) / window_count(i,t)
EMA(i,t)   = x_bar(i,t)                                  首次有效观察
EMA(i,t)   = rho * EMA(i,t-1) + (1-rho) * x_bar(i,t)    后续
rho        = 0.95
```

因此重复 ID 不会在同一窗口内重复衰减，结果与 env 排列顺序无关。热路径没有逐 env 的
Python loop。

## 5. Step error 定义

复用 motion command 已计算的目标和机器人状态，每个环境产生一个 scalar：

```text
body_error [m]
  = mean_b ||position_ref,b - position_robot,b||_2

joint_error [rad]
  = sqrt(mean_j (q_ref,j - q_robot,j)^2)

orientation_error [rad]
  = mean_b quaternion_geodesic_angle(q_ref,b, q_robot,b)
```

orientation 使用 quaternion geodesic angle，不使用 quaternion 分量 L2。默认 scale、
weight 和单位如下，全部可配置且标记为 provisional：

| 分量 | scale | E_segment weight |
|---|---:|---:|
| body | 0.30 m | 1.0 |
| joint | 0.50 rad | 1.0 |
| orientation | 0.40 rad | 1.0 |
| termination | 无量纲 | 1.0 |
| `1-completion` | 无量纲 | 0.5 |
| `1-success` | 无量纲 | 0.5 |

各 tracking component 在归一化后 clip 到 `[0, 5]`。缺失 outcome 不会被填成严重错误，
而是从 active-weight 分子和分母同时移除；同名 component 的定义始终不变。

## 6. Segment traversal outcome

Segment 区间采用 Stage-0 的右开区间 `[start, end)`。

- 正常越过边界：`termination=0, completion=1, success=1`；
- segment 内物理提前终止：`termination=1, success=0`，且
  `completion=observed_frames/required_remaining_frames`；
- 从 segment 中间开始时，completion 分母只取本次需要遍历的剩余长度；
- `minimum_segment_observed_fraction=0.20` 按完整 segment 长度检查。观察比例不足时仍
  更新 step tracking error，但 completion/success 记为缺失，避免从末尾一两帧开始产生
  伪成功；
- framework timeout 是行政删失：记录 `termination=0`，completion/success 均缺失，
  不让固定环境 horizon 变成学习失败信号；若 timeout 恰好发生在 segment 的自然末帧，
  该 segment 仍按自然完成结算，但 motion outcome 继续按行政删失处理。

这里的 success 表示自然离开 segment 或自然完成 motion，不是 evaluator 的成功率阈值。

## 7. Motion episode outcome

Motion outcome 始终归给当前 assignment 的 motion：

```text
termination     = 是否发生物理提前终止
completion      = observed_frames / expected_remaining_frames
success         = 是否自然完成 motion
```

从 motion 中间开始时，分母使用起点之后的预期剩余长度；自然完成强制 completion/success
为 1。物理提前终止保留真实 completion 并记 success=0。framework timeout 记录
termination=0，同时删失 completion/success。若 reset 原因不能解析，则按行政 truncation
处理，不冒充策略失败。

## 8. Raw error

令 active component 集为 `A_ms`：

```text
z_body    = clip(body_ema / 0.30 m, 0, 5)
z_joint   = clip(joint_ema / 0.50 rad, 0, 5)
z_ori     = clip(orientation_ema / 0.40 rad, 0, 5)
z_term    = termination_ema
z_comp    = 1 - completion_ema
z_success = 1 - success_ema

E_segment(ms) = sum_(k in A_ms) w_k z_k / sum_(k in A_ms) w_k
```

Segment 至少需要 32 个实际 reference observations，并且 tracking component 已初始化，
才可作为可靠 score；否则保持 finite 但标记 cold/invalid。

Motion raw error 使用 observation-count-weighted segment mean、unweighted linear P90 和
motion outcome：

```text
mean_E(m) = sum_s n_ms E_segment(ms) / sum_s n_ms
p90_E(m)  = P90_valid_s(E_segment(ms))

E_motion(m) = active_weighted_mean(
  mean_E(m), p90_E(m), termination_m,
  1-completion_m, 1-success_m
)

weights = [1.0, 0.25, 0.5, 0.5, 0.5]
```

Motion 至少需要 8 个 episode outcomes 和一个可靠 segment。否则 sampler 使用中性先验、
uniform floor 与 under-sampling bonus，而不是把初始 0 当成低误差。

M2/M3/M4 和 global-bin baseline 只使用这些在线 raw errors，不使用 difficulty。

## 9. Difficulty calibration 与 learning gap

M5/M6 加载模块二冻结的 `difficulty_bin(ms)`。只用当前训练过程中已经可靠的
`E_segment` 估计：

```text
mu_b    = weighted_mean(E_segment(ms) | bin(ms)=b)
sigma_b = max(weighted_population_std(E_segment(ms) | bin(ms)=b), 0.10)
```

默认每个 valid segment 等权；`bin_observation_weighted=true` 可改为 observation-count
加权。一个 bin 少于 32 个 valid segments 时回退到所有 valid Train segments 的全局
mean/std；全局也不足时该 bin 不可靠，不产生 gap。Validation/Test 从不参与估计。

全局 segment gap：

```text
G_global(ms) = clip((E_segment(ms) - mu_bin(ms)) / sigma_bin(ms), -5, 5)
```

Motion gap 只聚合正 excess：

```text
G_pos(ms) = max(G_global(ms), 0)

G_motion(m) = clip(active_weighted_mean(
  observation-weighted Mean_s(G_pos),
  P90_s(G_pos),
  termination_m,
  1-completion_m,
  1-success_m
), 0, 5)

weights = [1.0, 0.5, 0.5, 0.5, 0.5]
```

Motion 内局部相对 gap：

```text
G_local(ms) = clip(
  G_global(ms) - Median_valid_j(G_global(mj)),
  -5, 5
)
```

单 valid segment motion 的 `G_local=0`，无 valid segment 时 conditional sampler 均匀。
Motion 层回答“整条动作是否整体没学会”，segment 层回答“动作内部哪一段相对更弱”。
给同一 motion 的所有 `G_global` 加常数不会改变 `G_local`。segment sampler 直接使用
signed `G_local`；Motion 的绝对缺口只决定动作预算，条件分布只比较该 motion 内相对差异。

## 10. 概率、探索和 cap

所有 adaptive 方法共用同一概率变换：

```text
U_i       = 1 / sqrt(sample_count_i + 1)
logit_i   = (clip(score_i,-10,10) + 0.25 U_i) / 1.0
adaptive  = Softmax(logit over eligible items)
P_i       = 0.15 Uniform_i + 0.85 adaptive_i
```

随后执行 proportional water-filling/capped-simplex，而不是简单 clip 后再归一：

- motion cap：0.02；
- motion 内 conditional segment cap：1.00；
- global-bin 模式的 segment cap：1.00；
- cap 严格生效；若某个 eligible 集合只有 `N` 项而配置 cap 小于 `1/N`，初始化立即
  抛出不可行配置错误，不会静默放宽；
- 非法或完全不可靠 score 回退 uniform 并记录 fallback；
- quality Reject 和 empty motion 始终严格为 0；
- 其余 eligible items 保留 uniform floor。

层级模式的联合概率是：

```text
P(m,s) = P_motion(m) * P_segment(s | m)
```

概率仅在 PPO iteration 边界更新。前 1000 iterations 采样保持均匀，但统计仍增长；
score/error/gap 和 probability 默认每 50 iterations 刷新，EMA 每个 iteration 提交。
更新节拍由独立的 completed-window counter 驱动，而不是只依赖 runner 展示的 iteration
标签；因此 resume 后即使框架再次报告保存点的同一标签，也不会重复 warmup 或错过更新。
完整分布 entropy/max/min 也只在低频刷新或恢复时计算，reset 热路径只读取缓存。

## 11. Quality 集成与全局 bin 基线

M6 使用：

```text
Stage-0 legal start
intersection quality eligible segment
intersection non-empty motion
```

再在该 mask 上应用 learning-gap 概率。quality mask 不参与 `E_segment`、bin mean/std、
`G_global`、`G_motion` 或 `G_local`；同理，difficulty 只提供 bin。由于模块一 gate scope
是 `assignment_start`，从合法起点自然播放经过 Reject 区间仍可被诊断统计观察，但 Reject
片段永远不能成为新的 assignment 起点。

`GLOBAL_BIN_RAW_ERROR` 将所有 eligible segment flatten 后直接按 `E_segment` 抽样，
反查 motion，再在 segment 内均匀抽合法 start frame。它不使用 motion hierarchy 或
difficulty，名称中的 bin 指全局离散 segment bins，不是 difficulty bin。

## 12. 专用 RNG、checkpoint 与 resume

只有 adaptive 模式创建 `torch.Generator`，seed 默认为 42，并保留完整 device index。
该 generator 只用于 adaptive motion/segment/start-frame assignment，不推进 policy、domain
randomization 或全局 torch RNG。

checkpoint sidecar 保存：

- schema、module-three config hash、Stage-0 segment-index/pool identity；
- quality/difficulty metadata SHA256 与 mapping identity（启用时）；
- assignment/step/outcome counts、所有 EMA/initialized mask、pending window；
- 当前 runner iteration、已提交 window 数、最近 score/probability update window；
- `E_segment/E_motion`、bin mean/sigma/count/fallback、三类 gap；
- motion、conditional segment、global segment probability；
- fallback/cap counters 和 sampler generator state。

恢复时校验 shape、normalization、eligibility、有效 cap、配置与 metadata identity。统计、
概率和专用 sampler stream 精确恢复；RSL-RL wrapper 已经创建的临时 assignment 会被丢弃，
随后从恢复的 sampler RNG 抽取下一批，因此 assignment stream 连续。per-env 物理状态和
traversal cursor 不写入 sidecar，恢复后以新 assignment 重建；domain-randomization 的全局
RNG 也不是本模块状态，所以不承诺整个物理 rollout bitwise 相同。

确定性 evaluator 必须设置 `online_learning.enabled=false`，不能复用训练 cursor/RNG。

## 13. W&B 诊断

不上传 6000/21575 长数组。新增标量包括：

- `online/*`：valid ratio、observation/episode count、EMA update、cold count；
- `error/*`：segment mean/P50/P90/P99、motion mean/P90、六个 component mean；
- `gap/bin_{0..9}_{mu,sigma,valid_count}`、`gap/bin_fallback_count`；
- `gap/global_*`、`gap/motion_*`、`gap/local_*`、positive/clipped ratio；
- `sampling/mode`、warmup、update count、motion/segment entropy、max/min probability、
  effective count、uniform mix、fallback 和 probability sum error；
- `sampling/difficulty_bin_budget_{0..9}`（M5/M6）；
- 模块一已有的 `quality/reject_start_assignment_count`、
  `quality/reject_rollout_exposure_ratio`、`quality/excluded_motion_count`。

`sampling/mode` 是数值 code：M2=2、M3=3、M4=4、M5/M6=5、global-bin=7；正式方法名
同时保存在 W&B config。

## 14. CPU tests

核心测试不依赖 Isaac Sim，覆盖：

- batch permutation invariance、重复 ID 单次 EMA、首次 EMA 无零偏；
- current-segment attribution、正常边界、中段终止、中段开始、低观察比例与 timeout 删失；
- `E_segment/E_motion` 手算、cold start、bin mean/std、sparse fallback、sigma floor；
- 高难度高误差但 gap 不高的合成例；
- `G_global/G_motion/G_local`，local 平移不变性和单 segment motion；
- mask、uniform floor、temperature、under-sampling、fallback、signed local score、严格
  water-filling cap 与不可行 cap fail-fast；
- M2/M3/M4、M6 Reject mask、global-bin、warmup/update interval；
- sampler/controller checkpoint、概率 cap 校验和 RNG 序列恢复；
- M0/M1 legacy dispatch、pre-reset hook、iteration hook 和三方 mapping 接线。

运行：

```bash
cd /home/l/whole_body_tracking_new
/home/l/miniconda3/envs/hybrid_robot/bin/python -m unittest discover -s tests -p 'test_*.py'
```

## 15. random100 pilot 命令

本节只提供启动模板，不表示已经运行。公共参数：

```bash
cd /home/l/whole_body_tracking_new
conda activate hybrid_robot

MANIFEST=/home/l/whole_body_tracking_new/outputs/module2_difficulty_pilot_random100_seed42_v1/normalized_manifest.txt
DIFFICULTY=/home/l/whole_body_tracking_new/outputs/module2_difficulty_pilot_random100_seed42_v1/segment_difficulty_metadata.npz
QUALITY=/home/l/whole_body_tracking_new/outputs/module1_quality_pilot_random100_seed42_v1/segment_quality_metadata.npz

COMMON_ARGS=(
  --disable_fabric
  --task Tracking-Flat-G1-v0
  --motion_file "$MANIFEST"
  --headless
  --logger wandb
  --log_project_name whole_body_tracking_module3_pilot
  --num_envs 16
  --seed 42
  --max_iterations 2000
  env.commands.motion.research.segment.enabled=true
  env.commands.motion.research.segment.length_seconds=1.0
  env.commands.motion.research.online_learning.warmup_iterations=1000
  env.commands.motion.research.online_learning.probability_update_interval=50
  env.commands.motion.research.online_learning.sampler_seed=42
  env.commands.motion.research.sampling_statistics.enabled=true
  env.commands.motion.research.sampling_statistics.log_interval=1
  env.commands.motion.research.diversity_constraint.enabled=false
)
```

Pilot A（statistics only；不创建 adaptive RNG）：

```bash
env -u PYTHONPATH -u LD_LIBRARY_PATH WBT_DISABLE_ONNX_ON_SAVE=1 \
python scripts/rsl_rl/train.py "${COMMON_ARGS[@]}" \
  --run_name module3_stats_random100_seed42_v1 \
  --wandb_run_name module3_stats_random100_seed42_v1 \
  --wandb_run_id module3-stats-random100-seed42-v1 --wandb_resume never \
  env.commands.motion.research.method_name=M0 \
  env.commands.motion.research.motion_sampling.mode=uniform \
  env.commands.motion.research.segment_sampling.mode=uniform \
  env.commands.motion.research.quality_gate.enabled=false \
  env.commands.motion.research.difficulty_calibration.enabled=false \
  env.commands.motion.research.online_learning.enabled=true \
  env.commands.motion.research.online_learning.statistics_enabled=true
```

Pilot B（M2）与 Pilot C（M3）：

```bash
env -u PYTHONPATH -u LD_LIBRARY_PATH WBT_DISABLE_ONNX_ON_SAVE=1 \
python scripts/rsl_rl/train.py "${COMMON_ARGS[@]}" \
  --run_name module3_m2_random100_seed42_v1 \
  --wandb_run_name module3_m2_random100_seed42_v1 \
  --wandb_run_id module3-m2-random100-seed42-v1 --wandb_resume never \
  env.commands.motion.research.method_name=M2 \
  env.commands.motion.research.motion_sampling.mode=raw_error \
  env.commands.motion.research.segment_sampling.mode=uniform \
  env.commands.motion.research.quality_gate.enabled=false \
  env.commands.motion.research.difficulty_calibration.enabled=false \
  env.commands.motion.research.online_learning.enabled=true \
  env.commands.motion.research.online_learning.statistics_enabled=true

env -u PYTHONPATH -u LD_LIBRARY_PATH WBT_DISABLE_ONNX_ON_SAVE=1 \
python scripts/rsl_rl/train.py "${COMMON_ARGS[@]}" \
  --run_name module3_m3_random100_seed42_v1 \
  --wandb_run_name module3_m3_random100_seed42_v1 \
  --wandb_run_id module3-m3-random100-seed42-v1 --wandb_resume never \
  env.commands.motion.research.method_name=M3 \
  env.commands.motion.research.motion_sampling.mode=uniform \
  env.commands.motion.research.segment_sampling.mode=raw_error \
  env.commands.motion.research.quality_gate.enabled=false \
  env.commands.motion.research.difficulty_calibration.enabled=false \
  env.commands.motion.research.online_learning.enabled=true \
  env.commands.motion.research.online_learning.statistics_enabled=true
```

Pilot D（M4）：

```bash
env -u PYTHONPATH -u LD_LIBRARY_PATH WBT_DISABLE_ONNX_ON_SAVE=1 \
python scripts/rsl_rl/train.py "${COMMON_ARGS[@]}" \
  --run_name module3_m4_random100_seed42_v1 \
  --wandb_run_name module3_m4_random100_seed42_v1 \
  --wandb_run_id module3-m4-random100-seed42-v1 --wandb_resume never \
  env.commands.motion.research.method_name=M4 \
  env.commands.motion.research.motion_sampling.mode=raw_error \
  env.commands.motion.research.segment_sampling.mode=raw_error \
  env.commands.motion.research.quality_gate.enabled=false \
  env.commands.motion.research.difficulty_calibration.enabled=false \
  env.commands.motion.research.online_learning.enabled=true \
  env.commands.motion.research.online_learning.statistics_enabled=true
```

M5：

```bash
env -u PYTHONPATH -u LD_LIBRARY_PATH WBT_DISABLE_ONNX_ON_SAVE=1 \
python scripts/rsl_rl/train.py "${COMMON_ARGS[@]}" \
  --run_name module3_m5_random100_seed42_v1 \
  --wandb_run_name module3_m5_random100_seed42_v1 \
  --wandb_run_id module3-m5-random100-seed42-v1 --wandb_resume never \
  env.commands.motion.research.method_name=M5 \
  env.commands.motion.research.motion_sampling.mode=learning_gap \
  env.commands.motion.research.segment_sampling.mode=relative_learning_gap \
  env.commands.motion.research.quality_gate.enabled=false \
  env.commands.motion.research.difficulty_calibration.enabled=true \
  env.commands.motion.research.difficulty_calibration.metadata_path="$DIFFICULTY" \
  env.commands.motion.research.difficulty_calibration.strict_metadata_match=true \
  env.commands.motion.research.online_learning.enabled=true \
  env.commands.motion.research.online_learning.statistics_enabled=true
```

M6：

```bash
env -u PYTHONPATH -u LD_LIBRARY_PATH WBT_DISABLE_ONNX_ON_SAVE=1 \
python scripts/rsl_rl/train.py "${COMMON_ARGS[@]}" \
  --run_name module3_m6_random100_seed42_v1 \
  --wandb_run_name module3_m6_random100_seed42_v1 \
  --wandb_run_id module3-m6-random100-seed42-v1 --wandb_resume never \
  env.commands.motion.research.method_name=M6 \
  env.commands.motion.research.motion_sampling.mode=learning_gap \
  env.commands.motion.research.segment_sampling.mode=relative_learning_gap \
  env.commands.motion.research.quality_gate.enabled=true \
  env.commands.motion.research.quality_gate.metadata_path="$QUALITY" \
  env.commands.motion.research.quality_gate.empty_motion_policy=exclude \
  env.commands.motion.research.quality_gate.strict_metadata_match=true \
  env.commands.motion.research.difficulty_calibration.enabled=true \
  env.commands.motion.research.difficulty_calibration.metadata_path="$DIFFICULTY" \
  env.commands.motion.research.difficulty_calibration.strict_metadata_match=true \
  env.commands.motion.research.online_learning.enabled=true \
  env.commands.motion.research.online_learning.statistics_enabled=true
```

500-iteration wiring pilot 应同时把 `--max_iterations` 改为 500、`warmup_iterations`
改为 100；否则 500 iterations 全部处于默认 warmup，无法检查自适应概率变化。

Resume 使用完全相同的 manifest、metadata、method、mode 和 online overrides，并追加：

```bash
--wandb_resume must \
--resume True \
--load_run '.*_module3_m6_random100_seed42_v1$' \
--checkpoint model_1999.pt \
--max_iterations 2
```

验收时检查：warmup 内均匀、之后每 50 iterations 更新、概率 finite/sum=1/cap 合法、
M6 Reject assignment=0、无 NaN/Inf、checkpoint 能恢复统计/概率/RNG，且 W&B 指标存在。

## 16. Provisional 参数与已知限制

当前默认参数尚未论文冻结，必须只用 Train/pilot 诊断后冻结。已知限制：

- optional large state snapshot 配置存在，但 v1 中启用会明确抛出 `NotImplementedError`；
- sparse difficulty bin 使用全局 Train-valid backoff，尚未实现邻近 bin 合并；
- random6000 含 47 条单-segment motion，因此严格 conditional segment cap 的 provisional
  默认值为 1.00；若配置更小且任一 eligible motion 无法归一，运行会 fail-fast；
- checkpoint 不保存 per-env simulator state，因此不是完整物理 rollout 的 bitwise replay；
- 尚未完成真实 Isaac/W&B random100 pilot；CPU tests 不能替代它；
- global-bin 是独立强基线，不自动命名为 M3；
- 本模块没有 cluster diversity。模块四只需在 Motion 概率之前加入 cluster budget/约束，
  无需改写 Segment error、difficulty calibration 或 local-gap 层。
