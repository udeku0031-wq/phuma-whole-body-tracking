# 阶段 0：Segment 与实验基础设施

本文说明阶段 0 已实现的公共基础设施、配置方式、持久化行为和最小验证流程。阶段 0 的目标是给后续研究模块提供稳定的 segment ID、共享统计、概率检查、日志和 checkpoint 接口，同时保持现有 random6000 训练采样路径不变。

## 范围与未实现功能

阶段 0 已实现：

- 按 motion 自身 FPS 建立固定时长的 segment 索引；
- motion/frame、局部 segment 和全局 segment 之间的双向映射；
- 所有并行环境共享的 motion/segment 分配计数；
- 供未来加权采样使用的概率规范化与合法性检查工具；
- 低维度的 `sampling/*` 训练指标；
- 采样统计随 RSL-RL checkpoint 保存和恢复；
- 统一的 M0--M7 研究配置入口及 W&B config 元数据。

阶段 0 **没有**实现以下算法：

- 质量门控；
- 固有难度标定；
- raw-error、learning-gap 或 relative-learning-gap 加权采样；
- 多样性约束；
- M1--M7 的具体实验方法。

当前只接受 `method_name=M0`、`motion_sampling.mode=uniform` 和 `segment_sampling.mode=uniform`。开启尚未实现的功能或选择其他模式会在环境初始化时抛出 `NotImplementedError`，不会静默退回 uniform。

阶段 0 不改变 observation、reward、termination、PPO 网络及超参数、domain randomization、数据 split 或 manifest。

## 统一研究配置

配置位于：

```text
env.commands.motion.research
```

Python 配置对象中的默认值为：

```yaml
method_name: M0
segment:
  enabled: true
  length_seconds: 1.0
quality_gate:
  enabled: false
difficulty_calibration:
  enabled: false
motion_sampling:
  mode: uniform
segment_sampling:
  mode: uniform
diversity_constraint:
  enabled: false
sampling_statistics:
  enabled: true
  log_interval: 100
probability_validation:
  epsilon: 1.0e-8
```

训练脚本使用 Hydra 接收覆盖值，命令行路径必须以 `env.` 开头。例如：

```bash
env.commands.motion.research.segment.length_seconds=0.5 \
env.commands.motion.research.sampling_statistics.log_interval=10
```

如果要同时关闭 segment 索引和统计，必须同时设置：

```bash
env.commands.motion.research.segment.enabled=false \
env.commands.motion.research.sampling_statistics.enabled=false
```

阶段 0 不允许在 `segment.enabled=false` 时单独开启 sampling statistics。segment 长度、日志间隔和概率 epsilon 也会在启动时进行范围检查。

这些字段随 `env.yaml`/`env.pkl` 保存。训练入口还会把下列低维度字段加入 runner 配置和 W&B config：

```text
method_name
segment_enabled
segment_length_seconds
quality_gate_enabled
difficulty_calibration_enabled
motion_sampling_mode
segment_sampling_mode
diversity_constraint_enabled
sampling_statistics_enabled
sampling_statistics_log_interval
probability_validation_epsilon
```

## Segment 定义与 ID 映射

公共实现位于 `whole_body_tracking/utils/sampling.py` 的 `FixedLengthSegmentIndex`。对 motion `m`：

```python
segment_frames[m] = max(1, round(fps[m] * segment_length_seconds))
num_segments[m] = ceil(num_frames[m] / segment_frames[m])
```

默认 `segment_length_seconds=1.0`。最后一个不足完整时长的尾部 segment 仍然合法，其结束位置会截断到 motion 总帧数。每个 segment 可查询：

```text
motion_id
local_segment_id
global_segment_id
start_frame
end_frame_exclusive
num_frames
motion_num_frames
fps
```

局部 ID 在每个 motion 内从 0 连续编号；全局 ID 按 motion pool 的既定顺序连续编号。核心映射为：

```python
local_segment_id = frame_id // segment_frames[motion_id]
global_segment_id = motion_segment_offsets[motion_id] + local_segment_id
```

反查全局 ID 时使用预计算的 `segment_motion_ids` 和 `segment_local_ids`。索引还保存 `segment_start_frames`、`segment_end_frames` 和 `motion_segment_offsets`，训练热路径不需要遍历 Python 列表。

`FixedLengthSegmentIndex` 本身支持：

- 每条 motion 使用不同 FPS；
- 只有 1 帧的极短 motion；
- 任意正的有限 segment 时长；
- 空批量映射和批量 Tensor 映射。

训练使用的 `MotionLoader` 仍保留 baseline 的输入约束：同一个训练 pool 中 motion 必须同 FPS，且每条 motion 至少包含 2 帧。索引层支持混合 FPS 和 1 帧，是为了保证元数据逻辑独立正确，并为后续数据管线演进预留能力；阶段 0 没有改变 motion 播放时序。

## Baseline 兼容策略

random6000 的 legacy 路径保持原有顺序：

```text
torch.randint 选择 motion
→ sample_uniform 选择原有随机 phase/start frame
→ 根据已选 motion_id 和 start frame 确定性计算 segment ID
→ 累计共享统计
→ 执行原有 reset 状态随机化
```

阶段 0 没有从 segment 反向抽取起始帧，也没有用 `torch.multinomial` 替换 multi-motion uniform 抽样。segment 映射、计数和汇总不调用随机函数；legacy uniform 路径也不调用概率合法性工具。因此在相同 seed 下，新基础设施不会额外推进 PyTorch 随机数状态。

统一入口 `_sample_motion_and_start_frame()` 当前只分发 `uniform + uniform`，并直接调用原有 `_adaptive_sampling()`。未来模式在这个边界接入；阶段 0 的入口自身不调用任何随机函数。

`MotionCommand` 保存每个环境最近一次分配得到的：

```text
assigned_local_segment_ids
assigned_global_segment_ids
```

它们表示“本次起始帧所在的 segment”，不会随 motion 的当前播放帧逐步改变。确定性评估接口不计入训练采样统计。

## 共享统计与概率工具

`SamplingStatistics` 是一个由全部并行环境共享的对象，使用批量 `torch.bincount` 更新：

```text
total_assignments
motion_sample_count
segment_sample_count
invalid_probability_fallback_count
```

它提供批量记录、计数副本、汇总、`state_dict`/`load_state_dict` 和清零接口。motion pool 的指纹包含有序的 pool 内相对文件标识、文件大小、帧数和 FPS；整体搬迁同一数据树不会仅因绝对根目录变化而失效，manifest 顺序或常见文件替换则会被视为不兼容。该轻量指纹不会读取并哈希数千个 NPZ 的全部内容，因此“原路径原大小替换为不同内容”仍需由上游数据 checksum 管理。

`normalize_and_validate_probabilities()` 供后续自适应采样使用。它要求非空一维概率向量，可选检查期望长度，并检查 finite、非负和总质量大于 epsilon。合法输入会被归一化；非法数值默认产生限频的 `RuntimeWarning` 并返回 uniform fallback，同时返回 `used_fallback=True`。传入 `fallback_statistics` 时，fallback 与共享计数递增会在同一次调用中完成；也可使用 `fallback="raise"` 直接报错。

阶段 0 的 legacy uniform sampler 不经过该工具。

## W&B 与训练指标

`MotionOnPolicyRunner` 每隔 `sampling_statistics.log_interval` 个 PPO iteration 查询一次汇总统计，并通过现有 writer 记录：

```text
sampling/total_assignments
sampling/motion_coverage
sampling/segment_coverage
sampling/max_motion_sample_fraction
sampling/max_segment_sample_fraction
sampling/mean_motion_sample_count
sampling/mean_segment_sample_count
sampling/invalid_probability_fallbacks
sampling/num_motions
sampling/num_segments
sampling/segment_length_seconds
```

其中：

```text
motion_coverage  = 已采样 motion 数 / motion 总数
segment_coverage = 已采样 segment 数 / segment 总数
```

只记录上述标量，不向 W&B 上传 motion 或 segment 计数数组。关闭 sampling statistics 后不生成这些指标。

RSL-RL 的 W&B writer 初始化时会先记录一次 `log_dir`，resume 后 W&B 的内部 step 也可能领先于 checkpoint iteration。runner 会在首次日志前计算一个固定的非负 step 偏移，并统一包装现有 writer 的标量入口，保证基础训练指标和 `sampling/*` 都保持单调且不会被 W&B 丢弃；PPO iteration 和 checkpoint 文件编号本身不变。

## Checkpoint 与 resume

采样状态直接嵌入现有 `model_N.pt`：

```python
checkpoint["infos"]["sampling_state"]
```

不会创建 `.sampling_state.pt` sidecar。保存内容包括：

- 状态格式版本和研究配置；
- segment 时长、逐 motion 帧数/FPS、segment frames/count/offset；
- 有序 motion pool 指纹；
- motion/segment 分配计数；
- total assignments 和非法概率 fallback 计数。

恢复时会校验状态版本、segment/statistics 开关、segment 时长、逐 motion 布局、全局 segment 数、计数形状及一致性，以及有序 motion pool 指纹。不兼容时明确报错，避免加载到错误的 manifest。

旧 checkpoint 中没有 `infos["sampling_state"]` 时仍可加载策略；runner 会发出 `RuntimeWarning`，清除环境构造期间的临时累计值，然后把当前已激活的环境 assignment 作为新统计起点。旧 checkpoint 文件格式没有被改写。

RSL-RL wrapper 会在 `runner.load()` 之前先 reset 一次环境。加载带采样状态的新 checkpoint 时，runner 先覆盖恢复累计值，再把这次新进程正在实际使用的当前 assignments 追加一次，避免恢复后永久漏计这些样本。

阶段 0 新增状态 **不包含**：

- Python、NumPy、PyTorch 或 Isaac Sim RNG 状态；
- 当前并行环境的物理状态；
- 当前环境的 motion ID、start frame 或播放进度。

因此 resume 会恢复策略、优化器和 RSL-RL 原有状态以及累计采样统计，但不是逐 bit 的环境轨迹续跑。新进程仍会重新初始化环境。

## 纯 Python 测试

测试不启动 Isaac Sim，覆盖 50 FPS 边界、混合 FPS、1 帧 motion、三种 ID 映射、非法 ID、批量统计、coverage、状态恢复、概率 fallback 和 baseline RNG/调用顺序。

在已安装 PyTorch 的环境中运行：

```bash
cd /home/l/whole_body_tracking_new
python -m unittest tests/test_sampling_infrastructure.py -v
```

项目的完整纯测试可运行：

```bash
python -m pytest -q
```

代码检查：

```bash
git diff --check
python -m compileall \
  source/whole_body_tracking/whole_body_tracking/utils/sampling.py \
  source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py \
  source/whole_body_tracking/whole_body_tracking/utils/my_on_policy_runner.py \
  scripts/rsl_rl/train.py
```

## 16 环境 smoke test

以下流程只取 random6000 manifest 的前 4 条有效记录，并把它们转换成绝对路径写入 `/tmp`。它不会修改正式 manifest。

```bash
cd /home/l/whole_body_tracking_new
conda activate hybrid_robot

awk -v root="$PWD" \
  'NF && $1 !~ /^#/ { print root "/" $0; if (++count == 4) exit }' \
  PHUMA_wbt_motions/manifests/experiments/random_seed42/random6000_seed42.txt \
  > /tmp/stage0_random4_smoke.txt

wc -l /tmp/stage0_random4_smoke.txt
```

`wc` 应输出 4。然后运行 16 个环境、2 个 iteration 的 fresh W&B smoke test；日志间隔设为 1，以便立即看到 `sampling/*`：

```bash
env -u PYTHONPATH -u LD_LIBRARY_PATH \
  WBT_DISABLE_ONNX_ON_SAVE=1 \
python scripts/rsl_rl/train.py \
  --task Tracking-Flat-G1-v0 \
  --motion_file /tmp/stage0_random4_smoke.txt \
  --headless \
  --logger wandb \
  --log_project_name whole_body_tracking_stage0_smoke \
  --run_name stage0_segment_smoke_seed42_env16_fresh_v1 \
  --wandb_run_name stage0_segment_smoke_seed42_env16_v1 \
  --wandb_run_id stage0-segment-smoke-seed42-env16-v1 \
  --wandb_resume never \
  --num_envs 16 \
  --seed 42 \
  --max_iterations 2 \
  env.commands.motion.research.sampling_statistics.log_interval=1
```

预期在 `logs/rsl_rl/g1_flat/<timestamp>_stage0_segment_smoke_seed42_env16_fresh_v1/` 得到 `model_1.pt`。检查控制台中 motion 和 segment 已加载、reward 没有 NaN，并在 W&B 中确认研究配置和上述 `sampling/*` 指标存在。

随后用相同 manifest、seed、W&B run ID 和研究配置恢复 1 个额外 iteration：

```bash
env -u PYTHONPATH -u LD_LIBRARY_PATH \
  WBT_DISABLE_ONNX_ON_SAVE=1 \
python scripts/rsl_rl/train.py \
  --task Tracking-Flat-G1-v0 \
  --motion_file /tmp/stage0_random4_smoke.txt \
  --headless \
  --logger wandb \
  --log_project_name whole_body_tracking_stage0_smoke \
  --run_name stage0_segment_smoke_seed42_env16_resume_v1 \
  --wandb_run_name stage0_segment_smoke_seed42_env16_v1 \
  --wandb_run_id stage0-segment-smoke-seed42-env16-v1 \
  --wandb_resume must \
  --num_envs 16 \
  --seed 42 \
  --resume True \
  --load_run '.*_stage0_segment_smoke_seed42_env16_fresh_v1$' \
  --checkpoint model_1.pt \
  --max_iterations 1 \
  env.commands.motion.research.sampling_statistics.log_interval=1
```

fresh 和 resume 使用不同的本地 `--run_name`，避免 checkpoint 搜索误选刚创建且尚无模型的 resume 目录；W&B name 和 run ID 则保持相同。恢复时不应出现“no sampling state” warning，且 `sampling/total_assignments` 应延续 checkpoint 中的累计值而不是从零开始。

不要用此 smoke 命令启动正式训练，也不要把 `/tmp` manifest、日志或 checkpoint 提交到 Git。

## 后续模块接入约定

后续实现应遵守以下边界：

1. `mode=uniform` 始终调用现有 legacy motion/start-frame 抽样，不重写其随机逻辑。
2. 新模式使用 `FixedLengthSegmentIndex` 的 Tensor 映射和元数据，不在训练热路径遍历 motion 文件。
3. 产生自适应概率时调用 `normalize_and_validate_probabilities(..., fallback_statistics=共享统计)`，使 fallback 与共享计数同步更新。
4. 所有环境继续共享一套 `SamplingStatistics`，不要创建每环境计数表。
5. 新状态加入现有 `sampling_state` 并提升状态版本，同时增加严格的兼容性检查和纯 Python 测试。
6. W&B 只增加稳定、低维度的汇总标量，不上传每 motion/segment 数组。
7. 质量、难度、学习缺口和多样性模块通过统一 research 配置组合，不复制 M0--M7 八套训练入口。
