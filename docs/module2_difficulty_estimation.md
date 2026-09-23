# 模块二：策略无关的片段固有难度估计

本文描述当前仓库中模块二的实际实现、离线拟合语义、训练期 load-only 接入和验证流程。模块二只根据最终转换后的 WBT/G1 机器人参考轨迹估计固有难度；它不衡量当前策略是否已经学会某个动作。

> `configs/difficulty/g1_segment_difficulty.yaml` 当前明确标记为 `provisional: true`。现有特征、权重、sole offset、接触阈值和分数映射必须先用 Train 统计及人工回放审查，再冻结为正式 Profile。不得根据 Validation 或 Test 调整它们。

## 研究边界：难度与质量、策略表现解耦

模块二的输入是 PHUMA 经重定向和格式转换后得到的最终 G1/WBT `.npz`。分数只来自轨迹本身的运动学量和由轨迹推断的接触状态；core 特征、Profile fit 和 transform API 不接受也不读取：

- policy tracking error、reward、success、termination、completion 或 PPO loss；
- checkpoint、训练迭代、当前策略输出或评测结果；
- 模块一的 `quality_score`、`quality_status`、pass/borderline/reject mask 或过滤清单。

离线 builder 可在分数和 bins 已经完全生成后，选择性加载模块一 NPZ 做映射检查和只读交叉统计；这一诊断入口不传入上述 core API。改变匹配的质量标签不能改变任何 segment/motion 难度值或 Profile，端到端测试对此有显式约束。

所有 manifest 中的 motion 和所有 Stage-0 segment 都参与难度特征提取，包括模块一可能判为 reject 的片段。模块二不删除、不替换 motion，不把不可用特征解释为“容易”，也不根据质量标签改变拟合样本。低质量片段可能得到高难度分，这是质量与难度解耦后的正常现象；以后只有在 M6/M7 中，冻结的质量门控和冻结的难度标签才会作为两个独立模块组合。

同样，持续而平滑的高速移动、快速旋转、单脚支撑或腾空可以是高难度但高质量的轨迹。模块二不检查关节越限、穿地、足滑是否构成数据缺陷；这些属于模块一。

当前训练端只加载和记录冻结的 `difficulty_score/bin`，不依据它们重采样。因此本阶段不是 M5 学习缺口采样，也不会改变 M0/M1 的 motion、起始帧或 segment 概率。

## 实现位置

- 暂定特征定义：`configs/difficulty/g1_segment_difficulty.yaml`
- 特征提取、Profile 拟合和冻结变换：`source/whole_body_tracking/whole_body_tracking/utils/difficulty.py`
- 训练 metadata schema 与严格一致性校验：`source/whole_body_tracking/whole_body_tracking/utils/difficulty_metadata.py`
- WBT schema、WXYZ 和 sole-frame 公共函数：`source/whole_body_tracking/whole_body_tracking/utils/quality.py`
- 训练期 load-only 接入：`source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py`
- W&B 标量日志与 checkpoint 接线：`source/whole_body_tracking/whole_body_tracking/utils/my_on_policy_runner.py`

## 输入 schema 与 Segment 边界

特征提取复用模块一验证后的 `MotionData`，核心输入为：

```text
fps                 scalar，有限且 > 0
joint_pos           (T, J)，rad
joint_vel           (T, J)，rad/s
body_pos_w          (T, B, 3)，world frame，m
body_quat_w         (T, B, 4)，WXYZ
body_lin_vel_w      (T, B, 3)，world frame，m/s
body_ang_vel_w      (T, B, 3)，当前模块不直接入分
joint_names         (J,)
body_names          (B,)
```

默认 `segment_length_seconds=1.0`。每条 motion 独立计算：

```text
segment_frames = max(1, round(fps * segment_length_seconds))
```

这里使用与 Stage 0 一致的 round-to-nearest-even 规则。Segment 是连续、无重叠、完整覆盖 motion 的右开区间 `[start_frame, end_frame_exclusive)`；最后不足一秒的尾段仍保留，实际时长记录为 `(end-start)/fps`。

## 必须先对完整 motion 求导

所有会跨帧的导数、四元数相对旋转、sole 速度、接触状态和接触切换都先在完整 motion 上计算，再按 segment 聚合。不能把每个 segment 单独求导，否则每个片段首帧会产生人为零值，跨边界的加速度和接触切换也会丢失。

当前实现的具体语义为：

- root 线速度直接读取配置的 `pelvis` 在 `body_lin_vel_w` 中的 world-frame 速度；root 线加速度在完整速度向量上用 `fps * (v[t] - v[t-1])` 计算。
- 关节速度直接读取 `joint_vel` 的绝对值；关节加速度在完整 `joint_vel` 上使用同一目标帧后向差分，再取绝对值。
- 手部速度直接读取配置 hand link 的 `body_lin_vel_w`；sole 速度从完整 sole world position 使用目标帧后向差分计算。
- root 角速度不直接使用 `body_ang_vel_w`，而是由连续 WXYZ root quaternion 的相对旋转计算；root 角加速度再对完整角速度向量做目标帧后向差分。
- 所有普通差分和四元数 transition `t-1 -> t` 都归到目标帧 `t`，避免中心差分把边界后的变化泄漏到前一个 segment；第 0 帧复制 `0 -> 1` 的首个差分。接触切换也归到切换后的目标帧。

这样，segment 只决定统计窗口，不改变底层运动信号。

## WXYZ、sole 参考点与接触去抖

### WXYZ 角速度

`body_quat_w` 按 WXYZ 解释并先单位化。相邻四元数使用最短路径相对旋转：当内积为负时翻转当前 quaternion，使 `q` 和 `-q` 表示同一姿态；随后计算 `conjugate(q[t-1]) * q[t]` 的 rotation vector，并乘 FPS 得到 `rad/s`。第 0 帧复制 `0 -> 1` transition；非有限或近零范数 quaternion 会成为 unavailable，而不会静默当作零运动。

### sole world position

左右脚使用：

```text
left_ankle_roll_link
right_ankle_roll_link
```

两者当前 provisional 本地 sole offset 均为：

```text
[0.04, 0.0, -0.037] m
```

计算为：

```text
sole_position_w = ankle_body_pos_w + R_WXYZ(ankle_body_quat_w) * sole_local_offset
```

不能把 ankle link origin 直接当作脚底，也不能把 WXYZ 当作 XYZW。

### hysteresis 与 debounce

接触只是由参考轨迹推断的 candidate contact，不是仿真接触力。相对 `ground_z_m=0.0`：

- 非接触态进入接触：sole 高度 `<= 0.06 m` 且 `abs(v_z) <= 0.20 m/s`；
- 接触态保持接触：sole 高度 `<= 0.08 m` 且 `abs(v_z) <= 0.30 m/s`；超过任一 release 阈值才离开；
- NaN/Inf 帧强制为非接触；
- `minimum_frames=max(1, round(0.06 * fps))`；候选新状态必须连续保持至少该帧数才确认，确认后 transition 归到候选 run 的首帧。未达到持续帧数的交替 jitter 保持此前稳定状态，不依赖原地 run 遍历顺序。

左右脚分别去抖。`contact_switch_count` 是去抖后左右脚布尔 transition 数之和；同一帧两只脚都切换时计 2。`foot_swing_speed_p95` 只统计对应脚的非接触帧；segment 没有 swing 帧时记为可用的 `0.0`。接触脚水平滑动不进入模块二分数。

## 实际 28 个特征和入分权重

当前所有特征的 `direction=+1`，即更大的标准化值增加难度。`weight=0` 表示只输出诊断，不进入 `difficulty_raw`；它仍可能是 required，并不等于该字段可以缺失。

| Feature | 聚合与单位 | Required | Weight |
| --- | --- | :---: | ---: |
| `root_linear_speed_mean` | root world 线速度模的帧均值，m/s | 是 | 0.00 |
| `root_linear_speed_p95` | root world 线速度模的帧 P95，m/s | 是 | 0.75 |
| `root_linear_acceleration_mean` | 完整 motion root 线加速度模的帧均值，m/s² | 是 | 0.00 |
| `root_linear_acceleration_p95` | 完整 motion root 线加速度模的帧 P95，m/s² | 是 | 0.75 |
| `root_angular_speed_mean` | WXYZ 相对旋转角速度模的帧均值，rad/s | 是 | 0.00 |
| `root_angular_speed_p95` | WXYZ 相对旋转角速度模的帧 P95，rad/s | 是 | 0.75 |
| `root_angular_acceleration_mean` | 完整 motion 角速度向量导数模的帧均值，rad/s² | 是 | 0.00 |
| `root_angular_acceleration_p95` | 完整 motion 角速度向量导数模的帧 P95，rad/s² | 是 | 0.75 |
| `joint_speed_mean` | 帧和关节维度上 `abs(joint_vel)` 的均值，rad/s | 是 | 0.00 |
| `joint_speed_p95` | 帧和关节维度上 `abs(joint_vel)` 的 P95，rad/s | 是 | 0.75 |
| `joint_acceleration_mean` | 完整 motion 关节速度导数绝对值的均值，rad/s² | 是 | 0.00 |
| `joint_acceleration_p95` | 完整 motion 关节速度导数绝对值的 P95，rad/s² | 是 | 0.75 |
| `joint_range_mean` | segment 内逐关节 `max(q)-min(q)` 的关节均值，rad | 是 | 0.50 |
| `joint_range_p90` | segment 内逐关节 range 的关节 P90，rad | 是 | 0.50 |
| `body_height_mean` | `pelvis` world z 的帧均值，m | 是 | 0.00 |
| `body_height_range` | segment 内 `pelvis` world z 的 max-min，m | 是 | 0.50 |
| `body_height_std` | segment 内 `pelvis` world z 的标准差，m | 是 | 0.50 |
| `hand_speed_p95` | 左右 hand link world 速度模的 P95，m/s | 否 | 0.50 |
| `foot_swing_speed_p95` | 去抖非接触帧 sole world 速度模的 P95，m/s | 否 | 0.50 |
| `end_effector_speed_mean` | 双手和双 sole world 速度模的均值，m/s | 否 | 0.00 |
| `end_effector_speed_p95` | 双手和双 sole world 速度模的 P95，m/s | 否 | 0.00 |
| `double_support_ratio` | 双脚均为 candidate contact 的帧比例 | 否 | 0.00 |
| `single_support_ratio` | 恰有一只脚 candidate contact 的帧比例 | 否 | 0.65 |
| `flight_ratio` | 双脚均非 candidate contact 的帧比例 | 否 | 0.65 |
| `left_contact_ratio` | 左脚 candidate contact 帧比例 | 否 | 0.00 |
| `right_contact_ratio` | 右脚 candidate contact 帧比例 | 否 | 0.00 |
| `contact_switch_count` | segment 内左右脚去抖 transition 总数，count | 否 | 0.00 |
| `contact_switch_rate_per_second` | switch count / segment 实际时长，1/s | 否 | 0.70 |

若所有带分权重的特征均完整且非 near-constant，配置权重和为 `9.50`。实际 Profile 的固定分母可能更小，因为 Train 上 near-constant 或覆盖不完整的 optional 特征会被全局禁用。

## Required、optional 与 missingness

前 17 个特征是 required，后 11 个是固定 optional profile。规则是：

1. `fit_transform` 时，每个 Train segment 的每个 required 特征都必须可用，否则拟合失败。
2. 每个 segment 的 optional coverage 必须至少为 `0.75`；覆盖率按固定 11 个 optional 特征计算，因此当前实际至少需要 9/11 项可用，并且不随未来新增诊断字段漂移。
3. optional 特征即使总体大部分可用，只要在任一 Train segment 不可用，其有效权重就在整个 Profile 中置 0。这样所有 segment 使用相同特征集和相同分母。
4. `transform` 时，所有冻结 active 特征必须可用，required 必须可用，optional coverage 也必须达到冻结阈值，否则明确报错。
5. unavailable entry 在 metadata 中以 `NaN + feature_available_mask=false` 表示；只有 available entry 要求原值和 z-score 有限。

每个统计量定义所需的输入必须全部 finite；不会丢掉 NaN/Inf 后用剩余帧继续聚合。contact/debounce 是完整 motion 状态机，因此任一 sole position 非有限时，该 motion 的全部 foot/contact 特征均置为 unavailable，绝不把坏帧解释成 flight。当前 hands 或 feet body 名称缺失时，对应 optional 特征不可用；配置的 root body `pelvis` 缺失则直接报错。缺失从不按 0 分处理，也不会按每个 segment 临时重归一化权重。

## Train-only robust 标准化、zero-MAD fallback 与固定 raw 分数

对 Train 中特征 `i` 的所有 available 值，拟合：

```text
center_i = median(x_i)
mad_scale_i = 1.4826 * median(abs(x_i - center_i))
fallback_scale_i = (P95(x_i) - P05(x_i)) / 3.289707253902945

if mad_scale_i >= 1e-5:
    fitted_scale_i = mad_scale_i
else:
    fitted_scale_i = fallback_scale_i

scale_i = max(fitted_scale_i, 1e-6)
z_(s,i) = clip((x_(s,i) - center_i) / scale_i, -5, 5)
```

`1.4826*MAD` 是主尺度。配置键为 `zero_mad_fallback_quantiles=[0.05,0.95]` 和 `zero_mad_fallback_scale_divisor=3.289707253902945`。接触比例等零膨胀特征可能有大量相同中位数，使 MAD 为 0，但尾部仍存在有意义变化；因此只有 MAD 低于 `near_constant_scale_threshold=1e-5` 时，才改用线性 P05/P95 的 fallback。该除数对应正态分布 P05--P95 宽度，使 fallback 与标准差量级一致。

若 fallback 恢复到 `>=1e-5`，特征继续入分并在 Profile warning 中记录使用了 zero-MAD fallback。只有选中的 `fitted_scale_i` 仍 `<1e-5`，即 MAD 与 P05--P95 fallback 都近零时，才记入 `near_constant_features` 并将整个 Profile 的有效权重置 0。零覆盖特征同样禁用并产生 warning。用于 z-score 的最终 scale 才应用 `1e-6` epsilon；epsilon 不会把真正 near-constant 的特征重新启用。

设冻结后有效权重为 `w_i*`、方向为 `d_i`，active 集合 `A={i | w_i*>0}`。片段 raw 分数是：

```text
W = sum_(i in A) w_i*                         # Profile 拟合时冻结
contribution_(s,i) = z_(s,i) * d_i * w_i* / W
difficulty_raw_s = sum_(i in A) contribution_(s,i)
```

`W` 不会随 segment 可用特征变化；transform 数据缺少任何 active 特征时失败，而不是缩小分母。这一固定分母保证同一 Profile 下 raw 分数可比较。

## 经验 CDF、10 bins 与 ties

`difficulty_score` 不是 sigmoid，也不是固定物理单位。它是 Train `difficulty_raw` 的冻结经验百分位：

- 先按 raw 排序；重复值共享 mid-rank 百分位；
- 多样本时 mid-rank 除以 `N-1`；只有一个样本时该 knot 为 `0.5`；
- transform 用冻结 raw knots 和 percentile knots 线性插值；低于/高于 Train 范围分别截为 `0/1`。

因此 score 越大表示“相对该 Train Profile 越难”，不同 Profile 的 score 不能在没有共同冻结标识时直接比较。

默认在 Train raw 上用 linear quantile 得到 0.1、0.2、…、0.9 九个边界，形成 `difficulty_bin=0..9`。变换使用：

```text
difficulty_bin = searchsorted(bin_edges, difficulty_raw, side="right")
```

所以 raw 恰好等于边界时归入较高 bin。若重复 raw 令多个 quantile edge 相等，Profile 保留这些 ties、给出 warning，并允许某些 bin 为空；实现不会随机拆分同分样本来强行填满十个箱。

## Motion 级聚合

Motion 难度从该 motion 的 segment `difficulty_score` 聚合，而不是从最大值聚合：

```text
mean_m  = duration-weighted mean(segment difficulty_score)
p90_m   = P90(segment difficulty_score)
D_m     = (0.5 * mean_m + 0.5 * p90_m) / (0.5 + 0.5)
bin_m   = min(floor(D_m * 10), 9)
```

duration-weighted mean 使不足一秒的尾段按实际时长贡献；P90 保留高难局部，避免单个极短最大值支配整条 motion。当前 runtime 只加载 segment score/bin，motion 聚合是离线报告和后续模块的公共工具，尚未改变 motion 采样。

## `fit_transform` 与 `transform` 的冻结语义

### `fit_transform`

只对 Train manifest 使用。流程是完整提取特征、拟合 median/MAD、冻结 active 特征及固定分母、拟合经验 CDF 和十等分 raw 边界，写出 Profile，然后用同一 Profile 变换该 manifest 并生成 metadata。

Profile 绑定：

- Train manifest SHA256；
- Train ordered pool fingerprint；
- segment schema 与长度；
- 28 个特征的顺序、单位、required/optional、原始和有效权重；
- config SHA256、Git commit、warnings；
- scaling、CDF、bin edges 和 motion 聚合权重。

100-motion prefix 只能拟合 pilot Profile；不得把它当作正式 random6000 Profile。正式 random6000 必须在完整 Train random6000 上重新 `fit_transform`，冻结后再复用。

Builder 可以接收 `--quality-metadata`，但实现顺序刻意保证 Profile 拟合和 transform 已全部完成后才加载质量 NPZ。该选项只检查 segment 映射、生成 quality×difficulty 交叉计数和增加人工 review bucket；它不会删除行、改变特征、分数、CDF 或 bins。`--dataset-metadata` 同样只提供 split/category/source 诊断；已知的非 Train split 会阻止 `fit_transform`，但类别和来源不参与评分。

### `transform`

加载已有 `difficulty_profile.json`，只提取目标 manifest 特征并应用冻结参数；不重新计算 median/MAD、active 权重、CDF 或 bin edges。它可用于在同一 Train 定义下生成其他数据池的标签，但不能依据 Validation/Test 的分布或表现回头修改 Profile。

`transform` 输出的 metadata 绑定目标 manifest/pool，同时保留训练 Profile 的 SHA256。Profile 不匹配、active 特征缺失或覆盖不足时应失败，不得自动 refit。

## 离线构建命令

离线入口只支持 CPU；`--seed` 只控制 review 行的确定性抽样，不进入分数。先确认 CLI：

```bash
cd /home/l/whole_body_tracking_new
/home/l/miniconda3/envs/hybrid_robot/bin/python \
  scripts/build_segment_difficulty_metadata.py --help
```

### random100 Train-prefix pilot

下面命令取 random6000 manifest 的稳定前 100 条，拟合一个仅用于检查的 pilot Profile，并把与 metadata 精确绑定的 prefix 写到输出目录：

```bash
cd /home/l/whole_body_tracking_new
conda activate hybrid_robot

/home/l/miniconda3/envs/hybrid_robot/bin/python \
  scripts/build_segment_difficulty_metadata.py \
  --manifest PHUMA_wbt_motions/manifests/experiments/random_seed42/random6000_seed42.txt \
  --output-dir outputs/module2_difficulty_pilot_random100_seed42_v1 \
  --difficulty-config configs/difficulty/g1_segment_difficulty.yaml \
  --segment-length-seconds 1.0 \
  --mode fit_transform \
  --max-motions 100 \
  --workers 8 \
  --device cpu \
  --seed 42 \
  --strict \
  --overwrite
```

该 Profile 绑定：

```text
outputs/module2_difficulty_pilot_random100_seed42_v1/normalized_manifest.txt
```

如果对这个 prefix 做 runtime load-only smoke，训练也必须传上述 `normalized_manifest.txt`，不能传完整 random6000。该 pilot 只用于特征分布、贡献、接触和回放检查，不能给正式 random6000 打标签。

### 完整 random6000 Train `fit_transform`

审查并修改 provisional 配置后，用完整 Train random6000 重新拟合正式候选 Profile：

```bash
cd /home/l/whole_body_tracking_new
conda activate hybrid_robot

/home/l/miniconda3/envs/hybrid_robot/bin/python \
  scripts/build_segment_difficulty_metadata.py \
  --manifest PHUMA_wbt_motions/manifests/experiments/random_seed42/random6000_seed42.txt \
  --output-dir outputs/module2_difficulty_random6000_seed42_v1 \
  --difficulty-config configs/difficulty/g1_segment_difficulty.yaml \
  --segment-length-seconds 1.0 \
  --mode fit_transform \
  --quality-metadata outputs/module1_quality_random6000_seed42_original_v1/segment_quality_metadata.npz \
  --workers 8 \
  --device cpu \
  --seed 42 \
  --strict \
  --overwrite
```

不带 `--max-motions` 时，metadata/Profile 的 `manifest_sha256` 绑定原始 6000 行 manifest；`normalized_manifest.txt` 仍作为可读副本输出，但正式 runtime 应继续传原始 random6000 manifest。

### 用冻结 random6000 Profile 做 `transform`

目标 manifest 必须使用同一特征配置、算法 schema 和 1 秒 segment；该命令不会 refit：

```bash
cd /home/l/whole_body_tracking_new

TARGET_DIFFICULTY_MANIFEST=/absolute/path/to/frozen_target_manifest.txt
TARGET_DIFFICULTY_OUTPUT=outputs/module2_difficulty_target_random6000_profile_v1

/home/l/miniconda3/envs/hybrid_robot/bin/python \
  scripts/build_segment_difficulty_metadata.py \
  --manifest "$TARGET_DIFFICULTY_MANIFEST" \
  --output-dir "$TARGET_DIFFICULTY_OUTPUT" \
  --difficulty-config configs/difficulty/g1_segment_difficulty.yaml \
  --segment-length-seconds 1.0 \
  --mode transform \
  --profile outputs/module2_difficulty_random6000_seed42_v1/difficulty_profile.json \
  --workers 8 \
  --device cpu \
  --seed 42 \
  --strict \
  --overwrite
```

`fit_transform` 禁止同时传 `--profile`；`transform` 必须传 `--profile`。Transform 会核对 config SHA256、algorithm schema 和 segment length，并把冻结 Profile 原样复制进目标输出目录。

### 本轮实际构建结果

上述两个 `fit_transform` 已在当前工作树完成：

| Pool | Motions | Segments | Bin counts (`0..9`) |
| --- | ---: | ---: | --- |
| random100 prefix | 100 | 357 | `[36, 36, 35, 36, 36, 35, 36, 35, 36, 36]` |
| full random6000 | 6000 | 21575 | `[2158, 2157, 2158, 2157, 2157, 2158, 2157, 2158, 2157, 2158]` |

两次构建的 28 项特征 coverage 均为 `1.0`，`near_constant_features=[]`。以下 8 项零膨胀接触/支撑特征使用了 zero-MAD P05--P95 fallback，并因此保留有效尺度：

```text
foot_swing_speed_p95
double_support_ratio
single_support_ratio
flight_ratio
left_contact_ratio
right_contact_ratio
contact_switch_count
contact_switch_rate_per_second
```

random100 的 `difficulty_raw` 为 min/mean/P50/P90/P95/max = `-0.704920/0.436258/0.174386/1.535353/2.104580/4.106110`；full random6000 对应为 `-0.794562/0.413433/0.253464/1.551335/1.964788/4.186335`。两者的 Train empirical-CDF score 均为 P50=`0.5`、P90=`0.9`、P95=`0.95`。

random100 和 full random6000 的可选 quality metadata 映射均校验通过，交叉统计分别为 `segment_count_before=segment_count_after=357` 和 `21575`，说明质量标签没有删除任何 segment。以上是 provisional 配置的可复现构建事实，不表示权重和接触阈值已经论文冻结。

## 输出 schema、哈希与冻结规则

每次 builder 产生：

| 文件 | 用途 |
| --- | --- |
| `segment_difficulty_metadata.csv` | 逐 segment 的 ID、provenance、raw/score/bin、28 项原值/z/available 和前三项正贡献，供审查，不在训练热路径加载 |
| `segment_difficulty_metadata.npz` | runtime 严格加载的紧凑 segment metadata |
| `motion_difficulty_metadata.csv` | 逐 motion 的 duration-weighted mean、P90、0.5/0.5 分数和 bin |
| `motion_difficulty_metadata.npz` | 紧凑 motion 聚合结果，当前训练端不加载 |
| `difficulty_profile.json` | Train-fitted scaling、effective weights、CDF、bin edges、hash 和 warning；transform 时复制冻结 Profile |
| `difficulty_summary.json` | 全局/类别/来源分布、bin、相关性、near-constant、warning 和可选 quality 交叉统计 |
| `difficulty_feature_statistics.csv` | 逐特征单位、覆盖率、median、最终 fitted scale、配置/有效权重、分位统计及与 score 的 Spearman 相关；fallback 选择原因见 Profile warnings |
| `difficulty_review_segments.csv` | lowest/highest、逐 bin、边界、中位附近以及可选 quality 交叉回放样本 |
| `difficulty_config_resolved.json` | 应用 segment 长度覆盖后的 canonical 配置 |
| `normalized_manifest.txt` | 原始顺序副本；`--max-motions` 时是 metadata 实际绑定的 prefix manifest，不是难度过滤清单 |

训练实际加载的是紧凑、无原始轨迹的 `segment_difficulty_metadata.npz`，schema 为 `wbt.segment_difficulty.v1`。required arrays 为：

```text
schema_version
algorithm_schema_version
segment_schema_version
segment_length_seconds
manifest_sha256
manifest_motion_count
profile_sha256
difficulty_config_sha256
pool_fingerprint
num_bins

motion_keys
motion_lengths
motion_fps
motion_segment_offsets

global_segment_id
motion_id
local_segment_id
start_frame
end_frame_exclusive
duration_seconds
difficulty_raw
difficulty_score
difficulty_bin

feature_names
feature_values
feature_z
feature_available_mask
available_feature_count
optional_feature_coverage
near_constant_features
```

数组按 manifest motion 顺序和每条 motion 的 local segment 顺序排列；`global_segment_id` 必须是连续的 `0..N-1`。`motion_segment_offsets`、起止帧和 duration 必须能由 Stage 0 的 FPS/segment 长度精确重建。`difficulty_score` 必须有限且在 `[0,1]`，bin 必须在 `[0,num_bins-1]`。

`manifest_sha256` 绑定实际 target manifest，`pool_fingerprint` 绑定有序 motion pool 的相对路径、文件大小、帧数和 FPS，`difficulty_config_sha256` 绑定 provisional/frozen 配置，`profile_sha256` 绑定 Train-fitted Profile；loader 还计算整个 metadata 文件的 `metadata_sha256`。正式实验必须复用同一冻结 Profile、config 和 NPZ，而不是临时重新拟合或重新打包。

## Runtime 是 load-only，不改变采样

训练配置位于 `env.commands.motion.research.difficulty_calibration`：

```yaml
difficulty_calibration:
  enabled: false
  metadata_path: ""
  strict_metadata_match: true
  expected_num_bins: 10
```

启用时必须同时启用 Stage-0 segment，并提供本地 `.txt` manifest 和 metadata path。初始化会验证：

- metadata schema、所有数组 shape/range/NaN 约定；
- manifest 文件 SHA256、motion key 数量、顺序、帧数和 FPS；
- segment schema、长度、offset、global/motion/local ID、起止帧及实际时长；
- ordered pool fingerprint 和 `expected_num_bins`。

默认 `strict_metadata_match=true`，任一数据映射不一致都会在环境初始化阶段失败。`false` 只供诊断，运行时会 warning、`metadata_match_ok=0`，不得作为正式模块二结果；bin 数不一致无论 strict 设置如何都会失败。

验证通过后，完整 segment `difficulty_score` 和 `difficulty_bin` 被一次性复制为 device tensor，并通过 current/assigned segment 属性只读访问。它们不参与 `_resample_command`、不调用随机数，也不改变 uniform motion/start-frame 选择。当前研究配置仍只开放 M0/M1 方法名和 uniform sampling；打开该开关只代表“加载与观测难度标签”，不代表 M5 已实现。

## random100 disabled/enabled Isaac smoke 与 TRACE 比较

先完成 random100 离线 builder，确保以下两个文件存在：

```text
outputs/module2_difficulty_pilot_random100_seed42_v1/normalized_manifest.txt
outputs/module2_difficulty_pilot_random100_seed42_v1/segment_difficulty_metadata.npz
```

两次 fresh smoke 使用同一 prefix manifest、seed、环境数、网络、训练预算和 uniform sampler，只切换 difficulty load-only 开关。两边都记录已经采样完成后的前 2048 个 assignment；trace recorder 不调用 RNG。

这两条命令仍使用 `method_name=M0`，因为当前运行时只开放 M0/M1 方法名。它们是模块二接线 smoke，不是正式 M0：正式 M0 必须保持 difficulty 关闭。

### Disabled：不加载 difficulty metadata

```bash
cd /home/l/whole_body_tracking_new
conda activate hybrid_robot

env -u PYTHONPATH -u LD_LIBRARY_PATH \
  WBT_DISABLE_ONNX_ON_SAVE=1 \
python scripts/rsl_rl/train.py \
  --disable_fabric \
  --task Tracking-Flat-G1-v0 \
  --motion_file /home/l/whole_body_tracking_new/outputs/module2_difficulty_pilot_random100_seed42_v1/normalized_manifest.txt \
  --headless \
  --logger wandb \
  --log_project_name whole_body_tracking_module2_smoke \
  --run_name module2_random100_loadonly_disabled_seed42_env16_v1 \
  --wandb_run_name module2_random100_loadonly_disabled_seed42_env16_v1 \
  --wandb_run_id module2-random100-loadonly-disabled-seed42-env16-v1 \
  --wandb_resume never \
  --num_envs 16 \
  --seed 42 \
  --max_iterations 10 \
  env.commands.motion.research.method_name=M0 \
  env.commands.motion.research.segment.enabled=true \
  env.commands.motion.research.segment.length_seconds=1.0 \
  env.commands.motion.research.quality_gate.enabled=false \
  env.commands.motion.research.difficulty_calibration.enabled=false \
  env.commands.motion.research.motion_sampling.mode=uniform \
  env.commands.motion.research.segment_sampling.mode=uniform \
  env.commands.motion.research.diversity_constraint.enabled=false \
  env.commands.motion.research.sampling_statistics.enabled=true \
  env.commands.motion.research.sampling_statistics.log_interval=1 \
  env.commands.motion.research.assignment_trace.enabled=true \
  env.commands.motion.research.assignment_trace.output_path=/tmp/module2_random100_difficulty_disabled_trace.csv \
  env.commands.motion.research.assignment_trace.max_entries=2048
```

### Enabled：只加载同一 pool 的冻结标签

```bash
cd /home/l/whole_body_tracking_new
conda activate hybrid_robot

env -u PYTHONPATH -u LD_LIBRARY_PATH \
  WBT_DISABLE_ONNX_ON_SAVE=1 \
python scripts/rsl_rl/train.py \
  --disable_fabric \
  --task Tracking-Flat-G1-v0 \
  --motion_file /home/l/whole_body_tracking_new/outputs/module2_difficulty_pilot_random100_seed42_v1/normalized_manifest.txt \
  --headless \
  --logger wandb \
  --log_project_name whole_body_tracking_module2_smoke \
  --run_name module2_random100_loadonly_enabled_seed42_env16_v1 \
  --wandb_run_name module2_random100_loadonly_enabled_seed42_env16_v1 \
  --wandb_run_id module2-random100-loadonly-enabled-seed42-env16-v1 \
  --wandb_resume never \
  --num_envs 16 \
  --seed 42 \
  --max_iterations 10 \
  env.commands.motion.research.method_name=M0 \
  env.commands.motion.research.segment.enabled=true \
  env.commands.motion.research.segment.length_seconds=1.0 \
  env.commands.motion.research.quality_gate.enabled=false \
  env.commands.motion.research.difficulty_calibration.enabled=true \
  env.commands.motion.research.difficulty_calibration.metadata_path=/home/l/whole_body_tracking_new/outputs/module2_difficulty_pilot_random100_seed42_v1/segment_difficulty_metadata.npz \
  env.commands.motion.research.difficulty_calibration.strict_metadata_match=true \
  env.commands.motion.research.difficulty_calibration.expected_num_bins=10 \
  env.commands.motion.research.motion_sampling.mode=uniform \
  env.commands.motion.research.segment_sampling.mode=uniform \
  env.commands.motion.research.diversity_constraint.enabled=false \
  env.commands.motion.research.sampling_statistics.enabled=true \
  env.commands.motion.research.sampling_statistics.log_interval=1 \
  env.commands.motion.research.assignment_trace.enabled=true \
  env.commands.motion.research.assignment_trace.output_path=/tmp/module2_random100_difficulty_enabled_trace.csv \
  env.commands.motion.research.assignment_trace.max_entries=2048
```

比较实际 assignment identity：

```bash
cd /home/l/whole_body_tracking_new
python scripts/compare_assignment_traces.py \
  /tmp/module2_random100_difficulty_disabled_trace.csv \
  /tmp/module2_random100_difficulty_enabled_trace.csv
```

预期输出：

```text
TRACE_MATCH rows=<实际记录行数> columns=assignment_index,env_id,motion_id,start_frame,local_segment_id,global_segment_id
```

`TRACE_MATCH` 证明 load-only 接线没有改变已经存在的抽样序列。Enabled run 还应看到 metadata 初始化成功、`difficulty/metadata_match_ok=1` 和 `difficulty/sampling_still_uniform=1`；两边 reward/loss/reset 均不得出现 NaN/Inf。

本轮开发环境的 NVIDIA driver 不可用，因此上述两次 10-iteration Isaac smoke 尚未实际执行，不能将其写成已通过。本轮已经完成的是纯 CPU 的静态加载前后 2048 assignments 对照，结果为 `TRACE_MATCH`；它证明静态 difficulty load 没有引入 sampler 分支或额外 RNG，但不能替代 Isaac 环境初始化、物理 rollout、W&B 和 checkpoint smoke。

完整 random6000 runtime smoke 使用同一对 disabled/enabled 命令，只需把两边 `--motion_file` 都换成原始：

```text
/home/l/whole_body_tracking_new/PHUMA_wbt_motions/manifests/experiments/random_seed42/random6000_seed42.txt
```

并把 enabled 的 metadata path 换成：

```text
/home/l/whole_body_tracking_new/outputs/module2_difficulty_random6000_seed42_v1/segment_difficulty_metadata.npz
```

同时使用新的 run name、W&B run ID 和 trace 路径，避免覆盖 random100 smoke。

## W&B 配置与指标

启动参数先记录：

```text
difficulty_calibration_enabled
difficulty_calibration_metadata_path
difficulty_calibration_strict_metadata_match
difficulty_calibration_expected_num_bins
```

metadata 校验完成后，W&B config 追加：

```text
difficulty_metadata_file
difficulty_metadata_sha256
difficulty_profile_sha256
difficulty_config_sha256
difficulty_manifest_sha256
difficulty_schema_version
difficulty_num_bins
difficulty_metadata_match_ok
difficulty_sampling_still_uniform=true
```

`sampling_statistics.log_interval` 同时作为 `difficulty/*` 的低频日志间隔；即使 sampling statistics 本身关闭，只要 difficulty 开启仍记录静态难度摘要：

```text
difficulty/enabled
difficulty/metadata_match_ok
difficulty/sampling_still_uniform
difficulty/num_segments
difficulty/num_motions
difficulty/num_bins
difficulty/score_mean
difficulty/score_std
difficulty/score_p10
difficulty/score_p50
difficulty/score_p90
difficulty/raw_mean
difficulty/raw_std
difficulty/optional_feature_coverage_mean
difficulty/available_feature_count_mean
difficulty/near_constant_feature_count
difficulty/bin_0_count ... difficulty/bin_9_count
difficulty/bin_0_ratio ... difficulty/bin_9_ratio
```

这些是冻结 metadata 的静态摘要，不是训练表现，也不能用来反向调策略误差、reward 或成功率。

## Checkpoint 与 resume

难度 identity 随现有 `checkpoint["infos"]["sampling_state"]` 保存，不创建 sidecar，也没有在线更新的难度统计。identity 包含：

```text
schema_version
algorithm_schema_version
segment_schema_version
segment_length_seconds
metadata_sha256
profile_sha256
difficulty_config_sha256
manifest_sha256
pool_fingerprint
manifest_motion_count
num_bins
```

`metadata_path` 也会记录，但 resume 比较时刻意忽略路径本身。因此冻结 NPZ 可以整体搬迁；文件 SHA256、Profile、config、manifest、pool、segment 语义和 bin 数必须完全一致。以下情况都会明确拒绝恢复：

- checkpoint 开启 difficulty、当前运行关闭，或反之；
- difficulty-enabled checkpoint 缺少 identity；
- 除 metadata path 外任一 identity 字段变化；
- 研究配置语义变化。

load-only 难度不保存 policy-independent 特征的在线副本，也不保存新 RNG 或采样概率。

## 纯 Python 测试

以下命令不启动 Isaac Sim，一次运行 core、metadata、builder 端到端和 runtime 静态接线测试：

```bash
cd /home/l/whole_body_tracking_new
PYTHONDONTWRITEBYTECODE=1 \
/home/l/miniconda3/envs/hybrid_robot/bin/python -m unittest discover \
  -s tests \
  -p 'test_*difficulty*.py' \
  -v
```

当前覆盖：

- `test_difficulty_estimation.py`：静止/慢/快排序、单支撑/腾空、孤立与连续交替 jitter 去抖、边界 transition 只归目标 segment、`q/-q` 恒角速、25/50/100 FPS、1 帧短尾、robust scale/near-constant/clip/direction/composite、CDF/ties/bin、损坏 Profile 拒绝、JSON roundtrip 和 duration-weighted motion 聚合；
- `test_difficulty_metadata.py`：NPZ roundtrip、identity/metrics、Stage-0 多 FPS 和短尾布局、hash/shape/bounds/bin/missingness corruption，以及 strict/non-strict mapping；
- `test_build_segment_difficulty_metadata.py`：subprocess `fit_transform` 的全部 10 个输出和 10 bins、相同 seed 的语义确定性、非 Train split 在拟合/写 Profile 前拒绝、不同匹配质量标签不改变 difficulty、`transform` 原样复用冻结 Profile并拒绝 segment-schema 错配；
- `test_difficulty_runtime_integration.py`：配置约束、disabled 不加载、enabled 只读 tensor、不进入 sampler/RNG、non-strict 仍强制精确布局、W&B 和 checkpoint identity 接线，以及 enabled resume 缺 identity 时拒绝。

本轮该定向命令实际结果为 `23 tests / OK`。Runtime integration 是纯 Python/AST 接线验证，不等同于前述尚待 NVIDIA driver 恢复后执行的 Isaac smoke。

除模块二定向测试外，还应运行完整纯 Python 回归，并确认 Stage 0/M0 的抽样 AST 或 assignment trace 没有变化：

```bash
cd /home/l/whole_body_tracking_new
PYTHONDONTWRITEBYTECODE=1 \
/home/l/miniconda3/envs/hybrid_robot/bin/python -m unittest discover -s tests -p 'test_*.py'
```

本轮完整纯 Python 回归实际结果为 `111 tests / OK`。

## 验收清单

- 代码搜索和 API 审查确认模块二没有 quality label、policy error、reward、success 或 checkpoint 表现输入；
- random100 对所有 motion 的导数、WXYZ 角速度、sole 位置、去抖 contact 和 switch attribution 做人工 spot-check；
- 站立/缓慢平移动作总体较低，快速转身、单脚支撑、腾空与落地类片段总体较高；这只是 Train 轨迹合理性审查，不使用策略表现设阈值；
- required 特征 100% 可用，逐 segment optional coverage 达标；zero-MAD 特征正确尝试 P05--P95 fallback，只有两种尺度都近零才禁用；incomplete optional 的全局禁用及 warning 符合预期；
- raw contribution 可按冻结 z、direction、effective weight 和固定分母逐行复算；
- score 可由冻结经验 CDF 复算，bin 可由 raw edges 和 `side="right"` 复算；ties/空 bin 有明确 warning 而非随机打散；
- segment metadata 的 ID、边界、duration、manifest/pool/profile/config hash 全部严格匹配；
- runtime `difficulty/metadata_match_ok=1`、`difficulty/sampling_still_uniform=1`，M0/M1 assignment trace 与关闭难度标签时一致；
- checkpoint fresh/resume、metadata relocation 和故意 mismatch 拒绝路径通过；
- random100 只作为 pilot；完整 random6000 Train Profile、配置、metadata NPZ、Git commit 和评测规则冻结后，才开始后续正式实验。

## 当前限制与 provisional 项

当前算法 schema 是 `wbt.intrinsic_difficulty.kinematic_contact.v2`；相对早期开发版，v2 明确冻结目标帧后向差分、持续确认 debounce 和 strict non-finite propagation，旧 Profile 会被拒绝。它应准确称为“运动学与接触代理的固有难度”，而不是完整动力学难度。尚未实现：

- body angular momentum proxy；
- COM 到支撑区域边缘的 support margin；
- 质心高度变化（当前是 `pelvis` 高度）；
- torque sensitivity、理论力矩、功率或动力学残差；
- 仿真接触力或真实接触标签；
- 学习得到的难度模型。

另外需要保留以下解释限制：

- sole offset、flat ground `z=0`、接触 hysteresis/debounce 阈值和全部权重仍为 provisional，必须通过 Train 回放冻结；
- 所有方向目前均为单调 `+1`，不能表达某些非单调关系；相关特征可能重复描述同一种高动态，线性加权没有因果含义；
- 经验 CDF 和 bins 是相对冻结 Train pool 的排名，不是跨 Profile 的绝对物理尺度；ties 可能产生空 bin；
- optional feature 只要 Train 覆盖不是 100% 就不会入分，这优先保证固定分母，但可能舍弃部分有效信息；
- contact 只来自几何高度和垂直速度，不能证明真实支撑、冲击或可执行性；
- Stage-0 `pool_fingerprint` 不哈希每个大型 NPZ 的完整字节；同路径、同大小、同帧数和同 FPS 的原地内容改写理论上无法被它识别。正式冻结后应把轨迹池设为只读，并额外保存数据集版本或内容清单；
- 完整 motion 至少需要两帧才能计算 quaternion transition 和接触速度；完整 motion 内的 1 帧尾 segment 可以正常复用全 motion 导数；
- runtime 当前只加载标签和记录摘要，motion/segment 难度校准采样、学习缺口及其在线统计属于后续模块，尚未实现。

配置仍为 `provisional: true` 时，任何生成物都只能用于离线统计、回放和短程 load-only pilot。只有在 Train-only 审查后把配置/Profile/metadata/Git commit 一起冻结，才能将其作为后续 M5/M6/M7 的正式固有难度基准。
