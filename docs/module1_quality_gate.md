# 模块一：片段质量审计与 M1 质量门控

本文描述当前仓库中模块一的实际实现、暂定阈值、训练语义和验证流程。模块一只处理最终转换后的 WBT/G1 机器人轨迹，不重新筛选原始 PHUMA 人体数据。

> 当前 `configs/quality/g1_segment_quality.yaml` 明确标记为 `provisional: true`。其中阈值只用于离线统计、人工回放和短程 pilot；完成这些检查并冻结配置前，不得将结果当作正式 M1，也不得用 Validation 或 Test 调参。

## 研究边界：质量不等于难度

模块一只判断最终参考机器人轨迹是否存在数值、转换、几何或明确物理限制问题，不使用 policy tracking error、reward、success/termination/completion、PPO loss 或 checkpoint 表现。持续且平滑的高速、高加速度、跳跃腾空、单腿支撑、快速旋转和极端姿态可能很难学，但不因此被判为低质量；只有非有限值、明确越限、字段不一致、孤立跳变、明显穿地或接触候选帧中的严重足滑等可解释缺陷才降低质量。

因此 `quality_score` 不是固有难度分，也不能回答当前策略能否学会该片段。后续难度或学习缺口模块必须使用独立字段和实验开关，不能反向修改冻结的质量标签。

## 实现位置

- 离线入口：`scripts/build_segment_quality_metadata.py`
- 暂定质量配置：`configs/quality/g1_segment_quality.yaml`
- 纯 NumPy 审计：`source/whole_body_tracking/whole_body_tracking/utils/quality.py`
- metadata schema 与一致性校验：`source/whole_body_tracking/whole_body_tracking/utils/quality_metadata.py`
- 合法起点索引：`source/whole_body_tracking/whole_body_tracking/utils/sampling.py`
- 训练门控、指标和 checkpoint：`source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py`
- runner 日志与恢复：`source/whole_body_tracking/whole_body_tracking/utils/my_on_policy_runner.py`
- 人工回放：`scripts/replay_npz.py`

## 离线审计流程

输入必须是按训练顺序排列的本地 `.txt` Train manifest。脚本拒绝文件名明显包含 Validation/Test 的 manifest；如果 split metadata 识别到非 Train motion，也会拒绝继续。`--strict` 还要求每条 motion 都能由 split metadata 明确证明属于 Train，unknown/空 split 会直接失败。不要对 Validation 或 Test 生成质量标签。

流程如下：

1. 按 `MotionLoader` 语义解析 manifest，保留 motion key 和顺序。
2. 预检每个 `.npz` 的 FPS、帧数和 `joint_names`；完整审计还要求 WBT 轨迹字段及 `body_names`。
3. 使用与训练相同的固定时长 Segment 索引，边界均为右开区间 `[start_frame, end_frame_exclusive)`。
4. 从仓库 G1 URDF 按 `.npz` 中的 `joint_names` 对齐关节位置和速度限制，不复制一套硬编码关节限位。
5. 在整条 motion 上先计算有限差分、加速度、jerk 和连续性信号，再按 segment 聚合。跨 segment 的跳变归到目标帧所在 segment，不会因切段而消失。
6. 计算每项 metric 的 severity、综合质量分和 `pass/borderline/reject` 状态。
7. 写出供人工审查的 CSV/JSON，以及训练实际加载的紧凑 NPZ。
8. 用 `quality_review_segments.csv` 选出的片段人工回放；检查汇总 warning，调整暂定配置后重新生成。阈值冻结后必须保留并复用最终的配置和 NPZ。

### 输入 `.npz` schema

审计和训练使用实际 WBT schema，其中 `T` 是帧数、`J` 是关节数、`B` 是 body 数：

```text
fps                 scalar 或单元素数组，有限且 > 0
joint_pos           (T, J)
joint_vel           (T, J)
body_pos_w          (T, B, 3)，world frame
body_quat_w         (T, B, 4)，WXYZ
body_lin_vel_w      (T, B, 3)，world frame
body_ang_vel_w      (T, B, 3)，world frame
joint_names         (J,)
body_names          (B,)
source_file         scalar provenance（不参与评分）
source_format       scalar provenance（不参与评分）
```

实际 PHUMA/G1 数据当前为 `J=29`、`B=30`。所有时序数组必须共享同一个 `T`；名称数量必须与对应维度一致。严重 schema/shape 错误会带文件路径终止预检，NaN/Inf 则作为 segment 级硬违规记录。

### 100-motion 离线 pilot

以下命令审计 random6000 的稳定前 100 条，并生成与 metadata 完全匹配的
`normalized_manifest.txt` 前缀副本。它只用于短程 prefix pilot，不是质量过滤后的清单：

```bash
cd /home/l/whole_body_tracking_new
conda activate hybrid_robot

/home/l/miniconda3/envs/hybrid_robot/bin/python \
  scripts/build_segment_quality_metadata.py \
  --manifest PHUMA_wbt_motions/manifests/experiments/random_seed42/random6000_seed42.txt \
  --output-dir outputs/module1_quality_pilot_random100_seed42_v1 \
  --quality-config configs/quality/g1_segment_quality.yaml \
  --segment-length-seconds 1.0 \
  --max-motions 100 \
  --workers 8 \
  --device cpu \
  --seed 42 \
  --strict \
  --overwrite
```

prefix pilot 如果使用 `--max-motions 100`，训练必须使用生成的
`outputs/module1_quality_pilot_random100_seed42_v1/normalized_manifest.txt`；
正式 random6000 实验应去掉 `--max-motions`，训练直接传原始
`PHUMA_wbt_motions/manifests/experiments/random_seed42/random6000_seed42.txt`。
质量审计只生成 metadata，不删除、不替换、不按质量过滤 manifest。
所有正式 M0--M7 使用同一个 6000 行原始 manifest；5998 只表示质量门控开启后的
runtime eligible motion 数，不是一份 5998 行清单。

正式原始 random6000 metadata 建议用独立输出目录重建：

```bash
cd /home/l/whole_body_tracking_new
conda activate hybrid_robot

/home/l/miniconda3/envs/hybrid_robot/bin/python \
  scripts/build_segment_quality_metadata.py \
  --manifest PHUMA_wbt_motions/manifests/experiments/random_seed42/random6000_seed42.txt \
  --output-dir outputs/module1_quality_random6000_seed42_original_v1 \
  --quality-config configs/quality/g1_segment_quality.yaml \
  --segment-length-seconds 1.0 \
  --workers 8 \
  --device cpu \
  --seed 42 \
  --strict \
  --overwrite
```

构建完成后重点看 `quality_summary.json` 和 `empty_eligible_motions.csv`：
原始 random6000 预期 `motion_count=6000`、`empty_eligible_motion_count=2`、
`quality_filter_removed_motion_count=0`、`training_compatible=true`。

## 质量公式与暂定阈值

对可用 metric (i)，原始值为 (x_i)，warning/reject 阈值为 (a_i,b_i)。severity 为：

```text
            0,                         x_i <= a_i
s_i(x_i) = (x_i - a_i) / (b_i - a_i), a_i < x_i < b_i
            1,                         x_i >= b_i
```

不可用或非有限的单项 metric 不进入加权分母。required metric 必须可用；optional 覆盖率按配置中的固定 `optional_metric_coverage_profile` 计算，而不是按当前实现的 metric 总数硬编码。综合质量分为：

```text
Q_(m,s) = clip(1 - sum_i(w_i * s_i) / sum_i(w_i), 0, 1)
```

求和只包含 `available=true` 的 metric。分数越高表示质量越好。

当前状态规则是：

- 任一 `hard_violation=true`：`reject`；
- 任一配置的 required metric 不可用、optional coverage 低于 `minimum_optional_metric_coverage=0.60`，或有效权重和为 0：`reject`；
- `Q < 0.55`：`reject`；
- severity 达到 1 的 metric 至少有 2 项：`reject`；
- 未 reject 且 `Q < 0.90`：`borderline`；
- 当前 `borderline_on_warning=true`，所以未 reject 但任一 metric severity 大于 0，也为 `borderline`；
- 其余为 `pass`。

下表是 `g1_segment_quality.yaml` 中的 provisional 值，不是论文最终阈值：

| Metric | 原始值/单位 | Warning | Reject | Weight | Reject 阈值是否硬违规 |
| --- | --- | ---: | ---: | ---: | --- |
| `nonfinite_values` | 非有限数占比 | 0 | 1e-12 | 2.00 | 是 |
| `quaternion_norm` | 四元数单位范数最大误差 | 0.005 | 0.10 | 1.50 | 是 |
| `joint_position_limits` | 超 URDF 位置限位最大弧度 | 0.02 | 0.20 | 1.50 | 是 |
| `joint_velocity_limits` | 超 URDF 速度限位最大 rad/s | 1.0 | 8.0 | 1.00 | 是 |
| `joint_velocity_consistency` | 差分速度与 `joint_vel` 的 P95 绝对误差；另查 max | 1.0 | 8.0 | 1.00 | 是 |
| `body_velocity_consistency` | 差分 body 速度与存储速度的 P95 向量误差；另查 max | 0.75 | 5.0 | 0.75 | 是 |
| `joint_acceleration_spike` | 局部孤立峰比例 | 1.0 | 6.0 | 0.50 | 否 |
| `joint_jerk_spike` | 局部孤立峰比例 | 1.0 | 6.0 | 0.50 | 否 |
| `root_linear_acceleration_spike` | 局部孤立峰比例 | 1.0 | 6.0 | 0.35 | 否 |
| `root_angular_acceleration_spike` | 局部孤立峰比例 | 1.0 | 6.0 | 0.35 | 否 |
| `root_position_continuity` | 局部孤立跳变比例 | 1.0 | 4.0 | 1.00 | 是 |
| `root_orientation_continuity` | 局部孤立跳变比例 | 1.0 | 3.0 | 1.00 | 是 |
| `body_position_continuity` | 局部孤立跳变比例 | 1.0 | 4.0 | 0.75 | 是 |
| `body_orientation_continuity` | 局部孤立跳变比例 | 1.0 | 3.0 | 0.75 | 是 |
| `joint_position_continuity` | 局部孤立跳变比例 | 1.0 | 3.0 | 1.00 | 是 |
| `ground_penetration` | sole 参考点最大穿地深度，m | 0.03 | 0.10 | 1.00 | 是 |
| `foot_sliding` | 推断接触帧水平 sole 速度 P95，m/s | 0.80 | 2.50 | 0.50 | 仅持续严重规则为硬违规 |

孤立峰指标使用“绝对下限与邻帧中位数乘数的较大者”作为基准，当前相对乘数均为 8。绝对下限分别为：关节加速度 `500 rad/s²`、关节 jerk `30000 rad/s³`、root 线加速度 `100 m/s²`、root 角加速度 `200 rad/s²`；root/body/joint 连续性的位置或角度下限依次为 `0.25 m/frame`、`0.60 rad/frame`、`0.30 m/frame`、`0.80 rad/frame`、`0.80 rad/frame`。

速度一致性同时使用持续误差 P95 通道和稀疏损坏 max 通道，最终 severity 取二者较大值。max warning/reject 暂定为：joint `4/20 rad/s`、body `3/15 m/s`。required metrics 暂定为 nonfinite、quaternion norm、两项 URDF joint limits、ground penetration 和 foot sliding；缺少它们不会被当作零 severity，而是保留分数解释并以 `insufficient_metrics` 拒绝。当前 optional coverage profile 固定为其余 11 个已实现指标，阈值为 0.60；未来新增诊断指标若未加入 profile，不会自动改变旧数据的 status。

足部配置使用 NPZ `body_pos_w/body_quat_w/body_names` 中的 `left_ankle_roll_link` 和 `right_ankle_roll_link`。这些 body pose 是 world frame；`ground.z_m=0.0` 也是 world z 坐标。审计不会把 link origin 直接当脚底，而是使用本地 WXYZ-frame sole local offset `[0.04, 0.0, -0.037] m` 得到配置的 sole 参考点；该 offset 来源说明写在 resolved config 的 `ground.sole_offset_note`，仍保持 provisional，需要通过回放继续人工核验。默认 `require_configured_foot_bodies=true`，左右脚 body 名称对不上会明确报错。当前仅用 sole 高度不超过 `0.06 m` 且垂直速度绝对值不超过 `0.20 m/s` 推断接触；这不是仿真接触力。若至少 10 个接触样本中，速度达到 `2.50 m/s` 的比例不低于 25%，则判为持续严重足滑并形成 hard violation；腾空高速移动不算足滑，单次短促落地不会仅凭一个高速样本触发持续严重规则。

### 指标数学语义

所有导数、速度一致性和连续性信号都先在完整 motion 上计算，再按右开 segment `[start_frame,end_frame_exclusive)` 聚合；因此跨 segment 边界的跳变会归到跳变后的目标 segment，不会被边界切掉。

| Metric | 输入字段 | 公式/聚合 | 阈值单位与 hard 条件 | unavailable 处理 |
| --- | --- | --- | --- | --- |
| `nonfinite_values` | 所有数值轨迹数组 | `NaN/Inf count / numeric value count` | fraction；达到 reject 即 hard | schema 字段缺失会审计失败 |
| `quaternion_norm` | `body_quat_w` | segment 内 `max(abs(norm(q)-1))` | absolute norm error；达到 reject 即 hard | 无有限四元数则 unavailable |
| `joint_position_limits` | `joint_pos`, `joint_names`, URDF | 对每关节算 `max(lower-q, q-upper, 0)`，取 segment max | rad excess；达到 reject 即 hard | URDF/关节限位缺失则 unavailable |
| `joint_velocity_limits` | `joint_vel`, `joint_names`, URDF | 对每关节算 `max(abs(qdot)-velocity_limit, 0)`，取 segment max | rad/s excess；达到 reject 即 hard | URDF/速度限位缺失则 unavailable |
| `joint_velocity_consistency` | `joint_pos`, `joint_vel`, `fps` | 完整 motion 差分 `dq/dt` 与 `joint_vel` 的绝对误差；segment P95 和 max 双通道，severity 取大者 | P95 rad/s，max 通道 `4/20 rad/s`；max/reject 可 hard | 无有限误差则 unavailable |
| `body_velocity_consistency` | `body_pos_w`, `body_lin_vel_w`, `fps` | 完整 motion 差分 `d body_pos_w/dt` 与存储速度的向量误差；segment P95 和 max 双通道 | P95 m/s，max 通道 `3/15 m/s`；max/reject 可 hard | 无有限误差则 unavailable |
| acceleration spike | `joint_vel` 或 root `body_lin_vel_w/body_ang_vel_w` | `score = current / max(absolute_floor, relative_multiplier * neighbor_median)`，取 segment max；平滑持续高动态不会因绝对值大而直接 reject | dimensionless；当前不 hard | 依赖 body 缺失则 unavailable |
| `joint_jerk_spike` | `joint_vel`, `fps` | 先对关节速度求加速度再求 jerk，同样使用局部孤立比例，取 segment max | dimensionless；当前不 hard | 无有限值则 unavailable |
| root/body continuity | `body_pos_w`, `body_quat_w` | frame-to-frame world 位置跳变或 quaternion geodesic 角度，再用局部孤立比例取 segment max；`q` 和 `-q` 等价 | dimensionless；达到 reject 即 hard | root body 缺失时 root 指标 unavailable |
| `joint_position_continuity` | `joint_pos`, URDF continuous 标记 | continuous joint 先 unwrap，再算 frame-to-frame 绝对跳变和局部孤立比例 | dimensionless；达到 reject 即 hard | 无有限跳变则 unavailable |
| `ground_penetration` | `body_pos_w`, `body_quat_w`, `body_names`, ground config | `sole_world = body_pos_w + R(body_quat_w) * sole_local_offset`；`max(max(ground_z - sole_z, 0))` | m depth；达到 reject 即 hard | 默认 foot body 缺失直接报错 |
| `foot_sliding` | 同 ground metric，外加 `fps` | 只在 `sole_height <= contact_height_threshold` 且 `abs(vertical_speed) <= threshold` 的接触候选帧统计水平 sole speed P95；persistent severe 才 hard | m/s；persistent severe: 接触样本数与严重比例同时达阈值 | 默认 foot body 缺失直接报错 |

这些指标刻意区分“持续高动态”和“孤立转换尖峰”：高动态但连续的动作不应仅因绝对速度/加速度大被拒；单帧尖峰和存储字段损坏通过 max 通道或孤立比例暴露。

## 已实现和未实现的审计指标

已实现的是上表 17 项，包括 schema/nonfinite、四元数、URDF 关节限制、存储速度一致性、局部尖峰、轨迹连续性、穿地和基于运动学推断的足滑。

当前没有实现：

- 显式自碰撞检测；
- 理论力矩；
- 功率；
- 动力学残差；
- 基于接触力的支撑状态合理性；
- 独立的“非接触时异常贴地”评分。

因此当前结果应称为“运动学与几何质量审计”。在没有相应数据或仿真计算前，不要在论文或日志中宣称已经实现上述扩展指标。

## 输出 schema、哈希与冻结规则

每次审计产生：

| 文件 | 用途 |
| --- | --- |
| `segment_quality_metadata.csv` | 可读的逐 segment 完整审计表，不在训练热路径加载 |
| `segment_quality_metadata.npz` | 训练加载的紧凑、无轨迹数据 metadata |
| `quality_summary.json` | 全局、逐 motion、类别、来源组、每 metric、空 motion 和 warning 汇总 |
| `quality_config_resolved.json` | 已应用 CLI segment 长度覆盖后的完整配置 |
| `quality_review_segments.csv` | 各阈值附近及高/低质量片段、Isaac 回放命令和轻量视频命令 |
| `normalized_manifest.txt` | 原始顺序的规范化 manifest 副本；`--max-motions` 时为前缀 pilot 清单，不是质量过滤清单 |
| `empty_eligible_motions.csv` | 没有合法 M1 起点的 motion 清单、状态计数和主要 reject 原因 |

CSV 的核心字段包括 manifest/motion/segment ID、motion key/path、类别和来源、`start_frame`、`end_frame_exclusive`、FPS、帧数、`quality_score`、`quality_status`、hard/coverage/reason/trigger 信息。每个实现 metric 还提供：

```text
<metric>_raw_value
<metric>_severity
<metric>_available
<metric>_hard_violation
```

并保留关节限位、速度一致性、加速度、jerk、左右脚最低高度、穿地、足滑和连续性的可读诊断列。

训练 NPZ 的 schema 为 `wbt.segment_quality.v1`，包含：

```text
schema_version
segment_schema_version
segment_length_seconds
manifest_sha256
manifest_motion_count
quality_config_sha256
pool_fingerprint
motion_keys
motion_lengths
motion_fps
motion_segment_offsets
global_segment_id
motion_id
local_segment_id
start_frame
end_frame_exclusive
quality_score
quality_status
pass_mask
borderline_mask
reject_mask
```

其中：

- `manifest_sha256` 绑定实际使用的 manifest：全量 random6000 绑定原始清单，prefix pilot 绑定 `normalized_manifest.txt`；
- `quality_config_sha256` 绑定 canonical resolved quality config；
- `pool_fingerprint` 绑定有序 motion pool、帧数、FPS 和文件身份信息；
- loader 另外计算整个 `segment_quality_metadata.npz` 的 `metadata_sha256`；
- `quality_summary.json` 记录 URDF SHA256、Git commit、审计错误和 provenance。

相同输入和配置保证分数、状态、有序数组及 review 选择可重复；生成时间戳和 NPZ ZIP 容器字节不属于这一逻辑确定性承诺。正式实验和 resume 应复用同一个冻结 NPZ 文件，而不是临时重新打包一份内容看似相同的 NPZ。

## 训练配置与 Hydra overrides

配置位于 `env.commands.motion.research`。M0 必须关闭质量门控；M1 必须开启质量门控。当前只实现 `M0` 和 `M1`，motion/segment sampling 只实现 `uniform`。

质量门控字段及默认值：

```yaml
quality_gate:
  enabled: false
  metadata_path: ""
  reject_statuses: [reject]
  include_borderline: true
  strict_metadata_match: true
  empty_motion_policy: error
  gate_scope: assignment_start
```

约束：

- M1 必须启用 Segment 和 sampling statistics；
- M1 必须提供本地 `.txt` manifest 和 `metadata_path`；registry 单 motion 入口不支持 M1 metadata 绑定；
- `reject_statuses` 必须包含 `reject`，不得包含 `pass`；
- `include_borderline=true` 时不能同时把 `borderline` 加入 `reject_statuses`；
- 目前只接受 `gate_scope=assignment_start`；
- 正式 M1/M6/M7 必须使用 `strict_metadata_match=true` 和 `empty_motion_policy=exclude`。

训练启动时逐项验证 manifest 文件 SHA256、motion 数量与顺序、逐 motion 帧数和 FPS、segment 长度/schema/起止帧/全局数量，以及有序 pool fingerprint。任一项不一致时，默认在环境初始化阶段报错，不会把标签静默套到另一套数据上。`strict_metadata_match=false` 只保留给诊断并会产生非正式实验 warning。

如果要排除 borderline，使用下面的覆盖即可；无需改写 metadata：

```text
env.commands.motion.research.quality_gate.include_borderline=false
```

M1 不按 `quality_score` 加权，也不让 pass 比 borderline 概率更高。冻结状态只形成允许/拒绝 mask；选定 motion 后，pass 与允许的 borderline 起点同等参与 uniform start-frame 抽样。这样 M0/M1 的控制变量只有“排除 reject 起点”，不会混入质量加权策略。

## `assignment_start` 的精确定义与边界

M1 门控的是 episode assignment 的起始帧，不是整段播放期间的逐帧屏蔽。

对长度为 (T) 的 motion，legacy 起点域是 `[0, T-1)`，即整数帧 `0..T-2`。这是因为原 baseline 使用：

```text
start = int(uniform_phase * (T - 1)), uniform_phase in [0, 1)
```

对 segment `[a,b)`，若状态允许，合法起点数为：

```text
max(0, min(b, T - 1) - a)
```

若状态不允许，合法起点数为 0。因此：

- `reject` segment 永远不会成为 assignment 的起始 segment；
- motion 最后一帧 `T-1` 本来就不是 legacy 起点；
- 仅包含最后一帧的尾部 segment 即使是 `pass`，也没有合法 assignment start；
- motion 被均匀地从“至少有一个合法起点”的 motion 集合中选择；
- 在选定 motion 内，所有合法起始帧均匀，而不是先均匀选 segment；长 segment 因合法帧多，会有更多起点概率；
- sampler 固定使用一次 motion `torch.randint` 和一次 legacy `sample_uniform`，由无随机数的前缀索引映射到合法帧，不使用 rejection loop。

assignment 以后仍按原训练逻辑逐帧播放到 motion 结束，不会在 segment 边界自动 reset，也不会在即将进入 reject segment 时截断。因此：

- `sampling/reject_segment_sample_count` 和 `quality/reject_start_assignment_count` 统计“assignment 起点所在 segment”，M1 中应始终为 0；
- `sampling/current_reject_reference_fraction` 统计当前播放帧落在 reject segment 的环境比例，可能大于 0；
- `quality/reject_reference_frame_count` 累计训练期间参考帧落在 reject segment 的次数；
- `quality/reject_rollout_exposure_ratio = reject_reference_frame_count / reference_frame_count`，可能大于 0；
- 后两个 rollout exposure 指标大于 0 不代表门控失效，而是 `assignment_start` 语义的预期结果。

如果未来要求整段轨迹永不进入 reject 区域，需要新增截断、segment episode 或跨边界重采样语义；当前版本没有实现，不能悄悄改变。

## 空 motion 策略

“空 motion”指没有任何质量允许且属于 legacy 起点域的帧；一个只有 allowed 单帧尾段但该帧为 `T-1` 的 motion 也可能为空。

- `empty_motion_policy=error`：环境初始化立即报错并列出 motion ID，适合严格调试阈值和 metadata。
- `empty_motion_policy=exclude`：不修改 manifest，不删除 motion loader 中的原始 motion，只在质量门控开启时把空 motion 的 runtime 采样概率设为 0；剩余 eligible motions 重新归一化后均匀采样。正式 M1/M6/M7 使用这个策略。
- 如果所有 motion 都为空，无论配置如何都会报错。

离线 `quality_summary.json` 中的 `empty_eligible_motion_count` 是数据事实，不是 metadata 构建失败。当前原始 random6000 预期为 2；训练是否允许由 `empty_motion_policy` 决定。不要通过替换 motion 或生成 5998 行 manifest 来“消除”这个数字。

## W&B 配置和指标

`sampling_statistics.log_interval` 控制 `sampling/*` 和 `quality/*` 的共同记录间隔。不会上传逐 motion/segment 数组。

静态质量指标：

```text
quality/num_pass_segments
quality/num_borderline_segments
quality/num_reject_segments
quality/pass_ratio
quality/borderline_ratio
quality/reject_ratio
quality/mean_quality_score
quality/min_quality_score
quality/num_empty_eligible_motions
quality/excluded_motion_count
quality/eligible_motion_ratio
quality/metadata_match_ok
quality/reject_start_assignment_count
quality/reject_reference_frame_count
quality/reference_frame_count
quality/reject_rollout_exposure_ratio
```

assignment 和当前引用指标：

```text
sampling/quality_gate_enabled
sampling/eligible_segment_coverage
sampling/pass_segment_sample_count
sampling/borderline_segment_sample_count
sampling/reject_segment_sample_count
sampling/reject_start_assignment_count
sampling/current_reject_reference_fraction
```

它们与阶段 0 的 `sampling/total_assignments`、motion/segment coverage、最大采样占比、平均计数和概率 fallback 指标一起记录。

数据池大小指标：

```text
dataset/manifest_motion_count
dataset/effective_motion_count
dataset/excluded_motion_count
dataset/eligible_motion_ratio
```

W&B config 还保存 quality gate 开关、metadata path、reject statuses、borderline/strict/empty/scope 配置，并在环境完成 metadata 校验后加入：

```text
quality_metadata_file
quality_metadata_sha256
quality_config_sha256
quality_manifest_sha256
quality_schema_version
quality_gate_scope
quality_include_borderline
quality_empty_motion_policy
```

正式原始 random6000 的 M1 应看到 `quality/metadata_match_ok=1`、
`quality/num_empty_eligible_motions=2`、`quality/excluded_motion_count=2`、
`dataset/manifest_motion_count=6000`、`dataset/effective_motion_count=5998`、
`quality/reject_start_assignment_count=0` 和 `sampling/reject_segment_sample_count=0`。
`quality/reject_rollout_exposure_ratio` 若大于 0，应按 `assignment_start` 边界解释：起点被门控，但 reference 仍可能随后播放经过 reject segment。

## Checkpoint 与 resume

质量状态随现有 checkpoint 内的 `checkpoint["infos"]["sampling_state"]` 保存，不创建 sidecar。内容包括：

- 研究配置和 gate 语义；
- Segment 布局；
- motion/segment assignment 计数和概率 fallback 计数；
- quality rollout exposure 累计计数：`reference_frame_count` 和 `reject_reference_frame_count`；
- quality metadata schema、segment schema、metadata SHA256、quality config SHA256、manifest SHA256、pool fingerprint、`empty_motion_policy`、eligible motion mask hash、manifest/effective/excluded motion count。

恢复时会严格比较上述 identity；metadata 路径本身可以变化，所以冻结文件整体搬迁后仍可恢复，但实际 NPZ SHA256、配置、manifest 和 pool 必须一致。任何 identity 不一致都会明确报错，不能把 M1 统计加载到另一套标签或 manifest。

`QualityGatedStartIndex` 完全由当前 Segment 索引和冻结 mask 推导，没有额外可变状态。runner 恢复累计统计后，会追加新进程环境初始化时已经产生的 active assignments。旧 checkpoint 缺少 `quality_exposure` 时按 0 初始化，兼容模块一早期 checkpoint；旧 checkpoint 没有 sampling state 时仍能加载策略，但会 warning，并从当前 active assignments 建立新统计；这种恢复不具备旧质量 metadata identity 证明，不应直接作为正式 M1 连续实验。

checkpoint 不保存 RNG、并行环境物理状态、当前播放进度或逐 bit 轨迹；resume 会重新初始化环境。

## Assignment trace 与 M0 RNG 等价性检查

默认训练不会保存逐 assignment 明细。若要证明两个短程 smoke 在相同 seed、manifest、`num_envs` 和配置下采样序列一致，可以临时打开：

```text
env.commands.motion.research.assignment_trace.enabled=true
env.commands.motion.research.assignment_trace.output_path=/tmp/wbt_assignment_trace.csv
env.commands.motion.research.assignment_trace.max_entries=2048
```

trace 是只读观察器：它在采样已经完成后记录前 N 条 assignment，不调用 `torch.randint`、`sample_uniform` 或 `torch.multinomial`，不会改变训练使用的随机序列。CSV 字段为：

```text
assignment_index, env_id, motion_id, start_frame,
local_segment_id, global_segment_id, pool_fingerprint, run_label
```

比较两个 trace：

```bash
python scripts/compare_assignment_traces.py \
  /tmp/wbt_assignment_trace_a.csv \
  /tmp/wbt_assignment_trace_b.csv
```

输出 `TRACE_MATCH` 表示逐行一致；输出 `TRACE_MISMATCH` 会打印前若干个不一致行。正式训练保持 `assignment_trace.enabled=false`。

## 人工回放

`quality_review_segments.csv` 的 `recommended_play_command` 已按每一行生成准确 Isaac 回放命令。例如：

```bash
env -u PYTHONPATH -u LD_LIBRARY_PATH \
python scripts/replay_npz.py \
  --motion_file "/absolute/or/project-relative/path/from-quality_review_segments.csv" \
  --start_frame 100 \
  --end_frame_exclusive 150
```

`replay_npz.py` 严格校验：

```text
0 <= start_frame < end_frame_exclusive <= motion_length
```

并只循环播放该右开区间。人工审核至少覆盖高分 pass、pass/borderline 边界、borderline/reject 边界和最低分 reject；重点检查关节跳变、根/身体不连续、穿地、足滑和误杀的合理高难动作。

如果本机 GPU 扛不住 Isaac Sim，可以用 `recommended_video_command` 或下面的轻量离线视频命令。这个脚本不启动 Isaac，只从 NPZ 的 `body_pos_w/body_quat_w/body_names` 画侧视、俯视和脚底高度，适合快速判断穿地、足滑和大跳变：

```bash
python scripts/render_npz_preview.py \
  --motion_file "/absolute/or/project-relative/path/from-quality_review_segments.csv" \
  --start_frame 100 \
  --end_frame_exclusive 150 \
  --output /tmp/wbt_npz_previews/review_segment.mp4
```

离线视频只用于人工质检，不替代 Isaac 中的真实几何外观或控制器动力学回放。

## 纯 Python 测试

以下测试不启动 Isaac Sim：

```bash
cd /home/l/whole_body_tracking_new
PYTHONDONTWRITEBYTECODE=1 \
/home/l/miniconda3/envs/hybrid_robot/bin/python -m unittest \
  tests/test_quality_audit.py \
  tests/test_quality_metadata.py \
  tests/test_quality_gate_index.py \
  tests/test_quality_gate_integration.py \
  tests/test_build_segment_quality_metadata.py
```

它们覆盖配置和 schema、质量公式及状态、segment 边界跳变、非有限值、URDF 限位、穿地/足滑、metadata hash/顺序匹配、合法起点映射、尾帧边界、空 motion、M0/M1 配置约束、RNG 调用结构和 checkpoint 接线。

完整纯 Python 回归还应运行：

```bash
PYTHONDONTWRITEBYTECODE=1 \
/home/l/miniconda3/envs/hybrid_robot/bin/python -m unittest discover -s tests -p 'test_*.py'
```

## M0/M1 W&B pilot

先设置同一个 W&B entity。当前 RSL-RL writer 读取 `WANDB_USERNAME`，W&B SDK 读取 `WANDB_ENTITY`，因此两者都设置：

```bash
export WANDB_USERNAME=longxianli222-northeastern-university
export WANDB_ENTITY=longxianli222-northeastern-university
```

下面两个 pilot 使用相同的原始 random6000 manifest、seed、网络和 500-iteration 预算，只切换质量门控。若要先做更短的启动检查，可临时把两条命令同时改为 `--max_iterations 10` 并使用新的 W&B run ID；10 iterations 只算 smoke，不算 M1-pilot。

### M0：门控关闭

```bash
export WANDB_USERNAME=longxianli222-northeastern-university
export WANDB_ENTITY=longxianli222-northeastern-university

env -u PYTHONPATH -u LD_LIBRARY_PATH \
  WANDB_USERNAME=longxianli222-northeastern-university \
  WANDB_ENTITY=longxianli222-northeastern-university \
  WBT_DISABLE_ONNX_ON_SAVE=1 \
python scripts/rsl_rl/train.py \
  --disable_fabric \
  --task Tracking-Flat-G1-v0 \
  --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/manifests/experiments/random_seed42/random6000_seed42.txt \
  --headless \
  --logger wandb \
  --log_project_name whole_body_tracking_module1_pilot \
  --run_name module1_m0_random6000_seed42_pilot_v1 \
  --wandb_run_name module1_m0_random6000_seed42_pilot_v1 \
  --wandb_run_id module1-m0-random6000-seed42-pilot-v1 \
  --wandb_resume never \
  --num_envs 16 \
  --seed 42 \
  --max_iterations 500 \
  env.commands.motion.research.method_name=M0 \
  env.commands.motion.research.segment.enabled=true \
  env.commands.motion.research.segment.length_seconds=1.0 \
  env.commands.motion.research.quality_gate.enabled=false \
  env.commands.motion.research.difficulty_calibration.enabled=false \
  env.commands.motion.research.motion_sampling.mode=uniform \
  env.commands.motion.research.segment_sampling.mode=uniform \
  env.commands.motion.research.diversity_constraint.enabled=false \
  env.commands.motion.research.sampling_statistics.enabled=true \
  env.commands.motion.research.sampling_statistics.log_interval=1
```

### M1：门控开启

默认 `reject_statuses=("reject",)` 已由 typed config 固定；pilot 不需要额外覆盖 tuple。

```bash
export WANDB_USERNAME=longxianli222-northeastern-university
export WANDB_ENTITY=longxianli222-northeastern-university

env -u PYTHONPATH -u LD_LIBRARY_PATH \
  WANDB_USERNAME=longxianli222-northeastern-university \
  WANDB_ENTITY=longxianli222-northeastern-university \
  WBT_DISABLE_ONNX_ON_SAVE=1 \
python scripts/rsl_rl/train.py \
  --disable_fabric \
  --task Tracking-Flat-G1-v0 \
  --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/manifests/experiments/random_seed42/random6000_seed42.txt \
  --headless \
  --logger wandb \
  --log_project_name whole_body_tracking_module1_pilot \
  --run_name module1_m1_random6000_seed42_pilot_fresh_v1 \
  --wandb_run_name module1_m1_random6000_seed42_pilot_v1 \
  --wandb_run_id module1-m1-random6000-seed42-pilot-v1 \
  --wandb_resume never \
  --num_envs 16 \
  --seed 42 \
  --max_iterations 500 \
  env.commands.motion.research.method_name=M1 \
  env.commands.motion.research.segment.enabled=true \
  env.commands.motion.research.segment.length_seconds=1.0 \
  env.commands.motion.research.quality_gate.enabled=true \
  env.commands.motion.research.quality_gate.metadata_path=/home/l/whole_body_tracking_new/outputs/module1_quality_random6000_seed42_original_v1/segment_quality_metadata.npz \
  env.commands.motion.research.quality_gate.include_borderline=true \
  env.commands.motion.research.quality_gate.strict_metadata_match=true \
  env.commands.motion.research.quality_gate.empty_motion_policy=exclude \
  env.commands.motion.research.quality_gate.gate_scope=assignment_start \
  env.commands.motion.research.difficulty_calibration.enabled=false \
  env.commands.motion.research.motion_sampling.mode=uniform \
  env.commands.motion.research.segment_sampling.mode=uniform \
  env.commands.motion.research.diversity_constraint.enabled=false \
  env.commands.motion.research.sampling_statistics.enabled=true \
  env.commands.motion.research.sampling_statistics.log_interval=1
```

M1 不应因两条 empty eligible motion 失败；它们应在 `empty_motion_policy=exclude` 下被 runtime 排除，manifest 仍保持 6000 行。

### M1 resume 检查

500 iterations 的最终模型应为 `model_499.pt`。用相同 manifest、metadata、W&B run ID 和所有 Hydra overrides 再运行 2 个额外 iterations：

```bash
export WANDB_USERNAME=longxianli222-northeastern-university
export WANDB_ENTITY=longxianli222-northeastern-university

env -u PYTHONPATH -u LD_LIBRARY_PATH \
  WANDB_USERNAME=longxianli222-northeastern-university \
  WANDB_ENTITY=longxianli222-northeastern-university \
  WBT_DISABLE_ONNX_ON_SAVE=1 \
python scripts/rsl_rl/train.py \
  --disable_fabric \
  --task Tracking-Flat-G1-v0 \
  --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/manifests/experiments/random_seed42/random6000_seed42.txt \
  --headless \
  --logger wandb \
  --log_project_name whole_body_tracking_module1_pilot \
  --run_name module1_m1_random6000_seed42_pilot_resume_v1 \
  --wandb_run_name module1_m1_random6000_seed42_pilot_v1 \
  --wandb_run_id module1-m1-random6000-seed42-pilot-v1 \
  --wandb_resume must \
  --resume True \
  --load_run '.*_module1_m1_random6000_seed42_pilot_fresh_v1$' \
  --checkpoint model_499.pt \
  --num_envs 16 \
  --seed 42 \
  --max_iterations 2 \
  env.commands.motion.research.method_name=M1 \
  env.commands.motion.research.segment.enabled=true \
  env.commands.motion.research.segment.length_seconds=1.0 \
  env.commands.motion.research.quality_gate.enabled=true \
  env.commands.motion.research.quality_gate.metadata_path=/home/l/whole_body_tracking_new/outputs/module1_quality_random6000_seed42_original_v1/segment_quality_metadata.npz \
  env.commands.motion.research.quality_gate.include_borderline=true \
  env.commands.motion.research.quality_gate.strict_metadata_match=true \
  env.commands.motion.research.quality_gate.empty_motion_policy=exclude \
  env.commands.motion.research.quality_gate.gate_scope=assignment_start \
  env.commands.motion.research.difficulty_calibration.enabled=false \
  env.commands.motion.research.motion_sampling.mode=uniform \
  env.commands.motion.research.segment_sampling.mode=uniform \
  env.commands.motion.research.diversity_constraint.enabled=false \
  env.commands.motion.research.sampling_statistics.enabled=true \
  env.commands.motion.research.sampling_statistics.log_interval=1
```

resume 后不应出现 `no sampling state` warning；累计 assignment 计数应延续并包含新进程初始化的 active assignments，quality identity 校验必须通过，W&B step 应保持单调。

## Pilot 验收清单

- 离线 `audit_errors` 为空，`training_compatible=true`；
- 人工回放覆盖各质量层级，确认阈值没有把合理高难动作当作坏数据；
- 原始 random6000 下 `empty_eligible_motion_count=2`、`excluded_motion_count=2`、`dataset/effective_motion_count=5998`，但 manifest 仍为 6000 行；
- M0 仍走原 baseline sampler；M1 没有 rejection loop 或额外 RNG；
- M1 的 `sampling/reject_segment_sample_count` 始终为 0；
- `sampling/current_reject_reference_fraction` 按 `assignment_start` 语义解释；
- eligible segment coverage 随训练增长，pass/borderline 仍能被抽到；
- reward、概率和训练损失无 NaN/Inf，reset 正常；
- W&B config 中 hashes、schema 和 gate 配置完整，指标按 interval 出现；
- checkpoint 保存、resume、metadata relocation 和故意 mismatch 拒绝路径均已验证；
- 冻结阈值、quality config、metadata NPZ、manifest、Git commit 和评测规则后，才开始正式 M1。
