# 模块四：G1 轨迹聚类与多样性约束三级采样

## 1. 目标、边界与实现位置

模块四解决的是：模块三可能长期把训练预算集中到少数高学习缺口、但彼此相似的
Motion 上，使其他机器人运动模式得不到足够覆盖。实现只在模块三的 Motion 抽样之前
增加一个冻结的 cluster 层：

```text
Cluster -> Motion within Cluster -> Segment within Motion -> Start Frame
```

代码职责分开如下：

- `utils/motion_clustering.py`：从模块二的原始 segment feature 聚合 Motion feature，
  拟合/应用 robust scaler、可选 PCA、NumPy KMeans++，并冻结 Profile；
- `utils/cluster_metadata.py`：保存、加载和严格验证 motion-to-cluster NPZ；
- `utils/diversity_sampling.py`：计算 cluster 预算，并执行
  `cluster -> motion -> segment` 条件采样；
- `tasks/tracking/mdp/commands.py`：只在 diversity 开启时加载 metadata、校验四方映射，
  并把 cluster ID 接入现有在线控制器；
- `utils/online_learning.py`：继续计算模块三的误差与 gap，只把已有
  `G_motion/G_local` 交给 M7 的条件层；
- `scripts/run_module4_gpu_pilots.py`：仅生成或运行 random100 的 A/B/C 短程 Pilot，
  不提供正式训练入口。

本模块不引入新的质量、难度、误差或 gap 定义，也不改变 PPO、reward、observation、
termination、1 秒 segment 或原始 manifest。`diversity_constraint.enabled=false` 时不加载
cluster metadata、不创建 cluster sampler、不增加 cluster RNG 调用；M0--M6 仍走模块三
原路径，旧 assignment trace 仍保持 8 列。

### 为什么不能按来源目录分簇

`dance/fitness/humanml/GRAB/idea400/haa500` 等目录或 category label 描述数据来源，
不是最终 G1 机器人实际执行出的运动几何。把它们当 cluster 会把采集体系或人工标签当成
运动相似性，既可能把不同机器人运动混在一起，也可能把相似运动人为拆开，并造成难以解释
的数据来源预算偏置。

因此 cluster 拟合只能读取转换后的最终 G1、policy-independent 运动学和接触 feature。
source category 只允许出现在离线构建后的组成诊断和人工复查表中；修改 source、quality、
difficulty score/bin 或 policy error 不得改变 cluster ID 或 Profile SHA256。

## 2. 实际 Motion 聚类 Feature

配置文件为 `configs/diversity/g1_motion_clustering.yaml`。当前 17 项 source feature
展开为 30 个 Motion feature；展开名固定为
`<source>__<aggregation>`。`required/optional` 是 source 级声明，并继承到它的全部
aggregation。

| # | Source segment feature | Aggregation | Unit | Availability |
| ---: | --- | --- | --- | --- |
| 1 | `root_linear_speed_p95` | duration-weighted mean, P90 | m/s | required |
| 2 | `root_linear_acceleration_p95` | duration-weighted mean, P90 | `m/s^2` | required |
| 3 | `root_angular_speed_p95` | duration-weighted mean, P90 | rad/s | required |
| 4 | `root_angular_acceleration_p95` | duration-weighted mean, P90 | `rad/s^2` | required |
| 5 | `joint_speed_p95` | duration-weighted mean, P90 | rad/s | required |
| 6 | `joint_acceleration_p95` | duration-weighted mean, P90 | `rad/s^2` | required |
| 7 | `joint_range_mean` | duration-weighted mean | rad | required |
| 8 | `joint_range_p90` | duration-weighted mean, P90 | rad | required |
| 9 | `body_height_range` | duration-weighted mean, P90 | m | required |
| 10 | `body_height_std` | duration-weighted mean, P90 | m | required |
| 11 | `hand_speed_p95` | duration-weighted mean, P90 | m/s | optional |
| 12 | `foot_swing_speed_p95` | duration-weighted mean, P90 | m/s | optional |
| 13 | `end_effector_speed_p95` | duration-weighted mean, P90 | m/s | optional |
| 14 | `single_support_ratio` | duration-weighted mean | `frame_fraction` | optional |
| 15 | `double_support_ratio` | duration-weighted mean | `frame_fraction` | optional |
| 16 | `flight_ratio` | duration-weighted mean | `frame_fraction` | optional |
| 17 | `contact_switch_rate_per_second` | duration-weighted mean, P90 | 1/s | optional |

实现还支持配置化的 `duration_weighted_std`，但默认 17 项定义没有启用它。聚合直接复用
模块二 NPZ 的原始 `feature_values`、`feature_available_mask`、`duration_seconds` 和
`motion_segment_offsets`，不会重新从轨迹复制速度、接触或关节 feature 的计算。

### Duration weighting 与缺失规则

对 Motion `m` 的 duration-weighted mean：

```text
x_bar(m,j) = sum_s duration(m,s) * x(m,s,j)
             / sum_s duration(m,s)
```

这里使用模块二保存的每个 segment 实际 `duration_seconds`；尾部不足 1 秒的短 segment
不会获得完整 segment 的同等权重。P90 是该 Motion 内 segment 值的线性 P90，目前不按
duration 加权。若某个 source feature 在一条 Motion 的任一 segment 缺失，则该 Motion
对应 aggregation 整体保持 unavailable/NaN，而不是用 0 或部分时间区间冒充完整 Motion。

required feature 在任一 Train Motion 缺失会 fail-fast。optional feature 的 Train 覆盖率
必须至少为 `0.75`；达到阈值但不是 100% 的 optional 列仍只保留作诊断，不进入欧氏距离，
避免隐式填充缺失物理量。

## 3. Train-only 稳健变换

对每个展开 feature `j`，只在拟合用 Train manifest 上计算：

```text
center_j = median(x_j)
mad_scale_j = 1.4826 * median(|x_j - center_j|)
```

若 MAD 小于 `near_constant_scale_threshold=1e-5`，实现使用 P05--P95 fallback：

```text
fallback_scale_j = (P95_j - P05_j) / 3.289707253902945
scale_j = max(selected_scale_j, robust_scale_epsilon=1e-6)
z_j = clip((x_j - center_j) / scale_j, -5, 5)
```

fallback 仍小于阈值的 feature 被标为 near-constant，并从距离空间移除；这类列保留在
metadata/Profile 中供审查。零覆盖 optional 列同样禁用。active Train 列必须对每条
Motion 都 finite，否则停止构建。原始 missing 始终保留为 NaN 加 availability mask，
不会先填 0 再参与距离。

### 可选 PCA

默认 `use_pca=false`，所以当前聚类空间就是 active robust-scaled feature。实现提供
NumPy SVD PCA，可通过固定 `pca_components` 或
`explained_variance_target` 二选一启用。PCA 仍只在 Train 拟合，并冻结：

```text
mean
components
explained_variance
explained_variance_ratio
```

每个 component 以绝对值最大的 loading 为 pivot；若 pivot 为负则整行乘 `-1`，从而消除
SVD 的符号不确定性。`transform` 只应用冻结状态，不会重新拟合 PCA。

## 4. NumPy KMeans++ 与稳定 cluster ID

仓库没有为模块四新增 scikit-learn 依赖。当前实现标识为
`numpy.kmeans_plusplus.lloyd.canonical.v1`：

- NumPy `SeedSequence([random_seed, initialization])` 生成独立初始化；
- KMeans++ 选初始中心，Lloyd 迭代，空簇用最远且 donor 簇仍有多个样本的行确定性修复；
- 默认 `K=8`、seed 42、`n_init=20`、`max_iterations=300`、
  tolerance `1e-6`；
- 先按最小 inertia 选解；近似并列时以 centroid 扁平序列和 label 序列稳定破平；
- 最终 centroid 按 feature vector 字典序排序，旧 label 通过
  `canonical_label_remap` 重编号为 `0..K-1`。

相同输入、配置和 seed 应产生相同 centroid 顺序、motion `cluster_id` 和 Profile
SHA256。人工写的 cluster 描述不得用来重新编号。

### `fit_transform` 与 `transform`

`fit_transform` 只能用于 Train：

```text
segment raw feature
-> Motion aggregation
-> fit robust scaler
-> optional fit PCA
-> fit KMeans
-> canonicalize labels
-> nearest frozen centroid assignment
-> save Profile + target metadata
```

`transform` 用于其他冻结 manifest：

```text
segment raw feature
-> frozen Motion aggregation definition
-> frozen scaler
-> frozen PCA（若启用）
-> nearest frozen centroid assignment
```

`transform` 不重估 center/scale、PCA、centroid 或 label order。required/optional coverage、
feature 定义和 active feature 可用性仍按冻结 Profile 校验。K、Profile 或 config 不得根据
Validation/Test 表现重选。

`transform` 还要求当前 cluster config 的 hash、算法 schema 和 K 与冻结 Profile 完全一致，
且不允许 `--diagnose-k`。命令行 `--seed` 只控制 review 随机样本和 sampled silhouette；
真正的 KMeans seed 来自冻结 config 的 `random_seed`。

## 5. Profile、metadata 与映射身份

### Cluster Profile

`cluster_profile.json` 的 schema 是 `wbt.motion_clustering_profile.v1`，包含：

- 30 个展开 feature 的 name、source、aggregation、unit、required/optional；
- minimum optional coverage、center、scale、coverage、near-constant、active mask、
  robust clip；
- PCA enabled/state；
- KMeans 实现 identity、seed、n_init、max iterations、tolerance、inertia、
  iteration count、converged；
- canonical centroids、label remap、`num_clusters`；
- Train manifest SHA256、ordered pool fingerprint、difficulty metadata/profile SHA256、
  config SHA256、Git commit 和 warnings。

Profile SHA256 是对确定性 canonical JSON 的哈希。

### Runtime NPZ

训练只加载 `motion_cluster_metadata.npz`，schema 是 `wbt.motion_cluster.v1`。required
arrays 为：

```text
schema_version
algorithm_schema_version
manifest_sha256
manifest_motion_count
pool_fingerprint
difficulty_metadata_sha256
difficulty_profile_sha256
cluster_profile_sha256
cluster_config_sha256
num_clusters

motion_keys
motion_lengths
motion_fps
motion_segment_offsets
motion_id
cluster_id
cluster_sizes

centroids
feature_names
motion_feature_matrix
feature_available_mask
standardized_feature_matrix
```

NPZ 不保存完整轨迹。loader 校验 shape、dtype、finite/NaN 约定、唯一有序
`motion_id=0..N-1`、每条 Motion 恰好一个合法 `cluster_id`、cluster size 一致及所有
SHA256 格式。运行时再与 Stage-0 校验 manifest SHA256、motion 顺序、frame count、FPS、
segment offsets、pool fingerprint、difficulty metadata/profile identity 和预期 K。
layout 错配即使 `strict_metadata_match=false` 也不能放过；该开关只允许显式放宽 provenance，
且这样的 run 不得作为正式 M7。

M7 同时检查：

```text
Stage-0 <-> Quality <-> Difficulty <-> Cluster
```

cluster 是 Motion 级 metadata，但它仍绑定相同的 motion order 和全局 segment offsets，
避免把 cluster ID 附到错误轨迹。

### 离线审查输出

离线 builder 完成身份检查和计算后才开始生成固定 9 个文件：

| 文件 | 用途 |
| --- | --- |
| `motion_cluster_metadata.csv` | 逐 Motion 的 manifest/motion identity、duration、segment count、cluster、centroid distance、30 项 raw/z feature 和诊断字段 |
| `motion_cluster_metadata.npz` | runtime 严格加载的紧凑映射 |
| `cluster_profile.json` | Train-fitted scaler/PCA/KMeans 冻结状态 |
| `cluster_summary.json` | cluster size、Gini、silhouette、距离、feature、difficulty/quality/source 诊断与 warning |
| `cluster_feature_statistics.csv` | 每个展开 feature 的 source、aggregation、unit、coverage、center/scale、active/near-constant |
| `cluster_centroids.csv` | 每簇 distance-space centroid 与可解释的原始 z-space 坐标；top feature 见 summary/review |
| `cluster_review_motions.csv` | nearest/boundary/farthest/random Motion 与 replay 命令 |
| `cluster_config_resolved.json` | canonical resolved clustering config |
| `normalized_manifest.txt` | 实际有序 Motion pool；`--max-motions` 时为稳定 prefix |

Motion CSV 的固定核心列为：

```text
schema_version, algorithm_schema_version, manifest_index, motion_id
motion_key, motion_path, duration_seconds, segment_count
cluster_id, distance_to_centroid, normalized_distance_to_centroid
boundary_margin, cluster_size, source_category_diagnostic_only
```

随后对每个展开 feature 写 `<name>`、`z_<name>`、`available_<name>`。

`normalized_distance_to_centroid` 仅用于当前输出 manifest 内的人工 outlier 排序：它按
该批次每簇 distance median 归一，不是跨 manifest 可比较的冻结量。cluster assignment
只使用冻结 centroid 的原始 distance，不读取这个诊断列。

CSV/JSON 用于人工审查，训练热路径不加载。`cluster_summary.json` 中的
source/quality/difficulty 交叉统计只能是 post-hoc diagnostics，并必须明确这些字段没有
参与拟合。

builder 从不打开 Motion `.npz` 或重算运动学；manifest 只用于有序 identity、path 和
source-category 诊断，拟合数据始终来自冻结的模块二 metadata。manifest、difficulty
metadata 和 difficulty Profile 的身份绑定始终严格。可选 quality metadata 只做映射/诊断，
其 provenance 是否允许放宽由 `--strict` 控制。

## 6. Cluster 预算与三级概率

quality gate 和合法 start mask 之后，令 `n_c` 为 cluster `c` 中 eligible Motion 数，
`C` 为非空 eligible cluster 数，`f` 为
`minimum_budget_fraction_of_uniform`，`alpha` 为
`cluster_size_exponent`：

```text
minimum_share = f / C

size_component(c) = n_c^alpha / sum_j n_j^alpha

P(c) = minimum_share
       + (1 - C * minimum_share) * size_component(c)
```

默认 `f=0.5, alpha=0.5`，即每个 eligible cluster 至少获得均匀 cluster share 的一半，
其余预算按 eligible cluster size 的平方根分配。空 cluster 的概率严格为 0，预算仅在
剩余非空 cluster 中重算；只有一个 eligible cluster 时概率为 1。实现验证 finite、非负、
sum=1 和 floor。v1 的 `count_aware_correction=false`，因此 observed deficit 只诊断，
不会反馈改变 `P(c)`。

正式因子分解为：

```text
P(c,m,s) = P(c) * P(m | c) * P(s | m)

P_global(m) = P(cluster(m)) * P(m | cluster(m))
```

### `P(c)`

只由 eligible cluster size、floor 和 exponent 决定。它不读取 `G_motion`、`G_local`、
raw error、reward、success 或来源类别，避免同一高误差运动模式在 cluster 和 Motion
两层被重复放大。

### `P(m | c)`

只在选中 cluster 的 eligible Motion 内复用模块三的 `G_motion`、under-sampling bonus、
temperature、uniform mix、fallback 和 water-filling cap：

```text
U_m = 1 / sqrt(motion_sample_count_m + 1)
score_m = G_motion_m
```

Reject/empty Motion 概率为 0。配置 motion cap 对很小 cluster 可能不可行，所以仅对受影响
cluster 使用：

```text
effective_cap(c) = max(configured_motion_cap, 1 / eligible_motion_count(c))
```

放宽次数作为诊断记录，不改变 M5/M6 的全局 Motion cap 语义。

### `P(s | m)`

直接复用 M6 的 signed `G_local`、segment under-sampling、uniform mix、temperature、
quality eligibility 和 conditional cap。选定 segment 后，只在它的 legacy 合法
start frame 中均匀抽样。模块四不重定义 local gap。

## 7. Quality、Difficulty、Gap 与 Diversity 严格解耦

| 层 | 唯一职责 | 不允许做的事 |
| --- | --- | --- |
| Quality | 生成 assignment-start eligible mask；排除 Reject/empty Motion | 不进入 feature 距离、cluster ID、误差或 gap |
| Difficulty | 提供冻结 difficulty bin，校准同难度当前策略误差 | difficulty score/bin 不参与聚类，也不直接分预算 |
| Learning gap | `G_motion` 控制 `P(m|c)`；signed `G_local` 控制 `P(s|m)` | 不控制 `P(c)`，不反写 quality/difficulty |
| Diversity | 用冻结 G1 轨迹 cluster 和 eligible size 控制 `P(c)` | 不读取 policy error/gap/reward/source |

M7 配置契约是：

```text
method_name = M7
quality gate = on, empty_motion_policy = exclude
quality include_borderline = true
difficulty calibration = on
motion mode = learning_gap
segment mode = relative_learning_gap
diversity = on
```

## 8. Warmup、更新时间与抽样 RNG

v1 固定 `diversity_during_warmup=true`：

```text
warmup 内：
  P(c)       = diversity target
  P(m | c)   = cluster 内 uniform
  P(s | m)   = Motion 内 uniform

warmup 后：
  P(c)       = 同一个固定 diversity target
  P(m | c)   = 模块三 G_motion 条件概率
  P(s | m)   = 模块三 G_local 条件概率
```

因此 M7 从训练开始就维持 cluster 覆盖，而 Motion/Segment 学习缺口仍在模块三 warmup
结束后启用。正式默认 warmup 1000 iterations、每 50 iterations 更新；Pilot helper
默认分别缩短为 50 和 10。EMA 仍每个 PPO iteration 提交，概率不在 simulation step
重算。

M7 使用一个专用 `torch.Generator`，严格按 cluster、Motion、Segment、start-frame 顺序
推进。它不消费全局 torch RNG。diversity 关闭时仍使用模块三原 sampler；不会增加一次
cluster draw。M7 assignment trace 在原 8 列后追加：

```text
cluster_id
cluster_probability
motion_probability_conditional
motion_probability_global
segment_probability_conditional
```

这些值只观察已抽到的结果，不额外抽随机数。

## 9. Checkpoint 与 resume

sampling sidecar 除模块三原状态外，保存：

- cluster metadata/profile/config/schema identity 和 motion-to-cluster mapping；
- eligible cluster mask、每簇 eligible Motion 数、有效 conditional cap；
- 固定 target/current cluster probability；
- `P(m|c)`、global Motion probability、`P(s|m)`；
- cluster sample count、observed share、budget deficit；
- probability/fallback/iteration counters；
- 专用 sampler generator state。

恢复时重新验证当前 Stage-0、quality、difficulty、cluster 和研究配置身份，再恢复概率、
计数、更新 cadence 和 RNG；不重新聚类、重编号或 warmup。恢复后下一批
cluster/Motion/Segment assignment 接在保存时的专用 RNG stream 之后。checkpoint 不保存
per-env simulator state或 domain-randomization 全局 RNG，因此不承诺完整物理 rollout
bitwise replay。

diversity 关闭时 checkpoint 不要求 cluster identity，并继续兼容模块三状态；开启时缺失或
不匹配的 cluster identity 必须拒绝。

## 10. W&B 与轻量诊断

不上传 6000/21575 长数组。M7 在保留模块三 `online/*`、`error/*`、`gap/*`、
`sampling/*`、`quality/*`、`difficulty/*` 的同时记录：

- static：`diversity/enabled`、`metadata_match_ok`、`num_clusters`、
  eligible/empty cluster count、min/max cluster size、cluster size Gini；
- 每簇：`cluster_<c>_target_share`、`observed_share`、`sample_count`、
  `budget_deficit`；
- coverage/collapse：当前 target `P(c)` 的 cluster entropy/normalized entropy、实际
  sampled coverage、
  max/min-nonzero sampled fraction、target-observed L1/JS、probability sum error、
  cluster fallback count；
- cluster 内 Motion：mean/min conditional entropy、max conditional Motion
  probability、概率超过 0.1/0.25/0.5 的 Motion 数；
- 其他：conditional cap relax count、global Motion/conditional Segment
  entropy 与 max/min probability。

同一 sampler summary 也保留在 `sampling/*`，`diversity/*` 是 M7 专用镜像与静态身份。
大数组 snapshot 仍未实现，默认关闭。

## 11. CPU tests

模块四核心测试不依赖 Isaac Sim：

```bash
cd /home/l/whole_body_tracking_new

/home/l/miniconda3/envs/hybrid_robot/bin/python -m unittest \
  tests.test_motion_clustering \
  tests.test_cluster_metadata \
  tests.test_build_motion_cluster_metadata \
  tests.test_diversity_sampling \
  tests.test_diversity_runtime_integration \
  tests.test_module4_gpu_pilot_scripts
```

覆盖范围包括：

- duration weighting、P90/std、required/optional coverage、missing/NaN/Inf、
  near-constant 与默认 17 项配置；
- 三簇合成恢复、相同 seed 确定性、无关 source/quality/difficulty/policy 字段不变性；
- PCA 符号、KMeans centroid/label canonicalization、Profile round-trip 和冻结 transform；
- builder 固定 9 项输出、fit 重复确定性、冻结 transform、上游 identity fail-fast，
  以及 difficulty score/bin 改动不影响 cluster；
- NPZ schema、hash、ID/range/cluster size/centroid、严格 layout/provenance 和原子保存；
- budget floor、empty cluster、exponent 0/1、单 eligible cluster；
- `P(c)P(m|c)P(s|m)`、cluster 与 gap 独立、Reject/empty mask、条件 cap；
- 非连续 group 抽样、长期小簇覆盖、target-observed 收敛、warmup/update cadence；
- 专用 RNG、cluster count 和 assignment resume；
- `num_clusters=1` 时 M7 的 Motion/Segment 概率与 M6 一致；
- M0--M6 diversity guard、metadata 仅按需加载、四方 wiring、checkpoint identity；
- diversity-off trace 8 列兼容、M7 trace 因子字段，以及 A/B/C/resume 命令契约。

全仓回归仍应运行：

```bash
/home/l/miniconda3/envs/hybrid_robot/bin/python -m unittest discover \
  -s tests -p 'test_*.py'
git diff --check
```

本轮最终结果为 `211 tests / OK`，`compileall` 与 `git diff --check` 均通过。

## 12. random100 / random6000 离线构建

下列命令是构建模板；运行结果见下一节，未运行前不得填写数值。

### random100 Train-prefix `fit_transform`

```bash
cd /home/l/whole_body_tracking_new
conda activate hybrid_robot

python scripts/build_motion_cluster_metadata.py \
  --manifest outputs/module2_difficulty_pilot_random100_seed42_v1/normalized_manifest.txt \
  --difficulty-metadata outputs/module2_difficulty_pilot_random100_seed42_v1/segment_difficulty_metadata.npz \
  --difficulty-profile outputs/module2_difficulty_pilot_random100_seed42_v1/difficulty_profile.json \
  --quality-metadata outputs/module1_quality_pilot_random100_seed42_v1/segment_quality_metadata.npz \
  --cluster-config configs/diversity/g1_motion_clustering.yaml \
  --output-dir outputs/module4_clusters_random100_seed42_v1 \
  --mode fit_transform \
  --seed 42 \
  --strict \
  --overwrite
```

该 Profile 仅用于开发验证，不能作为 random6000 的正式候选 Profile。

### 完整 random6000 Train `fit_transform`

```bash
python scripts/build_motion_cluster_metadata.py \
  --manifest PHUMA_wbt_motions/manifests/experiments/random_seed42/random6000_seed42.txt \
  --difficulty-metadata outputs/module2_difficulty_random6000_seed42_v1/segment_difficulty_metadata.npz \
  --difficulty-profile outputs/module2_difficulty_random6000_seed42_v1/difficulty_profile.json \
  --quality-metadata outputs/module1_quality_random6000_seed42_original_v1/segment_quality_metadata.npz \
  --cluster-config configs/diversity/g1_motion_clustering.yaml \
  --output-dir outputs/module4_clusters_random6000_seed42_v1 \
  --mode fit_transform \
  --seed 42 \
  --strict \
  --overwrite
```

可选 Train-only K 诊断必须显式请求，且不能自动改变冻结 K：

```bash
python scripts/build_motion_cluster_metadata.py <同上参数> \
  --diagnose-k 6 8 10 12
```

### 冻结 Profile `transform`

```bash
python scripts/build_motion_cluster_metadata.py \
  --manifest /absolute/path/to/frozen_target_manifest.txt \
  --difficulty-metadata /absolute/path/to/target_segment_difficulty_metadata.npz \
  --difficulty-profile /absolute/path/to/target_difficulty_profile.json \
  --cluster-config configs/diversity/g1_motion_clustering.yaml \
  --output-dir outputs/module4_clusters_target_v1 \
  --mode transform \
  --profile outputs/module4_clusters_random6000_seed42_v1/cluster_profile.json \
  --seed 42 \
  --strict \
  --overwrite
```

### 本轮实际离线结果

以下数值直接来自本轮 builder 产物；没有运行 GPU Pilot，也没有据此调 K：

| Pool | Motions | Segments | Cluster sizes `0..7` | Min/Max | Gini | Silhouette |
| --- | ---: | ---: | --- | --- | ---: | ---: |
| random100 | 100 | 357 | `[5, 38, 8, 15, 4, 10, 17, 3]` | 3/38 | 0.427500 | 0.299712 |
| random6000 | 6000 | 21575 | `[1902, 456, 503, 948, 603, 557, 728, 303]` | 303/1902 | 0.299458 | 0.277857 |

random6000 在正式 `include_borderline=true` 口径下有 5998 条 eligible Motion；被排除的是
motion 1477 和 2288。各簇 eligible 数为
`[1902, 456, 503, 948, 603, 555, 728, 303]`。按默认 `f=0.5, alpha=0.5` 得到固定
cluster target：

```text
[0.166000, 0.113178, 0.115725, 0.135570,
 0.120776, 0.118409, 0.126532, 0.103810]
```

random6000 centroid 的首要正/负 feature 及一条最近/最远代表如下；完整 nearest、
boundary、farthest 清单及 replay 命令保存在 `cluster_review_motions.csv`：

| Cluster | Top positive | Top negative | Nearest example | Farthest/outlier example |
| ---: | --- | --- | --- | --- |
| 0 | `double_support_ratio__duration_weighted_mean` | `end_effector_speed_p95__p90` | `standing_while_Taking_the_bus_2...` | `play_trombone_10...` |
| 1 | `single_support_ratio__duration_weighted_mean` | `double_support_ratio__duration_weighted_mean` | `Knee_Tuck_To_Kick_L...` | `Shaolin_Kung_Fu...kicks...` |
| 2 | `joint_acceleration_p95__p90` | `foot_swing_speed_p95__duration_weighted_mean` | `humanml/013098...` | `humanml/000668...` |
| 3 | `contact_switch_rate_per_second__duration_weighted_mean` | `joint_speed_p95__p90` | `Shaolin_KungFu_Staff...` | `Emotion_Happy...` |
| 4 | `single_support_ratio__duration_weighted_mean` | `double_support_ratio__duration_weighted_mean` | `Butt_Kicks_Toe_Touch...` | `figure_skate_I_spin...` |
| 5 | `body_height_std__p90` | `double_support_ratio__duration_weighted_mean` | `Single_To_Double_Butt_Kicks...` | `Short_Weapon_Nachahmung...` |
| 6 | `contact_switch_rate_per_second__duration_weighted_mean` | `double_support_ratio__duration_weighted_mean` | `walking_and_Removing_the_hat...` | `custom/squat_chunk_0017...` |
| 7 | `joint_acceleration_p95__p90` | `double_support_ratio__duration_weighted_mean` | `LAFAN1/run2/subject1_chunk_0042...` | `humanml/001600...` |

K 诊断仅用 Train：K=6/8/10/12 的 sampled silhouette 分别为
0.271090/0.273505/0.263156/0.224865，`auto_selected=false`；冻结配置仍为 K=8。
random6000 没有 near-constant feature，唯一 warning 是第 27 列使用了 zero-MAD
P05--P95 fallback scale 0.0377371。random100 的小簇 warning 和 fallback scale
0.0490337 均保留在其 summary。两份 summary 均明确 source category、quality、
difficulty score/bin 和 policy statistics 未参与拟合。

本轮实际产物位于：

```text
outputs/module4_clusters_random100_seed42_v1/
outputs/module4_clusters_random6000_seed42_v1/
```

出现 `<1%` 小簇、`>50%` 大簇、低 silhouette 或 centroid 过近时只记录 warning，不自动
换 K 或重聚类。

## 13. random100 GPU Pilot 与 resume

先用 `--dry-run` 审查完整 Hydra overrides：

```bash
cd /home/l/whole_body_tracking_new
conda activate hybrid_robot

python scripts/run_module4_gpu_pilots.py --dry-run --pilots A,B,C
```

确认 metadata 已构建并人工审查后，分别运行：

```bash
# Pilot A：Diversity only；cluster target + cluster 内 uniform Motion/start frame。
python scripts/run_module4_gpu_pilots.py --pilots A

# Pilot B：M6 对照；quality+difficulty+gap，diversity 关闭。
python scripts/run_module4_gpu_pilots.py --pilots B

# Pilot C：M7；M6 + cluster diversity。
python scripts/run_module4_gpu_pilots.py --pilots C
```

helper 默认 random100、32 envs、seed/sampler seed 42、500 iterations、warmup 50、
update interval 10、minimum segment observations 4、minimum motion episodes 2，并启用
4096 行 assignment trace。M6/M7 默认都使用 `quality_include_borderline=true`，使对照
口径一致，并与完整 random6000 的 5998 条 eligible Motion 契约一致；需要严格 pass-only
诊断时可显式传 `--no-quality-include-borderline`。允许的 Pilot iteration 范围是
1--2000；它不会启动正式
34000-iteration 训练。

helper 的 random100 debug `motion_probability_cap` 默认是 0.5，并在 A/B/C 中保持相同。
random100 各簇都少于 50 条 Motion；若沿用正式默认 0.02，则
`effective_cap(c)=1/n_c` 会使簇内分布只能严格均匀，Pilot C 就无法观察 warmup 后的
`G_motion` 适应。这个 debug override 不修改正式 random6000 配置默认值。

Pilot A 的配置名是 `segment_sampling.mode=uniform`；在 diversity sampler 内，它仍显式
按均匀 `P(s|m)` 抽一次 segment，再在该 segment 的合法 start frame 内均匀抽样。因此
A/C 都严格执行 `cluster -> motion -> segment -> start`，trace 记录的概率就是实际生成
分布。这个行为只属于 diversity-enabled 路径，不修改 M0--M6 的 legacy uniform RNG。

M7 checkpoint resume：

```bash
python scripts/run_module4_gpu_pilots.py --dry-run \
  --suite resume \
  --resume-run '<现有 M7 run directory 或 load_run pattern>' \
  --checkpoint model_500.pt \
  --resume-iterations 100

python scripts/run_module4_gpu_pilots.py \
  --suite resume \
  --resume-run '<现有 M7 run directory 或 load_run pattern>' \
  --checkpoint model_500.pt \
  --resume-iterations 100
```

Pilot 只验证 wiring：metadata match、各层概率 finite/sum=1、Reject start=0、warmup
行为、cluster coverage/预算、checkpoint 和 RNG 连续性。短程 Pilot 不能用来声明性能提升。

### 本轮 RTX 4060 实测

环境为 RTX 4060 Laptop GPU、driver 580.159.03、PyTorch 2.5.1+cu124。A/B/C 均使用
random100、16 env、seed 42、500 iterations、TensorBoard、本地 trace 和 debug cap 0.5；
三个训练进程均正常退出：

| Pilot | Run directory | Steps | Time | 关键结果 |
| --- | --- | ---: | ---: | --- |
| A / Diversity-only | `2026-07-24_10-19-43_module4_diversity_only_random100_seed42_debug500` | 192000 | 466.96 s | cluster match=1，coverage=1，probability sum error=0，最终 target L1=0.011165 |
| B / M6 | `2026-07-24_10-28-19_module4_m6_random100_seed42_debug500` | 192000 | 450.48 s | quality/difficulty match=1，Reject start=0，fallback=0，无 `diversity/*` tag |
| C / M7 | `2026-07-24_10-36-43_module4_m7_random100_seed42_debug500` | 192000 | 445.06 s | 三方 match=1，coverage=1，Reject start=0，probability sum error=0，fallback=0，最终 target L1=0.016917 |

M7 warmup 的 TensorBoard 实测为：

```text
step 49: warmup=1, max P(m|c)=0.333333, max P(s|m)=0.500000
step 50: warmup=0, max P(m|c)=0.396844, max P(s|m)=0.851949
cluster_0_target_share: 0.10530485 -> 0.10530485（不变）
```

4096 行 trace 的同窗口比较：

| 指标 | M6 隐式 cluster | M7 |
| --- | ---: | ---: |
| cluster coverage | 8/8 | 8/8 |
| target-observed L1 | 0.433282 | 0.045330 |
| max cluster share | 0.377686 | 0.183350 |
| min cluster share | 0.039307 | 0.100098 |
| Reject start / layout violation | 0 / 0 | 0 / 0 |

M7 trace 另有 `P(c)P(m|c)=P_global(m)` 违规 0、non-finite 行 0，并有 2731 行已使用
warmup 后的非均匀条件 Motion 概率。B 的 trace 仍是原 8 列；A/C 为 13 列。

随后从 C 的 `model_499.pt` 运行 100-iteration resume：

```text
run: 2026-07-24_10-45-08_module4_m7_random100_seed42_resume100
steps: 38400
time: 95.85 s
learning iteration: 499 -> 598
cluster sample total: 27010 -> 34249
probability update count: 46 -> 55
quality/difficulty/cluster match: 1/1/1
coverage=1, Reject start=0, probability sum error=0, fallback=0
```

这证明 GPU 路径能恢复 cluster 概率、累计计数和更新 cadence；专用 sampler RNG 的逐 assignment
精确连续性另由 CPU checkpoint 测试验证。较早的 `10-11-03` A run 是修复 uniform Segment
显式抽样前的中止产物，不计入本节验收。

## 14. 人工 cluster 复查

`cluster_review_motions.csv` 应为每簇提供至少：

- 距 centroid 最近 10 条；
- cluster 边界 5 条；
- 最远/outlier 5 条；
- 随机 10 条。

逐行检查 motion path、centroid distance、top centroid feature、difficulty/quality/source
诊断和 builder 给出的 replay 命令。人工目标是判断：

- 同簇是否是相似的最终 G1 运动，而不是相同来源目录；
- 是否有一个 cluster 只包含异常数据；
- 极小/极大 cluster 是否合理；
- centroid 代表 Motion 与 outlier 是否符合 feature 解释。

可给 cluster 写“慢速移动”“腾空”等临时论文描述，但 runtime/metadata 始终只使用
`cluster_0..cluster_7`；不得据此重命名或重新编号。

## 15. M7 与 M6 的唯一差异

正式对照必须满足：

```text
M7 = M6 + frozen cluster diversity layer before Motion sampling
```

两者必须使用相同 quality metadata/config、difficulty metadata、`G_motion`、`G_local`、
EMA、warmup、update interval、PPO、seed、环境数和总训练步数。唯一变化是：

```text
M6: global P(m) -> P(s|m)
M7: P(c) -> P(m|c) -> P(s|m)
```

sampler-level 的 `num_clusters=1` 回归夹具中 `P(c)=1`，M7 的 Motion/Segment 概率在
浮点误差内等于 M6；正式配置仍要求至少两个 cluster。diversity disabled 的 M0--M6
不读取 metadata，assignment trace header 仍与模块三一致。

## 16. 性能与开销

在本机对完整 random6000 重跑 builder（包含 K=6/8/10/12 诊断）：

```text
elapsed = 17.56 s
max RSS = 188208 KiB
9 个输出文件逐字节复现
runtime NPZ = 2.3 MiB
```

GPU Pilot 平均 iteration wall time 为 A 0.934 s、B 0.901 s、C 0.890 s；当前短程噪声下
没有观察到 cluster 层带来的可分辨吞吐下降，也不据此声明性能加速。运行时不执行 scaler、
PCA 或 KMeans，只加载冻结 ID 并维护长度为 K/N/S 的概率、计数和 mask。

## 17. Provisional 参数、已知限制与正式冻结清单

当前以下参数均为 provisional，而非论文定稿：

```text
K = 8
17 source / 30 expanded features
aggregation definitions
optional coverage = 0.75
robust clip = 5
P05--P95 MAD fallback
PCA = off
n_init = 20
minimum cluster size warning = 20 motions
minimum cluster budget fraction = 0.5
cluster size exponent = 0.5
```

已知限制：

- v1 只实现固定 target；`count_aware_correction=true` 明确拒绝；
- v1 只实现 `sqrt_size_with_floor` runtime budget mode；
- silhouette/K 诊断只适合 Train，不能用 Validation/Test 调参；
- optional 列只要 Train coverage 不完整就从距离空间移除，尚未实现显式 missingness model；
- NumPy KMeans 适合当前 6000 Motion 规模，但不是通用大规模/流式聚类器；
- cluster 名称没有语义，人工描述不参与运行时；
- assignment resume 连续不等于完整 simulator rollout bitwise 连续；
- 本文档中的 builder/Pilot 数值只对应本轮列出的冻结产物与 run directory；
- Test 集不得用于聚类拟合、K/feature/budget 选择或本阶段 Pilot。

正式实验前必须冻结并归档：

- Train manifest、6000 Motion/21575 segment mapping 与 pool fingerprint；
- 模块一 quality NPZ/config、模块二 difficulty NPZ/Profile；
- 17 source feature、aggregation、unit、required/optional 与 scaler/fallback；
- PCA 开关/维数（若启用）、K、seed、KMeans 参数与 canonicalization；
- cluster Profile、metadata NPZ、config SHA256、Git commit 和每簇 size；
- `minimum_budget_fraction_of_uniform`、`cluster_size_exponent`、cap、warmup 和 update cadence；
- M6/M7 除 cluster 层外完全一致的训练配置；
- random6000 cluster summary、representative/outlier replay review 和 warning 处置记录；
- CPU 全回归、diversity-off trace、`K=1` M7=M6、factorization 与 resume 证据；
- Pilot A/B/C 的 W&B/trace/checkpoint 验收结果；
- 正式评测协议和 Test 不参与开发选择的声明。
