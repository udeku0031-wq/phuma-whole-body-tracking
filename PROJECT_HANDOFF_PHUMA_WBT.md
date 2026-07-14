# PHUMA + Whole Body Tracking 项目交接文档

更新时间：2026-07-14  
项目路径：`/home/l/whole_body_tracking_new`

## 1. 项目目标和当前进度

本项目目标是把 `HybridRobotics/whole_body_tracking` 从官方默认的单 motion tracking 流程，扩展成可以使用 PHUMA 大规模 G1 动作数据集训练的 multi-motion whole-body tracking 框架。

官方 `whole_body_tracking` 默认流程是：

```text
一个 motion.npz -> 一次训练 -> 一个 tracking policy
```

当前本地项目已经扩展为：

```text
多个 PHUMA .npy
    -> 转换成 whole_body_tracking 可读取的 .npz
    -> 生成 motion manifest 清单
    -> 多阶段 curriculum training
    -> 得到一个共享的 G1 multi-motion tracking policy
```

当前进度：

- PHUMA G1 原始数据已转换完成，共 `76086` 个动作片段。
- 转换后的 WBT 格式数据位于 `PHUMA_wbt_motions/g1_all`，约 `19G`。
- 已完成 4 阶段训练。
- 最终模型已经可以通过 `play.py` 加载并播放，视频测试效果可用。
- 当前模型属于“多动作跟踪策略”，不是高层语义任务策略。

## 2. 目前已经实现的效果

已经实现：

- PHUMA 原始 `.npy` 到 WBT `.npz` 的自动转换。
- 支持单个 `.npz`、目录、`.txt` manifest 三种 motion 输入。
- 支持多 motion library 训练。
- 支持每个并行环境随机抽取 motion id 和随机起始帧。
- 支持从上一阶段 checkpoint 继续训练下一阶段。
- 支持 headless 数值测试。
- 支持 headless 录制视频。
- 已完成 PHUMA 四阶段训练，最终 checkpoint 可正常播放。

最终模型可以理解为：

```text
PHUMA 多源动作库训练出的 Unitree G1 全身动作跟踪策略
```

它能根据输入的参考 motion 模仿多类别动作，例如基础行走、fitness、humanml、kungfu、music、perform、dance 等动作片段。它不会自主理解“跳舞/搬东西”等语言任务，只会跟踪给定参考动作轨迹。

## 3. 关键文件路径和作用

### 数据和清单

```text
PHUMA/data/g1
```

PHUMA 原始 G1 数据目录，里面是 `.npy` 动作文件。该目录按数据来源/动作类别划分，例如 `humanml`、`fitness`、`LAFAN1`、`dance` 等。

```text
PHUMA_wbt_motions/g1_all
```

转换后的 WBT 格式动作数据目录。结构基本继承 `PHUMA/data/g1`，只是 `.npy` 被转换成 `.npz`。

```text
PHUMA_wbt_motions/manifests/stage1_easy_1000.txt
PHUMA_wbt_motions/manifests/stage2_core_2000.txt
PHUMA_wbt_motions/manifests/stage3_mixed_4000.txt
PHUMA_wbt_motions/manifests/stage4_all_6000.txt
```

四阶段训练使用的 motion 清单。每一行是一个 `.npz` 路径。

### 脚本

```text
scripts/phuma_to_npz.py
```

PHUMA 到 WBT 格式的核心转换脚本。读取 PHUMA `.npy`，输出 WBT 可训练的 `.npz`。主要处理：

- `root_trans`
- `root_ori`
- `dof_pos`
- G1 23 DOF 到 WBT 29 DOF 的关节补齐
- Isaac 中 forward kinematics 后记录 body pose / velocity

```text
scripts/batch_csv_to_npz.py
```

批量转换 CSV motion 的辅助脚本，主要用于原始 LAFAN1/CSV motion 到 WBT motion 的处理。

```text
scripts/csv_to_npz.py
```

官方 CSV 到 `.npz` 的转换脚本，已做过稳定性和批处理相关修改。

```text
scripts/replay_npz.py
```

用于测试 `.npz` motion 文件是否能被 Isaac / WBT 正常加载和回放。现在支持本地 `--motion_file`。

```text
scripts/rsl_rl/train.py
```

训练入口。原版主要依赖 `--registry_name` 从 WandB 下载单个 motion；当前版本支持 `--motion_file` 加载本地单个 `.npz`、目录或 manifest。

```text
scripts/rsl_rl/play.py
```

播放/测试训练好的 policy。当前版本支持 `--motion_file`、`--video`、`--max_steps`、`--progress_interval`、`--skip_export`。

### 核心环境代码

```text
source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py
```

最关键的修改文件。实现 multi-motion loader 和 multi-motion sampling。

主要类：

- `MotionLoader`：加载单个 `.npz`、目录或 manifest，并拼接成 motion library。
- `MotionCommand`：给每个环境采样 motion id 和起始 time step。

核心逻辑：

```text
motion library 中有 N 个 motion
每个 env reset 时随机抽一个 motion_id
再从该 motion 的帧范围中随机抽起始 time_step
训练时 time_step 逐步加 1
到 motion 末尾后重新采样
```

```text
source/whole_body_tracking/whole_body_tracking/tasks/__init__.py
```

任务注册和导入相关修改，减少不必要 task import 对当前环境的影响。

```text
source/whole_body_tracking/whole_body_tracking/utils/my_on_policy_runner.py
```

RSL-RL runner 兼容性相关修改，用于适配当前环境中的 `rsl_rl` 版本。

## 4. 已经修改过的代码

主要修改内容如下：

1. 增加 PHUMA 转换脚本：

```text
scripts/phuma_to_npz.py
```

2. 增加批量 CSV 转换脚本：

```text
scripts/batch_csv_to_npz.py
```

3. 修改训练入口：

```text
scripts/rsl_rl/train.py
```

修改点：

- `--registry_name` 不再是唯一输入。
- 新增 `--motion_file`。
- 支持本地 `.npz`、目录、manifest。

4. 修改播放入口：

```text
scripts/rsl_rl/play.py
```

修改点：

- 支持本地 `--motion_file`。
- 支持 `--max_steps`。
- 支持 `--progress_interval`。
- 支持 `--skip_export`。
- 支持 headless video。

5. 修改 replay 脚本：

```text
scripts/replay_npz.py
```

修改点：

- 支持本地 `.npz`。
- 支持 registry 和 local motion 二选一。

6. 修改 motion command：

```text
source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py
```

修改点：

- 支持多 motion 加载。
- 支持 manifest。
- 支持目录递归加载。
- 支持 motion id 采样。
- 支持每个 env 不同 motion / time step。
- 增加数据 shape 校验。

7. 修改 runner / task import：

```text
source/whole_body_tracking/whole_body_tracking/utils/my_on_policy_runner.py
source/whole_body_tracking/whole_body_tracking/tasks/__init__.py
```

用于适配本地 IsaacLab / rsl_rl 环境。

## 5. 当前可运行命令

建议所有 Isaac 相关命令都使用：

```bash
env -u PYTHONPATH -u LD_LIBRARY_PATH python ...
```

这样可以避免 ROS/Gazebo 或其他环境变量污染 Isaac。

### 5.1 转换全部 PHUMA 数据

```bash
cd ~/whole_body_tracking_new
conda activate hybrid_robot

env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/phuma_to_npz.py \
  --input_dir PHUMA/data/g1 \
  --pattern "*.npy" \
  --output_dir PHUMA_wbt_motions/g1_all \
  --output_fps 50 \
  --headless \
  --device cuda:0 \
  --compressed
```

当前已经转换完成，一般不需要重复执行。

### 5.2 生成四阶段训练清单

```bash
cd ~/whole_body_tracking_new
conda activate hybrid_robot

mkdir -p PHUMA_wbt_motions/manifests
ROOT="$(pwd)/PHUMA_wbt_motions/g1_all"

find "$ROOT/LocoMuJoCo" "$ROOT/LAFAN1" \
  -name "*.npz" | shuf -n 1000 \
  > PHUMA_wbt_motions/manifests/stage1_easy_1000.txt

find "$ROOT/LocoMuJoCo" "$ROOT/LAFAN1" "$ROOT/humanml" "$ROOT/fitness" "$ROOT/EgoBody" \
  -name "*.npz" | shuf -n 2000 \
  > PHUMA_wbt_motions/manifests/stage2_core_2000.txt

find "$ROOT/LocoMuJoCo" "$ROOT/LAFAN1" "$ROOT/humanml" "$ROOT/fitness" "$ROOT/EgoBody" \
     "$ROOT/GRAB" "$ROOT/aist" "$ROOT/dance" "$ROOT/kungfu" "$ROOT/perform" "$ROOT/music" \
  -name "*.npz" | shuf -n 4000 \
  > PHUMA_wbt_motions/manifests/stage3_mixed_4000.txt

find "$ROOT" -name "*.npz" | shuf -n 6000 \
  > PHUMA_wbt_motions/manifests/stage4_all_6000.txt
```

当前清单数量：

```text
stage1_easy_1000.txt     1000
stage2_core_2000.txt     2000
stage3_mixed_4000.txt    4000
stage4_all_6000.txt      6000
```

### 5.3 阶段 1 训练

```bash
cd ~/whole_body_tracking_new
conda activate hybrid_robot

env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/rsl_rl/train.py \
  --task Tracking-Flat-G1-v0 \
  --motion_file PHUMA_wbt_motions/manifests/stage1_easy_1000.txt \
  --headless \
  --logger wandb \
  --log_project_name whole_body_tracking_phuma \
  --run_name phuma_stage1_easy1000 \
  --num_envs 1024 \
  --max_iterations 10000
```

### 5.4 阶段 2 训练

本次实际使用的阶段 1 权重：

```text
logs/rsl_rl/g1_flat/2026-07-08_22-53-52_phuma_stage1_easy1000_resume3500/model_6000.pt
```

命令：

```bash
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/rsl_rl/train.py \
  --task Tracking-Flat-G1-v0 \
  --motion_file PHUMA_wbt_motions/manifests/stage2_core_2000.txt \
  --headless \
  --logger wandb \
  --log_project_name whole_body_tracking_phuma \
  --run_name phuma_stage2_core2000_from6000 \
  --num_envs 1024 \
  --max_iterations 14000 \
  --resume True \
  --load_run 2026-07-08_22-53-52_phuma_stage1_easy1000_resume3500 \
  --checkpoint model_6000.pt
```

### 5.5 阶段 3 训练

本次实际使用的阶段 2 权重：

```text
logs/rsl_rl/g1_flat/2026-07-09_00-44-31_phuma_stage2_core2000_from6000/model_13999.pt
```

命令：

```bash
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/rsl_rl/train.py \
  --task Tracking-Flat-G1-v0 \
  --motion_file PHUMA_wbt_motions/manifests/stage3_mixed_4000.txt \
  --headless \
  --logger wandb \
  --log_project_name whole_body_tracking_phuma \
  --run_name phuma_stage3_mixed4000_from14000 \
  --num_envs 1024 \
  --max_iterations 23999 \
  --resume True \
  --load_run 2026-07-09_00-44-31_phuma_stage2_core2000_from6000 \
  --checkpoint model_13999.pt
```

### 5.6 阶段 4 训练

本次实际使用的阶段 3 权重：

```text
logs/rsl_rl/g1_flat/2026-07-09_06-33-32_phuma_stage3_mixed4000_from14000/model_23998.pt
```

命令：

```bash
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/rsl_rl/train.py \
  --task Tracking-Flat-G1-v0 \
  --motion_file PHUMA_wbt_motions/manifests/stage4_all_6000.txt \
  --headless \
  --logger wandb \
  --log_project_name whole_body_tracking_phuma \
  --run_name phuma_stage4_all6000_from24000 \
  --num_envs 1024 \
  --max_iterations 33998 \
  --resume True \
  --load_run 2026-07-09_06-33-32_phuma_stage3_mixed4000_from14000 \
  --checkpoint model_23998.pt
```

### 5.7 数值测试最终模型

```bash
cd ~/whole_body_tracking_new
conda activate hybrid_robot

env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/rsl_rl/play.py \
  --task Tracking-Flat-G1-v0 \
  --motion_file PHUMA_wbt_motions/manifests/stage4_all_6000.txt \
  --num_envs 1 \
  --headless \
  --max_steps 2000 \
  --progress_interval 200 \
  --load_run 2026-07-09_22-42-36_phuma_stage4_all6000_from24000 \
  --checkpoint model_33997.pt \
  --skip_export
```

### 5.8 录制最终模型视频

单个较长 dance motion 示例：

```bash
cd ~/whole_body_tracking_new
conda activate hybrid_robot

env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/rsl_rl/play.py \
  --task Tracking-Flat-G1-v0 \
  --motion_file PHUMA_wbt_motions/g1_all/dance/subset_0000/Apink_Mr_Chu_chunk_0004.npz \
  --num_envs 1 \
  --headless \
  --video \
  --video_length 220 \
  --max_steps 220 \
  --progress_interval 50 \
  --rendering_mode performance \
  --load_run 2026-07-09_22-42-36_phuma_stage4_all6000_from24000 \
  --checkpoint model_33997.pt \
  --skip_export
```

视频输出目录：

```text
logs/rsl_rl/g1_flat/2026-07-09_22-42-36_phuma_stage4_all6000_from24000/videos/play/
```

### 5.9 回放/检查单个 motion

```bash
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py \
  --motion_file PHUMA_wbt_motions/g1_all/dance/subset_0000/Apink_Mr_Chu_chunk_0004.npz \
  --headless \
  --max_steps 100 \
  --progress_interval 50
```

## 6. 数据集路径、motion 格式、checkpoint 路径

### 6.1 数据集路径

```text
PHUMA/data/g1
```

PHUMA 原始 G1 `.npy` 数据，约 `2.3G`。

```text
PHUMA_wbt_motions/g1_all
```

转换后的 WBT `.npz` 数据，约 `19G`。

```text
PHUMA_wbt_motions/manifests
```

训练清单目录，约 `1.4M`。

### 6.2 PHUMA 原始 `.npy` 格式

PHUMA 原始文件是 Python dict 格式，主要字段：

```text
root_trans    (T, 3)
root_ori      (T, 4)
dof_pos       (T, 23) 或 (T, 29)
fps           scalar
```

其中 `T` 是帧数。

### 6.3 WBT 转换后 `.npz` 格式

当前 WBT motion `.npz` 主要字段：

```text
fps                 (1,)
joint_pos           (T, 29)
joint_vel           (T, 29)
body_pos_w          (T, 30, 3)
body_quat_w         (T, 30, 4)
body_lin_vel_w      (T, 30, 3)
body_ang_vel_w      (T, 30, 3)
joint_names         (29,)
body_names          (30,)
source_file         scalar
source_format       scalar
```

训练时：

- 一个 `.npz` 约等于一个 motion clip。
- manifest 中每一行是一个 `.npz`。
- 多个 `.npz` 被加载为 motion library。
- 每个环境 reset 时随机抽 motion id 和起始帧。

### 6.4 最终 checkpoint

最终模型：

```text
logs/rsl_rl/g1_flat/2026-07-09_22-42-36_phuma_stage4_all6000_from24000/model_33997.pt
```

上一阶段关键 checkpoint：

```text
logs/rsl_rl/g1_flat/2026-07-08_22-53-52_phuma_stage1_easy1000_resume3500/model_6000.pt
logs/rsl_rl/g1_flat/2026-07-09_00-44-31_phuma_stage2_core2000_from6000/model_13999.pt
logs/rsl_rl/g1_flat/2026-07-09_06-33-32_phuma_stage3_mixed4000_from14000/model_23998.pt
logs/rsl_rl/g1_flat/2026-07-09_22-42-36_phuma_stage4_all6000_from24000/model_33997.pt
```

## 7. 已知问题和注意事项

### 7.1 不要删除 `~/docker/isaac-sim`

虽然当前 Isaac 主要从 conda 环境加载，但实测将 `~/docker` 改名后会触发：

```text
carb::cpp::bad_optional_access
Fatal Python error: Aborted
```

恢复 `~/docker` 后正常。因此暂时不要删除 `~/docker/isaac-sim`。

### 7.2 8GB 显存容易 OOM

当前机器显存有限，建议：

- 训练使用 `--num_envs 1024`。
- 如果 OOM，改成 `--num_envs 512`。
- 播放视频时关闭 Firefox / VSCode / 其他 Isaac 进程。
- 视频测试尽量用单个 motion，不要一次加载全部 `g1_all`。

### 7.3 不要直接上传数据集到 Git

以下目录很大，不应该提交到 Git/Gitee：

```text
PHUMA/
PHUMA_wbt_motions/
LAFAN1_Retargeting_Dataset/
logs/
wandb/
artifacts/
```

这些目录已加入 `.gitignore`。

### 7.4 PHUMA motion 多数很短

很多 PHUMA chunk 只有几十到两百帧。50 FPS 下：

```text
100 帧 = 2 秒
200 帧 = 4 秒
```

因此视频只有 2-4 秒是正常现象，不一定是播放失败。

### 7.5 当前 stage4 是按文件随机抽样，不是类别均衡抽样

`stage4_all_6000.txt` 是从全部 `76086` 个 `.npz` 中随机抽 `6000` 个，因此大类别占比更高，小类别样本较少。

当前 stage4 类别分布：

```text
humanml       1807
fitness       1450
idea400        785
game_motion    486
music          234
LAFAN1         179
EgoBody        161
kungfu         160
perform        158
haa500         149
GRAB           146
aist           117
LocoMuJoCo      63
custom          46
humman          37
animation       13
dance            9
```

这意味着当前最终模型更偏向 `humanml`、`fitness`、`idea400` 等大类，对 `dance`、`animation` 等小类覆盖较少。

### 7.6 WandB 状态可能显示 Crashed

训练结束时 Isaac / WandB 有时会显示 `Crashed` 或 UsdContext warning，但 checkpoint 已保存且能正常 play。判断是否成功以本地 checkpoint 和 play 测试为准。

### 7.7 Debug 箭头不是模型错误

play 视频中红绿蓝箭头是 motion/body/velocity 等调试可视化标记，不代表训练失败。

## 8. 下一步新功能建议

接下来如果要在“数据处理”上做创新，建议优先做以下方向。

### 8.1 高质量 motion 筛选

目标：从 `76086` 个 `.npz` 中筛出更适合训练的 motion。

可以设计评分指标：

- 帧数是否太短。
- root 位置是否突变。
- 关节速度是否异常大。
- body velocity 是否异常大。
- 脚底是否穿地。
- 身体高度是否异常。
- 关节是否接近极限。
- motion 是否有明显跳帧/不连续。

输出：

```text
PHUMA_wbt_motions/manifests/stage4_clean_6000.txt
```

然后从当前最终模型继续微调。

### 8.2 类别均衡采样

当前 stage4 是纯随机抽样，大类占比过高。可以改成：

```text
每类最多抽 K 个
小类全部保留
大类限量采样
```

例如：

```text
humanml: 600
fitness: 600
idea400: 600
dance: 全部 117
animation: 全部 115
LocoMuJoCo: 全部 794 或抽 600
```

这样更适合验证“多类别动作泛化”。

### 8.3 难度分级 curriculum

不要只按文件夹分阶段，可以按 motion 质量和难度自动分级：

```text
easy: 低速度、低角速度、接触稳定、短步态
medium: 日常动作、fitness、humanml
hard: dance、kungfu、music、perform、大幅度上肢动作
```

可以生成：

```text
stage1_auto_easy.txt
stage2_auto_medium.txt
stage3_auto_hard.txt
stage4_auto_balanced.txt
```

### 8.4 片段级采样权重

当前采样逻辑是随机 motion + 随机起始帧。后续可以改成：

```text
高质量时间片更容易被采样
失败率高但仍合理的片段提高采样
异常片段降低采样或过滤
```

这比只筛整个 `.npz` 更细。

### 8.5 动作类别标签与条件化训练

当前模型不知道 motion 来自哪个类别。可以给 motion 增加 category label：

```text
dance / fitness / humanml / LAFAN1 / ...
```

然后尝试：

- 在 observation 中加入 category embedding。
- 训练 category-conditioned policy。
- 测试同一模型在不同类别下的表现差异。

### 8.6 自动评测脚本

建议写一个 batch evaluation 脚本，对多个类别分别测试：

```text
success rate
mean episode length
body_pos error
body_rot error
joint_pos error
termination reason
```

输出 CSV 或 JSON，方便写论文/报告。

### 8.7 视频自动抽样评测

每个类别自动挑选若干 motion 录制视频：

```text
videos/eval/dance/
videos/eval/fitness/
videos/eval/kungfu/
videos/eval/music/
```

这样可以直观看出哪些类别学得好，哪些类别需要重新采样或微调。

## 9. 当前结论

本项目已经完成从单 motion whole-body tracking 到 PHUMA multi-motion tracking 的核心扩展，代码链路已经跑通，数据转换、训练、测试、视频播放均已验证。

后续最有价值的工作不是继续盲目扩大数据量，而是围绕 PHUMA 数据做更聪明的数据处理：

```text
质量筛选
类别均衡
难度分级
片段级采样
自动评测
```

这些方向更容易形成明确的创新点，也更可能提升最终模型的稳定性和泛化效果。
