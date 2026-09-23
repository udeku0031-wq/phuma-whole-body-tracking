#!/usr/bin/env bash
set -euo pipefail

cd /home/l/whole_body_tracking_new

if command -v conda >/dev/null 2>&1; then
  # Make `conda activate` available in non-interactive shells.
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate hybrid_robot
fi

PY="/home/l/miniconda3/envs/hybrid_robot/bin/python"

TRAIN_MANIFEST="PHUMA_wbt_motions/manifests/experiments/random_seed42/random6000_seed42.txt"
QUALITY_METADATA="outputs/module1_quality_random6000_seed42_original_v1/segment_quality_metadata.npz"
CLUSTER_METADATA="outputs/module4_clusters_random6000_seed42_v1/motion_cluster_metadata.npz"

export WANDB_MODE=online
export WANDB_USERNAME=longxianli222-northeastern-university
export WANDB_ENTITY=longxianli222-northeastern-university
export CUDA_VISIBLE_DEVICES=0
export __NV_PRIME_RENDER_OFFLOAD=1
export __GLX_VENDOR_LIBRARY_NAME=nvidia
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json

mkdir -p formal_logs

case "${1:-}" in
  q-only|Q-only|qonly)
    RUN_NAME="formal_v1_ablation_Qonly_n6000_trainseed42"
    RUN_ID="formal-v1-ablation-qonly-n6000-trainseed42"
    METHOD_NAME="Q-only"
    QUALITY_ENABLED=true
    QUALITY_PATH="$QUALITY_METADATA"
    QUALITY_EMPTY_POLICY="exclude"
    DIVERSITY_ENABLED=false
    DIVERSITY_PATH=""
    MOTION_MODE="raw_error"
    SEGMENT_MODE="raw_error"
    ;;
  d-only|D-only|donly)
    RUN_NAME="formal_v1_ablation_Donly_n6000_trainseed42"
    RUN_ID="formal-v1-ablation-donly-n6000-trainseed42"
    METHOD_NAME="D-only"
    QUALITY_ENABLED=false
    QUALITY_PATH=""
    QUALITY_EMPTY_POLICY="error"
    DIVERSITY_ENABLED=true
    DIVERSITY_PATH="$CLUSTER_METADATA"
    MOTION_MODE="raw_error"
    SEGMENT_MODE="raw_error"
    ;;
  globalraw|GlobalRaw|global-raw)
    RUN_NAME="formal_GlobalRaw_n6000_trainseed42"
    RUN_ID="formal-globalraw-n6000-trainseed42"
    METHOD_NAME="GlobalRaw"
    QUALITY_ENABLED=false
    QUALITY_PATH=""
    QUALITY_EMPTY_POLICY="error"
    DIVERSITY_ENABLED=false
    DIVERSITY_PATH=""
    MOTION_MODE="uniform"
    SEGMENT_MODE="global_bin_raw_error"
    ;;
  globalraw-q|GlobalRaw-Q|globalrawq)
    RUN_NAME="formal_v1_ablation_GlobalRawQ_n6000_trainseed42"
    RUN_ID="formal-v1-ablation-globalrawq-n6000-trainseed42"
    METHOD_NAME="GlobalRaw-Q"
    QUALITY_ENABLED=true
    QUALITY_PATH="$QUALITY_METADATA"
    QUALITY_EMPTY_POLICY="exclude"
    DIVERSITY_ENABLED=false
    DIVERSITY_PATH=""
    MOTION_MODE="uniform"
    SEGMENT_MODE="global_bin_raw_error"
    ;;
  all)
    "$0" q-only
    "$0" d-only
    "$0" globalraw
    "$0" globalraw-q
    exit 0
    ;;
  *)
    echo "Usage: $0 {q-only|d-only|globalraw|globalraw-q|all}" >&2
    exit 2
    ;;
esac

LOG_PATH="formal_logs/${RUN_NAME}.log"

echo "[INFO] Training ${RUN_NAME}"
echo "[INFO] method=${METHOD_NAME} motion=${MOTION_MODE} segment=${SEGMENT_MODE}"

WBT_DISABLE_ONNX_ON_SAVE=1 \
env -u PYTHONPATH -u LD_LIBRARY_PATH "$PY" \
  scripts/rsl_rl/train.py \
  --disable_fabric \
  --task Tracking-Flat-G1-v0 \
  --motion_file "$TRAIN_MANIFEST" \
  --headless \
  --logger wandb \
  --log_project_name whole_body_tracking_formal_v1 \
  --num_envs 3072 \
  --seed 42 \
  --max_iterations 34000 \
  --run_name "$RUN_NAME" \
  --wandb_run_name "$RUN_NAME" \
  --wandb_run_id "$RUN_ID" \
  --wandb_resume never \
  --device cuda:0 \
  env.sim.use_fabric=false \
  env.commands.motion.research.segment.enabled=true \
  env.commands.motion.research.segment.length_seconds=1.0 \
  env.commands.motion.research.method_name="$METHOD_NAME" \
  env.commands.motion.research.motion_sampling.mode="$MOTION_MODE" \
  env.commands.motion.research.segment_sampling.mode="$SEGMENT_MODE" \
  env.commands.motion.research.quality_gate.enabled="$QUALITY_ENABLED" \
  env.commands.motion.research.quality_gate.metadata_path="$QUALITY_PATH" \
  env.commands.motion.research.quality_gate.include_borderline=true \
  env.commands.motion.research.quality_gate.empty_motion_policy="$QUALITY_EMPTY_POLICY" \
  env.commands.motion.research.quality_gate.strict_metadata_match=true \
  env.commands.motion.research.difficulty_calibration.enabled=false \
  env.commands.motion.research.difficulty_calibration.metadata_path="" \
  env.commands.motion.research.difficulty_calibration.strict_metadata_match=true \
  env.commands.motion.research.difficulty_calibration.expected_num_bins=10 \
  env.commands.motion.research.joint_gap.enabled=false \
  env.commands.motion.research.diversity_constraint.enabled="$DIVERSITY_ENABLED" \
  env.commands.motion.research.diversity_constraint.metadata_path="$DIVERSITY_PATH" \
  env.commands.motion.research.diversity_constraint.strict_metadata_match=true \
  env.commands.motion.research.diversity_constraint.expected_num_clusters=8 \
  env.commands.motion.research.diversity_constraint.budget_mode=sqrt_size_with_floor \
  env.commands.motion.research.diversity_constraint.minimum_budget_fraction_of_uniform=0.5 \
  env.commands.motion.research.diversity_constraint.cluster_size_exponent=0.5 \
  env.commands.motion.research.diversity_constraint.diversity_during_warmup=true \
  env.commands.motion.research.diversity_constraint.count_aware_correction=false \
  env.commands.motion.research.online_learning.enabled=true \
  env.commands.motion.research.online_learning.statistics_enabled=true \
  env.commands.motion.research.online_learning.warmup_iterations=1000 \
  env.commands.motion.research.online_learning.probability_update_interval=50 \
  env.commands.motion.research.online_learning.min_segment_observations=32 \
  env.commands.motion.research.online_learning.min_motion_episodes=8 \
  env.commands.motion.research.online_learning.sampler_seed=42 \
  env.commands.motion.research.adaptive_sampling.uniform_mix=0.15 \
  env.commands.motion.research.adaptive_sampling.temperature=1.0 \
  env.commands.motion.research.adaptive_sampling.under_sampling_weight=0.25 \
  env.commands.motion.research.adaptive_sampling.motion_probability_cap=0.02 \
  env.commands.motion.research.adaptive_sampling.segment_probability_cap=1.0 \
  env.commands.motion.research.sampling_statistics.enabled=true \
  env.commands.motion.research.sampling_statistics.log_interval=100 \
  agent.save_interval=500 \
  2>&1 | tee -a "$LOG_PATH"
