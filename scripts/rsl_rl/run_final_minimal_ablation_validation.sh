#!/usr/bin/env bash
set -euo pipefail

cd /home/l/whole_body_tracking_new

if command -v conda >/dev/null 2>&1; then
  # Make `conda activate` available in non-interactive shells.
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate hybrid_robot
fi

PY="/home/l/miniconda3/envs/hybrid_robot/bin/python"
VALIDATION_MANIFEST="PHUMA_wbt_motions/manifests/splits_v1/validation_full.txt"
TEST_MANIFEST="PHUMA_wbt_motions/manifests/splits_v1/test.txt"
CHECKPOINTS="model_10000.pt,model_15000.pt,model_20000.pt,model_25000.pt,model_30000.pt,model_33500.pt,model_33999.pt"

export CUDA_VISIBLE_DEVICES=0
export __NV_PRIME_RENDER_OFFLOAD=1
export __GLX_VENDOR_LIBRARY_NAME=nvidia
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json

case "${1:-}" in
  q-only|Q-only|qonly)
    RUN_NAME="formal_v1_ablation_Qonly_n6000_trainseed42"
    OUTPUT_ROOT="evaluations/final_minimal/Qonly_seed42"
    ;;
  d-only|D-only|donly)
    RUN_NAME="formal_v1_ablation_Donly_n6000_trainseed42"
    OUTPUT_ROOT="evaluations/final_minimal/Donly_seed42"
    ;;
  globalraw|GlobalRaw|global-raw)
    RUN_NAME="formal_GlobalRaw_n6000_trainseed42"
    OUTPUT_ROOT="evaluations/final_minimal/GlobalRaw_seed42"
    ;;
  globalraw-q|GlobalRaw-Q|globalrawq)
    RUN_NAME="formal_v1_ablation_GlobalRawQ_n6000_trainseed42"
    OUTPUT_ROOT="evaluations/final_minimal/GlobalRawQ_seed42"
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

RUN_DIR="$(
  find logs/rsl_rl/g1_flat -maxdepth 1 -type d -name "*_${RUN_NAME}" | sort | tail -n 1
)"

if [ -z "$RUN_DIR" ]; then
  echo "No run directory found for ${RUN_NAME}" >&2
  exit 1
fi

echo "[INFO] Validation run_dir=${RUN_DIR}"
echo "[INFO] output_root=${OUTPUT_ROOT}"

env -u PYTHONPATH -u LD_LIBRARY_PATH "$PY" \
  scripts/rsl_rl/finalize_direct_random6000_eval.py \
  --task Tracking-Flat-G1-v0 \
  --run-dir "$RUN_DIR" \
  --checkpoints "$CHECKPOINTS" \
  --validation-manifest "$VALIDATION_MANIFEST" \
  --test-manifest "$TEST_MANIFEST" \
  --output-root "$OUTPUT_ROOT" \
  --num-envs 3072 \
  --seed 42 \
  --progress-interval 50 \
  --episode-length-s 60.0 \
  --device cuda:0 \
  --disable-fabric \
  --deterministic \
  --disable-randomization \
  --resume \
  --skip-test
