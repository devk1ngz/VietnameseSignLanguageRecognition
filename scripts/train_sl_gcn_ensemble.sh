#!/usr/bin/env bash
#
# Train the 4 SL-GCN streams sequentially, then run the multi-stream ensemble
# evaluation. Designed to run inside a tmux session on GPU 1 with a small
# dataloader worker count so it does not disturb other users' jobs.
#
#   tmux new -s sl_gcn_ens -d 'bash scripts/train_sl_gcn_ensemble.sh'
#
set -uo pipefail

REPO=/data_hdd_16t/namvu2/VietnameseSignLanguageRecognition

# --- Environment (see project memory: HF_HOME/PATH gotcha) ---
export HF_HOME="$REPO/hf_cache"                 # writable cache in the repo
export PATH="$REPO/.venv/bin:$PATH"             # use the project venv
export CUDA_VISIBLE_DEVICES=1                    # train on GPU 1 only
export TOKENIZERS_PARALLELISM=false

cd "$REPO/src" || exit 1

STREAMS=(
  sl_gcn_joint_multicam
  sl_gcn_bone_multicam
  sl_gcn_joint_motion_multicam
  sl_gcn_bone_motion_multicam
)

echo "########## SL-GCN multi-stream ensemble pipeline ##########"
echo "GPU: $CUDA_VISIBLE_DEVICES | workers: 4 (set in configs) | start: $(date)"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader || true

# --- 1. Train each stream sequentially ---
for name in "${STREAMS[@]}"; do
  echo ""
  echo "===== [$(date)] TRAIN: $name ====="
  python train.py --config_path "configs/training/${name}.yaml"
  status=$?
  if [ $status -ne 0 ]; then
    echo "!!!!! [$(date)] $name FAILED (exit $status) — stopping pipeline." >&2
    exit $status
  fi
done

# --- 2. Ensemble evaluation (late fusion of softmax probabilities) ---
MODEL_PATHS=()
for name in "${STREAMS[@]}"; do
  MODEL_PATHS+=("experiments/${name}")
done

for split in validation test; do
  echo ""
  echo "===== [$(date)] ENSEMBLE EVAL ($split) ====="
  python ensemble_evaluate.py \
    --dataset visl_400 \
    --data_dir ../data/processed/vsl_400 \
    --subset cam_1_2_3 \
    --eval_set "$split" \
    --batch_size 64 \
    --model_paths "${MODEL_PATHS[@]}"
done

# --- 3. Fuse the 4 streams into a SINGLE deployable ONNX ---
echo ""
echo "===== [$(date)] EXPORT SINGLE ENSEMBLE ONNX ====="
python convert_ensemble_to_onnx.py \
  --streams "${MODEL_PATHS[@]}" \
  --gloss-csv ../data/processed/vsl_400/gloss.csv \
  --output experiments/sl_gcn_ensemble_multicam/sl_gcn_ensemble.onnx

echo ""
echo "########## [$(date)] ALL DONE ##########"
echo "Ensemble ONNX: src/experiments/sl_gcn_ensemble_multicam/sl_gcn_ensemble.onnx"
