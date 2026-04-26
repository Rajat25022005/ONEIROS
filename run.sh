#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/base.yaml}"
WANDB="${WANDB:-false}"
SEED="${SEED:-42}"

WANDB_FLAG=""
if [ "$WANDB" = "true" ]; then
  WANDB_FLAG="--wandb"
fi

# ── TPU environment ─────────────────────────────────────────────────────────
export PJRT_DEVICE=TPU          # tells PyTorch/XLA to use TPU backend
export XLA_USE_BF16=1           # use BF16 everywhere on TPU (free speedup)
export XLA_TENSOR_ALLOCATOR_MAXSIZE=100000000   # avoids OOM on large batches

echo "=============================================="
echo " Think-in-Silence — TPU Training Pipeline"
echo " Config  : $CONFIG"
echo " Seed    : $SEED"
echo " Device  : $PJRT_DEVICE"
echo "=============================================="
echo ""

echo "[Stage 1] JEPA training..."
python main.py --config "$CONFIG" --seed "$SEED" $WANDB_FLAG

echo ""
echo "[Stage 2] Decoder training..."
python train_decoder.py --config "$CONFIG" --seed "$SEED" $WANDB_FLAG

echo ""
echo "[Stage 3] Joint fine-tuning..."
python finetune.py --config "$CONFIG" --seed "$SEED" $WANDB_FLAG

echo ""
echo "[Eval] Running full evaluation..."
python eval.py --config "$CONFIG" --eval_type all

echo ""
echo "Done. Results saved to results/"
