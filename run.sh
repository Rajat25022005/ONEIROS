#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/base.yaml}"
WANDB="${WANDB:-false}"
SEED="${SEED:-42}"

WANDB_FLAG=""
if [ "$WANDB" = "true" ]; then
  WANDB_FLAG="--wandb"
fi

# ── Detect runtime: TPU or GPU ───────────────────────────────────────
if [ "${PJRT_DEVICE:-}" = "TPU" ]; then
  # ── TPU training path ─────────────────────────────────────────────
  export XLA_USE_BF16=1
  export XLA_TENSOR_ALLOCATOR_MAXSIZE=100000000

  echo "=============================================="
  echo " Hypnos — TPU Training Pipeline"
  echo " Config  : $CONFIG"
  echo " Device  : TPU (PJRT)"
  echo "=============================================="
  echo ""

  echo "[Stage 1] JEPA training (TPU)..."
  python -m train.stage1_jepa_tpu --config "$CONFIG"

  echo ""
  echo "[Stage 2] Decoder training (TPU)..."
  python -m train.stage2_decoder_tpu --config "$CONFIG"

  echo ""
  echo "Done. Checkpoints saved to checkpoints/"

else
  # ── GPU / CPU training path (original) ─────────────────────────────
  echo "=============================================="
  echo " Hypnos — GPU/CPU Training Pipeline"
  echo " Config  : $CONFIG"
  echo " Seed    : $SEED"
  echo "=============================================="
  echo ""

  echo "[Stage 1] JEPA training..."
  python -m train.stage1_jepa --config "$CONFIG"

  echo ""
  echo "[Stage 2] Decoder training..."
  python -m train.stage2_decoder --config "$CONFIG"

  echo ""
  echo "Done. Checkpoints saved to checkpoints/"
fi
