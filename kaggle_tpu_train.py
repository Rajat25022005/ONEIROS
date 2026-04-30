"""
kaggle_tpu_train.py — Kaggle TPU v5e-8 Training Launcher for Hypnos

Run this in a Kaggle notebook with 'TPU v5e-8' selected as accelerator.
Ensure the ONEIROS repo is uploaded as a dataset or cloned via git.

Usage (in notebook cell):
    %run kaggle_tpu_train.py
"""

import os
import sys

# ── TPU environment ───────────────────────────────────────────────────
os.environ["PJRT_DEVICE"] = "TPU"
os.environ["XLA_USE_BF16"] = "1"
os.environ["XLA_TENSOR_ALLOCATOR_MAXSIZE"] = "100000000"

# Ensure project root is on path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

print("==============================================")
print(" Hypnos — TPU v5e-8 Training Pipeline")
print(f" Project root: {project_root}")
print(f" PJRT_DEVICE:  {os.environ.get('PJRT_DEVICE')}")
print(f" XLA_USE_BF16: {os.environ.get('XLA_USE_BF16')}")
print("==============================================")
print()

# ── Stage 1: JEPA pre-training ────────────────────────────────────────
print("[Pipeline] Starting Stage 1: JEPA training...")
from train.stage1_jepa_tpu import train_stage1_tpu

train_stage1_tpu("configs/hypnos_1.4b_tpu.yaml")
print("[Pipeline] Stage 1 complete.")
print()

# ── Stage 2: Decoder training ─────────────────────────────────────────
print("[Pipeline] Starting Stage 2: Decoder training...")
from train.stage2_decoder_tpu import train_stage2_tpu

train_stage2_tpu("configs/hypnos_1.4b_tpu.yaml")
print("[Pipeline] Stage 2 complete.")
print()

print("==============================================")
print(" All stages complete.")
print(" Checkpoints saved to checkpoints/")
print("==============================================")
