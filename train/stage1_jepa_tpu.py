"""
train/stage1_jepa_tpu.py — Stage 1: JEPA pre-training on TPU v5e-8

Two-phase training:
  Phase 1 (CPU): Pre-compute all backbone hidden states and cache them.
                 Mamba's sequential scan can't compile on XLA, so we run
                 it once on CPU and save the outputs.
  Phase 2 (TPU): Train ThoughtBlock on cached embeddings at full TPU speed.
                 Only the small ThoughtBlock + EMATeacher run on TPU.

This avoids the 18s/step bottleneck from running Mamba on CPU every step.

Usage (Kaggle notebook):
    from train.stage1_jepa_tpu import train_stage1_tpu
    train_stage1_tpu("configs/hypnos_1.4b_tpu.yaml")
"""

import os
import sys
import yaml
import argparse
import time
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ── JEPA loss with anti-collapse regularization ──────────────────────

def variance_loss(z: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    """
    Variance regularization (from VICReg).

    Forces each dimension of the representation to have std >= gamma.
    Without this, the model can collapse all representations to a single
    point and achieve zero invariance loss trivially.
    """
    std = z.std(dim=0)
    return F.relu(gamma - std).mean()


def covariance_loss(z: torch.Tensor) -> torch.Tensor:
    """
    Covariance regularization (from VICReg).

    Decorrelates the dimensions of the representation. Without this,
    the model can achieve low invariance loss by using only a few
    dimensions (dimensional collapse).
    """
    batch_size, dim = z.shape
    z_centered = z - z.mean(dim=0)
    cov = (z_centered.T @ z_centered) / (batch_size - 1)

    # zero out the diagonal (we only penalize off-diagonal correlations)
    off_diag = cov - torch.diag(cov.diag())
    return (off_diag ** 2).sum() / dim


def jepa_loss(
    z_student: torch.Tensor,
    z_teacher: torch.Tensor,
    var_weight: float = 1.0,
    cov_weight: float = 0.04,
) -> torch.Tensor:
    """
    JEPA loss with VICReg-style anti-collapse regularization.

    Components:
      1. Invariance: smooth L1 between normalized student/teacher (original)
      2. Variance:   force each latent dim to have non-trivial spread
      3. Covariance: decorrelate latent dims to prevent dimensional collapse

    Without (2) and (3), the loss collapses to ~0 as the student learns
    to map all inputs to the same point in latent space.
    """
    z_s = F.normalize(z_student, dim=-1)
    z_t = F.normalize(z_teacher, dim=-1)

    # invariance: student should match teacher
    inv_loss = F.smooth_l1_loss(z_s, z_t)

    # variance: prevent representation collapse
    var_loss = variance_loss(z_s) + variance_loss(z_t)

    # covariance: prevent dimensional collapse
    cov_loss = covariance_loss(z_s) + covariance_loss(z_t)

    total = inv_loss + var_weight * var_loss + cov_weight * cov_loss
    return total


# ── Phase 1: Pre-cache backbone embeddings ───────────────────────────

def precache_backbone_embeddings(backbone, tokenizer, config):
    """
    Run Mamba backbone once on CPU for all examples.
    Returns cached (hidden_ctx, hidden_tgt) tensors.

    This is the key optimization: instead of running the 1.4B backbone
    every training step (18s/step on CPU), we run it once upfront (~30 min)
    and then train ThoughtBlock purely on TPU (~0.1s/step).
    """
    from datasets import load_dataset

    train_cfg = config["training"]
    max_seq_len = train_cfg["max_seq_len"]

    print("[Stage1-TPU] ═══ Phase 1: Pre-caching backbone embeddings on CPU ═══")
    print("[Stage1-TPU] Loading and tokenizing dataset...")
    sys.stdout.flush()

    dataset_name = train_cfg.get("dataset", "HuggingFaceFW/fineweb-edu")
    dataset_config = train_cfg.get("dataset_config", None)
    dataset_split = train_cfg.get("dataset_split", "train")

    ds = load_dataset(
        dataset_name,
        name=dataset_config,
        split=dataset_split,
    )

    # Pre-tokenize
    def tokenize_fn(examples):
        tokens = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_seq_len,
            return_tensors=None,
        )
        return {"input_ids": tokens["input_ids"]}

    ds = ds.map(
        tokenize_fn,
        batched=True,
        batch_size=1000,
        remove_columns=ds.column_names,
        desc="Tokenizing",
    )
    ds.set_format("torch")

    all_ids = ds["input_ids"]
    mid = max_seq_len // 2
    n_examples = len(all_ids)

    print(f"[Stage1-TPU] Dataset: {n_examples} examples, seq_len={max_seq_len}")
    print(f"[Stage1-TPU] Running Mamba-1.4B on CPU for all examples...")
    print(f"[Stage1-TPU] This takes ~20-30 min but only happens once.")
    sys.stdout.flush()

    # Process in batches on CPU
    cache_batch_size = 8  # CPU batch size for backbone
    all_hidden_ctx = []
    all_hidden_tgt = []

    start = time.time()
    backbone.eval()

    for i in range(0, n_examples, cache_batch_size):
        batch_ids = all_ids[i : i + cache_batch_size]

        context_ids = batch_ids[:, :mid]
        target_ids = batch_ids[:, mid:]

        with torch.no_grad():
            h_ctx, _ = backbone.encode(context_ids)
            h_tgt, _ = backbone.encode(target_ids)

        # Mean-pool over sequence length to get fixed-size embeddings
        # Shape: (batch, hidden_size)
        all_hidden_ctx.append(h_ctx.mean(dim=1).cpu())
        all_hidden_tgt.append(h_tgt.mean(dim=1).cpu())

        done = min(i + cache_batch_size, n_examples)
        if done % 500 == 0 or done == n_examples:
            elapsed = time.time() - start
            rate = done / elapsed if elapsed > 0 else 0
            eta = (n_examples - done) / rate if rate > 0 else 0
            print(
                f"[Stage1-TPU]   Cached {done:>6d}/{n_examples} "
                f"({100*done/n_examples:.1f}%) | "
                f"{rate:.1f} ex/s | ETA: {eta:.0f}s"
            )
            sys.stdout.flush()

    hidden_ctx = torch.cat(all_hidden_ctx, dim=0)
    hidden_tgt = torch.cat(all_hidden_tgt, dim=0)

    elapsed = time.time() - start
    print(f"[Stage1-TPU] ═══ Phase 1 complete: {elapsed:.0f}s ═══")
    print(f"[Stage1-TPU] Cached shapes: ctx={hidden_ctx.shape}, tgt={hidden_tgt.shape}")
    sys.stdout.flush()

    return hidden_ctx, hidden_tgt


# ── Phase 2: Train ThoughtBlock on TPU ────────────────────────────────

def train_stage1_tpu(config_path: str):
    """
    Stage 1 JEPA training using SPMD on TPU v5e-8.

    Phase 1: Pre-cache backbone embeddings on CPU (~30 min, once)
    Phase 2: Train ThoughtBlock on TPU at full speed (~0.1s/step)
    """
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.runtime as xr
    import torch_xla.distributed.spmd as xs
    from torch_xla.distributed.spmd import Mesh

    with open(config_path) as f:
        config = yaml.safe_load(f)

    train_cfg = config["training"]
    model_cfg = config["model"]

    # ── Load backbone on CPU ──────────────────────────────────────────
    from hypnos.model.backbone import MambaBackbone

    backbone = MambaBackbone(
        model_cfg.get("backbone", "state-spaces/mamba-1.4b-hf"),
        use_slow_path=model_cfg.get("use_slow_path", True),
    )
    backbone.to(torch.device("cpu"))
    for param in backbone.parameters():
        param.requires_grad_(False)
    backbone.eval()

    # ── Phase 1: cache embeddings on CPU ──────────────────────────────
    hidden_ctx, hidden_tgt = precache_backbone_embeddings(
        backbone, backbone.tokenizer, config
    )

    # Free backbone memory — no longer needed
    del backbone
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    import gc; gc.collect()
    print("[Stage1-TPU] Backbone freed from memory.")

    # ── Phase 2: TPU training on cached embeddings ────────────────────
    print("[Stage1-TPU] ═══ Phase 2: Training ThoughtBlock on TPU ═══")

    # Enable SPMD mode
    xr.use_spmd()
    device = xm.xla_device()
    num_devices = xr.global_runtime_device_count()
    print(f"[Stage1-TPU] Device: {device} | TPU chips: {num_devices}")

    # Create 1D mesh for data parallelism
    mesh = Mesh(
        np.arange(num_devices),
        (num_devices,),
        ("data",),
    )

    # ── build trainable models (TPU only) ─────────────────────────────
    from hypnos.model.thought_block import ThoughtBlock
    from hypnos.model.ema_teacher import EMATeacher

    hidden_size = hidden_ctx.shape[-1]  # 2048

    thought_block = ThoughtBlock(
        input_dim=hidden_size,
        latent_dim=model_cfg.get("latent_dim", 512),
        k_steps=model_cfg.get("k_steps", 8),
    ).to(device)

    ema_teacher = EMATeacher(
        student=thought_block,
        tau=model_cfg.get("ema_tau", 0.999),
    ).to(device)

    # ── data pipeline from cached embeddings ──────────────────────────
    global_batch_size = train_cfg["batch_size"] * num_devices

    cached_dataset = TensorDataset(hidden_ctx, hidden_tgt)
    dataloader = DataLoader(
        cached_dataset,
        batch_size=global_batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=2,
    )

    # ── optimizer + scheduler ─────────────────────────────────────────
    optimizer = AdamW(
        thought_block.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg.get("weight_decay", 0.01),
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=train_cfg["max_steps"],
    )

    # ── checkpointing setup ───────────────────────────────────────────
    checkpoint_dir = Path("checkpoints/stage1")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ── training loop (fully on TPU) ──────────────────────────────────
    total_loss = 0.0
    start_time = time.time()
    global_step = 0

    print(
        f"[Stage1-TPU] Starting JEPA training for {train_cfg['max_steps']} steps..."
    )
    print(
        f"[Stage1-TPU] Per-chip batch: {train_cfg['batch_size']} | "
        f"Global batch: {global_batch_size} | "
        f"SPMD across {num_devices} chips"
    )
    print("[Stage1-TPU] All computation now on TPU — expect ~0.1-0.5s/step")
    sys.stdout.flush()

    for epoch in range(500):  # max epochs (will break by max_steps)
        for h_ctx, h_tgt in dataloader:
            if global_step >= train_cfg["max_steps"]:
                break

            # Move cached embeddings to TPU
            h_ctx = h_ctx.to(device)
            h_tgt = h_tgt.to(device)

            # Shard across TPU chips
            xs.mark_sharding(h_ctx, mesh, ("data", None))
            xs.mark_sharding(h_tgt, mesh, ("data", None))

            optimizer.zero_grad()

            # student path
            z_student, _ = thought_block(h_ctx)

            # teacher path (no grad)
            with torch.no_grad():
                z_teacher, _ = ema_teacher(h_tgt)

            # loss + backward
            loss = jepa_loss(z_student, z_teacher)
            loss.backward()

            # gradient clipping
            torch.nn.utils.clip_grad_norm_(
                thought_block.parameters(),
                train_cfg["grad_clip"],
            )

            optimizer.step()
            scheduler.step()

            # EMA update
            ema_teacher.update(thought_block)
            xm.mark_step()

            total_loss += loss.item()
            global_step += 1

            # logging
            if global_step % train_cfg.get("log_every", 5) == 0:
                avg_loss = total_loss / global_step
                elapsed = time.time() - start_time
                steps_per_sec = global_step / elapsed
                eta = (train_cfg["max_steps"] - global_step) / steps_per_sec
                print(
                    f"[Stage1-TPU] Step {global_step:>5d} | "
                    f"Loss: {loss.item():.4f} | "
                    f"Avg: {avg_loss:.4f} | "
                    f"LR: {scheduler.get_last_lr()[0]:.2e} | "
                    f"{steps_per_sec:.1f} steps/s | "
                    f"ETA: {eta:.0f}s"
                )
                sys.stdout.flush()

            # checkpointing
            if global_step % train_cfg.get("checkpoint_every", 500) == 0:
                ckpt_path = checkpoint_dir / f"stage1_step{global_step}.pt"
                xm.save(
                    {
                        "thought_block": thought_block.state_dict(),
                        "ema_teacher": ema_teacher.teacher.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "step": global_step,
                        "loss": total_loss / global_step,
                    },
                    str(ckpt_path),
                )
                print(f"[Stage1-TPU] Checkpoint saved: {ckpt_path}")

        if global_step >= train_cfg["max_steps"]:
            break

    # ── final save ────────────────────────────────────────────────────
    final_path = checkpoint_dir / "stage1_final.pt"
    xm.save(
        {
            "thought_block": thought_block.state_dict(),
            "ema_teacher": ema_teacher.teacher.state_dict(),
            "step": global_step,
            "loss": total_loss / max(global_step, 1),
        },
        str(final_path),
    )
    elapsed = time.time() - start_time
    print(f"[Stage1-TPU] ═══ Training complete in {elapsed:.0f}s ═══")
    print(f"[Stage1-TPU] Final checkpoint: {final_path}")


# ── CLI ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hypnos Stage 1 JEPA Training (TPU v5e-8)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/hypnos_1.4b_tpu.yaml",
        help="Path to TPU config YAML",
    )
    args = parser.parse_args()
    train_stage1_tpu(args.config)
