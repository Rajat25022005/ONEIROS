"""
train/stage1_jepa_tpu.py — Stage 1: JEPA pre-training on TPU v5e-8

Single-process SPMD training of the ThoughtBlock to predict future
latent states, supervised by the EMA teacher's coherence signal.
Uses PyTorch/XLA SPMD to shard data across all 8 TPU chips automatically.

Backbone is frozen — we only train reasoning, not world knowledge.

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
from torch.utils.data import DataLoader, Dataset

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


# ── Fixed-shape dataset for XLA ──────────────────────────────────────

class FixedLengthTextDataset(Dataset):
    """
    Pre-tokenized dataset with fixed sequence length.

    All sequences are padded/truncated to max_seq_len to avoid XLA
    graph recompilation from shape changes.
    """

    def __init__(self, tokenizer, config):
        from datasets import load_dataset

        train_cfg = config["training"]
        self.max_seq_len = train_cfg["max_seq_len"]

        print("[Stage1-TPU] Loading and tokenizing dataset...")

        dataset_name = train_cfg.get("dataset", "HuggingFaceFW/fineweb-edu")
        dataset_config = train_cfg.get("dataset_config", None)
        dataset_split = train_cfg.get("dataset_split", "train")

        ds = load_dataset(
            dataset_name,
            name=dataset_config,
            split=dataset_split,
        )

        # Pre-tokenize all examples to fixed length
        def tokenize_fn(examples):
            tokens = tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=self.max_seq_len,
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

        self.input_ids = ds["input_ids"]
        print(
            f"[Stage1-TPU] Dataset ready: {len(self.input_ids)} examples, "
            f"fixed length {self.max_seq_len}"
        )

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        ids = self.input_ids[idx]
        mid = self.max_seq_len // 2
        context = ids[:mid]
        target = ids[mid:]
        return context, target


# ── Training ─────────────────────────────────────────────────────────

def train_stage1_tpu(config_path: str):
    """
    Stage 1 JEPA training using SPMD on TPU v5e-8.

    SPMD (Single Program Multiple Data) runs as a single process.
    The XLA compiler automatically shards data across all 8 TPU chips
    based on sharding annotations — no xmp.spawn needed.
    """
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.runtime as xr
    import torch_xla.distributed.spmd as xs
    from torch_xla.distributed.spmd import Mesh

    # Enable SPMD mode — must be called before any XLA tensor creation
    xr.use_spmd()

    device = xm.xla_device()
    num_devices = xr.global_runtime_device_count()
    print(f"[Stage1-TPU] Device: {device} | TPU chips: {num_devices}")

    # Create 1D mesh for data parallelism across all chips
    mesh = Mesh(
        np.arange(num_devices),
        (num_devices,),
        ("data",),
    )

    with open(config_path) as f:
        config = yaml.safe_load(f)

    train_cfg = config["training"]
    model_cfg = config["model"]

    # ── build models ──────────────────────────────────────────────────
    from hypnos.model.backbone import MambaBackbone
    from hypnos.model.thought_block import ThoughtBlock
    from hypnos.model.ema_teacher import EMATeacher

    # Backbone stays on CPU — Mamba's sequential scan doesn't compile
    # on XLA (hangs). Since backbone is frozen, CPU is fine. We only
    # transfer the small hidden states (B, 2048) to TPU.
    cpu_device = torch.device("cpu")
    backbone = MambaBackbone(
        model_cfg.get("backbone", "state-spaces/mamba-1.4b-hf"),
        use_slow_path=model_cfg.get("use_slow_path", True),
    )
    backbone.to(cpu_device)
    for param in backbone.parameters():
        param.requires_grad_(False)
    backbone.eval()
    print("[Stage1-TPU] Backbone on CPU (frozen, avoids XLA compilation)")

    # Trainable components on TPU
    thought_block = ThoughtBlock(
        input_dim=backbone.hidden_size,
        latent_dim=model_cfg.get("latent_dim", 512),
        k_steps=model_cfg.get("k_steps", 8),
    ).to(device)

    ema_teacher = EMATeacher(
        student=thought_block,
        tau=model_cfg.get("ema_tau", 0.999),
    ).to(device)

    print("[Stage1-TPU] ThoughtBlock + EMATeacher on TPU")

    # ── data pipeline ─────────────────────────────────────────────────
    # Global batch size — SPMD shards this across chips automatically
    global_batch_size = train_cfg["batch_size"] * num_devices

    dataset = FixedLengthTextDataset(backbone.tokenizer, config)

    dataloader = DataLoader(
        dataset,
        batch_size=global_batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4,
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

    # ── training loop ─────────────────────────────────────────────────
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
    sys.stdout.flush()

    for epoch in range(100):  # max epochs (will break by max_steps)
        for context_ids, target_ids in dataloader:
            if global_step >= train_cfg["max_steps"]:
                break

            # ── backbone encode on CPU (avoids XLA compilation) ───────
            with torch.no_grad():
                hidden_ctx, _ = backbone.encode(context_ids)
                hidden_tgt, _ = backbone.encode(target_ids)

            # ── transfer hidden states to TPU ─────────────────────────
            hidden_ctx = hidden_ctx.to(device)
            hidden_tgt = hidden_tgt.to(device)

            # Shard across TPU chips
            xs.mark_sharding(hidden_ctx, mesh, ("data", None))
            xs.mark_sharding(hidden_tgt, mesh, ("data", None))

            optimizer.zero_grad()

            # student path (on TPU)
            z_student, _ = thought_block(hidden_ctx)

            # teacher path (on TPU, no grad)
            with torch.no_grad():
                z_teacher, _ = ema_teacher(hidden_tgt)

            # loss + backward (on TPU)
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
            if global_step % train_cfg.get("log_every", 50) == 0:
                avg_loss = total_loss / global_step
                elapsed = time.time() - start_time
                print(
                    f"[Stage1-TPU] Step {global_step:>5d} | "
                    f"Loss: {loss.item():.4f} | "
                    f"Avg: {avg_loss:.4f} | "
                    f"LR: {scheduler.get_last_lr()[0]:.2e} | "
                    f"Time: {elapsed:.0f}s"
                )

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
    print(f"[Stage1-TPU] Training complete. Final checkpoint: {final_path}")


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
