"""
train/stage1_jepa.py — Stage 1: JEPA pre-training

Trains the ThoughtBlock to predict future latent states from current
context, supervised by the EMA teacher's coherence signal.  Backbone
is frozen — we only train reasoning, not world knowledge.

Usage:
    python -m train.stage1_jepa --config configs/hypnos_130m.yaml
"""

import os
import sys
import yaml
import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hypnos.model.backbone import MambaBackbone
from hypnos.model.thought_block import ThoughtBlock
from hypnos.model.ema_teacher import EMATeacher


# ── JEPA loss ─────────────────────────────────────────────────────────

def jepa_loss(
    z_student: torch.Tensor,
    z_teacher: torch.Tensor,
) -> torch.Tensor:
    """
    JEPA loss: smooth L1 between normalized student and teacher latents.

    Normalization prevents the trivial collapse-to-zero solution.
    Smooth L1 (Huber) is robust to outliers.
    """
    z_s = F.normalize(z_student, dim=-1)
    z_t = F.normalize(z_teacher, dim=-1)
    return F.smooth_l1_loss(z_s, z_t)


# ── data ──────────────────────────────────────────────────────────────

def get_batches(backbone, config, device):
    """
    Yield batches of (context_ids, target_ids) pairs.

    Uses HuggingFace datasets streaming for memory efficiency.
    Falls back to random data if datasets unavailable.
    """
    train_cfg = config["training"]
    batch_size = train_cfg["batch_size"]
    max_seq_len = train_cfg["max_seq_len"]

    try:
        from datasets import load_dataset

        dataset = load_dataset(
            train_cfg.get("dataset", "roneneldan/TinyStories"),
            split="train",
            streaming=True,
        )

        batch_ids = []
        for example in dataset:
            text = example.get("text", "")
            if not text:
                continue
            ids = backbone.tokenize(text, device)
            if ids.shape[1] < 4:
                continue
            # pad/truncate to max_seq_len
            if ids.shape[1] > max_seq_len:
                ids = ids[:, :max_seq_len]
            batch_ids.append(ids)

            if len(batch_ids) >= batch_size:
                # pad batch to same length
                max_len = max(t.shape[1] for t in batch_ids)
                padded = []
                for t in batch_ids:
                    if t.shape[1] < max_len:
                        pad = torch.zeros(
                            1, max_len - t.shape[1],
                            dtype=torch.long, device=device
                        )
                        t = torch.cat([t, pad], dim=1)
                    padded.append(t)

                batch = torch.cat(padded, dim=0)
                mid = batch.shape[1] // 2
                context = batch[:, :mid]
                target = batch[:, mid:]
                yield context, target
                batch_ids = []

    except Exception as e:
        print(f"[Stage1] Dataset loading failed: {e}")
        print("[Stage1] Falling back to random data.")
        vocab_size = config["model"].get("vocab_size", 50257)
        for _ in range(train_cfg["max_steps"]):
            context = torch.randint(
                0, vocab_size, (batch_size, max_seq_len // 2), device=device
            )
            target = torch.randint(
                0, vocab_size, (batch_size, max_seq_len // 2), device=device
            )
            yield context, target


# ── training loop ─────────────────────────────────────────────────────

def train_stage1(config_path: str):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    train_cfg = config["training"]
    model_cfg = config["model"]
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"[Stage1] Device: {device}")

    # build models
    backbone = MambaBackbone(
        model_cfg.get("backbone", "state-spaces/mamba-130m-hf")
    )
    backbone.to(device)

    thought_block = ThoughtBlock(
        input_dim=backbone.hidden_size,
        latent_dim=model_cfg.get("latent_dim", 256),
        k_steps=model_cfg.get("k_steps", 8),
    ).to(device)

    ema_teacher = EMATeacher(
        student=thought_block,
        tau=model_cfg.get("ema_tau", 0.999),
    ).to(device)

    # freeze backbone — only train reasoning
    for param in backbone.parameters():
        param.requires_grad_(False)

    # optimizer + scheduler
    optimizer = AdamW(
        thought_block.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg.get("weight_decay", 0.01),
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=train_cfg["max_steps"],
    )

    # checkpointing
    checkpoint_dir = Path("checkpoints/stage1")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # training
    total_loss = 0.0
    start_time = time.time()

    print(f"[Stage1] Starting JEPA training for {train_cfg['max_steps']} steps...")

    for step, (context_ids, target_ids) in enumerate(
        get_batches(backbone, config, device)
    ):
        if step >= train_cfg["max_steps"]:
            break

        optimizer.zero_grad()

        # student path
        hidden_ctx, _ = backbone.encode(context_ids)
        z_student, _ = thought_block(hidden_ctx)

        # teacher path (no grad)
        with torch.no_grad():
            hidden_tgt, _ = backbone.encode(target_ids)
            z_teacher, _ = ema_teacher(hidden_tgt)

        # loss
        loss = jepa_loss(z_student, z_teacher)
        loss.backward()

        # gradient clipping
        torch.nn.utils.clip_grad_norm_(
            thought_block.parameters(),
            train_cfg["grad_clip"],
        )

        optimizer.step()
        scheduler.step()
        ema_teacher.update(thought_block)

        total_loss += loss.item()

        # logging
        if (step + 1) % train_cfg.get("log_every", 50) == 0:
            avg_loss = total_loss / (step + 1)
            elapsed = time.time() - start_time
            print(
                f"[Stage1] Step {step + 1:>5d} | "
                f"Loss: {loss.item():.4f} | "
                f"Avg: {avg_loss:.4f} | "
                f"LR: {scheduler.get_last_lr()[0]:.2e} | "
                f"Time: {elapsed:.0f}s"
            )

        # checkpointing
        if (step + 1) % train_cfg.get("checkpoint_every", 1000) == 0:
            ckpt_path = checkpoint_dir / f"stage1_step{step + 1}.pt"
            torch.save(
                {
                    "thought_block": thought_block.state_dict(),
                    "ema_teacher": ema_teacher.teacher.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step + 1,
                    "loss": total_loss / (step + 1),
                },
                ckpt_path,
            )
            print(f"[Stage1] Checkpoint saved: {ckpt_path}")

    # final save
    final_path = checkpoint_dir / "stage1_final.pt"
    torch.save(
        {
            "thought_block": thought_block.state_dict(),
            "ema_teacher": ema_teacher.teacher.state_dict(),
            "step": step + 1 if "step" in dir() else 0,
            "loss": total_loss / max(step + 1, 1) if "step" in dir() else 0,
        },
        final_path,
    )
    print(f"[Stage1] Training complete. Final checkpoint: {final_path}")


# ── CLI ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hypnos Stage 1 JEPA Training")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/hypnos_130m.yaml",
        help="Path to config YAML",
    )
    args = parser.parse_args()
    train_stage1(args.config)
