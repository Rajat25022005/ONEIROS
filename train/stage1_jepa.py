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
            train_cfg.get("dataset", "HuggingFaceFW/fineweb-edu"),
            name=train_cfg.get("dataset_config", None),
            split=train_cfg.get("dataset_split", "train"),
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


# ── dataset ───────────────────────────────────────────────────────────

class FixedLengthDataset(torch.utils.data.Dataset):
    """Pre-tokenized dataset with fixed sequence length for GPU training."""

    def __init__(self, config, tokenizer):
        from datasets import load_dataset

        train_cfg = config["training"]
        max_seq_len = train_cfg["max_seq_len"]

        print("[Stage1] Loading and tokenizing dataset...")

        ds = load_dataset(
            train_cfg.get("dataset", "HuggingFaceFW/fineweb-edu"),
            name=train_cfg.get("dataset_config", None),
            split=train_cfg.get("dataset_split", "train"),
        )

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
            num_proc=4,
        )
        ds.set_format("torch")

        self.input_ids = ds["input_ids"]
        self.mid = max_seq_len // 2
        print(f"[Stage1] Dataset ready: {len(self.input_ids)} examples")

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        ids = self.input_ids[idx]
        return ids[:self.mid], ids[self.mid:]


# ── training loop ─────────────────────────────────────────────────────

def train_stage1(config_path: str):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    train_cfg = config["training"]
    model_cfg = config["model"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"[Stage1] Device: {device} | GPUs: {n_gpus}")

    # ── build models ──────────────────────────────────────────────────
    backbone = MambaBackbone(
        model_cfg.get("backbone", "state-spaces/mamba-1.4b-hf"),
        use_slow_path=model_cfg.get("use_slow_path", False),
    )
    backbone.to(device)
    for param in backbone.parameters():
        param.requires_grad_(False)
    backbone.eval()

    thought_block = ThoughtBlock(
        input_dim=backbone.hidden_size,
        latent_dim=model_cfg.get("latent_dim", 512),
        k_steps=model_cfg.get("k_steps", 8),
    ).to(device)

    ema_teacher = EMATeacher(
        student=thought_block,
        tau=model_cfg.get("ema_tau", 0.999),
    ).to(device)

    # multi-GPU DataParallel
    if n_gpus > 1:
        backbone = nn.DataParallel(backbone)
        thought_block = nn.DataParallel(thought_block)
        ema_teacher = nn.DataParallel(ema_teacher)
        print(f"[Stage1] Using DataParallel across {n_gpus} GPUs")

    # ── data pipeline ─────────────────────────────────────────────────
    # Use the backbone's tokenizer; unwrap DataParallel if needed
    tok = backbone.module.tokenizer if n_gpus > 1 else backbone.tokenizer

    dataset = FixedLengthDataset(config, tok)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=4,          # parallel CPU tokenization/loading
        pin_memory=True,        # faster CPU→GPU transfer
        prefetch_factor=2,      # prefetch next batch while GPU trains
        persistent_workers=True,
    )

    # ── optimizer + scheduler ─────────────────────────────────────────
    optimizer = AdamW(
        (thought_block.module if n_gpus > 1 else thought_block).parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg.get("weight_decay", 0.01),
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=train_cfg["max_steps"])
    scaler = torch.cuda.amp.GradScaler()  # for bf16/fp16 training

    # ── checkpointing setup ───────────────────────────────────────────
    checkpoint_dir = Path("checkpoints/stage1")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ── training ──────────────────────────────────────────────────────
    total_loss = 0.0
    start_time = time.time()
    global_step = 0

    print(f"[Stage1] Starting JEPA training for {train_cfg['max_steps']} steps...")
    print(f"[Stage1] Batch size: {train_cfg['batch_size']} | "
          f"Effective: {train_cfg['batch_size'] * max(n_gpus, 1)}")

    for epoch in range(100):
        for context_ids, target_ids in dataloader:
            if global_step >= train_cfg["max_steps"]:
                break

            context_ids = context_ids.to(device, non_blocking=True)
            target_ids = target_ids.to(device, non_blocking=True)

            optimizer.zero_grad()

            # bf16 autocast — keeps GPU tensor cores busy
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                # student path
                with torch.no_grad():
                    hidden_ctx, _ = backbone.encode(context_ids) if not isinstance(backbone, nn.DataParallel) else backbone.module.encode(context_ids)
                z_student, _ = thought_block(hidden_ctx)

                # teacher path
                with torch.no_grad():
                    hidden_tgt, _ = backbone.encode(target_ids) if not isinstance(backbone, nn.DataParallel) else backbone.module.encode(target_ids)
                    z_teacher, _ = ema_teacher(hidden_tgt)

                loss = jepa_loss(z_student, z_teacher)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                (thought_block.module if n_gpus > 1 else thought_block).parameters(),
                train_cfg["grad_clip"],
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            (ema_teacher.module if n_gpus > 1 else ema_teacher).update(
                thought_block.module if n_gpus > 1 else thought_block
            )

            total_loss += loss.item()
            global_step += 1

            # logging
            if global_step % train_cfg.get("log_every", 50) == 0:
                avg_loss = total_loss / global_step
                elapsed = time.time() - start_time
                steps_per_sec = global_step / elapsed
                print(
                    f"[Stage1] Step {global_step:>5d} | "
                    f"Loss: {loss.item():.4f} | "
                    f"Avg: {avg_loss:.4f} | "
                    f"LR: {scheduler.get_last_lr()[0]:.2e} | "
                    f"{steps_per_sec:.2f} steps/s"
                )

            # checkpointing
            if global_step % train_cfg.get("checkpoint_every", 500) == 0:
                ckpt_path = checkpoint_dir / f"stage1_step{global_step}.pt"
                torch.save(
                    {
                        "thought_block": (thought_block.module if n_gpus > 1 else thought_block).state_dict(),
                        "ema_teacher": (ema_teacher.module if n_gpus > 1 else ema_teacher).teacher.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "step": global_step,
                        "loss": total_loss / global_step,
                    },
                    ckpt_path,
                )
                print(f"[Stage1] Checkpoint saved: {ckpt_path}")

        if global_step >= train_cfg["max_steps"]:
            break

    # final save
    final_path = checkpoint_dir / "stage1_final.pt"
    torch.save(
        {
            "thought_block": (thought_block.module if n_gpus > 1 else thought_block).state_dict(),
            "step": global_step,
            "loss": total_loss / max(global_step, 1),
        },
        final_path,
    )
    print(f"[Stage1] Training complete. Final checkpoint: {final_path}")


# ── CLI ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/hypnos_1.4b.yaml")
    args = parser.parse_args()
    train_stage1(args.config)


