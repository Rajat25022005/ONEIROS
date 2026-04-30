"""
train/stage2_decoder_tpu.py — Stage 2: Decoder training on TPU v5e-8

Two-phase training:
  Phase 1 (CPU): Pre-compute backbone + ThoughtBlock embeddings and cache.
  Phase 2 (TPU): Train LatentDecoder on cached z-vectors at full TPU speed.

Usage (Kaggle notebook):
    from train.stage2_decoder_tpu import train_stage2_tpu
    train_stage2_tpu("configs/hypnos_1.4b_tpu.yaml")
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


# ── Phase 1: Pre-cache backbone + ThoughtBlock embeddings ────────────

def precache_embeddings(backbone, thought_block, tokenizer, config):
    """
    Run frozen backbone + ThoughtBlock once on CPU.
    Returns cached (z_vectors, target_ids) for decoder training.
    """
    from datasets import load_dataset

    train_cfg = config["training"]
    max_seq_len = train_cfg["max_seq_len"]

    print("[Stage2-TPU] ═══ Phase 1: Pre-caching embeddings on CPU ═══")
    print("[Stage2-TPU] Loading and tokenizing dataset...")
    sys.stdout.flush()

    dataset_name = train_cfg.get("dataset", "HuggingFaceFW/fineweb-edu")
    dataset_config = train_cfg.get("dataset_config", None)
    dataset_split = train_cfg.get("dataset_split", "train")

    ds = load_dataset(
        dataset_name,
        name=dataset_config,
        split=dataset_split,
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
    )
    ds.set_format("torch")

    all_ids = ds["input_ids"]
    mid = max_seq_len // 2
    n_examples = len(all_ids)

    print(f"[Stage2-TPU] Dataset: {n_examples} examples, seq_len={max_seq_len}")
    print(f"[Stage2-TPU] Running backbone + ThoughtBlock on CPU...")
    sys.stdout.flush()

    cache_batch_size = 8
    all_z = []
    all_targets = []

    start = time.time()
    backbone.eval()
    thought_block.eval()

    for i in range(0, n_examples, cache_batch_size):
        batch_ids = all_ids[i : i + cache_batch_size]

        context_ids = batch_ids[:, :mid]
        target_ids = batch_ids[:, mid:]

        with torch.no_grad():
            hidden, _ = backbone.encode(context_ids)
            # Mean-pool over sequence length
            hidden_pooled = hidden.mean(dim=1)
            z, _ = thought_block(hidden_pooled)

        all_z.append(z.cpu())
        all_targets.append(target_ids.cpu())

        done = min(i + cache_batch_size, n_examples)
        if done % 500 == 0 or done == n_examples:
            elapsed = time.time() - start
            rate = done / elapsed if elapsed > 0 else 0
            eta = (n_examples - done) / rate if rate > 0 else 0
            print(
                f"[Stage2-TPU]   Cached {done:>6d}/{n_examples} "
                f"({100*done/n_examples:.1f}%) | "
                f"{rate:.1f} ex/s | ETA: {eta:.0f}s"
            )
            sys.stdout.flush()

    cached_z = torch.cat(all_z, dim=0)
    cached_targets = torch.cat(all_targets, dim=0)

    elapsed = time.time() - start
    print(f"[Stage2-TPU] ═══ Phase 1 complete: {elapsed:.0f}s ═══")
    print(f"[Stage2-TPU] Cached: z={cached_z.shape}, targets={cached_targets.shape}")
    sys.stdout.flush()

    return cached_z, cached_targets


# ── Phase 2: Train Decoder on TPU ─────────────────────────────────────

def train_stage2_tpu(
    config_path: str,
    stage1_checkpoint: str = None,
):
    """
    Stage 2 Decoder training using SPMD on TPU v5e-8.

    Phase 1: Pre-cache backbone + ThoughtBlock embeddings on CPU
    Phase 2: Train decoder on TPU at full speed
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

    # ── Load frozen models on CPU ─────────────────────────────────────
    from hypnos.model.backbone import MambaBackbone
    from hypnos.model.thought_block import ThoughtBlock
    from hypnos.model.decoder import LatentDecoder

    cpu_device = torch.device("cpu")

    backbone = MambaBackbone(
        model_cfg.get("backbone", "state-spaces/mamba-1.4b-hf"),
        use_slow_path=model_cfg.get("use_slow_path", True),
    )
    backbone.to(cpu_device)
    for param in backbone.parameters():
        param.requires_grad_(False)

    latent_dim = model_cfg.get("latent_dim", 512)
    thought_block = ThoughtBlock(
        input_dim=backbone.hidden_size,
        latent_dim=latent_dim,
        k_steps=model_cfg.get("k_steps", 8),
    ).to(cpu_device)

    # Load Stage 1 checkpoint
    ckpt_path = stage1_checkpoint or "checkpoints/stage1/stage1_final.pt"

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        thought_block.load_state_dict(ckpt["thought_block"])
        print(f"[Stage2-TPU] Loaded Stage 1 checkpoint: {ckpt_path}")
        print(f"[Stage2-TPU] Stage 1 loss was: {ckpt.get('loss', 'N/A')}")
    else:
        print(f"[Stage2-TPU] Warning: No Stage 1 checkpoint at {ckpt_path}")
        print("[Stage2-TPU] Training decoder with untrained ThoughtBlock.")

    for param in thought_block.parameters():
        param.requires_grad_(False)

    # ── Phase 1: cache embeddings on CPU ──────────────────────────────
    cached_z, cached_targets = precache_embeddings(
        backbone, thought_block, backbone.tokenizer, config
    )

    # Free CPU models
    del backbone, thought_block
    import gc; gc.collect()
    print("[Stage2-TPU] Freed backbone + ThoughtBlock from memory.")

    # ── Phase 2: TPU training ─────────────────────────────────────────
    print("[Stage2-TPU] ═══ Phase 2: Training Decoder on TPU ═══")

    xr.use_spmd()
    device = xm.xla_device()
    num_devices = xr.global_runtime_device_count()
    print(f"[Stage2-TPU] Device: {device} | TPU chips: {num_devices}")

    mesh = Mesh(
        np.arange(num_devices),
        (num_devices,),
        ("data",),
    )

    # Decoder on TPU
    decoder = LatentDecoder(
        latent_dim=latent_dim,
        vocab_size=model_cfg.get("vocab_size", 50280),
        hidden_dim=model_cfg.get("decoder_hidden", 1024),
        max_length=model_cfg.get("max_length", 128),
    ).to(device)

    trainable_params = sum(p.numel() for p in decoder.parameters())
    print(f"[Stage2-TPU] Decoder on TPU, trainable params: {trainable_params:,}")

    # ── data pipeline from cached embeddings ──────────────────────────
    global_batch_size = train_cfg["batch_size"] * num_devices

    cached_dataset = TensorDataset(cached_z, cached_targets)
    dataloader = DataLoader(
        cached_dataset,
        batch_size=global_batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=2,
    )

    # ── optimizer + scheduler ─────────────────────────────────────────
    max_steps = train_cfg.get("max_steps", 5000)

    optimizer = AdamW(
        decoder.parameters(),
        lr=train_cfg.get("learning_rate", 1e-4),
        weight_decay=train_cfg.get("weight_decay", 0.01),
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=max_steps)

    # ── checkpointing setup ───────────────────────────────────────────
    checkpoint_dir = Path("checkpoints/stage2")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ── training loop (fully on TPU) ──────────────────────────────────
    total_loss = 0.0
    start_time = time.time()
    global_step = 0
    decoder.train()

    print(
        f"[Stage2-TPU] Starting decoder training for {max_steps} steps..."
    )
    print(
        f"[Stage2-TPU] Per-chip batch: {train_cfg['batch_size']} | "
        f"Global batch: {global_batch_size} | "
        f"SPMD across {num_devices} chips"
    )
    print("[Stage2-TPU] All computation now on TPU — expect ~0.1-0.5s/step")
    sys.stdout.flush()

    for epoch in range(500):
        for z_batch, target_ids in dataloader:
            if global_step >= max_steps:
                break

            z_batch = z_batch.to(device)
            target_ids = target_ids.to(device)

            xs.mark_sharding(z_batch, mesh, ("data", None))
            xs.mark_sharding(target_ids, mesh, ("data", None))

            optimizer.zero_grad()

            # decoder: predict target tokens from z
            decoder_input = target_ids[:, :-1]
            decoder_target = target_ids[:, 1:]

            logits = decoder(z_batch, target_ids=decoder_input)

            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                decoder_target.reshape(-1),
                ignore_index=0,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                decoder.parameters(),
                train_cfg.get("grad_clip", 1.0),
            )

            optimizer.step()
            scheduler.step()
            xm.mark_step()

            total_loss += loss.item()
            global_step += 1

            # logging
            if global_step % train_cfg.get("log_every", 5) == 0:
                avg_loss = total_loss / global_step
                elapsed = time.time() - start_time
                steps_per_sec = global_step / elapsed
                eta = (max_steps - global_step) / steps_per_sec
                perplexity = min(
                    torch.exp(torch.tensor(avg_loss)).item(), 1e6
                )
                print(
                    f"[Stage2-TPU] Step {global_step:>5d} | "
                    f"Loss: {loss.item():.4f} | "
                    f"PPL: {perplexity:.1f} | "
                    f"LR: {scheduler.get_last_lr()[0]:.2e} | "
                    f"{steps_per_sec:.1f} steps/s | "
                    f"ETA: {eta:.0f}s"
                )
                sys.stdout.flush()

            # checkpointing
            if global_step % train_cfg.get("checkpoint_every", 500) == 0:
                ckpt_path = checkpoint_dir / f"stage2_step{global_step}.pt"
                xm.save(
                    {
                        "decoder": decoder.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "step": global_step,
                        "loss": total_loss / global_step,
                    },
                    str(ckpt_path),
                )
                print(f"[Stage2-TPU] Checkpoint saved: {ckpt_path}")

        if global_step >= max_steps:
            break

    # ── final save ────────────────────────────────────────────────────
    final_path = checkpoint_dir / "stage2_final.pt"
    xm.save(
        {
            "decoder": decoder.state_dict(),
            "step": global_step,
            "loss": total_loss / max(global_step, 1),
        },
        str(final_path),
    )
    elapsed = time.time() - start_time
    print(f"[Stage2-TPU] ═══ Training complete in {elapsed:.0f}s ═══")
    print(f"[Stage2-TPU] Final checkpoint: {final_path}")


# ── CLI ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hypnos Stage 2 Decoder Training (TPU v5e-8)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/hypnos_1.4b_tpu.yaml",
    )
    parser.add_argument(
        "--stage1-checkpoint",
        type=str,
        default=None,
    )
    args = parser.parse_args()
    train_stage2_tpu(args.config, args.stage1_checkpoint)
