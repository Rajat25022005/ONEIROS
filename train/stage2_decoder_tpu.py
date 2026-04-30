"""
train/stage2_decoder_tpu.py — Stage 2: Decoder training on TPU v5e-8

Single-process SPMD training of the LatentDecoder to convert ThoughtBlock's
latent output z_K back into readable text.  Uses PyTorch/XLA SPMD to
shard data across all 8 TPU chips automatically.

Backbone and ThoughtBlock are frozen — only the decoder learns.

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
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


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

        print("[Stage2-TPU] Loading and tokenizing dataset...")

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
            f"[Stage2-TPU] Dataset ready: {len(self.input_ids)} examples, "
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


# ── Sample generation ─────────────────────────────────────────────────

@torch.no_grad()
def _sample_generation(backbone, thought_block, decoder, device):
    """Generate a sample to monitor training quality."""
    import torch_xla.core.xla_model as xm

    prompts = [
        "Once upon a time",
        "The sun was setting",
        "She opened the door",
    ]
    prompt = prompts[int(time.time()) % len(prompts)]

    ids = backbone.tokenize(prompt, device)
    hidden, _ = backbone.encode(ids)
    z, _ = thought_block(hidden)
    output_ids = decoder(z)
    xm.mark_step()

    if backbone.tokenizer is not None:
        text = backbone.tokenizer.decode(
            output_ids[0], skip_special_tokens=True
        )
    else:
        text = str(output_ids[0].tolist()[:20])

    print(f'[Stage2-TPU] Sample: "{prompt}" → "{text[:80]}..."')


# ── Training ─────────────────────────────────────────────────────────

def train_stage2_tpu(
    config_path: str,
    stage1_checkpoint: str = None,
):
    """
    Stage 2 Decoder training using SPMD on TPU v5e-8.

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
    print(f"[Stage2-TPU] Device: {device} | TPU chips: {num_devices}")

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
    from hypnos.model.decoder import LatentDecoder

    backbone = MambaBackbone(
        model_cfg.get("backbone", "state-spaces/mamba-1.4b-hf"),
        use_slow_path=model_cfg.get("use_slow_path", True),
    )
    backbone.to(device)

    latent_dim = model_cfg.get("latent_dim", 512)
    k_steps = model_cfg.get("k_steps", 8)

    thought_block = ThoughtBlock(
        input_dim=backbone.hidden_size,
        latent_dim=latent_dim,
        k_steps=k_steps,
    ).to(device)

    decoder = LatentDecoder(
        latent_dim=latent_dim,
        vocab_size=model_cfg.get("vocab_size", 50280),
        hidden_dim=model_cfg.get("decoder_hidden", 1024),
        max_length=model_cfg.get("max_length", 128),
    ).to(device)

    # ── load Stage 1 checkpoint ───────────────────────────────────────
    ckpt_path = stage1_checkpoint or "checkpoints/stage1/stage1_final.pt"

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        thought_block.load_state_dict(ckpt["thought_block"])
        print(f"[Stage2-TPU] Loaded Stage 1 checkpoint: {ckpt_path}")
        print(f"[Stage2-TPU] Stage 1 loss was: {ckpt.get('loss', 'N/A')}")
    else:
        print(f"[Stage2-TPU] Warning: No Stage 1 checkpoint at {ckpt_path}")
        print("[Stage2-TPU] Training decoder with untrained ThoughtBlock.")

    # ── freeze backbone + thought block ───────────────────────────────
    for param in backbone.parameters():
        param.requires_grad_(False)
    for param in thought_block.parameters():
        param.requires_grad_(False)

    backbone.eval()
    thought_block.eval()

    trainable_params = sum(p.numel() for p in decoder.parameters())
    print(f"[Stage2-TPU] Decoder trainable params: {trainable_params:,}")

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

    # ── training loop ─────────────────────────────────────────────────
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

    for epoch in range(100):  # max epochs (will break by max_steps)
        for context_ids, target_ids in dataloader:
            if global_step >= max_steps:
                break

            # Move to XLA device
            context_ids = context_ids.to(device)
            target_ids = target_ids.to(device)

            # Shard batch across TPU chips (data parallelism)
            xs.mark_sharding(context_ids, mesh, ("data", None))
            xs.mark_sharding(target_ids, mesh, ("data", None))

            optimizer.zero_grad()

            # encode context through frozen backbone + thought block
            # mark_step after backbone to break XLA graph
            with torch.no_grad():
                hidden, _ = backbone.encode(context_ids)
            xm.mark_step()  # compile backbone separately
            with torch.no_grad():
                z_final, _ = thought_block(hidden)

            # decoder: predict target tokens from z_final
            # teacher forcing: feed target_ids shifted by 1
            decoder_input = target_ids[:, :-1]
            decoder_target = target_ids[:, 1:]

            logits = decoder(z_final, target_ids=decoder_input)

            # cross-entropy loss
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                decoder_target.reshape(-1),
                ignore_index=0,  # ignore padding
            )

            loss.backward()
            xm.mark_step()  # compile backward separately

            torch.nn.utils.clip_grad_norm_(
                decoder.parameters(),
                train_cfg.get("grad_clip", 1.0),
            )

            optimizer.step()
            scheduler.step()
            xm.mark_step()

            total_loss += loss.item()
            global_step += 1

            # ── logging ───────────────────────────────────────────────
            if global_step % train_cfg.get("log_every", 50) == 0:
                avg_loss = total_loss / global_step
                elapsed = time.time() - start_time
                perplexity = min(
                    torch.exp(torch.tensor(avg_loss)).item(), 1e6
                )
                print(
                    f"[Stage2-TPU] Step {global_step:>5d} | "
                    f"Loss: {loss.item():.4f} | "
                    f"Avg: {avg_loss:.4f} | "
                    f"PPL: {perplexity:.1f} | "
                    f"LR: {scheduler.get_last_lr()[0]:.2e} | "
                    f"Time: {elapsed:.0f}s"
                )

                # sample generation every 200 steps
                if global_step % 200 == 0:
                    _sample_generation(
                        backbone, thought_block, decoder, device
                    )

            # ── checkpointing ─────────────────────────────────────────
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
    print(f"[Stage2-TPU] Training complete. Final checkpoint: {final_path}")


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
        help="Path to Stage 1 checkpoint (default: checkpoints/stage1/stage1_final.pt)",
    )
    args = parser.parse_args()
    train_stage2_tpu(args.config, args.stage1_checkpoint)
