"""
train/stage2_decoder.py — Stage 2: Decoder training

Trains the LatentDecoder to convert ThoughtBlock's latent output z_K
back into readable text.  Backbone and ThoughtBlock are frozen — only
the decoder learns.

Usage:
    python -m train.stage2_decoder --config configs/hypnos_130m.yaml
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
from hypnos.model.decoder import LatentDecoder


# ── data ──────────────────────────────────────────────────────────────

def get_batches(backbone, config, device):
    """
    Yield batches of token ids for decoder training.

    Each batch is a full sequence — the decoder learns to reconstruct
    the second half given the ThoughtBlock's latent encoding of the first half.
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
            if ids.shape[1] < 8:
                continue
            if ids.shape[1] > max_seq_len:
                ids = ids[:, :max_seq_len]
            batch_ids.append(ids)

            if len(batch_ids) >= batch_size:
                max_len = max(t.shape[1] for t in batch_ids)
                padded = []
                for t in batch_ids:
                    if t.shape[1] < max_len:
                        pad = torch.zeros(
                            1, max_len - t.shape[1],
                            dtype=torch.long, device=device,
                        )
                        t = torch.cat([t, pad], dim=1)
                    padded.append(t)

                batch = torch.cat(padded, dim=0)
                mid = batch.shape[1] // 2
                context_ids = batch[:, :mid]
                target_ids = batch[:, mid:]
                yield context_ids, target_ids
                batch_ids = []

    except Exception as e:
        print(f"[Stage2] Dataset loading failed: {e}")
        print("[Stage2] Falling back to random data.")
        vocab_size = config["model"].get("vocab_size", 50257)
        for _ in range(train_cfg["max_steps"]):
            context = torch.randint(
                0, vocab_size, (batch_size, max_seq_len // 2), device=device,
            )
            target = torch.randint(
                0, vocab_size, (batch_size, max_seq_len // 2), device=device,
            )
            yield context, target


# ── training loop ─────────────────────────────────────────────────────

def train_stage2(config_path: str, stage1_checkpoint: str = None):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    train_cfg = config["training"]
    model_cfg = config["model"]

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"[Stage2] Device: {device} | GPUs available: {n_gpus}")

    # ── build models ──────────────────────────────────────────────────

    backbone = MambaBackbone(
        model_cfg.get("backbone", "state-spaces/mamba-1.4b-hf"),
        use_slow_path=model_cfg.get("use_slow_path", False),
    )
    backbone.to(device)

    latent_dim = model_cfg.get("latent_dim", 256)
    k_steps = model_cfg.get("k_steps", 8)

    thought_block = ThoughtBlock(
        input_dim=backbone.hidden_size,
        latent_dim=latent_dim,
        k_steps=k_steps,
    ).to(device)

    decoder = LatentDecoder(
        latent_dim=latent_dim,
        vocab_size=model_cfg.get("vocab_size", 50257),
        hidden_dim=model_cfg.get("decoder_hidden", 512),
        max_length=model_cfg.get("max_length", 128),
    ).to(device)

    # ── load Stage 1 checkpoint ───────────────────────────────────────

    if stage1_checkpoint is None:
        stage1_checkpoint = "checkpoints/stage1/stage1_step1000.pt"

    if os.path.exists(stage1_checkpoint):
        ckpt = torch.load(stage1_checkpoint, map_location=device)
        thought_block.load_state_dict(ckpt["thought_block"])
        print(f"[Stage2] Loaded Stage 1 checkpoint: {stage1_checkpoint}")
        print(f"[Stage2] Stage 1 loss was: {ckpt.get('loss', 'N/A')}")
    else:
        print(f"[Stage2] Warning: No Stage 1 checkpoint at {stage1_checkpoint}")
        print("[Stage2] Training decoder with untrained ThoughtBlock.")

    # ── freeze backbone + thought block ───────────────────────────────

    for param in backbone.parameters():
        param.requires_grad_(False)
    for param in thought_block.parameters():
        param.requires_grad_(False)

    backbone.eval()
    thought_block.eval()

    trainable_params = sum(p.numel() for p in decoder.parameters())
    print(f"[Stage2] Decoder trainable params: {trainable_params:,}")

    # ── optimizer ─────────────────────────────────────────────────────

    optimizer = AdamW(
        decoder.parameters(),
        lr=train_cfg.get("learning_rate", 1e-4),
        weight_decay=train_cfg.get("weight_decay", 0.01),
    )
    max_steps = train_cfg.get("max_steps", 10000)
    scheduler = CosineAnnealingLR(optimizer, T_max=max_steps)

    # ── checkpointing ─────────────────────────────────────────────────

    checkpoint_dir = Path("checkpoints/stage2")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ── training ──────────────────────────────────────────────────────

    total_loss = 0.0
    start_time = time.time()
    decoder.train()

    print(f"[Stage2] Starting decoder training for {max_steps} steps...")

    for step, (context_ids, target_ids) in enumerate(
        get_batches(backbone, config, device)
    ):
        if step >= max_steps:
            break

        optimizer.zero_grad()

        # encode context through frozen backbone + thought block
        with torch.no_grad():
            hidden, _ = backbone.encode(context_ids)
            z_final, _ = thought_block(hidden)

        # decoder: predict target tokens from z_final
        # teacher forcing: feed target_ids shifted by 1
        decoder_input = target_ids[:, :-1]   # input to decoder
        decoder_target = target_ids[:, 1:]   # what it should predict

        logits = decoder(z_final, target_ids=decoder_input)

        # cross-entropy loss
        # reshape: (batch * seq_len, vocab_size) vs (batch * seq_len)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            decoder_target.reshape(-1),
            ignore_index=0,  # ignore padding
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            decoder.parameters(),
            train_cfg.get("grad_clip", 1.0),
        )

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        # ── logging ───────────────────────────────────────────────────

        if (step + 1) % train_cfg.get("log_every", 50) == 0:
            avg_loss = total_loss / (step + 1)
            elapsed = time.time() - start_time
            perplexity = min(torch.exp(torch.tensor(avg_loss)).item(), 1e6)
            print(
                f"[Stage2] Step {step + 1:>5d} | "
                f"Loss: {loss.item():.4f} | "
                f"Avg: {avg_loss:.4f} | "
                f"PPL: {perplexity:.1f} | "
                f"LR: {scheduler.get_last_lr()[0]:.2e} | "
                f"Time: {elapsed:.0f}s"
            )

            # sample a generation every 200 steps
            if (step + 1) % 200 == 0:
                _sample_generation(
                    backbone, thought_block, decoder,
                    device, config,
                )

        # ── checkpointing ─────────────────────────────────────────────

        if (step + 1) % train_cfg.get("checkpoint_every", 1000) == 0:
            ckpt_path = checkpoint_dir / f"stage2_step{step + 1}.pt"
            torch.save(
                {
                    "decoder": decoder.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step + 1,
                    "loss": total_loss / (step + 1),
                },
                ckpt_path,
            )
            print(f"[Stage2] Checkpoint saved: {ckpt_path}")

    # ── final save ────────────────────────────────────────────────────

    final_path = checkpoint_dir / "stage2_final.pt"
    torch.save(
        {
            "decoder": decoder.state_dict(),
            "step": step + 1 if "step" in dir() else 0,
            "loss": total_loss / max(step + 1, 1) if "step" in dir() else 0,
        },
        final_path,
    )
    print(f"[Stage2] Training complete. Final checkpoint: {final_path}")


# ── sample generation ─────────────────────────────────────────────────

@torch.no_grad()
def _sample_generation(backbone, thought_block, decoder, device, config):
    """Generate a sample to monitor training quality."""
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

    if backbone.tokenizer is not None:
        text = backbone.tokenizer.decode(output_ids[0], skip_special_tokens=True)
    else:
        text = str(output_ids[0].tolist()[:20])

    print(f"[Stage2] Sample: \"{prompt}\" → \"{text[:80]}...\"")


# ── CLI ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hypnos Stage 2 Decoder Training"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/hypnos_130m.yaml",
    )
    parser.add_argument(
        "--stage1-checkpoint",
        type=str,
        default=None,
        help="Path to Stage 1 checkpoint (default: checkpoints/stage1/stage1_step1000.pt)",
    )
    args = parser.parse_args()
    train_stage2(args.config, args.stage1_checkpoint)
