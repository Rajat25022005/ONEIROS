"""
train/stage1_jepa.py — Stage-1 JEPA pre-training

Joint Embedding Predictive Architecture training: the model learns to
predict future latent states from current context without pixel/token
reconstruction, supervised by the EMA teacher's coherence signal.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import yaml
import os
import sys
from typing import Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.backbone import MambaBackbone
from model.thought_block import ThoughtBlock
from model.ema_teacher import EMATeacher
from model.decoder import LatentDecoder


class DummyTokenDataset(Dataset):
    """Placeholder dataset — replace with real tokenised corpus."""
    def __init__(self, n_samples=1000, seq_len=512, vocab_size=50280):
        self.data = torch.randint(0, vocab_size, (n_samples, seq_len))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def load_config(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def train_stage1(config_path: str):
    cfg = load_config(config_path)
    tcfg = cfg["training"]["stage1"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # models
    backbone = MambaBackbone(cfg["model"]).to(device)
    thought = ThoughtBlock(cfg["thought"]).to(device)
    teacher = EMATeacher(thought, decay=cfg["ema"]["decay"])
    teacher.to(device)

    # optimizer
    params = list(backbone.parameters()) + list(thought.parameters())
    optimizer = optim.AdamW(params, lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])

    # data
    dataset = DummyTokenDataset(seq_len=tcfg["max_seq_len"], vocab_size=cfg["model"]["vocab_size"])
    loader = DataLoader(dataset, batch_size=tcfg["batch_size"], shuffle=True)

    # training loop
    for epoch in range(tcfg["epochs"]):
        total_loss = 0.0
        for batch_idx, tokens in enumerate(loader):
            tokens = tokens.to(device)
            optimizer.zero_grad()

            h = backbone(tokens)
            # predict future: shift by 1
            h_context = h[:, :-1, :]
            h_target = h[:, 1:, :].detach()

            h_pred = thought(h_context, steps=cfg["thought"]["k_steps"])
            teacher_h = teacher(h_context, steps=cfg["thought"]["k_steps"])

            # JEPA loss: predict next latent
            jepa_loss = nn.functional.mse_loss(h_pred, h_target)
            coh_loss = teacher.coherence_loss(h_pred, teacher_h)

            loss = (tcfg["jepa_loss_weight"] * jepa_loss
                    + tcfg["coherence_loss_weight"] * coh_loss)

            loss.backward()
            nn.utils.clip_grad_norm_(params, tcfg["gradient_clip"])
            optimizer.step()
            teacher.update(thought)
            total_loss += loss.item()

        avg = total_loss / max(len(loader), 1)
        print(f"[Epoch {epoch+1:03d}] loss={avg:.4f}")


if __name__ == "__main__":
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "configs", "hypnos_130m.yaml")
    train_stage1(cfg_path)
