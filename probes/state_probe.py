"""
probes/state_probe.py — Interpretability tools for latent state analysis
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional
import numpy as np


class StateProbe:
    """Interpretability toolkit for analysing Hypnos latent states."""

    @staticmethod
    def cosine_trajectory(trace: List[torch.Tensor]) -> List[float]:
        """Pairwise cosine similarity between consecutive states in a trace."""
        sims = []
        for i in range(1, len(trace)):
            a = trace[i - 1].float().flatten()
            b = trace[i].float().flatten()
            sim = nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
            sims.append(sim)
        return sims

    @staticmethod
    def l2_drift(trace: List[torch.Tensor]) -> List[float]:
        """L2 distance between consecutive states."""
        drifts = []
        for i in range(1, len(trace)):
            d = (trace[i].float() - trace[i - 1].float()).norm().item()
            drifts.append(d)
        return drifts

    @staticmethod
    def pca_project(trace: List[torch.Tensor], n_components: int = 3):
        """PCA projection of trace states for visualization."""
        stacked = torch.stack([t.float().flatten() for t in trace])
        mean = stacked.mean(dim=0, keepdim=True)
        centered = stacked - mean
        U, S, V = torch.svd(centered)
        projected = centered @ V[:, :n_components]
        return projected.numpy()

    @staticmethod
    def entropy_profile(logits_seq: List[torch.Tensor]) -> List[float]:
        """Token-level entropy across a sequence of logit tensors."""
        entropies = []
        for logits in logits_seq:
            probs = torch.softmax(logits.float(), dim=-1)
            ent = -(probs * probs.clamp(min=1e-9).log()).sum(dim=-1).mean().item()
            entropies.append(ent)
        return entropies

    @staticmethod
    def attention_to_state(h: torch.Tensor) -> Dict[str, float]:
        """Basic statistics of a hidden state."""
        h_f = h.float()
        return {
            "mean": h_f.mean().item(),
            "std": h_f.std().item(),
            "norm": h_f.norm().item(),
            "min": h_f.min().item(),
            "max": h_f.max().item(),
        }
