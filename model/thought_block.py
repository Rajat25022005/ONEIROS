"""
model/thought_block.py — K-step latent reasoning

Implements iterative refinement in latent space: given a hidden state h,
the ThoughtBlock applies K learned transition steps *without* generating
tokens, allowing the model to "think" before speaking.
"""

import torch
import torch.nn as nn
from typing import Dict, Any


class ThoughtStep(nn.Module):
    """One refinement step: LayerNorm → MLP → residual."""

    def __init__(self, d_model: int, expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * expansion, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return h + self.mlp(self.norm(h))


class ThoughtBlock(nn.Module):
    """
    K-step latent reasoning block.

    Applies a shared (or per-step) transition function K times to the
    hidden state.  Supports both weight-shared and weight-unshared modes.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.d_model: int = config.get("d_model", 768)
        self.k_steps: int = config.get("k_steps", 4)
        self.share_weights: bool = config.get("share_weights", True)
        expansion: int = config.get("expansion", 4)
        dropout: float = config.get("dropout", 0.1)

        if self.share_weights:
            self.step = ThoughtStep(self.d_model, expansion, dropout)
        else:
            self.steps = nn.ModuleList(
                [ThoughtStep(self.d_model, expansion, dropout) for _ in range(self.k_steps)]
            )

        self.gate = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model),
            nn.Sigmoid(),
        )

    def forward(
        self,
        h: torch.Tensor,
        steps: int | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        h     : (B, L, D) input hidden state
        steps : override for number of thinking steps (default: self.k_steps)

        Returns
        -------
        h_out : (B, L, D) refined hidden state
        """
        k = steps if steps is not None else self.k_steps
        h_init = h

        for i in range(k):
            if self.share_weights:
                h = self.step(h)
            else:
                h = self.steps[min(i, len(self.steps) - 1)](h)

        # gated residual connection back to the input
        gate_val = self.gate(torch.cat([h_init, h], dim=-1))
        h_out = gate_val * h + (1 - gate_val) * h_init
        return h_out
