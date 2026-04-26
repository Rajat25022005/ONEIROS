"""
model/backbone.py — Mamba-130M backbone

Stubs the Mamba SSM backbone cleanly so the rest of the system can run
without a transformers / mamba-ssm dependency.  Drop-in replaceable with
a real Mamba checkpoint once available.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional


class MambaBlock(nn.Module):
    """Single Mamba-style selective-scan block (simplified stub)."""

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # projections
        self.in_proj = nn.Linear(d_model, 2 * d_model, bias=False)
        self.conv1d = nn.Conv1d(
            d_model, d_model,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=d_model,
        )
        self.x_proj = nn.Linear(d_model, d_state * 2, bias=False)
        self.dt_proj = nn.Linear(d_model, d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # SSM parameters
        self.A = nn.Parameter(torch.randn(d_model, d_state))
        self.D = nn.Parameter(torch.ones(d_model))

        self.norm = nn.LayerNorm(d_model)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, L, D)

        Returns
        -------
        (B, L, D) — same shape.
        """
        residual = x
        x = self.norm(x)

        xz = self.in_proj(x)
        x_part, z = xz.chunk(2, dim=-1)

        # conv branch
        x_conv = x_part.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :x_part.size(1)]
        x_conv = x_conv.transpose(1, 2)
        x_conv = self.act(x_conv)

        # gate
        y = x_conv * self.act(z)
        out = self.out_proj(y)
        return out + residual


class MambaBackbone(nn.Module):
    """
    Mamba-130M backbone stub.

    Embeds token ids → d_model, passes through N MambaBlocks,
    and returns the final hidden states.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.d_model: int = config.get("d_model", 768)
        self.n_layers: int = config.get("n_layers", 24)
        self.vocab_size: int = config.get("vocab_size", 50_280)

        self.embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.layers = nn.ModuleList(
            [MambaBlock(self.d_model) for _ in range(self.n_layers)]
        )
        self.norm_f = nn.LayerNorm(self.d_model)

    def forward(
        self,
        input_ids: torch.Tensor,
        h_prev: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        input_ids : (B, L)  token indices
        h_prev    : optional previous hidden state for warm-starting

        Returns
        -------
        h : (B, L, D)  contextualised hidden states
        """
        x = self.embedding(input_ids)
        if h_prev is not None:
            x = x + h_prev[:, : x.size(1), :]

        for layer in self.layers:
            x = layer(x)

        return self.norm_f(x)
