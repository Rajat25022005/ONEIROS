"""
gate/cognition_gate.py — Awake / Dream switch

A learned gating mechanism that decides whether the system should process
external input (awake) or enter autonomous internal reasoning (dream).
"""

import torch
import torch.nn as nn
from typing import Dict, Any
from enum import Enum, auto


class CognitionMode(Enum):
    AWAKE = auto()
    DREAM = auto()
    TRANSITION = auto()


class CognitionGate(nn.Module):
    """
    Cognition gate: a soft binary gate that modulates how much external
    input vs. internal dreaming drives the next state.

    g(h) → [0, 1]   where 0 = full dream,  1 = full awake.

    The gate uses the current hidden state plus an optional "stimulus"
    signal (e.g., entropy of input distribution) to decide the mode.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.d_model: int = config.get("d_model", 768)
        self.dream_threshold: float = config.get("dream_threshold", 0.3)
        self.awake_threshold: float = config.get("awake_threshold", 0.7)

        self.gate_net = nn.Sequential(
            nn.Linear(self.d_model + 1, self.d_model // 2),
            nn.GELU(),
            nn.Linear(self.d_model // 2, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        h: torch.Tensor,
        stimulus: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        h        : (B, D) pooled hidden state
        stimulus : (B, 1) scalar stimulus signal (default: zeros)

        Returns
        -------
        dict with 'gate_value' (B, 1) and 'mode' (CognitionMode).
        """
        if stimulus is None:
            stimulus = torch.zeros(h.size(0), 1, device=h.device)

        gate_input = torch.cat([h, stimulus], dim=-1)
        gate_value = self.gate_net(gate_input)  # (B, 1)

        mode = self._classify_mode(gate_value.mean().item())

        return {
            "gate_value": gate_value,
            "mode": mode,
        }

    def _classify_mode(self, g: float) -> CognitionMode:
        if g >= self.awake_threshold:
            return CognitionMode.AWAKE
        elif g <= self.dream_threshold:
            return CognitionMode.DREAM
        return CognitionMode.TRANSITION

    def blend(
        self,
        h_awake: torch.Tensor,
        h_dream: torch.Tensor,
        gate_value: torch.Tensor,
    ) -> torch.Tensor:
        """Blend awake and dream hidden states using the gate value."""
        return gate_value * h_awake + (1 - gate_value) * h_dream
