"""
model/decoder.py — Latent → Text decoder

Maps refined hidden states from the ThoughtBlock back into token-level
logits for text generation.
"""

import torch
import torch.nn as nn
from typing import Dict, Any


class LatentDecoder(nn.Module):
    """
    Decoder head: projects latent representations h ∈ ℝ^D to vocab logits.

    Architecture: LayerNorm → Linear(D, D) → GELU → Linear(D, V)
    Optionally ties weights with the backbone embedding matrix.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.d_model: int = config.get("d_model", 768)
        self.vocab_size: int = config.get("vocab_size", 50_280)
        self.tie_weights: bool = config.get("tie_weights", False)

        self.norm = nn.LayerNorm(self.d_model)
        self.proj = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
        )
        self.lm_head = nn.Linear(self.d_model, self.vocab_size, bias=False)

    def tie_to_embedding(self, embedding_weight: torch.Tensor) -> None:
        """Tie lm_head weights to the backbone embedding matrix."""
        self.lm_head.weight = embedding_weight

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        h : (B, L, D) latent hidden states

        Returns
        -------
        logits : (B, L, V) unnormalised token probabilities
        """
        x = self.norm(h)
        x = self.proj(x)
        logits = self.lm_head(x)
        return logits

    def decode_top_k(
        self,
        h: torch.Tensor,
        k: int = 5,
        temperature: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Convenience: return top-k token ids and their probabilities.

        Returns
        -------
        dict with 'token_ids' (B, L, k) and 'probs' (B, L, k)
        """
        logits = self.forward(h) / temperature
        probs = torch.softmax(logits, dim=-1)
        top_probs, top_ids = probs.topk(k, dim=-1)
        return {"token_ids": top_ids, "probs": top_probs}
