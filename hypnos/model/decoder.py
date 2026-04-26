"""
hypnos/model/decoder.py — Latent → Text decoder

Maps the final thought vector z_K back to token-level logits using a
4-layer Transformer decoder with cross-attention to the latent memory.
Supports teacher-forced training and greedy autoregressive decoding.
"""

import torch
import torch.nn as nn
from typing import Optional


class LatentDecoder(nn.Module):
    """
    Decoder head: latent_proj → TransformerDecoder → output_proj.

    The thought vector z becomes a 1-token "memory" that the decoder
    cross-attends to at every layer. This is the entire context for
    generating the response.
    """

    def __init__(
        self,
        latent_dim: int = 256,
        vocab_size: int = 50257,
        hidden_dim: int = 512,
        max_length: int = 128,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.max_length = max_length

        # project thought vector into decoder hidden space
        self.latent_proj = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        # 4-layer Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=4,
        )

        # final projection to vocabulary
        self.output_proj = nn.Linear(hidden_dim, vocab_size)

        # token and position embeddings
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Embedding(max_length, hidden_dim)

    def forward(
        self,
        z: torch.Tensor,
        target_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        z          : (B, latent_dim) thought vector from ThoughtBlock.
        target_ids : (B, L) target token ids for teacher forcing.
                     If None, runs greedy autoregressive decoding.

        Returns
        -------
        logits or generated token ids.
        """
        batch_size = z.shape[0]
        device = z.device

        # thought vector → 1-token memory for cross-attention
        memory = self.latent_proj(z).unsqueeze(1)  # (B, 1, hidden_dim)

        if target_ids is None:
            return self._greedy_decode(memory, device, batch_size)

        # teacher-forced forward pass
        seq_len = target_ids.shape[1]
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        tgt = self.token_embedding(target_ids) + self.pos_embedding(positions)

        # causal mask: prevent attending to future tokens
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            seq_len, device=device
        )

        out = self.transformer_decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask,
        )
        logits = self.output_proj(out)
        return logits

    @torch.no_grad()
    def _greedy_decode(self, memory, device, batch_size):
        """Autoregressive greedy decoding from the thought vector."""
        # start with BOS token (id=1)
        generated = torch.ones(batch_size, 1, dtype=torch.long, device=device)

        for step in range(self.max_length - 1):
            seq_len = generated.shape[1]
            positions = torch.arange(seq_len, device=device).unsqueeze(0)
            tgt = (
                self.token_embedding(generated)
                + self.pos_embedding(positions)
            )
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                seq_len, device=device
            )

            out = self.transformer_decoder(
                tgt=tgt, memory=memory, tgt_mask=tgt_mask
            )
            next_logits = self.output_proj(out[:, -1, :])
            next_token = next_logits.argmax(dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=1)

            # stop if all items generated EOS (id=2)
            if (next_token == 2).all():
                break

        return generated
