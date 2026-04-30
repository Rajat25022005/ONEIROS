"""
hypnos/model/backbone.py — Mamba-130M backbone

Wraps HuggingFace's MambaModel with persistent state (cache_params).
Falls back to a lightweight embedding stub when transformers is unavailable,
so the rest of the system can develop and test without downloading Mamba.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class MambaBackbone(nn.Module):
    """
    Mamba SSM backbone with persistent hidden state.

    The key property: Mamba's ``cache_params`` is the recurrent state h_t
    that evolves with every forward pass and can be saved/restored between
    sessions.  This is what makes Hypnos's persistent memory possible.
    """

    def __init__(
        self,
        model_name: str = "state-spaces/mamba-130m-hf",
        use_slow_path: bool = False,
    ):
        super().__init__()
        self.model_name = model_name
        self.use_slow_path = use_slow_path
        self._load_model()

    # ── model loading ─────────────────────────────────────────────────

    def _load_model(self):
        """Try loading the real Mamba model; fall back to stub on failure."""
        try:
            from transformers import MambaModel, MambaConfig, AutoTokenizer

            self.config = MambaConfig.from_pretrained(self.model_name)

            if self.use_slow_path:
                self.config.use_cuda = False

            self.model = MambaModel.from_pretrained(
                self.model_name,
                config=self.config,
                torch_dtype=torch.bfloat16,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.hidden_size = self.config.hidden_size
            self._stub_mode = False
            print(f"[Hypnos] Loaded {self.model_name} (hidden_size={self.hidden_size})")

        except Exception as e:
            print(f"[Hypnos] Warning: Could not load {self.model_name}: {e}")
            print("[Hypnos] Falling back to lightweight stub for development.")
            self._load_stub()

    def _load_stub(self):
        """Lightweight stub that mimics Mamba's interface."""
        self.hidden_size = 768
        self.tokenizer = None
        self.model = None
        self._stub_mode = True
        self._stub_embedding = nn.Embedding(50257, self.hidden_size)

    # ── forward pass ──────────────────────────────────────────────────

    def encode(
        self,
        input_ids: torch.Tensor,
        cache_params=None,
    ) -> Tuple[torch.Tensor, any]:
        """
        Encode token ids into a hidden representation + updated state.

        Parameters
        ----------
        input_ids    : (B, L) token indices.
        cache_params : Mamba recurrent state from a previous call.

        Returns
        -------
        hidden     : (B, hidden_size) — last-token hidden state.
        new_cache  : updated Mamba recurrent state (h_{t+1}).
        """
        if getattr(self, "_stub_mode", False):
            hidden = self._stub_embedding(input_ids).mean(dim=1)
            return hidden, cache_params

        outputs = self.model(
            input_ids=input_ids,
            cache_params=cache_params,
            use_cache=True,
            return_dict=True,
        )

        last_hidden = outputs.last_hidden_state[:, -1, :]
        new_cache = outputs.cache_params
        return last_hidden, new_cache

    def tokenize(self, text: str, device: torch.device) -> torch.Tensor:
        """Convert text to token ids on the given device."""
        if self.tokenizer is None:
            return torch.randint(0, 50257, (1, 16)).to(device)

        tokens = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        return tokens["input_ids"].to(device)

    def forward(self, input_ids, cache_params=None):
        """PyTorch convention: delegates to encode()."""
        return self.encode(input_ids, cache_params)
