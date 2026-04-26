"""
state/manager.py — Latent state (h_t) persistence

Manages the rolling buffer of hidden states, supports save/load to disk,
and provides utilities for state interpolation and retrieval.
"""

import torch
import os
import json
from typing import Dict, Any, Optional, List
from collections import deque
from datetime import datetime


class StateManager:
    """
    Manages a rolling history of latent hidden states h_t.

    Features
    --------
    - Fixed-size FIFO buffer of recent states
    - Save / load to disk (torch tensors + metadata JSON)
    - Interpolation between two states (for smooth transitions)
    - Mean-pool over the buffer for a "consolidated" state
    """

    def __init__(self, config: Dict[str, Any]):
        self.max_history: int = config.get("max_history", 64)
        self.d_model: int = config.get("d_model", 768)
        self._buffer: deque = deque(maxlen=self.max_history)
        self._timestamps: deque = deque(maxlen=self.max_history)

    # ── core operations ──────────────────────────────────────────────

    def push(self, h: torch.Tensor) -> None:
        """Append a new hidden state to the buffer (detached, on CPU)."""
        self._buffer.append(h.detach().cpu())
        self._timestamps.append(datetime.utcnow().isoformat())

    def latest(self) -> Optional[torch.Tensor]:
        """Return the most recent state, or None if the buffer is empty."""
        return self._buffer[-1] if self._buffer else None

    def history(self, n: Optional[int] = None) -> List[torch.Tensor]:
        """Return the last *n* states (default: all)."""
        items = list(self._buffer)
        if n is not None:
            items = items[-n:]
        return items

    def consolidated(self) -> Optional[torch.Tensor]:
        """Mean-pool all buffered states into a single tensor."""
        if not self._buffer:
            return None
        stacked = torch.stack(list(self._buffer), dim=0)
        return stacked.mean(dim=0)

    def clear(self) -> None:
        """Flush the buffer."""
        self._buffer.clear()
        self._timestamps.clear()

    # ── persistence ──────────────────────────────────────────────────

    def save(self, directory: str) -> None:
        """Save buffer to disk as individual .pt files + metadata."""
        os.makedirs(directory, exist_ok=True)
        meta = {
            "count": len(self._buffer),
            "d_model": self.d_model,
            "max_history": self.max_history,
            "timestamps": list(self._timestamps),
        }
        with open(os.path.join(directory, "state_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        for idx, h in enumerate(self._buffer):
            torch.save(h, os.path.join(directory, f"h_{idx:04d}.pt"))

    def load(self, directory: str) -> None:
        """Restore buffer from disk."""
        meta_path = os.path.join(directory, "state_meta.json")
        with open(meta_path) as f:
            meta = json.load(f)
        self.clear()
        for idx in range(meta["count"]):
            h = torch.load(os.path.join(directory, f"h_{idx:04d}.pt"))
            self._buffer.append(h)
        self._timestamps.extend(meta.get("timestamps", []))

    # ── utilities ────────────────────────────────────────────────────

    @staticmethod
    def interpolate(
        h_a: torch.Tensor,
        h_b: torch.Tensor,
        alpha: float = 0.5,
    ) -> torch.Tensor:
        """Spherical linear interpolation between two states."""
        h_a_n = torch.nn.functional.normalize(h_a.float(), dim=-1)
        h_b_n = torch.nn.functional.normalize(h_b.float(), dim=-1)
        dot = (h_a_n * h_b_n).sum(dim=-1, keepdim=True).clamp(-1, 1)
        omega = torch.acos(dot)
        sin_omega = torch.sin(omega).clamp(min=1e-8)
        coeff_a = torch.sin((1 - alpha) * omega) / sin_omega
        coeff_b = torch.sin(alpha * omega) / sin_omega
        return (coeff_a * h_a + coeff_b * h_b).to(h_a.dtype)

    def __len__(self) -> int:
        return len(self._buffer)

    def __repr__(self) -> str:
        return f"StateManager(len={len(self)}, max={self.max_history}, d={self.d_model})"
