"""
hypnos/state/manager.py — Persistent h_t management

Saves and restores Mamba's recurrent state between sessions so the
model accumulates experience indefinitely.  Tracks metadata (session
count, tokens processed, dream cycles) alongside the state tensor.
"""

import torch
import json
import time
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict


@dataclass
class StateMetadata:
    """Biography of the model's accumulated experience."""
    created_at: float
    updated_at: float
    session_count: int
    total_tokens_processed: int
    dream_cycles_completed: int
    version: str = "0.1.0"


class StateManager:
    """
    Manages h_t persistence: load, save, reset.

    State is stored as a PyTorch checkpoint containing:
    - ``h_t``: the Mamba recurrent state (cache_params)
    - ``metadata``: StateMetadata as a plain dict
    """

    def __init__(self, state_dir: str, device: torch.device):
        self.state_dir = Path(state_dir)
        self.device = device
        self.state_dir.mkdir(parents=True, exist_ok=True)

        self.state_path = self.state_dir / "h_t.pt"
        self.meta_path = self.state_dir / "metadata.json"

        self._cache: Optional[Any] = None
        self._metadata: Optional[StateMetadata] = None

    # ── load / save ──────────────────────────────────────────────────

    def load(self) -> Optional[Any]:
        """Load h_t from disk. Returns None on first session."""
        if not self.state_path.exists():
            print("[StateManager] No prior state found. Starting fresh.")
            self._metadata = StateMetadata(
                created_at=time.time(),
                updated_at=time.time(),
                session_count=0,
                total_tokens_processed=0,
                dream_cycles_completed=0,
            )
            return None

        try:
            checkpoint = torch.load(
                self.state_path,
                map_location=self.device,
                weights_only=False,
            )
            self._cache = checkpoint["h_t"]
            self._metadata = StateMetadata(**checkpoint["metadata"])
            self._metadata.session_count += 1

            age_hours = (time.time() - self._metadata.updated_at) / 3600
            print(
                f"[StateManager] State loaded. "
                f"Sessions: {self._metadata.session_count} | "
                f"Last updated: {age_hours:.1f}h ago | "
                f"Dream cycles: {self._metadata.dream_cycles_completed}"
            )
            return self._cache

        except Exception as e:
            print(f"[StateManager] Error loading state: {e}")
            print("[StateManager] Starting fresh.")
            self._metadata = StateMetadata(
                created_at=time.time(),
                updated_at=time.time(),
                session_count=0,
                total_tokens_processed=0,
                dream_cycles_completed=0,
            )
            return None

    def save(self, h_t: Any, tokens_processed: int = 0):
        """Save h_t + metadata to disk."""
        if self._metadata is None:
            self._metadata = StateMetadata(
                created_at=time.time(),
                updated_at=time.time(),
                session_count=0,
                total_tokens_processed=0,
                dream_cycles_completed=0,
            )

        self._metadata.updated_at = time.time()
        self._metadata.total_tokens_processed += tokens_processed
        self._cache = h_t

        try:
            checkpoint = {
                "h_t": h_t,
                "metadata": asdict(self._metadata),
            }
            torch.save(checkpoint, self.state_path)

            # also save human-readable metadata
            with open(self.meta_path, "w") as f:
                json.dump(asdict(self._metadata), f, indent=2)

        except Exception as e:
            print(f"[StateManager] Error saving state: {e}")

    # ── utilities ────────────────────────────────────────────────────

    def increment_dream_cycles(self, count: int = 1):
        """Increment the dream cycle counter."""
        if self._metadata:
            self._metadata.dream_cycles_completed += count

    def reset(self):
        """Erase all accumulated experience. Requires manual confirmation."""
        confirmation = input(
            "[StateManager] ⚠ This will erase all accumulated experience. "
            "Type 'RESET' to confirm: "
        )
        if confirmation == "RESET":
            if self.state_path.exists():
                self.state_path.unlink()
            if self.meta_path.exists():
                self.meta_path.unlink()
            self._cache = None
            self._metadata = None
            print("[StateManager] State erased.")
        else:
            print("[StateManager] Reset cancelled.")

    @property
    def has_state(self) -> bool:
        """True if a saved state exists on disk."""
        return self.state_path.exists()

    @property
    def session_count(self) -> int:
        return self._metadata.session_count if self._metadata else 0

    @property
    def metadata(self) -> Optional[StateMetadata]:
        return self._metadata

    def __repr__(self) -> str:
        return (
            f"StateManager(dir={self.state_dir}, "
            f"has_state={self.has_state}, "
            f"sessions={self.session_count})"
        )
