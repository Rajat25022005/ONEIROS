"""
hypnos/probes/state_probe.py — Interpretability tools

Snapshot, compare, and analyse Mamba's latent state h_t.
The closest you can get to reading the model's mind.
"""

import torch
import numpy as np
from typing import List, Optional, Dict
from pathlib import Path
import json


class StateProbe:
    """
    Interpretability toolkit for Hypnos latent states.

    - ``snapshot(label)`` — save a named copy of the current h_t
    - ``measure_drift(a, b)`` — cosine drift between two snapshots
    - ``compare(a, b)`` — full statistical comparison
    """

    def __init__(self, model, probe_dir: str = ".hypnos/probes"):
        self.model = model
        self.probe_dir = Path(probe_dir)
        self.probe_dir.mkdir(parents=True, exist_ok=True)
        self._snapshots: Dict[str, torch.Tensor] = {}

    def snapshot(self, label: str):
        """Take a named snapshot of the current flat state."""
        h = self._get_flat_state()
        if h is None:
            print("[Probe] No state to snapshot.")
            return

        self._snapshots[label] = h.clone()

        stats = {
            "label": label,
            "mean": h.mean().item(),
            "std": h.std().item(),
            "norm": h.norm().item(),
            "min": h.min().item(),
            "max": h.max().item(),
            "dim": h.shape[0],
        }

        # save to disk
        snap_path = self.probe_dir / f"snapshot_{label}.json"
        with open(snap_path, "w") as f:
            json.dump(stats, f, indent=2)

        print(
            f"[Probe] Snapshot '{label}': "
            f"dim={stats['dim']} | "
            f"norm={stats['norm']:.2f} | "
            f"mean={stats['mean']:.4f}"
        )

    def measure_drift(self, label_a: str, label_b: str) -> float:
        """
        Cosine drift between two snapshots.

        drift = 1 − cosine_similarity(a, b)
        0.0 = identical, 1.0 = perpendicular, 2.0 = opposite.
        """
        if label_a not in self._snapshots or label_b not in self._snapshots:
            available = list(self._snapshots.keys())
            print(f"[Probe] Snapshot not found. Available: {available}")
            return -1.0

        a = self._snapshots[label_a].float()
        b = self._snapshots[label_b].float()

        cosine_sim = torch.nn.functional.cosine_similarity(
            a.unsqueeze(0), b.unsqueeze(0)
        ).item()
        drift = 1.0 - cosine_sim

        l2 = (a - b).norm().item()

        print(
            f"[Probe] Drift '{label_a}' → '{label_b}': "
            f"cosine_drift={drift:.4f} | "
            f"L2={l2:.2f}"
        )
        return drift

    def compare(self, label_a: str, label_b: str) -> Dict[str, float]:
        """Full statistical comparison between two snapshots."""
        a = self._snapshots[label_a].float()
        b = self._snapshots[label_b].float()

        return {
            "cosine_similarity": torch.nn.functional.cosine_similarity(
                a.unsqueeze(0), b.unsqueeze(0)
            ).item(),
            "l2_distance": (a - b).norm().item(),
            "mean_diff": (a.mean() - b.mean()).item(),
            "std_diff": (a.std() - b.std()).item(),
            "norm_a": a.norm().item(),
            "norm_b": b.norm().item(),
        }

    def list_snapshots(self) -> List[str]:
        """Return all available snapshot labels."""
        return list(self._snapshots.keys())

    # ── internal ──────────────────────────────────────────────────────

    def _get_flat_state(self) -> Optional[torch.Tensor]:
        """Flatten Mamba's nested cache into a single vector."""
        cache = self.model._mamba_cache
        if cache is None:
            return None

        try:
            if isinstance(cache, torch.Tensor):
                return cache.flatten().float().cpu()

            if isinstance(cache, (list, tuple)):
                parts = []
                for layer_state in cache:
                    if isinstance(layer_state, torch.Tensor):
                        parts.append(layer_state.flatten().float().cpu())
                    elif isinstance(layer_state, (list, tuple)):
                        for s in layer_state:
                            if isinstance(s, torch.Tensor):
                                parts.append(s.flatten().float().cpu())
                if parts:
                    return torch.cat(parts)

            # HuggingFace Mamba cache object — extract tensors from attributes
            if hasattr(cache, "conv_states") and hasattr(cache, "ssm_states"):
                parts = []
                for cs in cache.conv_states:
                    if isinstance(cs, torch.Tensor):
                        parts.append(cs.flatten().float().cpu())
                for ss in cache.ssm_states:
                    if isinstance(ss, torch.Tensor):
                        parts.append(ss.flatten().float().cpu())
                if parts:
                    return torch.cat(parts)

            print("[Probe] Could not flatten cache — unknown structure.")
            return None

        except Exception as e:
            print(f"[Probe] Error flattening state: {e}")
            return None
