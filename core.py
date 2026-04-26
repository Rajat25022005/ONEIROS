"""
ONEIROS — core.py
Hypnos: the main orchestrator class that binds backbone, thought blocks,
EMA teacher, decoder, state manager, cognition gate, and dream loop
into a single coherent agent.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any

from model.backbone import MambaBackbone
from model.thought_block import ThoughtBlock
from model.ema_teacher import EMATeacher
from model.decoder import LatentDecoder
from state.manager import StateManager
from gate.cognition_gate import CognitionGate
from dream.loop import DreamLoop


class Hypnos(nn.Module):
    """
    Hypnos — A latent-reasoning agent with awake / dream duality.

    In *awake* mode Hypnos processes incoming tokens through a Mamba backbone,
    refines the hidden state via K-step latent thought blocks, and decodes
    text when needed.

    In *dream* mode the cognition gate closes the input stream and Hypnos
    autonomously evolves its latent state, consolidating knowledge and
    exploring novel representations under EMA-teacher coherence supervision.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config

        # ── core modules ──────────────────────────────────────────────
        self.backbone = MambaBackbone(config["model"])
        self.thought_block = ThoughtBlock(config["thought"])
        self.ema_teacher = EMATeacher(
            student=self.thought_block,
            decay=config["ema"]["decay"],
        )
        self.decoder = LatentDecoder(config["decoder"])

        # ── state & control ───────────────────────────────────────────
        self.state_manager = StateManager(config["state"])
        self.cognition_gate = CognitionGate(config["gate"])
        self.dream_loop = DreamLoop(
            thought_block=self.thought_block,
            ema_teacher=self.ema_teacher,
            state_manager=self.state_manager,
            config=config["dream"],
        )

        # ── latent state ──────────────────────────────────────────────
        self._h: Optional[torch.Tensor] = None

    # ── public API ────────────────────────────────────────────────────

    def forward(
        self,
        input_ids: torch.Tensor,
        dream: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        input_ids : (B, L) token ids — ignored when *dream* is True.
        dream     : if True, run autonomous dream loop instead of awake pass.

        Returns
        -------
        dict with keys 'logits', 'h_t', and optionally 'dream_metrics'.
        """
        if dream:
            return self._dream_step()
        return self._awake_step(input_ids)

    # ── private helpers ───────────────────────────────────────────────

    def _awake_step(self, input_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Encode → think → decode."""
        h = self.backbone(input_ids)
        h = self.thought_block(h, steps=self.config["thought"]["k_steps"])
        self._h = h
        self.state_manager.push(h)
        logits = self.decoder(h)
        return {"logits": logits, "h_t": h}

    def _dream_step(self) -> Dict[str, torch.Tensor]:
        """Run one autonomous dream cycle."""
        h_init = self._h if self._h is not None else self.state_manager.latest()
        dream_out = self.dream_loop.step(h_init)
        self._h = dream_out["h_t"]
        self.state_manager.push(self._h)
        return dream_out

    def save_state(self, path: str) -> None:
        """Persist latent state + model weights."""
        self.state_manager.save(path)
        torch.save(self.state_dict(), f"{path}/hypnos_weights.pt")

    def load_state(self, path: str) -> None:
        """Restore latent state + model weights."""
        self.state_manager.load(path)
        self.load_state_dict(torch.load(f"{path}/hypnos_weights.pt"))
        self._h = self.state_manager.latest()
