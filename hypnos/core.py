"""
hypnos/core.py — Main Hypnos class

The central orchestrator: builds the model stack, manages persistent
state, wires up the cognition gate and dream loop, and exposes the
public API (think, start, stop, save, load).
"""

import os
import yaml
import torch
import torch.nn as nn
from typing import Optional

from hypnos.model.backbone import MambaBackbone
from hypnos.model.thought_block import ThoughtBlock
from hypnos.model.ema_teacher import EMATeacher
from hypnos.model.decoder import LatentDecoder
from hypnos.state.manager import StateManager
from hypnos.gate.cognition_gate import CognitionGate
from hypnos.dream.loop import DreamLoop


class Hypnos:
    """
    Hypnos — A continuously reasoning AI with persistent memory.

    Usage::

        model = Hypnos.from_config("configs/hypnos_130m.yaml")
        model.start()          # activate gate + dream loop
        response = model.think("What is consciousness?")
        model.stop()           # clean shutdown
    """

    def __init__(self, config: dict):
        self.config = config
        self.device = self._select_device()
        print(f"[Hypnos] Device: {self.device}")

        self._build_model()

        # persistent state
        state_dir = config.get("state", {}).get("dir", ".hypnos/state")
        self.state_manager = StateManager(state_dir, self.device)
        self._mamba_cache = self.state_manager.load()

        # gate and dream loop (created in start())
        self._gate: Optional[CognitionGate] = None
        self._dream_loop: Optional[DreamLoop] = None

    # ── constructors ──────────────────────────────────────────────────

    @classmethod
    def from_config(cls, config_path: str) -> "Hypnos":
        """Create Hypnos from a YAML config file."""
        with open(config_path) as f:
            config = yaml.safe_load(f)
        return cls(config)

    # ── model building ────────────────────────────────────────────────

    def _build_model(self):
        """Instantiate all model components on the selected device."""
        model_cfg = self.config.get("model", {})

        # backbone
        backbone_name = model_cfg.get(
            "backbone", "state-spaces/mamba-130m-hf"
        )
        self.backbone = MambaBackbone(backbone_name)
        self.backbone.to(self.device)

        # thought block
        latent_dim = model_cfg.get("latent_dim", 256)
        k_steps = model_cfg.get("k_steps", 8)
        self.thought_block = ThoughtBlock(
            input_dim=self.backbone.hidden_size,
            latent_dim=latent_dim,
            k_steps=k_steps,
        ).to(self.device)

        # EMA teacher
        self.ema_teacher = EMATeacher(
            student=self.thought_block,
            tau=model_cfg.get("ema_tau", 0.999),
        ).to(self.device)

        # decoder
        self.decoder = LatentDecoder(
            latent_dim=latent_dim,
            vocab_size=model_cfg.get("vocab_size", 50257),
            hidden_dim=model_cfg.get("decoder_hidden", 512),
            max_length=model_cfg.get("max_length", 128),
        ).to(self.device)

    def _select_device(self) -> torch.device:
        """Auto-select best device: MPS > CUDA > CPU."""
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    # ── lifecycle ─────────────────────────────────────────────────────

    def start(self):
        """Activate the cognition gate and dream loop."""
        dream_cfg = self.config.get("dream", {})
        gate_cfg = self.config.get("gate", {})

        self._dream_loop = DreamLoop(
            model=self,
            k_steps=dream_cfg.get("k_steps", 4),
            coherence_threshold=dream_cfg.get("coherence_threshold", 0.3),
            consolidation_rate=dream_cfg.get("consolidation_rate", 0.1),
            stability_lambda=dream_cfg.get("stability_lambda", 0.01),
            max_cycles_per_session=dream_cfg.get(
                "max_cycles_per_session", 1000
            ),
            verbose=dream_cfg.get("verbose", True),
        )

        self._gate = CognitionGate(
            idle_threshold=gate_cfg.get("idle_threshold", 30.0),
            on_dream_start=self._on_dream_start,
            on_dream_end=self._on_dream_end,
            on_dream_step=self._dream_loop.step,
            verbose=gate_cfg.get("verbose", True),
        )
        self._gate.start()
        print("[Hypnos] System active. Gate and dream loop ready.")

    def stop(self):
        """Clean shutdown: stop gate, save state."""
        if self._gate:
            self._gate.stop()
        if self._mamba_cache is not None:
            self.state_manager.save(self._mamba_cache)
        print("[Hypnos] Shutdown complete.")

    # ── thinking ──────────────────────────────────────────────────────

    def think(
        self,
        text: str,
        k_steps: Optional[int] = None,
    ) -> str:
        """
        Process input text through the full pipeline.

        1. Notify gate (wake from dream if needed)
        2. Tokenize → Backbone encode (updates h_t)
        3. ThoughtBlock K-step reasoning
        4. Decoder → text output

        Parameters
        ----------
        text    : input text to respond to.
        k_steps : override reasoning depth (default: config value).

        Returns
        -------
        response : decoded text string.
        """
        # wake up
        if self._gate:
            self._gate.notify_input()
            if self._dream_loop:
                self._dream_loop.reset_session()

        # encode input
        input_ids = self.backbone.tokenize(text, self.device)

        with torch.no_grad():
            hidden, new_cache = self.backbone.encode(
                input_ids,
                cache_params=self._mamba_cache,
            )

        # update persistent state
        self._mamba_cache = new_cache
        self.state_manager.save(
            self._mamba_cache,
            tokens_processed=input_ids.shape[1],
        )

        # reason
        with torch.no_grad():
            z_final, trajectory = self.thought_block(
                hidden,
                k_override=k_steps,
            )

        # decode
        with torch.no_grad():
            output_ids = self.decoder(z_final)

        response = self._decode_tokens(output_ids)
        return response

    # ── checkpointing ─────────────────────────────────────────────────

    def save(self, checkpoint_dir: str):
        """Save trainable weights (ThoughtBlock + Decoder)."""
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save(
            self.thought_block.state_dict(),
            f"{checkpoint_dir}/thought_block.pt",
        )
        torch.save(
            self.decoder.state_dict(),
            f"{checkpoint_dir}/decoder.pt",
        )
        torch.save(
            self.ema_teacher.teacher.state_dict(),
            f"{checkpoint_dir}/ema_teacher.pt",
        )
        # save persistent state
        self.state_manager.save(self._mamba_cache)
        print(f"[Hypnos] Checkpoint saved to {checkpoint_dir}")

    def load_checkpoint(self, checkpoint_dir: str):
        """Load trainable weights from a checkpoint."""
        tb_path = f"{checkpoint_dir}/thought_block.pt"
        dec_path = f"{checkpoint_dir}/decoder.pt"
        ema_path = f"{checkpoint_dir}/ema_teacher.pt"

        if os.path.exists(tb_path):
            self.thought_block.load_state_dict(
                torch.load(tb_path, map_location=self.device)
            )
        if os.path.exists(dec_path):
            self.decoder.load_state_dict(
                torch.load(dec_path, map_location=self.device)
            )
        if os.path.exists(ema_path):
            self.ema_teacher.teacher.load_state_dict(
                torch.load(ema_path, map_location=self.device)
            )
        print(f"[Hypnos] Checkpoint loaded from {checkpoint_dir}")

    # ── private helpers ───────────────────────────────────────────────

    def _decode_tokens(self, output_ids: torch.Tensor) -> str:
        """Convert token ids back to text."""
        if self.backbone.tokenizer is not None:
            return self.backbone.tokenizer.decode(
                output_ids[0], skip_special_tokens=True
            )
        # stub mode: return raw ids as string
        return f"[stub output: {output_ids[0].tolist()[:20]}...]"

    def _on_dream_start(self):
        """Callback: dream mode activated."""
        pass

    def _on_dream_end(self):
        """Callback: dream mode deactivated."""
        if self._dream_loop:
            self._dream_loop.reset_session()

    def __repr__(self) -> str:
        return (
            f"Hypnos(device={self.device}, "
            f"hidden_size={self.backbone.hidden_size}, "
            f"latent_dim={self.thought_block.latent_dim}, "
            f"k_steps={self.thought_block.k_steps})"
        )
