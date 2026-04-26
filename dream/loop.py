"""
dream/loop.py — Autonomous dreaming loop
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List

from model.thought_block import ThoughtBlock
from model.ema_teacher import EMATeacher
from state.manager import StateManager


class DreamLoop(nn.Module):
    """Autonomous dream loop with coherence-bounded termination."""

    def __init__(self, thought_block, ema_teacher, state_manager, config):
        super().__init__()
        self.thought_block = thought_block
        self.ema_teacher = ema_teacher
        self.state_manager = state_manager
        self.max_dream_steps = config.get("max_dream_steps", 32)
        self.coherence_floor = config.get("coherence_floor", 0.4)
        self.noise_scale = config.get("noise_scale", 0.01)
        self.thought_steps = config.get("thought_steps_per_dream", 2)

    def step(self, h_init: torch.Tensor) -> Dict[str, Any]:
        h = h_init
        trace = [h.detach().cpu()]
        coherence_log = []
        terminated_by = "max_steps"

        for step_idx in range(self.max_dream_steps):
            noise = torch.randn_like(h) * self.noise_scale
            h_noisy = h + noise
            h_new = self.thought_block(h_noisy, steps=self.thought_steps)
            teacher_h = self.ema_teacher(h_noisy, steps=self.thought_steps)
            coherence = self.ema_teacher.coherence_score(h_new, teacher_h)
            mean_coh = coherence.mean().item()
            coherence_log.append(mean_coh)
            trace.append(h_new.detach().cpu())
            h = h_new
            if mean_coh < self.coherence_floor:
                terminated_by = "coherence_floor"
                break

        self.ema_teacher.update(self.thought_block)
        return {"h_t": h, "trace": trace, "coherence_log": coherence_log,
                "steps_taken": step_idx + 1, "terminated_by": terminated_by}

    def dream_n(self, h_init, n_cycles=3):
        results, h = [], h_init
        for _ in range(n_cycles):
            out = self.step(h)
            results.append(out)
            h = out["h_t"]
        return results
