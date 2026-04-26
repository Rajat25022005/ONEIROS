"""
model/ema_teacher.py — Exponential Moving Average teacher for coherence evaluation

The EMA teacher maintains a slow-moving copy of the student (ThoughtBlock)
and provides a coherence signal by comparing the student's latent trajectory
against the teacher's smoothed representation.
"""

import torch
import torch.nn as nn
from typing import Dict, Any
import copy


class EMATeacher(nn.Module):
    """
    Coherence evaluator via Exponential Moving Average.

    Maintains a momentum-updated copy of the student network and computes
    a coherence score between student and teacher representations.  Used
    as a self-supervised quality signal during both awake and dream phases.
    """

    def __init__(self, student: nn.Module, decay: float = 0.999):
        super().__init__()
        self.decay = decay

        # deep-copy the student; teacher weights are never trained directly
        self.teacher = copy.deepcopy(student)
        for p in self.teacher.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, student: nn.Module) -> None:
        """EMA update: θ_teacher ← decay * θ_teacher + (1 - decay) * θ_student."""
        for t_param, s_param in zip(
            self.teacher.parameters(), student.parameters()
        ):
            t_param.data.mul_(self.decay).add_(s_param.data, alpha=1.0 - self.decay)

    def forward(self, h: torch.Tensor, **kwargs) -> torch.Tensor:
        """Run teacher forward pass (no grad)."""
        with torch.no_grad():
            return self.teacher(h, **kwargs)

    def coherence_score(
        self,
        student_h: torch.Tensor,
        teacher_h: torch.Tensor,
    ) -> torch.Tensor:
        """
        Cosine-similarity based coherence between student and teacher states.

        Parameters
        ----------
        student_h : (B, L, D)
        teacher_h : (B, L, D)

        Returns
        -------
        score : (B,) coherence in [−1, 1], higher = more coherent.
        """
        s = student_h.mean(dim=1)  # (B, D)
        t = teacher_h.mean(dim=1)
        return nn.functional.cosine_similarity(s, t, dim=-1)

    def coherence_loss(
        self,
        student_h: torch.Tensor,
        teacher_h: torch.Tensor,
    ) -> torch.Tensor:
        """1 − coherence_score, averaged over the batch."""
        return (1.0 - self.coherence_score(student_h, teacher_h)).mean()
