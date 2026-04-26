"""
hypnos/model/ema_teacher.py — Exponential Moving Average teacher

Maintains a momentum-updated copy of the student ThoughtBlock.
Provides coherence scoring via cosine similarity — used as a
self-supervised quality signal in both awake and dream phases.
"""

import torch
import torch.nn as nn
import copy
from typing import Tuple


class EMATeacher(nn.Module):
    """
    EMA teacher: a slowly-updated copy of the student.

    θ_teacher = τ × θ_teacher + (1-τ) × θ_student

    At τ=0.999, after 1000 steps ~36.8% of the original teacher remains.
    This creates stable, slowly-moving targets for the student.
    """

    def __init__(self, student: nn.Module, tau: float = 0.999):
        super().__init__()
        self.tau = tau
        self.teacher = copy.deepcopy(student)

        # teacher never receives gradients — only updated via EMA
        for param in self.teacher.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, student: nn.Module):
        """EMA update: θ_t ← τ·θ_t + (1−τ)·θ_s."""
        for t_param, s_param in zip(
            self.teacher.parameters(),
            student.parameters(),
        ):
            t_param.data.mul_(self.tau).add_(
                s_param.data, alpha=1.0 - self.tau
            )

    @torch.no_grad()
    def forward(
        self,
        hidden: torch.Tensor,
        k_override: int = None,
    ) -> Tuple[torch.Tensor, list]:
        """Teacher forward pass (no gradients)."""
        return self.teacher(hidden, k_override=k_override)

    @torch.no_grad()
    def dream_forward(
        self,
        z_seed: torch.Tensor,
        k_override: int = None,
    ) -> Tuple[torch.Tensor, list]:
        """Teacher dream forward from a latent seed."""
        return self.teacher.dream_forward(z_seed, k_override=k_override)

    def coherence_score(
        self,
        student_z: torch.Tensor,
        teacher_z: torch.Tensor,
    ) -> torch.Tensor:
        """
        Cosine similarity between student and teacher latent states.

        Returns
        -------
        score : scalar tensor in [-1, 1].
                ~1 = coherent, ~0 = incoherent, <0 = contradictory.
        """
        return torch.nn.functional.cosine_similarity(
            student_z, teacher_z, dim=-1
        ).mean()
