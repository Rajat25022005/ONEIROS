"""
hypnos/model/thought_block.py — K-step latent reasoning

Implements iterative refinement in a compressed latent space (256-dim).
Given a hidden state from the backbone (768-dim), the ThoughtBlock
projects it down, runs K reasoning steps, and returns the refined
latent vector plus the full trajectory for JEPA training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class ThoughtStep(nn.Module):
    """
    One unit of reasoning: z_k → z_{k+1}.

    Architecture: LayerNorm → Linear(expand) → GELU → Linear(compress) + residual.
    The expansion factor (4×) gives the step more capacity for complex transforms.
    """

    def __init__(self, latent_dim: int, expansion: int = 4):
        super().__init__()
        hidden = latent_dim * expansion

        self.norm = nn.LayerNorm(latent_dim)
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Residual connection: output = input + transform(norm(input))."""
        return z + self.net(self.norm(z))


class ThoughtBlock(nn.Module):
    """
    K-step latent reasoning block.

    Projects backbone output (768D) into a compressed reasoning space (256D),
    applies K independent ThoughtSteps, and returns the final latent state
    plus the full trajectory [z_0, z_1, ..., z_K].
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 256,
        k_steps: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.k_steps = k_steps

        # project from backbone space to latent reasoning space
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )

        # K independent reasoning steps
        self.steps = nn.ModuleList([
            ThoughtStep(latent_dim) for _ in range(k_steps)
        ])

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden: torch.Tensor,
        k_override: int = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Project backbone output into latent space and run K reasoning steps.

        Parameters
        ----------
        hidden     : (B, input_dim) backbone hidden state.
        k_override : run fewer steps than max (for k-scaling experiments).

        Returns
        -------
        z_final    : (B, latent_dim) refined latent state after K steps.
        trajectory : list of K+1 tensors [z_0, z_1, ..., z_K].
        """
        k = k_override if k_override is not None else self.k_steps

        z = self.input_proj(hidden)
        z = self.dropout(z)

        trajectory = [z]
        for i in range(min(k, self.k_steps)):
            z = self.steps[i](z)
            trajectory.append(z)

        return z, trajectory

    def dream_forward(
        self,
        z_seed: torch.Tensor,
        k_override: int = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Dream-mode forward: reason from a raw latent seed.

        Skips input_proj (seed is already in latent space) and
        dropout (not training during dreaming).

        Parameters
        ----------
        z_seed     : (B, latent_dim) latent seed vector.
        k_override : number of reasoning steps.

        Returns
        -------
        z_final    : (B, latent_dim)
        trajectory : [z_0, ..., z_K]
        """
        k = k_override if k_override is not None else self.k_steps
        z = z_seed

        trajectory = [z]
        for i in range(min(k, self.k_steps)):
            z = self.steps[i](z)
            trajectory.append(z)

        return z, trajectory
