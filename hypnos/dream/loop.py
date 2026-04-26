"""
hypnos/dream/loop.py — Autonomous dreaming engine

Implements the dream cycle: sample a latent seed, run student and teacher
ThoughtBlocks, compare coherence, and consolidate if quality is above
threshold.  Called once per second by the CognitionGate while idle.
"""

import torch
import torch.nn.functional as F
import time
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from hypnos.core import Hypnos


class DreamLoop:
    """
    Dream loop: autonomous latent reasoning during idle periods.

    Each ``step()`` call:
    1. Samples a random latent seed z ~ N(0, 1)
    2. Student ThoughtBlock reasons K steps from z
    3. Teacher ThoughtBlock reasons K steps from the same z
    4. If cosine coherence ≥ threshold → consolidate into h_t
    5. Update the EMA teacher toward the student
    """

    def __init__(
        self,
        model: "Hypnos",
        k_steps: int = 4,
        coherence_threshold: float = 0.3,
        consolidation_rate: float = 0.1,
        stability_lambda: float = 0.01,
        max_cycles_per_session: int = 1000,
        verbose: bool = True,
    ):
        self.model = model
        self.k_steps = k_steps
        self.coherence_threshold = coherence_threshold
        self.consolidation_rate = consolidation_rate
        self.stability_lambda = stability_lambda
        self.max_cycles = max_cycles_per_session
        self.verbose = verbose

        self._cycle_count = 0
        self._consolidated_count = 0
        self._session_start: Optional[float] = None

    def step(self):
        """Run one dream cycle. Called by CognitionGate every ~1 second."""
        if self._cycle_count >= self.max_cycles:
            return

        if self._session_start is None:
            self._session_start = time.time()
            if self.verbose:
                print("[Dream] Dream session started.")

        self._cycle_count += 1
        device = self.model.device
        z_seed = self._sample_prior(device)

        # student reasons from the seed
        with torch.no_grad():
            z_student, student_traj = self.model.thought_block.dream_forward(
                z_seed, k_override=self.k_steps
            )

        # teacher reasons from the same seed
        with torch.no_grad():
            z_teacher, teacher_traj = self.model.ema_teacher.dream_forward(
                z_seed, k_override=self.k_steps
            )

        # coherence check
        coherence = self.model.ema_teacher.coherence_score(
            z_student, z_teacher
        ).item()

        # consolidate only if coherent enough
        consolidated = False
        if coherence >= self.coherence_threshold:
            self._consolidate(z_student)
            consolidated = True
            self._consolidated_count += 1

        if self.verbose and self._cycle_count % 10 == 0:
            print(
                f"[Dream] Cycle {self._cycle_count} | "
                f"Coherence: {coherence:.3f} | "
                f"Consolidated: {self._consolidated_count}/{self._cycle_count}"
            )

    def _sample_prior(self, device: torch.device) -> torch.Tensor:
        """Sample a random latent seed from the standard normal prior."""
        z = torch.randn(
            1,
            self.model.thought_block.latent_dim,
            device=device,
            dtype=torch.bfloat16,
        )
        return z

    def _consolidate(self, z_dream: torch.Tensor):
        """
        Write dream results back into persistent state.

        v1: increment dream counter and save state.
        Future: blend z_dream into h_t at consolidation_rate.
        """
        self.model.state_manager.increment_dream_cycles()
        self.model.state_manager.save(
            self.model._mamba_cache,
            tokens_processed=0,
        )

    def reset_session(self):
        """Reset counters when waking up. Called by Hypnos on input."""
        if self._cycle_count > 0 and self.verbose:
            duration = time.time() - (self._session_start or time.time())
            print(
                f"[Dream] Session ended. "
                f"Cycles: {self._cycle_count} | "
                f"Consolidated: {self._consolidated_count} | "
                f"Duration: {duration:.1f}s"
            )
        self._cycle_count = 0
        self._consolidated_count = 0
        self._session_start = None

    @property
    def cycle_count(self) -> int:
        return self._cycle_count

    @property
    def consolidated_count(self) -> int:
        return self._consolidated_count
