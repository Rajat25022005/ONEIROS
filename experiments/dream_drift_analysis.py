"""
experiments/dream_drift_analysis.py — Does dreaming change anything?

Runs N dream cycles and measures:
- Coherence trajectory (do dreams stay coherent over time?)
- State drift (how much does h_t change from dreaming?)
- Drift statistics at 25%, 50%, 75%, 100% of cycles

Usage:
    python -m experiments.dream_drift_analysis --config configs/hypnos_130m.yaml
"""

import os
import sys
import yaml
import json
import time
import argparse

import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hypnos.core import Hypnos
from hypnos.probes.state_probe import StateProbe


def run_analysis(
    config_path: str,
    n_cycles: int = 100,
    k_steps: int = 4,
    seed: int = 42,
):
    """Run dream drift analysis and save results."""
    torch.manual_seed(seed)

    print(f"[Experiment] Dream Drift Analysis")
    print(f"  Config: {config_path}")
    print(f"  Cycles: {n_cycles}")
    print(f"  K-steps: {k_steps}")
    print(f"  Seed: {seed}")
    print()

    # build model
    model = Hypnos.from_config(config_path)
    probe = StateProbe(model, probe_dir=".hypnos/experiments/dream_drift")

    # baseline snapshot
    probe.snapshot("baseline")

    # run dream cycles
    coherence_history = []
    start_time = time.time()

    for i in range(n_cycles):
        z_seed = torch.randn(
            1,
            model.thought_block.latent_dim,
            device=model.device,
            dtype=torch.bfloat16,
        )

        with torch.no_grad():
            z_student, _ = model.thought_block.dream_forward(
                z_seed, k_override=k_steps
            )
            z_teacher, _ = model.ema_teacher.dream_forward(
                z_seed, k_override=k_steps
            )

        coherence = model.ema_teacher.coherence_score(
            z_student, z_teacher
        ).item()
        coherence_history.append(coherence)

        # snapshot at milestone points
        if (i + 1) in [
            n_cycles // 4,
            n_cycles // 2,
            3 * n_cycles // 4,
            n_cycles,
        ]:
            label = f"cycle_{i + 1}"
            probe.snapshot(label)

        if (i + 1) % max(1, n_cycles // 10) == 0:
            print(
                f"  Cycle {i + 1:>4d}/{n_cycles} | "
                f"Coherence: {coherence:.4f}"
            )

    elapsed = time.time() - start_time

    # measure drifts
    drifts = {}
    for label in probe.list_snapshots():
        if label != "baseline":
            drift = probe.measure_drift("baseline", label)
            drifts[label] = drift

    # compile results
    results = {
        "n_cycles": n_cycles,
        "k_steps": k_steps,
        "seed": seed,
        "elapsed_seconds": elapsed,
        "coherence": {
            "mean": sum(coherence_history) / len(coherence_history),
            "min": min(coherence_history),
            "max": max(coherence_history),
            "std": (
                sum((c - sum(coherence_history) / len(coherence_history)) ** 2
                    for c in coherence_history)
                / len(coherence_history)
            ) ** 0.5,
            "history": coherence_history[:: max(1, n_cycles // 100)],
        },
        "state_drift": drifts,
    }

    # save
    results_dir = Path(".hypnos/experiments/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / "dream_drift_report.json"

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print()
    print(f"[Experiment] Results saved to {results_path}")
    print(f"  Mean coherence: {results['coherence']['mean']:.4f}")
    print(f"  Coherence range: [{results['coherence']['min']:.4f}, {results['coherence']['max']:.4f}]")
    print(f"  State drifts: {drifts}")
    print(f"  Elapsed: {elapsed:.1f}s")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hypnos Dream Drift Analysis"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/hypnos_130m.yaml",
    )
    parser.add_argument("--cycles", type=int, default=100)
    parser.add_argument("--k-steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_analysis(
        args.config,
        n_cycles=args.cycles,
        k_steps=args.k_steps,
        seed=args.seed,
    )
