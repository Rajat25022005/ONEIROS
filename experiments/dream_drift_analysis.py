"""
experiments/dream_drift_analysis.py — Analyse latent drift during dreaming

Runs Hypnos through multiple dream cycles and produces quantitative
analyses of how the hidden state evolves: cosine trajectory, L2 drift,
PCA projections, and coherence curves.
"""

import torch
import yaml
import os
import sys
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.backbone import MambaBackbone
from model.thought_block import ThoughtBlock
from model.ema_teacher import EMATeacher
from state.manager import StateManager
from dream.loop import DreamLoop
from probes.state_probe import StateProbe


def load_config(path: str):
    with open(path) as f:
        return yaml.safe_load(f)


def run_analysis(config_path: str, n_cycles: int = 5, seed: int = 42):
    torch.manual_seed(seed)
    cfg = load_config(config_path)
    device = torch.device("cpu")

    # initialise modules
    backbone = MambaBackbone(cfg["model"]).to(device)
    thought = ThoughtBlock(cfg["thought"]).to(device)
    teacher = EMATeacher(thought, decay=cfg["ema"]["decay"]).to(device)
    state_mgr = StateManager(cfg["state"])
    dream_loop = DreamLoop(thought, teacher, state_mgr, cfg["dream"])

    # create a seed hidden state from random tokens
    seed_tokens = torch.randint(0, cfg["model"]["vocab_size"], (1, 64))
    with torch.no_grad():
        h_init = backbone(seed_tokens)

    # run dream cycles
    print(f"Running {n_cycles} dream cycles...")
    results = dream_loop.dream_n(h_init, n_cycles=n_cycles)

    # analyse each cycle
    report = {"timestamp": datetime.utcnow().isoformat(), "cycles": []}
    probe = StateProbe()

    for i, res in enumerate(results):
        cosine_traj = probe.cosine_trajectory(res["trace"])
        l2_traj = probe.l2_drift(res["trace"])
        stats = probe.attention_to_state(res["h_t"])

        cycle_report = {
            "cycle": i + 1,
            "steps_taken": res["steps_taken"],
            "terminated_by": res["terminated_by"],
            "coherence_log": res["coherence_log"],
            "cosine_trajectory": cosine_traj,
            "l2_drift": l2_traj,
            "final_state_stats": stats,
        }
        report["cycles"].append(cycle_report)
        print(f"  Cycle {i+1}: {res['steps_taken']} steps, "
              f"terminated by {res['terminated_by']}, "
              f"final coherence={res['coherence_log'][-1]:.4f}")

    # save report
    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "dream_drift_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {out_path}")


if __name__ == "__main__":
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "configs", "hypnos_130m.yaml")
    run_analysis(cfg_path)
