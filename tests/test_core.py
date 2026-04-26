"""
tests/test_core.py — Automated tests for Hypnos components

Run: python -m pytest tests/ -v
"""

import time
import torch
import pytest


# ── ThoughtBlock ──────────────────────────────────────────────────────

def test_thought_block_forward():
    """ThoughtBlock produces correct shapes and trajectory length."""
    from hypnos.model.thought_block import ThoughtBlock

    tb = ThoughtBlock(input_dim=768, latent_dim=256, k_steps=4)
    hidden = torch.randn(2, 768)
    z_final, trajectory = tb(hidden)

    assert z_final.shape == (2, 256), f"Expected (2, 256), got {z_final.shape}"
    assert len(trajectory) == 5, f"Expected 5 (z_0 + 4 steps), got {len(trajectory)}"


def test_thought_block_k_override():
    """k_override runs fewer steps than the maximum."""
    from hypnos.model.thought_block import ThoughtBlock

    tb = ThoughtBlock(input_dim=768, latent_dim=256, k_steps=8)
    hidden = torch.randn(1, 768)

    z, traj = tb(hidden, k_override=3)
    assert len(traj) == 4, f"Expected 4 (z_0 + 3 steps), got {len(traj)}"


def test_thought_block_dream_forward():
    """dream_forward works from a raw latent seed."""
    from hypnos.model.thought_block import ThoughtBlock

    tb = ThoughtBlock(input_dim=768, latent_dim=256, k_steps=4)
    z_seed = torch.randn(1, 256)
    z_final, traj = tb.dream_forward(z_seed)

    assert z_final.shape == (1, 256)
    assert len(traj) == 5


# ── EMATeacher ────────────────────────────────────────────────────────

def test_ema_teacher_coherence():
    """Coherence score is between -1 and 1."""
    from hypnos.model.thought_block import ThoughtBlock
    from hypnos.model.ema_teacher import EMATeacher

    tb = ThoughtBlock(input_dim=768, latent_dim=256, k_steps=4)
    teacher = EMATeacher(student=tb, tau=0.999)

    z_a = torch.randn(2, 256)
    z_b = torch.randn(2, 256)
    score = teacher.coherence_score(z_a, z_b).item()

    assert -1.0 <= score <= 1.0, f"Score {score} out of [-1, 1]"


def test_ema_teacher_update():
    """EMA update moves teacher toward student."""
    from hypnos.model.thought_block import ThoughtBlock
    from hypnos.model.ema_teacher import EMATeacher

    tb = ThoughtBlock(input_dim=768, latent_dim=256, k_steps=4)
    teacher = EMATeacher(student=tb, tau=0.999)

    # get initial teacher params
    before = [p.clone() for p in teacher.teacher.parameters()]

    # modify student and update
    with torch.no_grad():
        for p in tb.parameters():
            p.add_(torch.randn_like(p) * 0.1)
    teacher.update(tb)

    # teacher should have changed
    changed = False
    for b, a in zip(before, teacher.teacher.parameters()):
        if not torch.allclose(b, a):
            changed = True
            break
    assert changed, "Teacher params should change after update"


def test_ema_self_coherence():
    """Student compared to itself (via teacher copy) should be highly coherent."""
    from hypnos.model.thought_block import ThoughtBlock
    from hypnos.model.ema_teacher import EMATeacher

    tb = ThoughtBlock(input_dim=768, latent_dim=256, k_steps=4)
    teacher = EMATeacher(student=tb, tau=0.999)

    hidden = torch.randn(1, 768)
    z_student, _ = tb(hidden)
    z_teacher, _ = teacher(hidden)

    score = teacher.coherence_score(z_student, z_teacher).item()
    assert score > 0.9, f"Self-coherence should be high, got {score}"


# ── LatentDecoder ─────────────────────────────────────────────────────

def test_decoder_shapes():
    """Decoder produces correct logit dimensions."""
    from hypnos.model.decoder import LatentDecoder

    dec = LatentDecoder(latent_dim=256, vocab_size=50257, hidden_dim=512)
    z = torch.randn(2, 256)
    target = torch.randint(0, 50257, (2, 10))

    logits = dec(z, target_ids=target)
    assert logits.shape == (2, 10, 50257), f"Expected (2, 10, 50257), got {logits.shape}"


def test_decoder_greedy_decode():
    """Greedy decode returns token ids of correct shape."""
    from hypnos.model.decoder import LatentDecoder

    dec = LatentDecoder(latent_dim=256, vocab_size=100, hidden_dim=64, max_length=16)
    z = torch.randn(1, 256)

    output = dec(z)
    assert output.shape[0] == 1
    assert output.shape[1] <= 16
    assert output.dtype == torch.long


# ── StateManager ──────────────────────────────────────────────────────

def test_state_manager_save_load(tmp_path):
    """Save and load round-trip preserves state."""
    from hypnos.state.manager import StateManager

    sm = StateManager(str(tmp_path / "state"), torch.device("cpu"))
    loaded = sm.load()
    assert loaded is None, "First load should return None"

    # save a fake state
    fake_state = torch.randn(3, 768)
    sm.save(fake_state, tokens_processed=42)

    # load in a new manager (simulates new session)
    sm2 = StateManager(str(tmp_path / "state"), torch.device("cpu"))
    loaded = sm2.load()
    assert loaded is not None, "Second load should return state"
    assert sm2.session_count == 1


def test_state_manager_metadata(tmp_path):
    """Metadata tracks tokens and sessions correctly."""
    from hypnos.state.manager import StateManager

    sm = StateManager(str(tmp_path / "state"), torch.device("cpu"))
    sm.load()
    sm.save(torch.randn(1, 10), tokens_processed=100)
    sm.save(torch.randn(1, 10), tokens_processed=200)

    assert sm.metadata.total_tokens_processed == 300


# ── CognitionGate ─────────────────────────────────────────────────────

def test_cognition_gate_lifecycle():
    """Gate transitions: awake → dream → awake on input."""
    from hypnos.gate.cognition_gate import CognitionGate

    gate = CognitionGate(idle_threshold=0.3, verbose=False)
    gate.start()

    assert gate.is_awake(), "Should start awake"

    # wait past threshold
    time.sleep(1.5)
    assert gate.is_dreaming(), "Should be dreaming after idle"

    # input wakes it up
    gate.notify_input()
    assert gate.is_awake(), "Should be awake after input"

    gate.stop()


def test_cognition_gate_callbacks():
    """Dream callbacks are called correctly."""
    from hypnos.gate.cognition_gate import CognitionGate

    events = []

    gate = CognitionGate(
        idle_threshold=0.3,
        on_dream_start=lambda: events.append("start"),
        on_dream_end=lambda: events.append("end"),
        on_dream_step=lambda: events.append("step"),
        verbose=False,
    )
    gate.start()
    time.sleep(2.0)

    gate.notify_input()  # should trigger end
    gate.stop()

    assert "start" in events, "Dream start callback should fire"
    assert "step" in events, "Dream step callback should fire"
    assert "end" in events, "Dream end callback should fire"


# ── Integration ───────────────────────────────────────────────────────

def test_full_pipeline():
    """Full pipeline: backbone → thought → teacher → decoder."""
    from hypnos.model.backbone import MambaBackbone
    from hypnos.model.thought_block import ThoughtBlock
    from hypnos.model.ema_teacher import EMATeacher
    from hypnos.model.decoder import LatentDecoder

    backbone = MambaBackbone()  # uses stub
    tb = ThoughtBlock(input_dim=backbone.hidden_size, latent_dim=256, k_steps=4)
    teacher = EMATeacher(student=tb)
    decoder = LatentDecoder(latent_dim=256, vocab_size=100, hidden_dim=64, max_length=16)

    # encode
    ids = torch.randint(0, 50257, (1, 16))
    hidden, cache = backbone.encode(ids)

    # think
    z, traj = tb(hidden)
    z_teacher, _ = teacher(hidden)

    # coherence
    score = teacher.coherence_score(z, z_teacher).item()
    assert -1 <= score <= 1

    # decode
    output = decoder(z)
    assert output.shape[0] == 1
