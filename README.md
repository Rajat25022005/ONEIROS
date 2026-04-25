# Oneiros

> *A continuously reasoning AI architecture that thinks when idle, remembers across sessions, and develops understanding from experience — not just training.*

---

## What Is This

Oneiros is an open-source research framework combining **Mamba's persistent state-space model** with **JEPA-style latent reasoning** to build the first architecture that:

- **Thinks autonomously when idle** — runs structured K-step latent reasoning during idle periods with no input, no output, no supervision
- **Remembers across sessions** — Mamba's hidden state `h_t` persists to disk, carrying accumulated experience forward indefinitely
- **Reasons privately** — all reasoning happens in a 256-dimensional latent space, never in token space
- **Develops from experience** — patterns from real interactions consolidate during dream cycles into the model's persistent state

This is not a chatbot. It is a framework for building AI systems with **continuous inner life**.

---

## Architecture

```
┌─────────────────────────────────────────────┐
│            PERSISTENT MAMBA STATE           │
│         h_t ∈ R^d  — never resets           │
│         saved to disk between sessions      │
└──────────────┬──────────────────────────────┘
               │
       ┌───────▼────────┐
       │  COGNITION     │
       │  GATE          │
       │                │
       │  input? AWAKE  │
       │  idle?  DREAM  │
       └───┬───────┬────┘
           │       │
      AWAKE│       │DREAM
           │       │
    ┌──────▼──┐  ┌─▼──────────────────────┐
    │ Encode  │  │ Sample latent from h_t │
    │ input   │  │ z ~ p(z | h_t)         │
    └──────┬──┘  └──────┬─────────────────┘
           │            │
           └──────┬─────┘
                  │
         ┌────────▼────────┐
         │  THOUGHT BLOCK  │
         │  K-step latent  │
         │  reasoning      │
         │  z_0→z_1→...z_K │
         └────────┬────────┘
                  │
         ┌────────▼────────┐
         │   EMA TEACHER   │
         │  evaluates      │
         │  reasoning      │
         │  quality        │
         └────────┬────────┘
                  │
         ┌────────▼────────┐
         │  WRITE BACK     │
         │  to h_t         │
         │  (awake: always)│
         │  (dream: gated) │
         └─────────────────┘
```

### Core Components

**Mamba-130M Backbone**
State-space model with persistent hidden state. Unlike transformers, Mamba's state evolves continuously — even with zero input. This is the foundation of autonomous cognition.

**ThoughtBlock**
K-step latent reasoning module operating in R^256. Inherited and extended from [think-in-silence](https://github.com/Rajat25022005/think-in-silence). Reasoning steps are pure vector transitions — no tokens, no readable intermediates.

**Persistent State Manager**
Serializes and restores `h_t` across sessions. The model picks up exactly where it left off. Experience accumulates indefinitely.

**Dream Loop**
Background process that activates when no input arrives for `N` seconds. Samples from the model's own latent prior, runs ThoughtBlock K-steps, evaluates with EMA teacher, writes coherent consolidations back into `h_t`. No external supervision. No output.

**EMA Teacher**
Exponential moving average of ThoughtBlock weights. Evaluates reasoning quality during both awake and dream phases without explicit labels.

---

## Quickstart

```bash
git clone https://github.com/Rajat25022005/oneiros
cd oneiros
pip install -r requirements.txt
```

### Run Awake Mode
```python
from oneiros import Oneiros

model = Oneiros.load("checkpoints/oneiros-130m")

response = model.think("What is the nature of time?", k_steps=8)
print(response)
# h_t automatically updated and saved
```

### Start Dream Loop
```python
from oneiros import Oneiros
from oneiros.dream import DreamLoop

model = Oneiros.load("checkpoints/oneiros-130m")

# Model will begin dreaming after 30s of no input
dream = DreamLoop(model, idle_threshold=30, k_steps=4)
dream.start()  # runs in background thread
```

### Inspect State
```python
# Check what h_t looks like after N conversations
from oneiros.probes import StateProbe

probe = StateProbe(model)
probe.visualize_attractors()   # plot latent attractor structure
probe.measure_drift(baseline)  # how much has state changed
```

---

## Hardware Requirements

| Setup | Minimum | Recommended |
|---|---|---|
| RAM | 8 GB | 16 GB |
| Device | CPU (slow) | Apple Silicon M1/M2/M3 |
| Storage | 2 GB | 5 GB |
| Python | 3.10+ | 3.11+ |

Developed and tested on **MacBook M2 Air 16GB**. Runs fully on Metal via MPS backend. No GPU required.

---

## Project Structure

```
oneiros/
├── oneiros/
│   ├── model/
│   │   ├── backbone.py          # Mamba-130M wrapper
│   │   ├── thought_block.py     # K-step latent reasoning
│   │   ├── ema_teacher.py       # EMA evaluation
│   │   └── decoder.py           # latent → text
│   ├── state/
│   │   ├── manager.py           # persistent h_t save/load
│   │   └── serializer.py        # state serialization
│   ├── dream/
│   │   ├── loop.py              # autonomous dream cycle
│   │   ├── sampler.py           # latent prior sampling
│   │   └── consolidator.py      # dream → h_t writeback
│   ├── gate/
│   │   └── cognition_gate.py    # awake/dream switching
│   └── probes/
│       └── state_probe.py       # interpretability tools
├── train/
│   ├── stage1_jepa.py           # JEPA pretraining
│   ├── stage2_decoder.py        # decoder training
│   └── stage3_joint.py          # joint fine-tuning
├── configs/
│   ├── oneiros_130m.yaml
│   └── dream_config.yaml
├── experiments/
│   └── dream_drift_analysis.py  # does dreaming change anything?
├── checkpoints/                 # saved model + state
├── tests/
├── requirements.txt
└── README.md
```

---

## Research Questions This Framework Is Built To Answer

1. **Does autonomous dreaming improve reasoning on subsequent awake tasks?**
2. **Do stable attractor patterns form in h_t over time — and what do they represent?**
3. **How much does persistent state change the model's behavior vs a stateless baseline?**
4. **Can dream consolidation substitute for explicit fine-tuning on new tasks?**
5. **What is the minimum K for meaningful latent reasoning?**

---

## Relation To Prior Work

| Project | What Oneiros Extends |
|---|---|
| [think-in-silence](https://github.com/Rajat25022005/think-in-silence) | Direct predecessor — ThoughtBlock and EMA teacher inherited |
| Mamba / Mamba2 | SSM backbone providing persistent state |
| DreamerV3 | Autonomous reasoning during idle — but Oneiros is not reward-driven |
| JEPA (LeCun) | Training objective for latent reasoning quality |
| Neural Turing Machines | Persistent memory concept — but Oneiros uses SSM not external memory |

---

## Roadmap

- [x] Architecture design
- [ ] **Phase 1** — Mamba backbone + persistent state (current)
- [ ] **Phase 2** — Dream loop + EMA-gated consolidation
- [ ] **Phase 3** — Dream training signal (predictive dreaming)
- [ ] **Phase 4** — Attractor analysis + interpretability probes
- [ ] **Phase 5** — Multi-session longitudinal experiments

---

## Safety & Design Philosophy

Oneiros is built with awareness that persistent, privately-reasoning systems carry unique alignment considerations. The framework includes:

- **No tool access by default** — the model has no network or filesystem access during operation
- **State monitoring** — `StateProbe` tools to observe attractor formation over time
- **Dream constraints** — configurable stability regularizer preventing runaway state drift
- **Corrigibility objective** — alignment-aware training signal in Stage 3

This is a research framework. Deploy thoughtfully.

---

## Contributing

Oneiros is designed to be a foundation others can build on. Contributions welcome across all components — especially dream training signals, state interpretability, and alternative backbone integrations.

```bash
git checkout -b feature/your-contribution
# build something interesting
git push origin feature/your-contribution
```

---

## Citation

```bibtex
@software{oneiros2025,
  author = {Rajat Malik},
  title = {Oneiros: Continuous Latent Reasoning with Autonomous Dream Consolidation},
  year = {2026},
  url = {https://github.com/Rajat25022005/oneiros}
}
```

---

## Author

**Rajat Malik** — ML Researcher  
[GitHub](https://github.com/Rajat25022005) · [HuggingFace](https://huggingface.co/rajat5039)

*Built on the shoulders of think-in-silence.*

---

> *"Oneiros" — in Greek mythology, the god of dreams. Son of Hypnos (sleep) and Nyx (night). The one who visits minds in the quiet hours and leaves something changed.*
