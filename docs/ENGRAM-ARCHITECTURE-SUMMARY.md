# Engram: Neuromorphic Architecture Summary

**License:** MIT. See [LICENSE](../LICENSE).

---

## 1. Overview

Engram is a biologically-inspired spiking neural network (SNN) architecture designed for embodied robotic intelligence. Unlike transformer-based approaches that require massive compute for inference, Engram implements six concurrent learning mechanisms operating on sparse spike-based computation, enabling real-time continual learning on commodity hardware.

**1M PoC deployment**: 1,001,800 neurons, 1.19 billion synapses, 24 synapse groups - running on a Hetzner CCX63 (48 vCPU, 192 GB RAM) at ~0.74 simulation steps/second with full plasticity enabled. Base configuration uses ~215,500 neurons; the 1M scale is an overlay (`deploy/docker-compose.1m.yml`).

---

## 2. Brain Regions (11 Regions)

The network implements a hierarchical cortical architecture with distinct functional regions:

| Region | Base Scale | 1M Scale | Function |
|--------|-----------|----------|----------|
| **Brainstem** | ~4,000 | ~18,600 | Arousal, homeostatic drives (hunger, curiosity, fatigue) |
| **Sensory Cortex** | ~50,000 | ~232,500 | Multi-modal input encoding (vision, audio, proprioception) |
| **Feature Layer** | ~20,000 | ~93,000 | Low-level feature extraction from sensory input |
| **Association Cortex** | ~50,000 | ~232,500 | Cross-modal binding, pattern recognition |
| **Concept Layer** | ~5,000 | ~23,200 | k-WTA sparse coding, abstract concept formation |
| **Predictive Layer** | ~25,000 | ~116,200 | Temporal prediction, prediction error computation |
| **Working Memory** | ~6,000 | ~27,900 | Short-term maintenance of active representations |
| **Motor Cortex** | ~25,000 | ~116,200 | 6 motor sub-ranges: locomotion, manipulation, head, speech, expression, cognitive |
| **Cerebellum** | ~25,000 | ~116,200 | Motor timing, fine coordination, error correction |
| **Reflex Arc** | ~2,500 | ~11,600 | Hardwired fast responses (withdrawal, startle) |
| **Meta-Controller** | ~3,000 | ~13,900 | Neuromodulatory regulation, attention gating |

Base scale totals ~215,500 neurons (default `docker-compose.yml`). 1M scale totals 1,001,800 neurons (`deploy/docker-compose.1m.yml`). Feature, Concept, and Meta-Controller layers are hierarchical additions enabled by configuration.

---

## 3. Six Concurrent Learning Mechanisms (Architecture Invariant 1)

All six mechanisms operate simultaneously every simulation step:

### (a) Spike-Timing-Dependent Plasticity (STDP)
- Configurable per synapse group: `tau_plus`, `tau_minus`, `a_plus`, `a_minus`
- Pre-before-post strengthens; post-before-pre weakens
- Compartment-aware: weight change scaled by target dendritic compartment activity

### (b) Eligibility Traces
- STDP changes accumulate in eligibility traces (not applied directly to weights)
- Exponential decay with `tau_eligibility` ~ 1000ms
- Gated by neuromodulators — traces are the "pending credit" that neuromodulation applies
- Active-set tracking: only non-zero entries are processed (sparse optimization)

### (c) BCM Metaplasticity
- Per-postsynaptic-neuron sliding threshold `theta`
- `scaling = epsilon / (theta + epsilon)` — active neurons become harder to potentiate
- Prevents runaway excitation; creates competitive input selection

### (d) 4-Channel Neuromodulation
- **Dopamine (DA)**: Reward signal, gates eligibility trace application
- **Acetylcholine (ACh)**: Attention/novelty, modulates sensory gain
- **Norepinephrine (NE)**: Arousal/urgency, scales overall learning rate
- **Serotonin (5-HT)**: Satiety/calm, dampens exploration
- Critical period baselines shift across developmental phases
- Meta-controller region provides top-down neuromodulatory influence

### (e) Homeostatic Synaptic Scaling
- Per-postsynaptic-neuron multiplicative normalization
- Prevents catastrophic forgetting by maintaining input competition
- Rate scales inversely with plasticity multiplier (less correction during high-plasticity phases)

### (f) Reward-Modulated STDP (R-STDP)
- Prediction error gates STDP weight changes on predictive pathways
- Extended to motor pathways via motor feedback loop
- Enables delayed credit assignment: motor fires → feedback arrives 50-500ms later → eligibility traces bridge the gap

---

## 4. Developmental Critical Periods (Architecture Invariant 2)

Five mandatory phases with distinct learning dynamics:

```
infant → toddler → juvenile → adolescent → mature
```

| Phase | Entry Condition | Key Characteristics |
|-------|----------------|---------------------|
| **Infant** | Birth | High plasticity, broad STDP windows, rapid sensory mapping |
| **Toddler** | Step count | Neuromodulator baselines shift via smooth interpolation |
| **Juvenile** | Step count | Balanced plasticity, association formation |
| **Adolescent** | Experience-dependent* | Pruning, myelination, identity tagging, widened STDP |
| **Mature** | Post-adolescent | Stable network, low plasticity, continual refinement |

*Adolescent entry requires ALL criteria simultaneously: concept differentiation score, sensory stability, feature STDP decline below threshold, and minimum step count. This is never hardcoded by step count alone.

**Adolescent structural modifications**:
- **Pruning**: Remove weak synapses (max 5% per round), skip myelinated/identity synapses
- **Myelination**: High-weight + stable synapses → plasticity reduced to 10%
- **Identity tagging**: Myelinated + survived pruning → plasticity reduced to 1% (near-permanent)
- **Neighborhood consolidation**: DA burst rescues nearby eligibility traces

---

## 5. Hybrid SNN-LLM Cognitive Architecture (Architecture Invariant 3)

The motor cortex includes a "cognitive" sub-range. When:
1. Cognitive motor neurons fire above threshold, AND
2. Prediction error remains high for a sustained window

...the system publishes a query to an LLM (via NATS). The LLM response is re-injected into the sensory pipeline with boosted gain. STDP on re-injected responses creates closed-loop learning.

The query decision is **emergent via STDP** — the network learns when querying is useful through reinforcement, not hardcoded IF-THEN rules.

---

## 6. Cross-Modal Binding with Instinctual Gain (Architecture Invariant 4)

- Arbitrary sensor modalities encoded in modality-specific sub-ranges of sensory cortex
- Dynamic allocation: new modalities get sub-ranges on first observation
- STDP on sensory→association binds temporally coincident cross-modal spikes
- Instinctual gain is multiplicative (always >= 1.0): novelty, change detection, cross-modal coincidence, habituation
- Phase-dependent: adolescent amplifies gains, mature dampens

---

## 7. Multi-Compartment Dendritic Processing (Architecture Invariant 5)

Four dendritic compartments per neuron:
1. **Apical distal** — top-down feedback, context
2. **Basal** — feedforward sensory input
3. **Apical proximal** — lateral/recurrent connections
4. **Perisomatic** — strongest drive, brainstem arousal

Each compartment has independent membrane dynamics with configurable soma coupling. Supralinear dendritic spikes in apical compartments provide 2.5x amplification. Every synapse group targets a specific compartment.

---

## 8. Motor Output and Sensorimotor Loop

### Motor Sub-Ranges (6 channels at 1M scale)
| Sub-Range | Cortex Fraction | Function |
|-----------|----------------|----------|
| Locomotion | 0–30% | Walking, balance, navigation |
| Manipulation | 30–60% | Grasping, reaching, tool use |
| Head | 60–80% | Gaze, head orientation |
| Speech | 80–83% | Vocalization tokens (256-token vocabulary) |
| Expression | 83–85% | Emotional expression |
| Cognitive | 85–100% | LLM query trigger (emergent) |

Speech and cognitive sub-ranges are enabled via environment variables (`NEURO_SPEECH_END`, `NEURO_EXPRESSION_END`) in the 1M overlay config. At base scale, only 4 sub-ranges are active (locomotion, manipulation, head, expression). At 1M scale, all 6 sub-ranges are active.

### Motor Feedback Loop
The sensorimotor loop is closed: brain fires motor command → actuator executes → proprioceptive feedback returns → R-STDP updates motor synapses. Motor echo DA boost (1.5x for 100 steps) keeps eligibility traces alive during the feedback delay window.

---

## 9. System Architecture

All components communicate via NATS message bus. The neuromorphic brain is one service among several:

```
Sensory Gateway → NATS → Neuromorphic Brain → NATS → Motor Decoder
                           ↕                           ↕
                    Moral Kernel ← Beliefs Graph    Actuators
                           ↕                           ↕
                    Safety Supervisor           Motor Feedback
```

- **Moral Kernel**: Immutable safety gatekeeper (ALLOW/TRANSFORM/DENY/DEFER)
- **Beliefs Graph**: Constitutional values (immutable floor) + learned norms
- **Safety Supervisor**: Risk analysis engine (advisory, no decision authority)
- **Sensory Gateway**: Host-side sensor discovery, CNN preprocessing, aggregation
- **Dashboard**: Real-time monitoring, human-in-the-loop approval for DEFER decisions

---

## 10. Key Performance Numbers (1M PoC Scale)

| Metric | Value |
|--------|-------|
| Total neurons | 1,001,800 |
| Total synapses | 1.19 billion |
| Synapse groups | 24 |
| Memory footprint | ~24 GB core + persistence overhead |
| Training rate | 0.74 steps/sec (full plasticity) |
| STDP parallelism | ThreadPoolExecutor (8 threads) |
| Persistence | Hybrid SQLite + .npy files |
| Save time (BGSAVE) | ~13 seconds for 9 GB |
| Sensory aggregation | 1,400 obs/sec → 4/sec (99.7% reduction) |
| Video preprocessing | Shared ONNX CNN, 64x64 grayscale, 576 features |
| Audio features | 13 MFCC coefficients, 10 Hz emission |

---

## 11. Why Neuromorphic Hardware?

The current implementation runs on conventional CPUs. The architecture is designed for direct mapping to neuromorphic hardware:

- **Sparse computation**: Only active neurons compute (typically 1-5% firing rate)
- **Local learning rules**: STDP and eligibility traces require only local pre/post information
- **Event-driven**: Spike-based processing maps directly to Loihi 2's asynchronous mesh
- **Per-neuron state**: Compartmental membrane potentials, BCM thresholds, and neuromodulatory receptors map to Loihi 2's programmable neuron models

**Expected benefits on Loihi 2**:
- 73-109x energy reduction vs CPU/GPU (based on published benchmarks for comparable tasks)
- Real-time operation: 1M neuron step in <10ms (vs current ~1+ seconds on CPU)
- On-chip learning: STDP + eligibility traces in silicon
- Deployment target: Lava framework compatibility for direct migration

---

*Document prepared for Intel Neuromorphic Research Community (INRC) application.*
