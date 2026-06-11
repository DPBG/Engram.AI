# Hierarchical Brain Architecture Upgrade — Research Synthesis

> Compiled 2026-02-19 from 5 parallel research agents covering: hierarchical cortical layers, meta-controller, abstract concept formation, concept bottleneck/transfer, and codebase audit.

---

## Executive Summary

The current neuromorphic core (8 flat regions, 187K neurons, 40M synapses, STDP/R-STDP) has strong biological foundations but lacks the depth and modulatory control needed for generalization, transfer learning, and abstract concept formation.

This upgrade adds **4 major capabilities** across **9 implementation phases**:

1. **Inhibitory neurons + E/I balance** — enables oscillations, sparse coding, competition
2. **Hierarchical cortical layers + concept bottleneck** — enables feature hierarchy, abstraction, transfer
3. **Meta-controller + neuromodulation** — enables adaptive plasticity, critical periods, learning-to-learn
4. **Three-factor learning rules (eligibility traces)** — enables delayed reward, credit assignment

**Scale change**: ~187K → ~220K neurons, ~40M → ~55M synapses, ~600MB → ~900MB RAM.

---

## Phase-by-Phase Implementation Plan

### Phase 1: Config Extension (LOW RISK)
**Files**: `config.py`
**Add**: `InhibitoryConfig`, `ConceptLayerConfig`, `MetaControllerConfig`, `EligibilityTraceConfig`, `NeuromodulatorConfig`
**Backward compatible**: All new dataclasses with defaults. Existing configs unchanged.

### Phase 2: Inhibitory Neurons (FOUNDATIONAL)
**Files**: `neurons.py`, `regions.py`
**What**: Add `is_inhibitory` mask to NeuronPopulation. Inhibitory neurons have faster tau (10ms vs 20ms), shorter refractory (1ms), and produce negative output current.
**Why first**: Oscillations and sparse coding depend on E/I balance. Everything else builds on this.
**Neuron split**: 80% excitatory / 20% inhibitory per region (matching cortical ratio).
**New internal synapse groups per region**: E→I, I→E, I→I (lateral inhibition circuits).

### Phase 3: Concept Layer with k-WTA (HIGH VALUE)
**Files**: `regions.py`, `config.py`, `network.py`
**What**: New `ConceptLayer` region (10K neurons) between association cortex and predictive/WM/motor. Enforces 2% sparsity via k-Winners-Take-All (200 active neurons per step).
**Compression**: 50:1 from association cortex activity → concept SDR.
**Capacity**: C(10000,200) ≈ 10^430 unique patterns.
**LIF params**: tau=40ms (slower integration), threshold=-52mV (harder to fire).
**6 new synapse groups**:
- association → concept (STDP, feedforward)
- concept lateral (STDP, binding within layer)
- concept → predictive (R-STDP)
- concept → working_memory (STDP)
- concept → motor (STDP)
- predictive → concept (STDP, top-down prediction)

### Phase 4: Feature Integration Layer (HIERARCHY)
**Files**: `regions.py`, `network.py`
**What**: New `FeatureLayer` (80K neurons) between sensory and association cortex.
**Why**: Creates the V4/IT-like intermediate that pools simple features into complex feature combinations before multi-modal binding.
**New synapse groups**:
- sensory → feature (STDP, broad connectivity)
- feature → association (STDP, convergent)

### Phase 5: Eligibility Traces (THREE-FACTOR LEARNING)
**Files**: `synapses.py`
**What**: Add `eligibility` shadow array (float32, same shape as weights.data) to every plastic SynapseGroup.
**Memory cost**: +160MB for 40M synapses.
**Equations**:
```
de/dt = -e/tau_e + STDP(t_pre, t_post)    # trace accumulates from spike coincidences
dw/dt = M(t) * e(t)                        # weight change gated by neuromodulator
```
**tau_e**: 500-2000ms (eligibility window for delayed reward).
**Key change**: STDP no longer directly updates weights. It updates eligibility traces. Weight changes happen only when a neuromodulatory signal arrives.

### Phase 6: Meta-Controller Region (META-LEARNING)
**Files**: `regions.py`, `network.py`, new `neuromodulation.py`
**What**: New `MetaController` region (6K neurons) that monitors network-wide statistics and outputs 4 neuromodulatory signals.
**Sub-populations**:
- Monitor (2K) — reads firing rates, weight changes, prediction error
- DA output (1K) — per-region plasticity multiplier [0, 5]
- ACh output (500) — feedforward vs recurrent balance [0, 3]
- NE output (500) — global gain/arousal [0.5, 3]
- 5-HT output (500) — consolidation brake [0, 1]
- Integrator (1.5K) — internal processing

**Effects**:
- DA: multiplies R-STDP modulation (amplify learning at active synapses)
- ACh: scales feedforward vs lateral connections (trust sensory vs internal)
- NE: multiplies all neural input current (arousal)
- 5-HT: reduces STDP amplitudes globally (consolidation mode)

**Learns via slow R-STDP**: tau=200ms, A+=0.003 (4x smaller than cortical)

### Phase 7: Critical Periods
**Files**: `neuromodulation.py`
**What**: Developmental schedule that shifts neuromodulatory baselines.
**Phases**:
1. Infant (0-10 min): Wide open — high ACh (2.5), high NE (2.0), moderate DA (2.0), low 5-HT (0.1)
2. Toddler (10-60 min): Guided exploration — ACh→1.5, NE→1.5, DA→1.5, 5-HT→0.4
3. Juvenile (1-6 hrs): Refinement — all modulators→1.2, 5-HT→0.6
4. Mature (6+ hrs): Stable — all→1.0, 5-HT→0.5

### Phase 8: BCM Metaplasticity
**Files**: `synapses.py`
**What**: Per-neuron modification threshold (theta_m) that adapts based on recent activity.
```
theta_m = EMA[post_rate^2]
A_plus_effective = A_plus * theta_m / (theta_m + epsilon)
```
**Effect**: Active neurons become harder to potentiate (prevents saturation). Quiet neurons become easier to potentiate (prevents death).

### Phase 9: Persistence + Integration Testing
**Files**: `persistence.py`, all test files
**What**: Extend SQLite schema for new regions, eligibility traces, neuromodulator state. Backward-compatible loading of old checkpoints.

---

## Architecture: Before vs After

### Before (8 regions, 15 connections)
```
Brainstem → Sensory → Association → Predictive
                ↓           ↓        (recurrent)
             Reflex     Working Mem
                ↓           ↓
              Motor ← Cerebellum
```

### After (12 regions, ~30 connections)
```
Brainstem → Sensory → Feature Layer → Association → Concept Layer (k-WTA bottleneck)
                ↓                          ↓              ↓           ↓
             Reflex                   (lateral E/I)   Predictive   Working Mem
                ↓                                     (recurrent)     ↓
              Motor ← Cerebellum                          ↓          ↓
                ↑                                    Meta-Controller
                └────────────────────────────────── (neuromodulation)
```

---

## Abstract Concept Formation (Emerges, Not Engineered)

Concepts are NOT pre-built. They emerge through the pipeline:

1. **Sensory features** (sensory cortex) — edges, frequencies, pressure
2. **Complex features** (feature layer) — shapes, phonemes, textures via STDP
3. **Multi-modal binding** (association cortex) — "red+round+smooth = apple" via lateral STDP + gamma oscillations from E/I balance
4. **Prototype extraction** (association cortex over time) — shared features across many "dog" experiences strengthen; unique features average out
5. **Abstract SDR** (concept layer) — 200/10000 sparse code, compressed, modality-invariant
6. **Temporal abstraction** (predictive layer) — causal sequences "A then B then C" across concept-level patterns
7. **Relational patterns** (predictive + working memory) — "bigger-than", "causes" emerge as temporal structure across many specific instances

**Numerosity** ("three-ness"): Many sets of three items → sensory cortex encodes individual items as activity bumps → feature layer develops "number of bumps" detectors via convergent STDP → concept layer receives these across modalities → neurons selective for "three" emerge. Inhibitory competition provides the normalization (response to COUNT, not total energy).

**Justice/fairness**: Embodied emotional grounding (drives/brainstem provide "distress" and "satisfaction") → social observation ("taking causes distress") → predictive layer learns causal relation → abstract rule "causing distress to others triggers punishment from group" → the concept. Requires full pipeline + long developmental timescale.

---

## Key Numbers

| Metric | Current | After Upgrade |
|--------|---------|---------------|
| Regions | 8 | 12 |
| Neurons | 187K | ~220K |
| Synapse groups | 15 | ~30 |
| Synapses | 40M | ~55M |
| RAM (weights) | ~600MB | ~900MB (incl. eligibility traces) |
| Step time | ~17ms | ~22ms (estimated) |
| New files | 0 | 1 (`neuromodulation.py`) |
| Modified files | 0 | 10 core + 11 test |

---

## Key References

- Fremaux & Gerstner (2016) — Three-factor learning rules (Frontiers Neural Circuits)
- Izhikevich (2007) — DA-modulated STDP with eligibility traces (Cerebral Cortex)
- Bellec et al. (2020) — e-prop with eligibility traces (Nature Communications)
- Quiroga et al. (2005) — Concept cells in MTL (Nature)
- Patterson et al. (2007) — Hub-and-spoke semantic memory (Nature Rev Neuroscience)
- Tishby et al. (1999) — Information Bottleneck principle
- Ahmad & Hawkins (2016) — Sparse distributed representations (Numenta)
- Dehaene (2005) — Cortical recycling for transfer learning
- Bienenstock, Cooper, Munro (1982) — BCM metaplasticity rule
- Brzosko et al. (2019) — Sequential ACh+DA neuromodulation (eLife)
