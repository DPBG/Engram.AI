# Engram Technical Architecture Diagrams

Detailed flow diagrams for the 4 core systems that define Engram's neuromorphic brain
and would be hardwired into a custom ASIC.

---

## Table of Contents

1. [System Overview — One Simulation Step](#1-system-overview)
2. [The 6 Learning Mechanisms — Full Pipeline](#2-the-6-learning-mechanisms)
3. [4-Compartment Dendritic Processing](#3-4-compartment-dendritic-processing)
4. [Developmental Phase Transitions](#4-developmental-phase-transitions)
5. [CSR Sparse Matrix Operations](#5-csr-sparse-matrix-operations)
6. [Custom ASIC Block Diagram](#6-custom-asic-block-diagram)

---

## 1. System Overview

### What Happens in ONE Simulation Step (~1ms brain time)

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║                        ONE SIMULATION STEP (~1ms)                            ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║  ┌─────────────┐                                                              ║
║  │ SENSORY      │  Camera, Mic, Touch, IMU, etc.                              ║
║  │ INPUT        │  Encoded as spike currents                                  ║
║  └──────┬──────┘                                                              ║
║         │                                                                     ║
║         ▼                                                                     ║
║  ┌──────────────────────────────────────────────────────────────────────┐     ║
║  │  PHASE 1: CURRENT ACCUMULATION                                       │     ║
║  │                                                                      │     ║
║  │  For each synapse group (24 groups):                                 │     ║
║  │    current[post] = CSR_matrix @ spikes[pre]    ← SpMV operation     │     ║
║  │                                                                      │     ║
║  │  For each neuron:                                                    │     ║
║  │    Route current to correct COMPARTMENT (apical/basal/perisomatic)  │     ║
║  │    Apply instinctual GAIN (novelty, habituation, cross-modal)       │     ║
║  │    Add sensory input current (if sensory neuron)                     │     ║
║  └──────────────────────┬───────────────────────────────────────────────┘     ║
║                         │                                                     ║
║                         ▼                                                     ║
║  ┌──────────────────────────────────────────────────────────────────────┐     ║
║  │  PHASE 2: DENDRITIC INTEGRATION                                      │     ║
║  │                                                                      │     ║
║  │  For each neuron (1,001,800 neurons):                               │     ║
║  │    1. Update 4 compartment voltages (leak + input current)          │     ║
║  │    2. Check dendritic spike thresholds (apical: 2.5x amplification) │     ║
║  │    3. Couple compartments → soma: v_soma += Σ(coupling × v_dend)   │     ║
║  │    4. Check soma threshold → FIRE or not                            │     ║
║  │    5. If fired: reset, enter refractory, record spike time          │     ║
║  └──────────────────────┬───────────────────────────────────────────────┘     ║
║                         │                                                     ║
║                         ▼                                                     ║
║  ┌──────────────────────────────────────────────────────────────────────┐     ║
║  │  PHASE 3: LEARNING (all 6 mechanisms, simultaneously)                │     ║
║  │                                                                      │     ║
║  │  For each synapse group with spiking pre OR post neurons:           │     ║
║  │    ┌──────────┐   ┌──────────────┐   ┌────────────┐                │     ║
║  │    │  STDP     │──▶│ Eligibility  │──▶│ Neuromod   │                │     ║
║  │    │  Δw calc  │   │ Trace += Δw  │   │ Gating     │                │     ║
║  │    └──────────┘   └──────────────┘   │ DA×ACh×NE  │                │     ║
║  │         │                             │ ×5HT       │                │     ║
║  │         ▼                             └─────┬──────┘                │     ║
║  │    ┌──────────┐                             │                       │     ║
║  │    │  BCM      │   scales a_plus/a_minus    │                       │     ║
║  │    │  θ update │◀──────────────────────────┘                       │     ║
║  │    └──────────┘                             │                       │     ║
║  │                                              ▼                       │     ║
║  │    ┌──────────────┐   ┌──────────────────────────┐                  │     ║
║  │    │ Compartment   │   │ Weight Update             │                  │     ║
║  │    │ Activity      │──▶│ w += elig × neuromod      │                  │     ║
║  │    │ Scaling       │   │     × bcm × compartment   │                  │     ║
║  │    └──────────────┘   └──────────┬───────────────┘                  │     ║
║  │                                   │                                  │     ║
║  │    ┌──────────────┐               │                                  │     ║
║  │    │ R-STDP        │               │  (motor groups only)            │     ║
║  │    │ × pred_error  │──────────────▶│                                  │     ║
║  │    └──────────────┘               │                                  │     ║
║  │                                   ▼                                  │     ║
║  │    ┌──────────────────────────────────────┐                          │     ║
║  │    │ Homeostatic Scaling                   │                          │     ║
║  │    │ Per-neuron row normalization           │                          │     ║
║  │    │ total = Σ(incoming weights)           │                          │     ║
║  │    │ weights *= target / total             │                          │     ║
║  │    └──────────────────────────────────────┘                          │     ║
║  └──────────────────────┬───────────────────────────────────────────────┘     ║
║                         │                                                     ║
║                         ▼                                                     ║
║  ┌──────────────────────────────────────────────────────────────────────┐     ║
║  │  PHASE 4: OUTPUT & FEEDBACK                                          │     ║
║  │                                                                      │     ║
║  │  Decode motor cortex spikes → locomotion, manipulation, speech, etc │     ║
║  │  Publish motor commands via NATS                                    │     ║
║  │  Check cognitive firing → query LLM if sustained prediction error   │     ║
║  │  Update developmental phase metrics                                 │     ║
║  │  Decay eligibility traces: elig *= exp(-dt/tau_trace)               │     ║
║  │  Update neuromodulator levels based on reward/novelty signals       │     ║
║  └──────────────────────────────────────────────────────────────────────┘     ║
║                                                                               ║
║  Total per step at 1M neurons, 1.19B synapses: ~1.27 seconds wall time      ║
╚═══════════════════════════════════════════════════════════════════════════════╝
```

---

## 2. The 6 Learning Mechanisms

### 2.1 How They Connect — The Learning Pipeline

```
                          ┌─────────────────────────┐
                          │    SPIKE EVENT            │
                          │  Pre neuron j fires, OR   │
                          │  Post neuron i fires      │
                          └────────────┬──────────────┘
                                       │
                                       ▼
                ┌─────────────────────────────────────────┐
                │         (1) STDP — Raw Weight Change     │
                │                                          │
                │  Δt = t_post - t_pre                     │
                │                                          │
                │  if Δt > 0 (pre before post = causal):   │
                │    Δw = +a_plus × exp(-Δt / tau_plus)    │
                │         ▲                                │
                │         │ scaled by (3) BCM              │
                │                                          │
                │  if Δt < 0 (post before pre = acausal):  │
                │    Δw = -a_minus × exp(+Δt / tau_minus)  │
                │         ▲                                │
                │         │ scaled by (3) BCM              │
                └────────────────┬────────────────────────┘
                                 │
                                 │  Δw does NOT go to weights yet!
                                 ▼
                ┌─────────────────────────────────────────┐
                │      (2) ELIGIBILITY TRACE               │
                │                                          │
                │  elig[synapse] += Δw                     │
                │                                          │
                │  Every step (regardless of spikes):      │
                │  elig[synapse] *= exp(-dt / 1000ms)      │
                │                                          │
                │  Trace persists ~1 second of brain time  │
                │  "Remembers" which synapses were active  │
                └────────────────┬────────────────────────┘
                                 │
                                 │  Trace is GATED by neuromodulators
                                 ▼
                ┌─────────────────────────────────────────┐
                │     (4) NEUROMODULATOR GATING            │
                │                                          │
                │  Δw_actual = elig[syn]                   │
                │              × DA    (reward signal)     │
                │              × f(ACh) (attention)        │
                │              × f(NE)  (arousal)          │
                │              × f(5HT) (patience)         │
                │                                          │
                │  DA  ∈ [0.2, 2.5]  ← reward/prediction  │
                │  ACh ∈ [0.3, 2.0]  ← novelty/attention  │
                │  NE  ∈ [0.1, 2.0]  ← urgency/arousal    │
                │  5HT ∈ [0.1, 1.5]  ← stability/calm     │
                │                                          │
                │  Non-zero floors prevent total shutdown   │
                └────────────────┬────────────────────────┘
                                 │
                        ┌────────┴────────┐
                        │                 │
                        ▼                 ▼
         ┌────────────────────┐  ┌─────────────────────┐
         │ (5) COMPARTMENT     │  │ (6) R-STDP           │
         │     SCALING         │  │ (motor groups only)  │
         │                     │  │                      │
         │ scale = activity    │  │ Δw *= pred_error     │
         │ of target           │  │                      │
         │ compartment at      │  │ High error → learn   │
         │ spike time          │  │ Low error → stable   │
         │                     │  │                      │
         │ Active compartment: │  │ Prediction error =   │
         │   full Δw           │  │ |expected - actual|  │
         │ Quiet compartment:  │  │ from motor outcome   │
         │   reduced Δw        │  │ feedback via NATS    │
         └─────────┬──────────┘  └──────────┬──────────┘
                   │                         │
                   └────────────┬────────────┘
                                │
                                ▼
                ┌─────────────────────────────────────────┐
                │        WEIGHT UPDATE                     │
                │                                          │
                │  w[syn] += Δw_actual × bcm × compart     │
                │                        × pred_error      │
                │                        (if R-STDP group) │
                │                                          │
                │  w[syn] = clip(w[syn], w_min, w_max)     │
                │           where w_min = 0.01 (never 0)   │
                └────────────────┬────────────────────────┘
                                 │
                                 ▼
                ┌─────────────────────────────────────────┐
                │   (3) BCM METAPLASTICITY — θ Update      │
                │                                          │
                │  θ[neuron_i] = EMA(firing_rate_i²)       │
                │                                          │
                │  High activity → θ rises → harder to     │
                │  potentiate → neuron becomes selective    │
                │                                          │
                │  Low activity → θ falls → easier to      │
                │  potentiate → neuron finds new inputs     │
                │                                          │
                │  Scaling:                                 │
                │  a_plus_eff  = a_plus  × ε/(θ+ε)        │
                │  a_minus_eff = a_minus × ε/(θ+ε)        │
                │                                          │
                │  Applied NEXT time STDP fires for this   │
                │  post-neuron (feeds back to step 1)      │
                └────────────────┬────────────────────────┘
                                 │
                                 ▼
                ┌─────────────────────────────────────────┐
                │   (5b) HOMEOSTATIC SYNAPTIC SCALING      │
                │   (runs periodically, not every step)    │
                │                                          │
                │  For each post-neuron i:                  │
                │    total = Σ w[all synapses → i]         │
                │    scale = target_total / total           │
                │    w[all synapses → i] *= scale          │
                │                                          │
                │  Effect:                                  │
                │    Strengthening some → weakens others   │
                │    Prevents runaway excitation            │
                │    Prevents catastrophic forgetting       │
                │    Old memories scaled down, not erased   │
                │                                          │
                │  Rate: REDUCED during infant/adolescent   │
                │         FULL during mature phase          │
                └─────────────────────────────────────────┘
```

### 2.2 Mechanism Interaction Matrix

```
Which mechanisms affect which:

                  ┌──────┬──────┬──────┬──────┬──────┬──────┐
                  │ STDP │Elig  │ BCM  │Neuro │Homeo │R-STDP│
                  │      │Trace │      │ mod  │static│      │
    ┌─────────────┼──────┼──────┼──────┼──────┼──────┼──────┤
    │ STDP        │  —   │writes│reads │      │      │      │
    │             │      │ to   │ from │      │      │      │
    ├─────────────┼──────┼──────┼──────┼──────┼──────┼──────┤
    │ Elig Trace  │reads │  —   │      │gated │      │      │
    │             │ from │      │      │ by   │      │      │
    ├─────────────┼──────┼──────┼──────┼──────┼──────┼──────┤
    │ BCM         │scales│      │  —   │      │      │      │
    │             │rates │      │      │      │      │      │
    ├─────────────┼──────┼──────┼──────┼──────┼──────┼──────┤
    │ Neuromod    │      │gates │      │  —   │      │      │
    │             │      │apply │      │      │      │      │
    ├─────────────┼──────┼──────┼──────┼──────┼──────┼──────┤
    │ Homeostatic │      │      │      │      │  —   │      │
    │             │      │      │      │      │      │      │
    ├─────────────┼──────┼──────┼──────┼──────┼──────┼──────┤
    │ R-STDP      │mult  │      │      │      │      │  —   │
    │             │ Δw   │      │      │      │      │      │
    └─────────────┴──────┴──────┴──────┴──────┴──────┴──────┘

    reads from / writes to = data dependency
    gates / scales / mult  = multiplicative modifier
    Homeostatic operates independently on final weights
```

### 2.3 Timing Diagram — One Synapse Over Time

```
Time (ms) →  0    5    10   15   20   25   30   ...  500  ...  1000  1500
             │    │    │    │    │    │    │         │        │     │
Pre spike:   ▲                                      ▲
Post spike:       ▲              ▲                        ▲
             │    │    │    │    │    │    │         │        │     │
STDP Δw:     │ +0.008  │    │ +0.004│    │         │   +0.007│     │
             │ (Δt=5ms)│    │(Δt=15)│    │         │  (Δt=5) │     │
             │    │    │    │    │    │    │         │        │     │
Elig Trace:  │    │    │    │    │    │    │         │        │     │
  value:     0  0.008 0.007 0.007 0.011 0.010       0.003   0.010 0.005
             │   ↑    ↓    │    ↑    ↓              ↓       ↑     ↓
             │ +=Δw  decay │  +=Δw  decay          decay  +=Δw  decay
             │    │    │    │    │    │    │         │        │     │
DA level:    1.0  1.0  1.0  1.0  1.0  1.0  1.0     2.0      1.5   1.0
             │    │    │    │    │    │    │         ↑        │     │
             │    │    │    │    │    │    │      REWARD!     │     │
             │    │    │    │    │    │    │         │        │     │
Weight Δ:    0    0    0    0    0    0    0      +0.006      │     │
             │    │    │    │    │    │    │      elig×DA     │     │
             │    │    │    │    │    │    │      0.003×2.0   │     │
             │    │    │    │    │    │    │         │        │     │
             ▼    ▼    ▼    ▼    ▼    ▼    ▼         ▼        ▼     ▼

KEY INSIGHT: The robot did something at t=0-20ms. Reward arrived at t=500ms.
The eligibility trace REMEMBERED which synapses were active (decayed but present).
When DA spiked, those synapses got strengthened. Without eligibility traces,
the 500ms delay would make learning impossible.
```

---

## 3. 4-Compartment Dendritic Processing

### 3.1 Single Neuron Anatomy

```
                    APICAL DISTAL COMPARTMENT
                    ┌───────────────────────┐
                    │  v_apical_distal       │
                    │  Receives: top-down    │
                    │  feedback from higher  │
                    │  cortical regions      │
                    │                        │
                    │  Dendritic spike:      │
                    │  if v > threshold →    │
                    │  v *= 2.5 (supralinear)│
                    │                        │
                    │  Synapse groups:       │
                    │  • association→concept │
                    │  • concept→meta        │
                    └───────────┬────────────┘
                                │
                         coupling = 0.3
                       (weak influence on soma)
                                │
                    ┌───────────┴────────────┐
                    │ APICAL PROXIMAL         │
                    │  v_apical_proximal      │
                    │  Integration zone       │
                    │                         │
                    │  Combines:              │
                    │  • Apical distal input  │
                    │  • Its own synaptic     │
                    │    input                │
                    │                         │
                    │  Dendritic spike:       │
                    │  if v > threshold →     │
                    │  v *= 2.5              │
                    └───────────┬─────────────┘
                                │
                         coupling = 0.5
                                │
    ┌───────────────────────────┴─────────────────────────────┐
    │                         SOMA                             │
    │                    (Cell Body)                            │
    │                                                          │
    │   v_soma += Σ (coupling_k × v_compartment_k)            │
    │                                                          │
    │   v_soma += leak current                                 │
    │   v_soma += noise                                        │
    │                                                          │
    │   IF v_soma > threshold AND not refractory:              │
    │     → SPIKE! (output = 1)                                │
    │     → v_soma = v_reset                                   │
    │     → refractory timer = refractory_period               │
    │     → record last_spike_time                             │
    │     → trigger STDP for all incoming synapses             │
    │                                                          │
    └──────┬─────────────────────────────────────────┬─────────┘
           │                                         │
    coupling = 0.7                            coupling = 0.9
    (strong feedforward)                      (very strong inhibition)
           │                                         │
    ┌──────┴───────────────┐              ┌──────────┴──────────┐
    │ BASAL COMPARTMENT     │              │ PERISOMATIC          │
    │  v_basal               │              │  v_perisomatic       │
    │                        │              │                      │
    │  Receives: feedforward │              │  Receives: inhibitory│
    │  input from same layer │              │  input from basket   │
    │  or lower regions      │              │  cells / interneurons│
    │                        │              │                      │
    │  Synapse groups:       │              │  Effect: powerful    │
    │  • sensory→association │              │  veto on firing      │
    │  • sensory→feature     │              │  (coupling near 1.0) │
    │  • feature→concept     │              │                      │
    │  No dendritic spikes   │              │  Creates competition │
    │  (linear summation)    │              │  among neurons       │
    └────────────────────────┘              └──────────────────────┘
```

### 3.2 Compartment-Aware STDP — Credit Assignment

```
Question: Post-neuron i fired. WHY? Which input pathway caused it?

┌─────────────────────────────────────────────────────────────────┐
│ Standard STDP (no compartments):                                 │
│   All incoming synapses get the same Δw                         │
│   → No way to know if it was the feedforward or feedback input  │
│   → POOR credit assignment                                      │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ Compartment-Aware STDP (Engram):                                  │
│                                                                   │
│   Synapse A → basal compartment                                  │
│     basal was active (v = 0.8) when post spiked                  │
│     Δw_A = STDP_Δw × 0.8 = FULL learning                       │
│     "Basal drove the spike — credit this input"                  │
│                                                                   │
│   Synapse B → apical distal compartment                          │
│     apical was quiet (v = 0.1) when post spiked                  │
│     Δw_B = STDP_Δw × 0.1 = MINIMAL learning                    │
│     "Apical didn't contribute — don't credit this input"         │
│                                                                   │
│   Result: The brain learns WHICH pathway caused the spike        │
│   This is the basis for hierarchical learning                    │
└─────────────────────────────────────────────────────────────────┘


Data flow during one post-spike STDP update:

  For synapse j → post-neuron i:

  ┌──────────────┐     ┌─────────────────┐     ┌───────────────┐
  │ Which         │     │ What was that    │     │ Final Δw       │
  │ compartment   │────▶│ compartment's    │────▶│ = STDP_Δw      │
  │ does syn j    │     │ voltage at       │     │   × compart_v  │
  │ target?       │     │ spike time?      │     │   × BCM scale  │
  │               │     │                  │     │   × neuromod   │
  │ target = 2    │     │ v[2] = 0.8       │     │   × (R-STDP?)  │
  │ (basal)       │     │ (active!)        │     │                │
  └──────────────┘     └─────────────────┘     └───────────────┘
```

### 3.3 Dendritic Spike Mechanism

```
Normal synaptic input (basal):
  v_basal += Σ(incoming currents)    ← linear summation
  Contribution to soma: 0.7 × v_basal

Dendritic spike (apical):

  v_apical += Σ(incoming currents)

                     threshold
                         │
  ───────────────────────┼───────────────────
                         │
  v_apical:  0.1  0.3   │0.5  0.4  0.3  ...   (sub-threshold: normal)
                         │
  v_apical:  0.1  0.3   │0.6  ← CROSSES THRESHOLD
                         │     │
                         │     ▼
                         │    v_apical = 0.6 × 2.5 = 1.5  ← SUPRALINEAR!
                         │
                         │    Contribution to soma: 0.3 × 1.5 = 0.45
                         │    (vs normal 0.3 × 0.6 = 0.18)
                         │
                         │    2.5x amplification!

  WHY THIS MATTERS:
  ┌────────────────────────────────────────────────────────┐
  │ Top-down context (from higher cortex) arrives at       │
  │ apical dendrites. When context is STRONG ENOUGH,       │
  │ it triggers a dendritic spike that dramatically boosts │
  │ the neuron's response — even if feedforward input      │
  │ (basal) is weak.                                       │
  │                                                        │
  │ This is how the brain implements ATTENTION:             │
  │ "I expect to see a cat" (top-down context) →           │
  │ apical dendritic spike → cat-detecting neurons fire    │
  │ more easily even with ambiguous visual input            │
  └────────────────────────────────────────────────────────┘
```

---

## 4. Developmental Phase Transitions

### 4.1 Full Phase State Machine

```
                    ┌───────────────────┐
                    │     INFANT         │
                    │  steps 0 → 60K     │
                    │                    │
                    │  DA=1.5  ACh=1.8   │
                    │  NE=1.2  5HT=0.3   │
                    │                    │
                    │  Max plasticity     │
                    │  Fast wiring        │
                    │  Homeostatic: LOW   │
                    └─────────┬──────────┘
                              │
                    step > 60,000
                    (time-based, smooth
                     interpolation of
                     neuromod baselines)
                              │
                              ▼
                    ┌───────────────────┐
                    │     TODDLER        │
                    │  steps 60K → 360K  │
                    │                    │
                    │  DA=1.2  ACh=1.5   │
                    │  NE=1.0  5HT=0.5   │
                    │                    │
                    │  Cross-modal bind   │
                    │  Motor exploration  │
                    │  Category forming   │
                    └─────────┬──────────┘
                              │
                    step > 360,000
                    (time-based)
                              │
                              ▼
                    ┌───────────────────┐
                    │     JUVENILE        │
                    │  steps 360K → ???   │
                    │                    │
                    │  DA=1.0  ACh=1.2   │
                    │  NE=0.8  5HT=0.8   │
                    │                    │
                    │  Feature extract    │
                    │  Concept formation  │
                    │  BCM stabilizing    │
                    │  STDP rates decline │
                    └─────────┬──────────┘
                              │
                    ┌─────────┴──────────────────────────────┐
                    │  EXPERIENCE-DEPENDENT GATE              │
                    │  ALL 4 criteria must be TRUE:           │
                    │                                         │
                    │  ┌─ □ Concept Differentiation ────────┐ │
                    │  │  Association layer neurons respond  │ │
                    │  │  to DISTINCT inputs, not uniform   │ │
                    │  │  Metric: entropy of response       │ │
                    │  │  vectors > threshold               │ │
                    │  └────────────────────────────────────┘ │
                    │  ┌─ □ Sensory Stability ──────────────┐ │
                    │  │  Change detection output has       │ │
                    │  │  declined below threshold          │ │
                    │  │  "World model is converging"       │ │
                    │  └────────────────────────────────────┘ │
                    │  ┌─ □ Feature STDP Decline ───────────┐ │
                    │  │  Average STDP weight change on     │ │
                    │  │  sensory→feature groups declining  │ │
                    │  │  "Basic features already learned"  │ │
                    │  └────────────────────────────────────┘ │
                    │  ┌─ □ Minimum Steps ──────────────────┐ │
                    │  │  Safety: don't enter adolescence   │ │
                    │  │  if metrics are met by random noise│ │
                    │  └────────────────────────────────────┘ │
                    │                                         │
                    │  ANY criterion FALSE → stay juvenile    │
                    │  ALL criteria TRUE  → enter adolescent  │
                    └─────────┬───────────────────────────────┘
                              │
                              ▼
          ┌───────────────────────────────────────────────────┐
          │                  ADOLESCENT                        │
          │           (variable duration)                      │
          │                                                    │
          │  DA=1.3  ACh=1.0  NE=1.5  5HT=0.6                │
          │  STDP widened: tau_plus=30ms, tau_minus=25ms       │
          │  a_plus scaled 1.5x (stronger potentiation)       │
          │                                                    │
          │  ┌─────────── STRUCTURAL MODIFICATIONS ──────────┐ │
          │  │  Every N steps, run these in sequence:         │ │
          │  │                                                │ │
          │  │  ╔══════════════════════════════════════════╗  │ │
          │  │  ║ 1. PRUNING                               ║  │ │
          │  │  ║                                           ║  │ │
          │  │  ║  Sort synapses by weight (ascending)     ║  │ │
          │  │  ║  Select bottom 5% for removal            ║  │ │
          │  │  ║  SKIP if: myelinated=True                ║  │ │
          │  │  ║  SKIP if: identity=True                  ║  │ │
          │  │  ║  DELETE from CSR matrix                   ║  │ │
          │  │  ║  Resize: data, indices, eligibility,     ║  │ │
          │  │  ║          stability, myelinated, identity  ║  │ │
          │  │  ║  Rebuild: indptr                          ║  │ │
          │  │  ╚══════════════════════════════════════════╝  │ │
          │  │                    │                            │ │
          │  │                    ▼                            │ │
          │  │  ╔══════════════════════════════════════════╗  │ │
          │  │  ║ 2. MYELINATION                           ║  │ │
          │  │  ║                                           ║  │ │
          │  │  ║  For each synapse:                        ║  │ │
          │  │  ║    if weight > high_threshold             ║  │ │
          │  │  ║    AND stability_counter > N steps:       ║  │ │
          │  │  ║      myelinated[syn] = True               ║  │ │
          │  │  ║      plasticity reduced to 10%            ║  │ │
          │  │  ║                                           ║  │ │
          │  │  ║  "This connection is proven reliable.     ║  │ │
          │  │  ║   Lock it in but allow slow adaptation."  ║  │ │
          │  │  ╚══════════════════════════════════════════╝  │ │
          │  │                    │                            │ │
          │  │                    ▼                            │ │
          │  │  ╔══════════════════════════════════════════╗  │ │
          │  │  ║ 3. IDENTITY TAGGING                      ║  │ │
          │  │  ║                                           ║  │ │
          │  │  ║  For each synapse:                        ║  │ │
          │  │  ║    if myelinated=True                     ║  │ │
          │  │  ║    AND survived pruning round:            ║  │ │
          │  │  ║      identity[syn] = True                 ║  │ │
          │  │  ║      plasticity reduced to 1%             ║  │ │
          │  │  ║                                           ║  │ │
          │  │  ║  "This is WHO I AM. Core skill.           ║  │ │
          │  │  ║   Nearly permanent. My personality."      ║  │ │
          │  │  ╚══════════════════════════════════════════╝  │ │
          │  │                    │                            │ │
          │  │                    ▼                            │ │
          │  │  ╔══════════════════════════════════════════╗  │ │
          │  │  ║ 4. NEIGHBORHOOD CONSOLIDATION            ║  │ │
          │  │  ║                                           ║  │ │
          │  │  ║  After pruning, DA burst signal sent     ║  │ │
          │  │  ║  to nearby synapses                      ║  │ │
          │  │  ║                                           ║  │ │
          │  │  ║  Effect: eligibility traces near          ║  │ │
          │  │  ║  pruned regions get RESCUED               ║  │ │
          │  │  ║  (multiplied by DA boost factor)          ║  │ │
          │  │  ║                                           ║  │ │
          │  │  ║  "Losing neighbors → strengthen           ║  │ │
          │  │  ║   surviving connections"                   ║  │ │
          │  │  ╚══════════════════════════════════════════╝  │ │
          │  └────────────────────────────────────────────────┘ │
          └───────────────────┬───────────────────────────────┘
                              │
                    Adolescent duration complete
                    (time-based after entry)
                              │
                              ▼
                    ┌───────────────────┐
                    │      MATURE        │
                    │   (indefinite)     │
                    │                    │
                    │  DA=0.8  ACh=0.8   │
                    │  NE=0.6  5HT=1.2   │
                    │                    │
                    │  High stability     │
                    │  Slow refinement    │
                    │  Homeostatic: FULL  │
                    │  Continual learning │
                    │  via eligibility +  │
                    │  un-myelinated syns │
                    │                    │
                    │  NEVER FROZEN.      │
                    │  Still learns.      │
                    │  Just slowly.       │
                    └────────────────────┘
```

### 4.2 Neuromodulator Baselines Across Phases

```
Level
2.0 │
    │  ╭─── ACh (attention)
1.8 │──╯
    │      ╲
1.5 │  DA───╲──────────╮
    │  (reward)  ╲      │
1.2 │  NE────────╲─────│───╮
    │              ╲    │   │
1.0 │               ╲───│───│─── DA
    │                 ╲  │   │
0.8 │                  ╲ │   │── ACh ── DA ── NE
    │                   ╲│   │
0.6 │                    ╲   │── 5HT ── ACh
    │                     ╲  │
0.5 │                      ╲ │
    │                       ╲│
0.3 │  5HT───────────────────╲
    │  (patience)              ╲──── 5HT (now HIGH = stable)
0.1 │                                 NE  (now LOW = calm)
    │
    └────────┬─────────┬──────────┬───────────┬──────────▶
          INFANT    TODDLER   JUVENILE  ADOLESCENT  MATURE

Key transitions:
  INFANT:      Maximum learning. ACh sky-high (attend to everything).
               5HT rock-bottom (no patience, absorb fast).

  TODDLER:     Slightly calmer. DA still high (reward-seeking).

  JUVENILE:    Balanced. Learning rates declining naturally.
               BCM θ values stabilizing.

  ADOLESCENT:  NE spikes (arousal for structural changes).
               DA elevated (reward pruning survivors).
               ACh drops (selective attention, not broad).
               STDP windows WIDEN (capture more temporal patterns).

  MATURE:      5HT dominates (patient, stable).
               All others reduced (slow, careful learning).
               Not frozen — just deliberate.
```

---

## 5. CSR Sparse Matrix Operations

### 5.1 CSR Structure Explained

```
Example: 5 post-neurons, 4 pre-neurons, 8 synapses

Dense view (what CSR represents):        Sparse Reality:
                                          Only 8 of 20 entries are non-zero
  post\pre   0     1     2     3          (60% sparse — real brain is 99.9%+)
    0      [0.5   0.0   0.3   0.0]
    1      [0.0   0.7   0.0   0.2]
    2      [0.0   0.0   0.0   0.0]        ← row 2 has no synapses
    3      [0.1   0.0   0.8   0.6]
    4      [0.0   0.4   0.0   0.0]

CSR Encoding:

  indptr:   [0,    2,    4,    4,    7,    8]
             │     │     │     │     │     │
             │     │     │     │     │     └─ end of row 4
             │     │     │     │     └─ start of row 4 (syn 7)
             │     │     │     └─ start of row 3 (syn 4) — NOTE: same as row 2 end
             │     │     └─ start of row 2 (syn 4) — empty row!
             │     └─ start of row 1 (syn 2)
             └─ start of row 0 (syn 0)

  indices:  [0,    2,    1,    3,    0,    2,    3,    1]
             │     │     │     │     │     │     │     │
             row 0       row 1       row 3            row 4
             pre=0 pre=2 pre=1 pre=3 pre=0 pre=2 pre=3 pre=1

  data:     [0.5,  0.3,  0.7,  0.2,  0.1,  0.8,  0.6,  0.4]
             │     │     │     │     │     │     │     │
             w(0→0) w(2→0) w(1→1) w(3→1) w(0→3) w(2→3) w(3→3) w(1→4)


Getting all synapses to post-neuron 3:
  start = indptr[3] = 4
  end   = indptr[4] = 7
  pre_neurons = indices[4:7] = [0, 2, 3]
  weights     = data[4:7]    = [0.1, 0.8, 0.6]
  → O(1) lookup! No scanning needed.
```

### 5.2 SpMV — The Core Operation

```
Input:  spikes = [1, 0, 1, 0]   (neurons 0 and 2 fired)

Operation: current = CSR_matrix @ spikes

Step-by-step (what the C code does):

  Post-neuron 0:
    start=0, end=2
    syn 0: pre=0, weight=0.5, spikes[0]=1 → 0.5 × 1 = 0.5 ✓
    syn 1: pre=2, weight=0.3, spikes[2]=1 → 0.3 × 1 = 0.3 ✓
    current[0] = 0.5 + 0.3 = 0.8

  Post-neuron 1:
    start=2, end=4
    syn 2: pre=1, weight=0.7, spikes[1]=0 → 0.7 × 0 = 0.0
    syn 3: pre=3, weight=0.2, spikes[3]=0 → 0.2 × 0 = 0.0
    current[1] = 0.0

  Post-neuron 2:
    start=4, end=4       ← empty row
    current[2] = 0.0

  Post-neuron 3:
    start=4, end=7
    syn 4: pre=0, weight=0.1, spikes[0]=1 → 0.1 × 1 = 0.1 ✓
    syn 5: pre=2, weight=0.8, spikes[2]=1 → 0.8 × 1 = 0.8 ✓
    syn 6: pre=3, weight=0.6, spikes[3]=0 → 0.6 × 0 = 0.0
    current[3] = 0.1 + 0.8 = 0.9

  Post-neuron 4:
    start=7, end=8
    syn 7: pre=1, weight=0.4, spikes[1]=0 → 0.4 × 0 = 0.0
    current[4] = 0.0

  Result: current = [0.8, 0.0, 0.0, 0.9, 0.0]

  At 1M scale: 1.19 BILLION of these lookups per step.
  SciPy's C code: ~4 seconds on single core.
  128 cores:       ~33ms.
  Custom ASIC:     ~microseconds (hardwired row processors).
```

### 5.3 Parallel Arrays — Everything Aligned to CSR

```
Index:      0     1     2     3     4     5     6     7
            ├─────┼─────┼─────┼─────┼─────┼─────┼─────┤
data:       │0.5  │0.3  │0.7  │0.2  │0.1  │0.8  │0.6  │0.4 │  ← weights
            ├─────┼─────┼─────┼─────┼─────┼─────┼─────┤
indices:    │ 0   │ 2   │ 1   │ 3   │ 0   │ 2   │ 3   │ 1  │  ← pre neuron
            ├─────┼─────┼─────┼─────┼─────┼─────┼─────┤
eligibility:│0.001│0.000│0.003│0.000│0.002│0.008│0.001│0.000│  ← trace
            ├─────┼─────┼─────┼─────┼─────┼─────┼─────┤
stability:  │ 450 │ 230 │ 890 │ 100 │ 50  │1200 │ 780 │ 340│  ← steps stable
            ├─────┼─────┼─────┼─────┼─────┼─────┼─────┤
myelinated: │  0  │  0  │  1  │  0  │  0  │  1  │  0  │  0 │  ← locked 10%
            ├─────┼─────┼─────┼─────┼─────┼─────┼─────┤
identity:   │  0  │  0  │  1  │  0  │  0  │  0  │  0  │  0 │  ← locked 1%
            ├─────┼─────┼─────┼─────┼─────┼─────┼─────┤
compartment:│ bas │ bas │ api │ per │ bas │ api │ api │ bas│  ← target
            └─────┴─────┴─────┴─────┴─────┴─────┴─────┘

All arrays indexed identically: syn_index 5 always refers to the same synapse
across ALL arrays. This is what makes per-synapse operations efficient.

At 1.19B synapses:
  data:        4.76 GB (float32)
  indices:     4.76 GB (int32)
  eligibility: 4.76 GB (float32)
  stability:   2.38 GB (int16)
  myelinated:  1.19 GB (bool → packed bits = 149 MB)
  identity:    1.19 GB (bool → packed bits = 149 MB)
  compartment: 1.19 GB (2-bit enum → 298 MB)
  indptr:      4 MB   (1M+1 × int32)
  ─────────────────────────────
  Total:       ~17 GB synapse state per group
               × 24 groups = theoretical max ~408 GB
               (actual: ~27 GB due to sparse group sizes)
```

### 5.4 Pruning — CSR Structural Modification

```
BEFORE PRUNING (row 3 of example):
  indptr: [..., 4, 7, ...]     ← row 3 has 3 synapses (index 4,5,6)
  indices: [..., 0, 2, 3, ...]  ← connected to pre-neurons 0, 2, 3
  data:    [..., 0.1, 0.8, 0.6, ...]

  Prune target: synapse 4 (weight=0.1, weakest)
  Check: myelinated[4]=False, identity[4]=False → OK to prune

AFTER PRUNING:
  indptr: [..., 4, 6, ...]     ← row 3 now has 2 synapses (index 4,5)
  indices: [..., 2, 3, ...]     ← pre-neurons 0 removed
  data:    [..., 0.8, 0.6, ...]

  ALL parallel arrays shift:
  eligibility: remove index 4, shift everything after left
  stability:   remove index 4, shift everything after left
  myelinated:  remove index 4, shift everything after left
  identity:    remove index 4, shift everything after left

  ALL indptr entries after row 3 decrement by 1

  This is EXPENSIVE: O(nnz) for the shift.
  That's why pruning is max 5% per round, only during adolescence.
  At 1.19B synapses: pruning round touches ~60M entries.
```

---

## 6. Custom ASIC Block Diagram (Aspirational)

> **Note**: This section describes a theoretical custom ASIC design. No hardware design or fabrication is underway. This documents how the software architecture could map to custom silicon in the future.

### 6.1 Chip Architecture — Purpose-Built for Engram

```
╔══════════════════════════════════════════════════════════════════════════╗
║                        Engram NEUROMORPHIC ASIC                          ║
║                     (28nm, estimated die size: ~100mm²)                  ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  ┌────────────────────────────────────────────────────────────────────┐  ║
║  │                    SRAM MEMORY BANKS (largest area)                 │  ║
║  │                                                                    │  ║
║  │  Bank 0: indptr arrays (all 24 groups)                            │  ║
║  │  Bank 1-4: indices arrays (round-robin across groups)             │  ║
║  │  Bank 5-8: weight data (float16 for area savings)                 │  ║
║  │  Bank 9-12: eligibility traces (float16)                          │  ║
║  │  Bank 13-14: stability/myelinated/identity (packed)               │  ║
║  │  Bank 15: neuron state (v_soma, v_dend[4], theta, refractory)     │  ║
║  │                                                                    │  ║
║  │  Total SRAM: 32-64 MB on-chip (1M neuron scale)                   │  ║
║  │  Overflow: DDR5 off-chip for larger-than-1M configurations        │  ║
║  └────────────────────────────────────────────────────────────────────┘  ║
║                                    │                                     ║
║                    ┌───────────────┼───────────────┐                     ║
║                    │               │               │                     ║
║                    ▼               ▼               ▼                     ║
║  ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐         ║
║  │  SpMV ENGINE 0    │ │  SpMV ENGINE 1    │ │  SpMV ENGINE N    │  ...  ║
║  │                   │ │                   │ │  (64-256 total)   │       ║
║  │  ┌─────────────┐ │ │  ┌─────────────┐ │ │  ┌─────────────┐ │       ║
║  │  │Row Processor │ │ │  │Row Processor │ │ │  │Row Processor │ │       ║
║  │  │             │ │ │  │             │ │ │  │             │ │       ║
║  │  │ Read indptr │ │ │  │ Read indptr │ │ │  │ Read indptr │ │       ║
║  │  │ Stream data │ │ │  │ Stream data │ │ │  │ Stream data │ │       ║
║  │  │ Dot product │ │ │  │ Dot product │ │ │  │ Dot product │ │       ║
║  │  │ w × spike   │ │ │  │ w × spike   │ │ │  │ w × spike   │ │       ║
║  │  └──────┬──────┘ │ │  └──────┬──────┘ │ │  └──────┬──────┘ │       ║
║  │         │        │ │         │        │ │         │        │       ║
║  │  ┌──────▼──────┐ │ │  ┌──────▼──────┐ │ │  ┌──────▼──────┐ │       ║
║  │  │ STDP Unit    │ │ │  │ STDP Unit    │ │ │  │ STDP Unit    │ │       ║
║  │  │             │ │ │  │             │ │ │  │             │ │       ║
║  │  │ Δt compute  │ │ │  │ Δt compute  │ │ │  │ Δt compute  │ │       ║
║  │  │ exp(-Δt/τ)  │ │ │  │ exp(-Δt/τ)  │ │ │  │ exp(-Δt/τ)  │ │       ║
║  │  │ LUT-based   │ │ │  │ LUT-based   │ │ │  │ LUT-based   │ │       ║
║  │  └──────┬──────┘ │ │  └──────┬──────┘ │ │  └──────┬──────┘ │       ║
║  │         │        │ │         │        │ │         │        │       ║
║  │  ┌──────▼──────┐ │ │  ┌──────▼──────┐ │ │  ┌──────▼──────┐ │       ║
║  │  │ Elig+Neuro  │ │ │  │ Elig+Neuro  │ │ │  │ Elig+Neuro  │ │       ║
║  │  │ Fused Unit  │ │ │  │ Fused Unit  │ │ │  │ Fused Unit  │ │       ║
║  │  │             │ │ │  │             │ │ │  │             │ │       ║
║  │  │ elig += Δw  │ │ │  │ elig += Δw  │ │ │  │ elig += Δw  │ │       ║
║  │  │ elig *= decay│ │ │  │ elig *= decay│ │ │  │ elig *= decay│ │       ║
║  │  │ Δw = elig   │ │ │  │ Δw = elig   │ │ │  │ Δw = elig   │ │       ║
║  │  │  × DA×ACh   │ │ │  │  × DA×ACh   │ │ │  │  × DA×ACh   │ │       ║
║  │  │  × NE×5HT   │ │ │  │  × NE×5HT   │ │ │  │  × NE×5HT   │ │       ║
║  │  │  × BCM(θ)   │ │ │  │  × BCM(θ)   │ │ │  │  × BCM(θ)   │ │       ║
║  │  │  × compart  │ │ │  │  × compart  │ │ │  │  × compart  │ │       ║
║  │  └──────┬──────┘ │ │  └──────┬──────┘ │ │  └──────┬──────┘ │       ║
║  │         │        │ │         │        │ │         │        │       ║
║  │  ┌──────▼──────┐ │ │  ┌──────▼──────┐ │ │  ┌──────▼──────┐ │       ║
║  │  │Homeostatic  │ │ │  │Homeostatic  │ │ │  │Homeostatic  │ │       ║
║  │  │Row Sum +    │ │ │  │Row Sum +    │ │ │  │Row Sum +    │ │       ║
║  │  │Normalize    │ │ │  │Normalize    │ │ │  │Normalize    │ │       ║
║  │  └─────────────┘ │ │  └─────────────┘ │ │  └─────────────┘ │       ║
║  └──────────────────┘ └──────────────────┘ └──────────────────┘       ║
║                                    │                                     ║
║                    ┌───────────────┼───────────────┐                     ║
║                    ▼               ▼               ▼                     ║
║  ┌──────────────────────────────────────────────────────────────────┐    ║
║  │              NEURON PROCESSING CLUSTERS (16-64 clusters)         │    ║
║  │                                                                  │    ║
║  │  Each cluster handles ~16K-64K neurons:                          │    ║
║  │  ┌────────────────────────────────────────────┐                  │    ║
║  │  │  4-Compartment Dendritic Engine             │                  │    ║
║  │  │                                             │                  │    ║
║  │  │  For each neuron in this cluster:            │                  │    ║
║  │  │    1. Update v_apical_distal  (leak + input) │                  │    ║
║  │  │    2. Update v_apical_proximal               │                  │    ║
║  │  │    3. Update v_basal                         │                  │    ║
║  │  │    4. Update v_perisomatic                   │                  │    ║
║  │  │    5. Check dendritic spike thresholds       │                  │    ║
║  │  │    6. Couple to soma: v += Σ(c_k × v_k)    │                  │    ║
║  │  │    7. Soma threshold check → spike/no-spike  │                  │    ║
║  │  │    8. Update BCM θ (running average)         │                  │    ║
║  │  └────────────────────────────────────────────┘                  │    ║
║  └──────────────────────────────────────────────────────────────────┘    ║
║                                    │                                     ║
║                                    ▼                                     ║
║  ┌──────────────────────────────────────────────────────────────────┐    ║
║  │                     GLOBAL CONTROL UNIT                           │    ║
║  │                                                                   │    ║
║  │  ┌─────────────────┐  ┌──────────────────┐  ┌────────────────┐   │    ║
║  │  │ Phase State      │  │ Neuromodulator    │  │ Instinct Gain  │   │    ║
║  │  │ Machine          │  │ Registers         │  │ Computer       │   │    ║
║  │  │                  │  │                   │  │                │   │    ║
║  │  │ 5 states         │  │ DA  = reg[0]      │  │ novelty gain   │   │    ║
║  │  │ 4 entry criteria │  │ ACh = reg[1]      │  │ change gain    │   │    ║
║  │  │ phase-dependent  │  │ NE  = reg[2]      │  │ cross-modal    │   │    ║
║  │  │ parameter banks  │  │ 5HT = reg[3]      │  │ habituation    │   │    ║
║  │  │                  │  │                   │  │                │   │    ║
║  │  │ Switches STDP    │  │ Updated per step  │  │ Multiplicative │   │    ║
║  │  │ params + neuromod│  │ from reward/error  │  │ gain ≥ 1.0     │   │    ║
║  │  │ baselines on     │  │ signals           │  │ (never suppress)│   │    ║
║  │  │ phase transition │  │                   │  │                │   │    ║
║  │  └─────────────────┘  └──────────────────┘  └────────────────┘   │    ║
║  │                                                                   │    ║
║  │  ┌─────────────────┐  ┌──────────────────┐                       │    ║
║  │  │ Pruning Engine   │  │ I/O Controller    │                       │    ║
║  │  │ (adolescent only)│  │                   │                       │    ║
║  │  │                  │  │ NATS interface     │                       │    ║
║  │  │ Sort by weight   │  │ Sensor input DMA   │                       │    ║
║  │  │ Select bottom 5% │  │ Motor output DMA   │                       │    ║
║  │  │ CSR rebuild      │  │ Metric export      │                       │    ║
║  │  │ (slow, rare)     │  │                   │                       │    ║
║  │  └─────────────────┘  └──────────────────┘                       │    ║
║  └──────────────────────────────────────────────────────────────────┘    ║
║                                                                          ║
║  Power Budget (estimated):                                               ║
║    SpMV engines (64):     ~1.5W                                         ║
║    Neuron clusters (16):  ~0.5W                                         ║
║    SRAM (32MB):           ~1.0W                                         ║
║    Control + I/O:         ~0.3W                                         ║
║    Total:                 ~3.3W  (vs Loihi 2: ~1W for simpler model)    ║
║                                                                          ║
║  Why 10-100x more efficient than general-purpose neuromorphic:          ║
║    Loihi:  Emulates eligibility traces in microcode (~20 instructions)  ║
║    Engram:  Hardwired elig+neuromod fused unit (1 clock cycle)           ║
║                                                                          ║
║    Loihi:  No native BCM, compartments, or R-STDP                       ║
║    Engram:  All hardwired, zero microcode overhead                        ║
║                                                                          ║
║    Loihi:  Generic neuron core, maps Engram imperfectly                  ║
║    Engram:  Every gate designed for exactly our 6 mechanisms              ║
╚══════════════════════════════════════════════════════════════════════════╝
```

### 6.2 Data Flow Through the ASIC — One Step

```
Clock    Operation                          Unit                  Data Movement
Cycle
─────────────────────────────────────────────────────────────────────────────
  1      Load spike vector from I/O         I/O Controller        DMA → spike reg
         Load neuromod registers            Global Control        4 × float16

  2-10   SpMV: all 24 synapse groups        64 SpMV Engines       SRAM → engines
         (pipelined, ~4 groups per engine)   (parallel rows)       (streaming)
         Each engine processes one row:
           read indptr[i], indptr[i+1]
           stream indices[start:end]
           stream data[start:end]
           dot product with spike vector
           accumulate current for neuron i

 11-14   Route currents to compartments     SpMV → Neuron         Engine output →
         Apply instinctual gain             Clusters              cluster input
         (multiply by gain registers)

 15-20   Dendritic integration              Neuron Clusters       Per-cluster
         Update 4 compartments per neuron   (16 parallel)         SRAM read/write
         Check dendritic spikes (2.5x)
         Couple to soma
         Threshold check → new spike vector

 21-30   STDP computation                   STDP Units            Spike times from
         For each spiking neuron:           (per SpMV engine)     neuron SRAM
           compute Δt for all incoming
           exp(-Δt/τ) via LUT
           BCM scaling from θ registers
           compartment activity lookup

 31-40   Eligibility update + neuromod gate Fused Elig+Neuro      SRAM read/write
         elig += Δw                         Units                 for elig arrays
         elig *= decay
         Δw_final = elig × DA × ACh × NE × 5HT × BCM × compartment
         weight += Δw_final
         clip(weight, w_min, w_max)

 41-45   Homeostatic scaling                Row Normalizers       Sum weights/row
         (every Nth step only)                                    then multiply

 46-48   BCM θ update                       Neuron Clusters       Per-neuron
         Phase metric update                Global Control        register update
         Export metrics                     I/O Controller        DMA → host

─────────────────────────────────────────────────────────────────────────────
Total:   ~48 clock cycles per simulation step

At 500 MHz clock: 48 cycles = 96 nanoseconds per step
                  → 10,000,000 steps per second
                  → 10,000x real-time (1ms brain = 100ns wall)

Compare:
  Current CPU (Hetzner):  1.27 seconds per step  = 0.79x real-time
  Custom ASIC:            0.000000096 seconds     = 10,000x real-time

  Speedup: ~13,000,000x

  THIS is why custom silicon matters.
```

---

*These diagrams document the computational core of Engram's neuromorphic brain.
Every operation shown here is currently implemented in Python/NumPy/SciPy and
running on a production server. A custom ASIC would hardwire these exact
operations in silicon, achieving million-fold speedup over software simulation.*
