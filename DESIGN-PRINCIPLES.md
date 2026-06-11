# Design Principles — Engram

Six architecture invariants define the neuromorphic cognitive core.
All neuromorphic code must conform to them (see `CLAUDE.md` for enforcement rules).

## 1. Integrated Multi-Mechanism Learning
6 mechanisms (STDP, eligibility traces, BCM, neuromodulation, homeostatic scaling,
R-STDP) operating **simultaneously** every simulation step.

## 2. Developmental Critical Periods
5 phases (infant → toddler → juvenile → adolescent → mature) with
experience-dependent transitions — never hardcoded by step count.

## 3. Hybrid SNN-LLM Cognitive Architecture
Emergent LLM queries via STDP reinforcement, not hardcoded IF-THEN logic.

## 4. Cross-Modal Binding with Instinctual Gain
Arbitrary modality binding; instinctual gain always multiplicative (>= 1.0,
never suppresses sensory input).

## 5. Multi-Compartment Dendritic Processing
4 compartments per neuron; every synapse group targets a specific compartment;
supralinear dendritic spikes in apical compartments.

## 6. Neuromodulatory Continual Learning
Eligibility traces ≥1000ms for delayed credit assignment; homeostatic scaling
prevents catastrophic forgetting; weight persistence via SQLite survives restarts.

---

## Implementation Files

| Principle | Primary Files |
|-----------|--------------|
| 1 | `synapses.py`, `neuromodulation.py` |
| 2 | `neuromodulation.py`, `network.py` |
| 3 | `decoding.py`, `cognitive_bridge.py`, `service.py` |
| 4 | `instincts.py`, `encoding.py` |
| 5 | `neurons.py`, `config.py`, `synapses.py` |
| 6 | `neuromodulation.py`, `synapses.py`, `persistence.py` |
