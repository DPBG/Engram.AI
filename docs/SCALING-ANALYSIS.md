# Scaling Analysis — Engram Neuromorphic Core

## Current Architecture (Phase 1)

- **~1.06M neurons**, ~1.31B synapses (with hierarchical layers)
- 11 brain regions, 48 synapse groups
- 4-compartment dendritic model (implemented: apical distal, basal, apical proximal, perisomatic)
- STDP + BCM + eligibility traces + neuromodulation + developmental phases
- Avg ~1,500 synapses/neuron (target: 7,000 with Approach C/D)

---

## Memory Model

### Per Neuron (with 4 dendritic compartments)

| Component | Bytes | Notes |
|-----------|-------|-------|
| v_membrane (soma) | 4 | float32 |
| v_dendrite (4 compartments) | 16 | 4 x float32 |
| refractory_timer | 4 | float32 |
| dend_refractory (4) | 16 | 4 x float32 |
| last_spike_time | 4 | float32 |
| spikes (bool) | 1 | bool |
| threshold, tau, etc. | 20 | per-neuron arrays |
| compartment_active | 4 | 4 x bool |
| **Total per neuron** | **~69 bytes** | |

### Per Synapse (all features enabled)

| Component | Bytes | Notes |
|-----------|-------|-------|
| CSR weight (float32) | 4 | the weight value |
| CSR column index (int32) | 4 | which pre-neuron |
| CSR indptr share | ~0.01 | amortized (n_post+1 entries) |
| Eligibility trace | 4 | float32 (if enabled) |
| Stability counter | 4 | int32 (adolescent phase) |
| Myelinated flag | 1 | bool |
| Identity flag | 1 | bool |
| BCM theta share | ~0.01 | per post-neuron, amortized |
| **Total per synapse** | **~18 bytes** | |

---

## Scaling Table

| Scale | Neurons | Synapses (7K/neuron) | Neuron RAM | Synapse RAM | **Total RAM** | Biological Analog |
|-------|---------|---------------------|------------|-------------|--------------|-------------------|
| Current | ~1.06M | 1.31B | 73 MB | 23.6 GB | **~34 GB actual** | Zebrafish |
| **Demo target** | **1-5M** | **7-35B** | **69-345 MB** | **126-630 GB** | **~130-650 GB** | **Zebrafish-Frog** |
| Fruit fly | 100K | 700M | 7 MB | 12.6 GB | **~13 GB** | Drosophila |
| Zebrafish | 1M | 7B | 69 MB | 126 GB | **~130 GB** | Danio rerio |
| Mouse cortex | 70M | 490B | 4.8 GB | 8.6 TB | **~8.6 TB** | Mus musculus |
| Macaque cortex | 1.6B | 11.2T | 110 GB | 197 TB | **~197 TB** | Primate (partial) |
| **Human brain** | **86B** | **150T** | **5.9 TB** | **2.64 PB** | **~2.6 PB** | Homo sapiens |

---

## Compute Requirements

STDP benchmark: ~42ms per update on a 7.5M-synapse group (single core, M-series Mac).

| Scale | Synapses | STDP time/step (1 core) | With 128 cores | Steps/sec |
|-------|----------|------------------------|----------------|-----------|
| Current (~1.06M) | 1.31B | ~7 sec | ~55ms | 18 |
| Demo (1-5M) | 7-35B | ~40-200 sec | ~300ms-1.5s | 0.7-3 |
| Zebrafish (1M) | 7B | ~40 sec | ~300ms | 3 |
| Mouse cortex (70M) | 490B | ~46 min | ~22 sec | 0.05 |
| Human brain (86B) | 150T | ~9.3 days | ~1.7 hours | 0.0002 |

### Real-time Factor (brain timestep = 1ms, so 1000 steps = 1 sec brain time)

| Scale | Steps/sec (128-core) | Real-time factor |
|-------|---------------------|-----------------|
| Current | 18 | 1 sec brain = 56 sec wall |
| Demo (1M) | 3 | 1 sec brain = 5.5 min wall |
| Mouse | 0.05 | 1 sec brain = 5.5 hours wall |
| **Human** | **0.0002** | **1 sec brain = 58 days wall** |

---

## Hardware Cost by Scale

### Cloud (Monthly)

| Milestone | Neurons | Synapses | Hardware | Cost/mo |
|-----------|---------|----------|----------|---------|
| **Phase 1 (NOW)** | ~1.06M | 1.31B | Hetzner CCX63 (192GB) | ~$170/mo |
| **Phase 2: Demo** | 1-5M | 7-35B | 1 server (1TB RAM) | $300-500/mo |
| **Phase 3: Mouse cortex** | 70M | 500B | Small cluster (8 nodes, 8TB) | $5K-15K/mo |
| **Phase 4: Primate** | 1-2B | 10T | GPU cluster (100 H100s) | $200K-500K/mo |
| **Phase 5: Human** | 86B | 150T | Neuromorphic or massive GPU | $2-5M/mo |

### On-Premises (Capital)

| Scale | Hardware | Upfront Cost | Power/Cooling/mo |
|-------|----------|-------------|-----------------|
| Demo (1-5M) | 1 workstation (1TB RAM) | $15-25K | ~$50 |
| Mouse cortex | 8-node cluster | $200-400K | ~$2K |
| Primate | 100 GPU nodes | $5-10M | ~$50K |
| Human | Custom neuromorphic | $50-100M+ | ~$200K |

### Neuromorphic Hardware (Verified Specs, 2026)

| System | Scale | Power | Cost (est.) | Speed | Source |
|--------|-------|-------|-------------|-------|--------|
| Intel Loihi 2 (1 chip) | 1M neurons, 120M synapses, 128 cores | ~1W | ~$5K (research access) | Real-time inference | Intel, open-neuromorphic.org |
| Intel Hala Point (1,152 Loihi 2) | 1.15B neurons, 128B synapses | 2,600W max | Research-only (Sandia Labs) | 20 petaops, real-time inference | Intel Newsroom, Apr 2024 |
| SpiNNaker 2 (1 chip) | 153 ARM cores + ML accelerators | Low (ARM-based) | Commercial via SpiNNcloud | Real-time | SpiNNcloud Systems |
| SpiNNaker 2 (Sandia "Braunfels") | Multi-chip system | TBD | Research deployment | Real-time | Sandia, Aug 2025 |
| IBM NorthPole | 256 cores, 224MB SRAM, 12nm | ~74W | Research-only | Inference-only (no on-chip learning) | Science, Oct 2023 |
| BrainChip Akida 2.0 | Commercial M.2 modules | Ultra-low (<1W) | Shipping commercially | Real-time inference | BrainChip |
| **Human-scale** | 86B neurons | ~50-200 kW | **$500M-$1B** | **Target: real-time** | Projection |

**Measured energy comparisons (published, not projected):**
- Loihi peg-in-hole robotics: **73x less energy** than CPU (52 uJ vs 3,800 uJ per inference) — IEEE 2024
- Loihi keyword spotting: **109x less energy** than GPU — Intel Labs
- Hala Point: **>15 TOPS/W** for conventional DNN inference — Intel Newsroom
- NSLLM 1.5B on FPGA: **19.8x energy efficiency** over A800 GPU — NSR 2026

**Note:** "1000x efficiency" claims from some sources are marketing projections. Measured advantages range from 10-200x depending on workload sparsity. Energy advantage is greatest for sparse, event-driven workloads and diminishes for dense computation.

---

## Demo Target: 1-5M Neurons

### Hardware Requirements

| Config | Neurons | Synapses | RAM Needed | Hardware Option | Cost |
|--------|---------|----------|------------|----------------|------|
| 1M (zebrafish) | 1M | 7B | ~130 GB | Hetzner CCX63 (192GB) | ~$170/mo |
| 2M | 2M | 14B | ~260 GB | AWS r6i.8xlarge (256GB) x2 | $600/mo |
| 5M (frog) | 5M | 35B | ~650 GB | AWS r6i.metal (1TB) | $1,200/mo |

### Region Scaling (current ~1.06M neurons, proportional)

| Region | Current (~1.06M) | At 5M |
|--------|---------------|-------|-------|
| Sensory Cortex | 200K | 940K |
| Association Cortex | 200K | 940K |
| Motor Cortex | 116K | 545K |
| Predictive Layer | 100K | 470K |
| Cerebellum | 100K | 470K |
| Feature Layer | 200K | 940K |
| Working Memory | 50K | 235K |
| Brainstem | 15K | 70K |
| Reflex Arc | 10K | 47K |
| Concept Layer | 50K | 235K |
| Meta-Controller | 20K | 94K |

---

## Investor Key Points

1. **Architecture scales** — CSR sparse matrices, compartmental neurons, developmental learning rules all work at any scale. No rewrite needed.

2. **Bottleneck is hardware, not algorithms** — STDP, eligibility traces, BCM, pruning, myelination, neuromodulation, dendritic compartments all scale linearly with synapse count.

3. **$300/mo proves the concept** — Rat hippocampus scale (1-5M neurons) demonstrates associative learning no LLM can do. Sufficient for investor demos.

4. **Neuromorphic hardware is the endgame** — Intel Loihi, IBM, SpiNNaker are investing billions in chips that run SNNs natively. Our software maps directly to these chips.

5. **The IP moat is the learning rules** — STDP + BCM + eligibility traces + neuromodulation + developmental phases + dendritic compartments. No one else combines all of these. The hardware is a commodity.

6. **Cost trajectory** — $0 (laptop) → $300/mo (demo) → $15K/mo (mouse) → $500K/mo (primate) → neuromorphic hardware partnership for human-scale.

---

## Comparison with Major Projects

| Project | Scale | Approach | Cost | Real-time? |
|---------|-------|----------|------|-----------|
| Human Brain Project (EU) | 1B neurons | SpiNNaker hardware | ~$1B over 10 years | Near |
| Blue Brain (EPFL) | 31K (detailed) | NEURON simulator | ~$100M+ | No (1000x slower) |
| DeepSouth (Sydney) | 228T synapses target | Custom neuromorphic | ~$40M | Target real-time |
| Intel Hala Point | 1.15B neurons | Loihi 2 ASIC array | Billions R&D | Yes (inference) |
| SpiNNaker 2 (Sandia) | Multi-chip | ARM-based neuromorphic | Research grant funded | Yes |
| **Engram** | **~1.06M → 5M → 86B** | **Software SNN + future HW** | **$220/mo → $300/mo → TBD** | **Not yet** |

Our advantage: We have the **complete learning rule stack** (STDP + BCM + eligibility + neuromodulation + developmental phases + dendritic compartments + pruning + myelination). Most of these projects have the hardware but simpler learning rules. We can port our rules to their hardware.

**Key context:** Intel Hala Point and SpiNNaker 2 are research deployments (both at Sandia National Labs as of 2025). Neither has the biological software stack (developmental phases, neuromodulation, homeostatic regulation) that Engram implements. Neuromorphic VC funding exceeded $200M in 2025 — the market needs software, not just hardware.
