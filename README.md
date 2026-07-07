<div align="center">

<img src="docs/engram-logo.svg" alt="Engram logo" width="112" />

# Engram

### A self-aware, continuously-learning neuromorphic AI

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![Developed by Gittensor · Bittensor SN74](https://img.shields.io/badge/Developed%20by-Gittensor%20·%20Bittensor%20SN74-6e40c9.svg)](https://gittensor.io)

<sub>Contributed & developed by <strong>Gittensor</strong> — <strong>Bittensor Subnet 74</strong>.</sub>

<img src="docs/engram-hero.jpg" alt="Engram — a humanoid body and a ~1M-neuron spiking neural network perceiving, moving, and learning in real time" width="100%" />

<em>A ~1M-neuron spiking neural network and a humanoid body that perceive, move, and learn in real time — developmental, STDP-based learning across 13 brain regions and six biological mechanisms, with no batch training.</em>

<br/>

[Quick Start](#quick-start) · [Architecture](#architecture) · [Dashboard](#dashboard) · [Known Limitations](#known-limitations) · [Contributing](CONTRIBUTING.md)

</div>

---

Engram detects its environment, learns from every interaction, and improves over time — on whatever machine it runs on. Open `localhost:8080` and you see a living system: a neuromorphic brain learning inside a MuJoCo physics simulation, not a static admin panel.

> **Project status:** Engram is an **active research project**, not production-ready
> software. The architecture is implemented and unit-tested, but not yet validated
> at scale or in real time on physical robots. It is released under the MIT License
> to invite contributors — see [Known Limitations](#known-limitations) for an honest
> account of what works and what doesn't yet, and [CONTRIBUTING.md](CONTRIBUTING.md)
> to get involved.

---

## Core Principles

| Principle | Implementation |
|---|---|
| **Environment Awareness** | Detects and adapts to ANY machine — OS, hardware, services, APIs, capabilities |
| **Skills as API Calls** | Abstracted skill registry — `env.detect`, `brain.chat`, `env.docker`, etc. |
| **Data Flywheel** | Every interaction feeds back: teleoperation → observation → deployment → learning |
| **Unified Intelligence** | One system with categorized skills, not disconnected services |
| **Self-Improvement** | Background health checks, anomaly detection, knowledge accumulation |

### The Data Flywheel

Engram learns from 4 sources:

```
    ┌─── Teleoperation ───┐
    │   (human chat)       │
    ▼                      │
┌──────┐              ┌──────┐
│  AI  │◄────────────│ Data │
│  🧠   │             │  📊  │
└──────┘              └──────┘
    │                      ▲
    ▼                      │
    ├─── Observation ──────┤   (NATS messages, system monitoring)
    ├─── Deployment ───────┤   (self-generated health checks)
    └─── Simulation ───────┘   (synthetic/test data)
```

Every chat message, every system observation, every health check feeds the flywheel. The system gets smarter as it runs.

---

## Investor Demos

Five interactive demo pages live at `brain-viz/demos/`. Open the index to navigate:

```bash
open brain-viz/demos/index.html
```

| Demo | File | What It Shows |
|------|------|---------------|
| **Live Brain** | `brain-viz/index.html` | Real-time 3D brain with lightning synapses, connects live to Hetzner |
| **Brain Reaction** | `brain-viz/demos/reaction.html` | Drop an image or trigger a stimulus, watch the neural cascade |
| **Development Timeline** | `brain-viz/demos/timeline.html` | Timelapse from birth to maturity across all 5 developmental phases |
| **Speech Babble** | `brain-viz/demos/speech.html` | Live token stream from 3,486 speech motor neurons |
| **Energy Comparison** | `brain-viz/demos/energy.html` | GPT-4o vs Engram vs human brain energy per inference |

If the dashboard is running, access via `http://localhost:8080/brain-viz/demos/index.html`.

---

## Quick Start

Engram runs with **pure Python — no Docker required**. A launcher (`run.py`)
downloads and manages NATS automatically, then starts the services as native
processes. See **[RUN-LOCAL.md](RUN-LOCAL.md)** for the complete guide.

```bash
# 1. (recommended) create a virtualenv and install deps
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows  (use: source .venv/bin/activate on macOS/Linux)
python run.py --install

# 2. start the core profile (NATS auto-managed + brain, kernel, dashboard, ...)
python run.py

# 3. open the dashboard
#    http://localhost:8080

# Other options:
python run.py --list              # show all services and what they need
python run.py --profile full      # also start Qdrant/Ollama-backed services
python run.py --only kernel,planner
```

**Optional infrastructure** (only for the `full` profile):
- **Ollama** (AI chat / cognitive bridge) — install from https://ollama.com, then
  `ollama pull deepseek-coder:6.7b`.
- **Qdrant** (vector memory) — run the `qdrant` server, then `python run.py --profile full`.

---

## Dashboard

### System Awareness
Deep environment detection — not just "what OS":
- Hardware: CPU model, cores, RAM, disk, GPU
- Services: What's listening on which ports
- APIs: NATS, Ollama, Qdrant availability
- Capabilities: What the system can do (`docker_orchestration`, `local_llm`, `nats_messaging`, etc.)
- Docker: Container CPU, memory, network metrics in real-time

### Skill Registry
Abstracted, categorized skills with execution tracking:

| Category | Skills |
|---|---|
| 👁️ **Perception** | Environment Detection, Resource Monitor, Container Orchestration |
| 🧠 **Cognition** | Conversational Reasoning, Self-Improvement Loop, Knowledge Base |
| 📡 **Communication** | NATS Message Bus, WebSocket Stream |

Each skill tracks: call count, error rate, average latency, last invocation.

### Teleoperation (Chat)
The chat interface is the human guidance channel:
- LLM has full system context (hardware, resources, capabilities, services)
- Maintains conversation history
- Every message feeds the data flywheel
- Connects to Ollama or any OpenAI-compatible endpoint

### Data Flywheel Visualization
SVG ring chart showing knowledge growth by source:
- 🔵 Teleoperation (human chat interactions)
- 🟢 Observation (NATS messages, system events)
- 🟠 Deployment (self-generated health check data)
- 🟣 Simulation (synthetic/test data)

---

## Architecture

### The Cognitive Stack

Engram has two brains: a **neuromorphic cognitive core** (spiking neural network that IS the intelligence) and an **LLM** (external knowledge source, like a teacher/book).

The neuromorphic core is a spiking neural network (currently **1,056,800 neurons** training on Hetzner; hierarchical layers active across 13 regions, 48 synapse groups, 1.31B synapses) with Hebbian/STDP learning. Intelligence emerges from:
1. **Multi-modal binding** — simultaneous sensory inputs bind together through temporal correlation
2. **Predictive world model** — learns temporal sequences, prediction errors drive learning
3. **Homeostatic drives** — energy, damage, temperature, fatigue create needs and motivations

```
Layer 5: External Knowledge (LLM/Ollama)        <- "teacher" — consulted, not core
            |
Layer 4: Higher Cortex (Planner/Coordinator)     <- planning, reasoning
            |  via NATS
Layer 3: Association + Prediction (Neuromorphic)  <- multi-modal binding, world model
            |
Layer 2: Sensory + Motor Cortex (Neuromorphic)   <- perception, movement
            |
Layer 1: Brainstem + Reflexes (Neuromorphic)      <- survival, drives, innate responses
            |  via NATS
         Kernel (safety gate) -> Motor Feedback Adapter -> Body (MuJoCo / real hardware)
```

> This layering diagram is logical (from the brain's perspective). Physically, these layers run as parallel Docker services communicating via NATS. See `docs/ARCHITECTURE.md` for the runtime component structure.

### Brain Regions (1M PoC: 1,001,800 neurons, 11 regions, 24 synapse groups, 1.19B synapses)

| Region | Neurons | Function |
|---|---|---|
| Brainstem | 15K | Homeostatic drives (energy, damage, temp, fatigue) |
| Reflex Arc | 10K | Fast hardwired responses (pain withdrawal, grip, startle) |
| Sensory Cortex | 200K | Encodes raw input (visual, auditory, tactile, proprioceptive) |
| Motor Cortex | 100K | Movement generation (locomotion, manipulation, head, expression) |
| Cerebellum | 100K | Motor coordination via efference copy |
| Association Cortex | 200K | Multi-modal binding — concept formation via STDP |
| Predictive Layer | 100K | Temporal sequence learning + world model (R-STDP) |
| Working Memory | 25K | Sustained attention via strong recurrent connections |

**Hierarchical additions** (Phase 8 — implemented, currently active via docker-compose.yml):

| Region | Neurons (active) | Target (full scale) | Function |
|---|---|---|---|
| Feature Layer | 20K | 80K | Intermediate feature integration (V4/IT-like), pools simple features into complex combinations |
| Concept Layer | 5K | 10K | Information bottleneck with k-WTA (2% sparsity). Abstract, modality-invariant concept SDRs |
| Meta-Controller | 3K | 6K | Neuromodulatory hub — outputs DA, ACh, NE, 5-HT signals to gate plasticity across all regions |
| Pattern Separator | 10K | 10K | Sparse expansion for episodic binding (dentate gyrus analog) |
| Global Workspace | 5K | 10K | Cross-region broadcast hub for conscious access (global workspace theory) |

### How Learning Works

The brain uses a **three-factor learning rule**:

```
Factor 1 (pre-synaptic):  When did the input neuron fire?
Factor 2 (post-synaptic): When did the output neuron fire?
Factor 3 (neuromodulator): Should this connection change?

Step 1: Spike coincidence → eligibility trace (temporary molecular tag)
Step 2: Neuromodulator (DA/ACh/NE/5-HT) → converts trace to permanent weight change
No modulator within ~1 second → trace decays, no learning
```

This enables **delayed reward** (the brain can learn from consequences that arrive seconds later) and **selective learning** (the meta-controller decides what to learn and what to consolidate).

### How Abstract Concepts Form

Concepts are NOT pre-built or labeled. They emerge through the processing hierarchy:

1. **Sensory features** → edges, frequencies, pressure (sensory cortex)
2. **Complex features** → shapes, phonemes, textures (feature layer, via STDP)
3. **Multi-modal binding** → "red + round + smooth = apple" (association cortex, via lateral STDP + gamma oscillations)
4. **Prototype extraction** → shared features across many "dog" experiences strengthen; unique features average out (association cortex, over time)
5. **Abstract SDR** → 200/10,000 sparse code, compressed, modality-invariant (concept layer, via k-WTA competition)
6. **Temporal abstraction** → "A causes B" patterns across concept-level representations (predictive layer, via R-STDP)
7. **Transfer** → same concept neurons activate for novel instances that share learned features (cross-domain generalization)

### Motor Feedback Loop (MuJoCo Virtual Body)

The brain closes the sensorimotor loop: motor cortex fires → motor command goes through the Kernel → physical simulation → proprioceptive feedback → STDP learning.

```
Motor cortex fires "locomotion 0.5"
   │
   ▼
Kernel (ALLOW/DENY/TRANSFORM)
   │
   ▼
Motor Feedback Adapter ──────────────────────┐
   │                                          │
   ├─ [real hardware]  ── actuator.heartbeat ─┤  (per-channel auto-detection)
   │                                          │
   └─ [MuJoCo body]   ── physics simulation ──┘
          │
          ▼
   motor.outcome.{channel}  →  proprioceptive injection  →  STDP learns
```

**MuJoCo virtual body**: Configurable robot body simulated with MuJoCo physics (default: 22-body humanoid). Motor channels map to joint actuators via `MotorFeedbackConfig.channel_actuators`; speech and expression use stochastic feedback. The body persists physics state and derives all joint mappings, fall thresholds, and proprioceptive vector size dynamically from the loaded model.

**Per-channel auto-detection**: If a real actuator publishes heartbeats (`actuator.heartbeat.{channel}`), that channel routes to hardware. Otherwise it routes to MuJoCo. Plug in a real arm → manipulation goes real, locomotion stays virtual. Unplug → falls back to MuJoCo after timeout.

**R-STDP on motor pathways**: Prediction error gates motor learning — successful motor outcomes reinforce the firing patterns that produced them. Motor echo tracker provides DA boost for 100 steps after motor fire, keeping eligibility traces alive during the feedback delay.

### Governance Pipeline

Four services form an autonomous self-improvement and safety pipeline. All proposals (action or code) must pass through the Kernel before execution.

```
Unknown device detected
   │
   ▼
Gateway ──publish──> device.unknown
                        │
                        ▼
               Coordinator ──publish──> knowledge.gap
                                           │
                                           ▼
                                    Meta-Programmer (Ollama)
                                           │
                                    code.proposal
                                           │
                                           ▼
                                    Kernel ◄──request──► Safety Supervisor
                                      │
                          ┌───────────┼───────────┐
                          │           │           │
                        ALLOW     TRANSFORM     DENY
                          │           │
                          ▼           ▼
                     Sandbox test   Apply transforms,
                     & deploy       then sandbox test
```

| Service | Role | Key NATS Subjects |
|---|---|---|
| **Kernel** | Immutable safety gate. Evaluates all proposals against envelope limits, risk thresholds, and belief norms. | `proposal.new`, `code.proposal` → `decision.{trace_id}` |
| **Safety Supervisor** | Risk analysis via request-reply. Checks dangerous patterns, self-referential code, protected paths. | `safety.analyze.action`, `safety.analyze.code` |
| **Coordinator** | Routes unknown devices to meta-programmer, manages sensor discovery, deduplicates knowledge gaps. | `device.unknown`, `task.request` |
| **Meta-Programmer** | Generates plugin code via Ollama, runs in sandboxed containers, submits to Kernel for approval. | `knowledge.gap` → `code.proposal` |

### System Architecture

```
+---------------------------------------------------------------------+
|                       Engram Dashboard                                |
|                     http://localhost:8080                             |
|                                                                      |
|  +----------+  +------------------+  +-------------------+          |
|  |  System   |  |  Teleoperation   |  |  Skill Registry   |          |
|  |  Aware-   |  |  (Chat + LLM)    |  |  (categorized)    |          |
|  |  ness     |  |                  |  |                   |          |
|  |  Gauges   |  |  <- WebSocket -> |  |  Containers       |          |
|  |  Flywheel |  |                  |  |  Message Bus       |          |
|  |  Insights |  |  Knowledge <---- |--|-->Flywheel Data   |          |
|  +----------+  +------------------+  +-------------------+          |
|                         |                                            |
|              +----------+----------+                                 |
|              |   FastAPI Backend   |                                 |
|              |   SkillRegistry     | <- Tracks all skill calls       |
|              |   KnowledgeBase     | <- Data flywheel storage        |
|              |   SystemDetection   | <- Deep env awareness           |
|              |   SelfMonitor       | <- Health loop (60s)            |
|              +----------+----------+                                 |
|                         |                                            |
|  +----------------------+------------------------------+            |
|  |                   NATS (Message Bus)                 |            |
|  +-----------------------------------------------------+            |
|     |        |         |         |        |         |               |
|  +------+ +------+ +--------+ +------+ +------+ +-----------+     |
|  |Coord-| |Safety| |Kernel  | |Meta- | |Ollama| |Neuromorphic|     |
|  |inator| |Super-| |(safety | |Prog- | |      | |1.06M SNN  |     |
|  |      | |visor | | gate)  | |rammer| |      | |+MuJoCo    |     |
|  +------+ +------+ +--------+ +------+ +------+ +-----------+     |
|                                  |                    |             |
|                               +------+         +----------+       |
|                               |Qdrant|         | Gateway  |       |
|                               +------+         | (sensors)|       |
|                                                 +----------+       |
+---------------------------------------------------------------------+
```

---

## API Reference

### Core
| Endpoint | Method | Description |
|---|---|---|
| `/api/health` | GET | Health + uptime + skill calls + knowledge count |
| `/api/system` | GET | Deep system detection + live metrics |

### Skills
| Endpoint | Method | Description |
|---|---|---|
| `/api/skills` | GET | Full skill registry (by category, call counts, latency) |
| `/api/skills/log` | GET | Recent skill execution log |

### Knowledge (Flywheel)
| Endpoint | Method | Description |
|---|---|---|
| `/api/flywheel` | GET | Data flywheel stats (sources, growth rate) |
| `/api/knowledge` | GET | Knowledge entries (filterable by source) |

### Teleoperation
| Endpoint | Method | Description |
|---|---|---|
| `/api/chat` | POST | Send message → LLM response (feeds flywheel) |
| `/api/chat/history` | GET | Conversation history |

### Monitoring
| Endpoint | Method | Description |
|---|---|---|
| `/api/metrics` | GET | Docker container metrics |
| `/api/messages` | GET | Recent NATS messages |
| `/api/insights` | GET | Self-monitor insights |
| `/ws` | WS | Real-time stream (metrics, chat, flywheel, skills) |

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `NATS_URL` | `nats://nats:4222` | NATS server |
| `OLLAMA_URL` | `http://ollama:11434` | Ollama LLM |
| `LLM_MODEL` | `llama3.2` | Chat model |
| `OPENAI_API_URL` | — | Optional OpenAI-compatible endpoint |
| `OPENAI_API_KEY` | — | API key |
| `DASHBOARD_PORT` | `8080` | Dashboard port |
| `NEURO_MOTOR_FEEDBACK` | `0` | Enable motor feedback loop (MuJoCo virtual body) |
| `NEURO_COGNITIVE_ENABLED` | `0` | Enable cognitive action channel (brain→LLM queries) |
| `NEURO_SPEECH_END` | `0.80` | Speech motor sub-range end (>0.80 enables speech output) |

---

## Development

```bash
# Run just the dashboard (auto-reload code by restarting run.py)
python run.py --only dashboard

# Run everything; logs stream to the console, prefixed per-service
python run.py

# Test APIs
curl http://localhost:8080/api/skills | python -m json.tool
curl http://localhost:8080/api/flywheel | python -m json.tool
```

---

## Tech Stack

| Component | Technology |
|---|---|
| **Dashboard** | FastAPI + vanilla JS + WebSocket |
| **Message Bus** | NATS |
| **LLM** | Ollama (local) or OpenAI-compatible API |
| **Physics** | MuJoCo (CPU, headless) — configurable robot body for motor feedback |
| **Vector DB** | Qdrant |
| **Database** | SQLite |
| **Orchestration** | Pure-Python launcher (`run.py`) — no containers |

---

## Scientific Foundations & References

The neuromorphic cognitive core draws from established computational neuroscience research. Key concepts and their origins:

### Learning Rules
| Concept | Reference | How We Use It |
|---|---|---|
| **STDP** | Bi & Poo (1998). Synaptic modifications in cultured hippocampal neurons. *J. Neuroscience* | Core two-factor learning rule — spike timing drives synaptic weight changes |
| **Three-factor learning** | Fremaux & Gerstner (2016). Neuromodulated STDP and theory of three-factor learning rules. *Frontiers in Neural Circuits* | STDP → eligibility trace → neuromodulator gates weight update |
| **DA-modulated STDP** | Izhikevich (2007). Solving the distal reward problem through linkage of STDP and dopamine signaling. *Cerebral Cortex* | Dopamine multiplies eligibility traces for reward-driven learning |
| **e-prop** | Bellec et al. (2020). A solution to the learning dilemma for recurrent networks of spiking neurons. *Nature Communications* | Eligibility trace formulation and decay dynamics |
| **BCM metaplasticity** | Bienenstock, Cooper & Munro (1982). Theory for the development of neuron selectivity. *J. Neuroscience* | Per-neuron sliding threshold prevents weight saturation |

### Architecture & Representations
| Concept | Reference | How We Use It |
|---|---|---|
| **Sparse Distributed Representations** | Ahmad & Hawkins (2016). How do neurons operate on sparse distributed representations? *Numenta Technical Report* | Concept layer uses SDRs (200/10K active) for high-capacity pattern storage |
| **Information Bottleneck** | Tishby, Pereira & Bialek (1999). The Information Bottleneck method. *37th Allerton Conference* | 50:1 compression in concept layer forces abstraction |
| **Concept cells** | Quiroga et al. (2005). Invariant visual representation by single neurons in the human brain. *Nature* | Inspiration for sparse, modality-invariant concept neurons |
| **Hub-and-spoke model** | Patterson, Nestor & Rogers (2007). Where do you know what you know? *Nature Reviews Neuroscience* | Concept layer as amodal hub receiving modality-specific inputs |

### Neuromodulation & Development
| Concept | Reference | How We Use It |
|---|---|---|
| **Sequential neuromodulation** | Brzosko, Zannone, Bhatt et al. (2019). Sequential neuromodulation of Hebbian plasticity offers mechanism for effective reward-based navigation. *eLife* | ACh + DA sequential gating of plasticity |
| **Critical periods** | Hensch (2005). Critical period plasticity in local cortical circuits. *Nature Reviews Neuroscience* | Developmental schedule: infant (wide-open) → mature (reward-gated) |

### Concept Formation & Transfer
| Concept | Reference | How We Use It |
|---|---|---|
| **Cortical recycling** | Dehaene & Cohen (2007). Cultural recycling of cortical maps. *Neuron* | Transfer learning — same neural circuits repurposed for new domains |
| **Numerosity** | Nieder & Dehaene (2009). Representation of number in the brain. *Annual Review of Neuroscience* | Inspiration for emergent number-sense via convergent feature detection |
| **Temporal binding** | Singer & Gray (1995). Visual feature integration and the temporal correlation hypothesis. *Annual Review of Neuroscience* | E/I balance produces gamma oscillations that bind multi-modal features |

### Inhibitory Circuits
| Concept | Reference | How We Use It |
|---|---|---|
| **E/I balance** | Isaacson & Scanziani (2011). How inhibition shapes cortical activity. *Neuron* | 80/20 excitatory/inhibitory split, lateral inhibition for competition |
| **k-WTA competition** | Maass (2000). On the computational power of winner-take-all. *Neural Computation* | Concept layer enforces 2% sparsity via k-Winners-Take-All |

### Reminiscence Bump & Adolescent Plasticity
The brain's developmental model includes an "adolescent" supercharged learning phase inspired by the reminiscence bump — the neurobiological phenomenon where experiences during ages 10-25 are wired into core identity with extraordinary durability.

| Concept | Reference | How We Use It |
|---|---|---|
| **Reminiscence bump** | Rubin, Wetzler & Nebes (1986). *Autobiographical Memory* | Developmental phase where foundational knowledge is wired in permanently |
| **Music-evoked autobiographical memory** | Janata (2009). Neural architecture of music-evoked autobiographical memories. *Cerebral Cortex* | MPFC hub integrates sensory experience, memory, and identity — model for cross-modal identity binding |
| **DA widens STDP windows** | Brzosko, Mierau & Bhatt (2019). Neuromodulation of STDP. *Neuron* | During adolescent phase, elevated DA widens tau_plus/tau_minus for broader coincidence detection |
| **Synaptic tagging & capture** | Frey & Morris (1997). *Nature*; Moncada & Viola (2007). *J. Neuroscience* | Strong emotional signals rescue nearby eligibility traces — temporal neighborhood consolidation |
| **Adolescence as critical period** | Larsen & Luna (2018). Adolescence as a neurobiological critical period. *Neuropsychopharmacology* | Association cortex has its critical period during adolescence, not infancy |
| **Hierarchical critical periods** | Larsen et al. (2025). *Neuropsychopharmacology* | Critical periods unfold sensory-first, association-last — matches our layer hierarchy |
| **Synaptic pruning** | Petanjek et al. (2011). Extraordinary neoclassical pruning in human PFC. *PNAS* | "Sculpt or lose" — eliminate weak synapses, strengthen survivors. Creates foundational identity wiring |
| **Musical reminiscence bump (global)** | Jakubowski et al. (2025). Memory bumps across the lifespan. *Memory* | 2,000 participants, 84 countries — peak at age 17. Music as identity-binding stimulus |

---

## Developer Setup

```bash
# 1. Clone
git clone https://github.com/engramai/engram && cd engram

# 2. Environment
cp .env.example .env
# Edit .env with your values — see .env.example for all options (NATS_TOKEN is only needed for production deployment)

# 3. Install dependencies (pure Python, no Docker)
python run.py --install

# 4. Start services locally (NATS is downloaded & managed automatically)
python run.py

# 5. Open dashboard
#    http://localhost:8080
```

See **[RUN-LOCAL.md](RUN-LOCAL.md)** for profiles, optional Qdrant/Ollama, and troubleshooting.

### Project Structure

| Directory | What It Is | Key Tech |
|-----------|-----------|----------|
| `neuromorphic/` | Spiking neural network core (the brain) | NumPy, SciPy, uv |
| `sdk/` | Python SDK (BaseService, SensorPlugin, ActuatorPlugin) | Python, NATS |
| `dashboard/` | Web monitoring UI (standalone, no SDK dep) | FastAPI, vanilla JS |
| `sensory-gateway/` | Host-side sensor discovery + streaming | Python, OpenCV, pyaudio |
| `kernel/` | Safety gate (immutable, all proposals pass through) | Python, NATS |
| `meta-programmer/` | Self-evolution agent (LLM code generation) | Python, Ollama |
| `safety-supervisor/` | Risk analysis for proposals | Python, NATS |
| `coordinator/` | Task orchestration + planning | Python, NATS |
| `deploy/` | Production deployment scripts (cloud provisioning, sync) | Bash |
| `brain-viz/` | 3D brain visualization demos | Three.js |

### Running Tests

```bash
# Neuromorphic tests (primary)
cd neuromorphic && uv run python -m pytest tests/ -v -p no:anchorpy

# Restart the dashboard after changes
python run.py --only dashboard
```

---

## Further Reading

| Document | Content |
|----------|---------|
| [`ROADMAP.md`](ROADMAP.md) | Technical plan & current-state audit — phases M1–M7, tracking issues, status |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Full system design, component relationships, NATS message schemas |
| [`DESIGN-PRINCIPLES.md`](DESIGN-PRINCIPLES.md) | 6 core architecture invariants and implementation file map |
| [`docs/SENSORY-GATEWAY.md`](docs/SENSORY-GATEWAY.md) | Gateway architecture, sensor types, discovery |
| [`docs/META-PROGRAMMER.md`](docs/META-PROGRAMMER.md) | Self-evolution agent system |
| [`docs/KERNEL-CRASH-RECOVERY.md`](docs/KERNEL-CRASH-RECOVERY.md) | Threat model for the Kernel process dying mid-decision, and how callers still fail closed |
| [`docs/DECISION-KEY-ROTATION.md`](docs/DECISION-KEY-ROTATION.md) | Zero-downtime rotation procedure for the decision-bus signing key |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | How to contribute, PR process, code standards |

> **Naming:** Engram is the product/brand name and the domain is `engram.ai`. "Engram" was a former brand name and "ActiveLearningAI" is the original project directory name — both are still retained in a few places (the GitHub org/repos, the `activelearning` Python packages, and the legal entity Engram Incorporated). Public-facing materials use Engram.

---

## Known Limitations

Engram is a research system, and we believe in being honest about its current state:

- **Performance / scale.** The core simulation is Python + NumPy/SciPy. It runs
  and learns, but it is **not real-time** at the ~1M-neuron scale on commodity
  hardware. Real-time embodied use will require GPU/compiled kernels or
  neuromorphic hardware (e.g. Loihi, SpiNNaker).
- **Validation.** Emergent concept formation and continual learning are
  implemented and unit-tested, but large-scale, quantitative evidence that the
  system learns *useful* abstractions is still a work in progress.
- **Meta-Programmer.** Autonomous code generation uses a local LLM and runs in a
  sandbox, but generated drivers are experimental, not production-grade.
- **Hardware integration.** Built-in sensors cover camera, microphone, and serial
  devices; IMU / depth / real-actuator support is partial or planned.
- **Contribution opportunities.** The `good first issue` backlog (issues labelled
  [`good first issue`](https://github.com/DPBG/Engram.AI/labels/good%20first%20issue))
  lists small, well-scoped tasks across `neuromorphic/`, `sdk/`, and
  `sensory-gateway/`. Contributions are especially welcome.

If something doesn't match the docs, please open an issue — accurate docs are a priority.

## Contributing

Contributions are welcome! A few essentials:

- **Open pull requests against the [`dev`](https://github.com/DPBG/Engram.AI/tree/dev) branch**, not `main`. `dev` is the
  integration branch; `main` is the stable/release branch that `dev` is merged
  into for releases.
- Branch from `dev` (`git checkout dev && git checkout -b feature/your-feature`),
  make your change, and open the PR with **base: `dev`**.
- Run the tests before pushing — see [CONTRIBUTING.md](CONTRIBUTING.md) for the
  full workflow, code standards, and the architecture invariants in
  [CLAUDE.md](CLAUDE.md).

## Gittensor Driven Development

Engram is developed by **[Gittensor](https://gittensor.io)** — **Bittensor Subnet 74 (SN74)**, a decentralized network that rewards open-source contributions. Its contributors deliver Engram's features and fixes through the [`dev`-branch PR workflow](CONTRIBUTING.md).

**Without Gittensor, this project couldn't have come so far** — its contributor network drives Engram's development forward.

## License

Engram is released under the [MIT License](LICENSE).

Copyright (c) 2026 Engram Incorporated.
