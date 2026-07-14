# System Architecture

> **Primary Language**: Python 3.11+
> **Message Bus**: NATS (self-hosted)
> **LLM**: Local via Ollama (deepseek-coder, codellama, etc.)
> **Containerization**: Docker Compose (dev), Kubernetes-ready (prod)

This document defines the unified architecture for Engram, resolving inconsistencies across other docs and establishing the canonical reference.

---

## Design Principles

1. **Self-hosted first** - Everything runs on your hardware, no external APIs required (except optional Claude/OpenAI for complex reasoning)
2. **Container isolation** - Each component in its own container for security and modularity
3. **Python-first** - Consistent language across all components (Kernel, Meta-Programmer, SDK)
4. **Message-driven** - All components communicate via NATS pub/sub, no direct dependencies
5. **Single machine, multi-machine ready** - Works on one M2, scales to cluster later

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         YOUR MACHINE (M2 Mac)                               │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                          DOCKER NETWORK                               │ │
│  │                                                                       │ │
│  │  ┌─────────────────────────────────────────────────────────────────┐ │ │
│  │  │                         NATS (Message Bus)                      │ │ │
│  │  │                    nats://localhost:4222                        │ │ │
│  │  └───────────────────────────┬─────────────────────────────────────┘ │ │
│  │                              │                                       │ │
│  │         ┌────────────────────┼────────────────────┐                 │ │
│  │         │                    │                    │                 │ │
│  │         ▼                    ▼                    ▼                 │ │
│  │  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐           │ │
│  │  │  OPEN ZONE  │     │ CLOSED ZONE │     │   INFRA     │           │ │
│  │  │             │     │(internal net)│     │             │           │ │
│  │  │ ┌─────────┐ │     │ ┌─────────┐ │     │ ┌─────────┐ │           │ │
│  │  │ │   SDK   │ │     │ │ Kernel  │ │     │ │ Ollama  │ │           │ │
│  │  │ │(library)│ │     │ │ Cluster │ │     │ │ (LLM)   │ │           │ │
│  │  │ └─────────┘ │     │ └─────────┘ │     │ └─────────┘ │           │ │
│  │  │ ┌─────────┐ │     │ ┌─────────┐ │     │ ┌─────────┐ │           │ │
│  │  │ │ Planner │ │     │ │ Meta-   │ │     │ │ Qdrant  │ │           │ │
│  │  │ │         │ │     │ │Programmer│ │     │ │(Vector) │ │           │ │
│  │  │ └─────────┘ │     │ └─────────┘ │     │ └─────────┘ │           │ │
│  │  │ ┌─────────┐ │     │ ┌─────────┐ │     │ ┌─────────┐ │           │ │
│  │  │ │ Memory  │ │     │ │ Safety  │ │     │ │ SQLite  │ │           │ │
│  │  │ │ Service │ │     │ │Supervisor│ │     │ │ (shared)│ │           │ │
│  │  │ └─────────┘ │     │ └─────────┘ │     │ └─────────┘ │           │ │
│  │  │ ┌─────────┐ │     │             │     │ ┌─────────┐ │           │ │
│  │  │ │ Beliefs │ │     │             │     │ │Dashboard│ │           │ │
│  │  │ │ Graph   │ │     │             │     │ │(Monitor)│ │           │ │
│  │  │ └─────────┘ │     │             │     │ └─────────┘ │           │ │
│  │  │             │     │             │     │             │           │ │
│  │  └─────────────┘     └─────────────┘     └─────────────┘           │ │
│  │                                                                     │ │
│  │  ┌─────────────────────────────────────────────────────────────┐   │ │
│  │  │              SANDBOX ZONE (Ephemeral Containers)            │   │ │
│  │  │                                                             │   │ │
│  │  │   ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐       │   │ │
│  │  │   │Sandbox-1│  │Sandbox-2│  │Sandbox-3│  │   ...   │       │   │ │
│  │  │   │(testing │  │(testing │  │(testing │  │         │       │   │ │
│  │  │   │ code A) │  │ code B) │  │ code C) │  │         │       │   │ │
│  │  │   └─────────┘  └─────────┘  └─────────┘  └─────────┘       │   │ │
│  │  │                                                             │   │ │
│  │  │   - No network access                                       │   │ │
│  │  │   - Read-only filesystem (except /tmp)                      │   │ │
│  │  │   - Resource limits (CPU, memory, time)                     │   │ │
│  │  │   - Destroyed after use                                     │   │ │
│  │  └─────────────────────────────────────────────────────────────┘   │ │
│  │                                                                     │ │
│  │  ┌─────────────────────────────────────────────────────────────┐   │ │
│  │  │                    SHARED VOLUMES                           │   │ │
│  │  │  /data/sqlite/    - Unified database                        │   │ │
│  │  │  /data/vectors/   - Qdrant storage                          │   │ │
│  │  │  /data/staging/   - Code staging area                       │   │ │
│  │  │  /data/models/    - Ollama model cache                      │   │ │
│  │  │  /data/tasks/     - Learned task code                       │   │ │
│  │  │  /data/plugins/   - Deployed plugins                        │   │ │
│  │  │  /data/overrides/ - Human override code                     │   │ │
│  │  └─────────────────────────────────────────────────────────────┘   │ │
│  └───────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Component Details

### Component Relationships: Coordinator vs Planner

The system has two key orchestration components with distinct responsibilities:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    COMPONENT HIERARCHY                               │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                     COORDINATOR                              │    │
│  │  (High-level orchestration)                                 │    │
│  │                                                              │    │
│  │  Responsibilities:                                          │    │
│  │  - Task lookup from vector DB                               │    │
│  │  - Decides: execute cached task OR generate new             │    │
│  │  - Sensor fusion (weighted by priority)                     │    │
│  │  - Learning mode management                                 │    │
│  │  - Autopilot mode control                                   │    │
│  │  - Routes to Meta-Programmer for knowledge gaps             │    │
│  │                                                              │    │
│  │  Contains: SensorManager, TaskIndex, LLMCache               │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              │                                       │
│                              │ Delegates low-level planning          │
│                              ▼                                       │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                       PLANNER                                │    │
│  │  (Low-level action planning)                                │    │
│  │                                                              │    │
│  │  Responsibilities:                                          │    │
│  │  - Converts observations to ActionProposals                 │    │
│  │  - Executes task code (from Coordinator)                    │    │
│  │  - Manages action sequences                                 │    │
│  │  - Handles Kernel decisions (ALLOW/TRANSFORM/DENY/DEFER)    │    │
│  │  - Scheduler modes (EXECUTION/LEARNING/EXPLORATION/SAFE_HALT)│    │
│  │                                                              │    │
│  │  Publishes to: proposal.new                                 │    │
│  │  Subscribes to: decision.{trace_id}, observation.*          │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              │                                       │
│                              │ All proposals pass through            │
│                              ▼                                       │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                     MORAL KERNEL                             │    │
│  │  (Immutable safety gatekeeper)                              │    │
│  │                                                              │    │
│  │  Returns: ALLOW / TRANSFORM / DENY / DEFER                  │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**Key distinction:**
- **Coordinator** decides WHAT to do (which task, cached or new)
- **Planner** decides HOW to do it (action sequences, timing)
- **Kernel** decides IF it's allowed (safety validation)

### Learning Controller

The Learning Controller is a **component within the Coordinator** (not a separate container) that manages demonstration learning:

```python
# coordinator/learning_controller.py
class LearningController:
    """
    Manages demonstration learning phases.
    Lives inside the Coordinator container.
    """

    def __init__(self, sensor_manager: SensorManager, meta_programmer: MetaProgrammer):
        self.sensor_manager = sensor_manager
        self.meta_programmer = meta_programmer
        self.bus = EventBus()
        self.current_demo: Optional[DemoSession] = None

    async def start(self):
        """Subscribe to learning-related NATS subjects."""
        await self.bus.subscribe("learning.demo.start", self.on_demo_start)
        await self.bus.subscribe("learning.demo.frame", self.on_demo_frame)
        await self.bus.subscribe("learning.feedback", self.on_feedback)

    async def on_demo_start(self, msg):
        """Human initiated a demonstration."""
        event = DemoStartEvent.from_dict(msg.data)
        self.current_demo = DemoSession(
            task_name=event.task_name,
            started_at=time.time(),
            frames=[],
            feedback=[]
        )
        logger.info(f"Demonstration started: {event.task_name}")

    async def on_demo_frame(self, msg):
        """Camera captured a demonstration frame."""
        if self.current_demo:
            frame = DemoFrame.from_dict(msg.data)
            self.current_demo.frames.append(frame)

    async def on_feedback(self, msg):
        """Human provided feedback during learning."""
        feedback = HumanFeedback.from_dict(msg.data)
        if self.current_demo:
            self.current_demo.feedback.append(feedback)

    async def complete_demo(self) -> LearnedTask:
        """Finalize demonstration and generate task code."""
        if not self.current_demo:
            raise ValueError("No active demonstration")

        # Ask Meta-Programmer to generate task code from demo
        task = await self.meta_programmer.generate_from_demonstration(
            self.current_demo
        )

        # Index in vector DB
        await self.index_task(task)

        self.current_demo = None
        return task
```

### Neuromorphic Brain

The neuromorphic brain (`neuromorphic/`) is the core spiking neural network that processes all sensory input and produces motor output. It implements a biologically realistic SNN with multi-compartment dendritic processing, developmental critical periods, and six simultaneous learning mechanisms (see `DESIGN-PRINCIPLES.md`).

**Current scale (1M deployment):**

| Metric | Count |
|--------|-------|
| Total neurons | ~1,056,800 |
| Brain regions | Up to 13 (8 base + feature, concept, pattern separator, meta-controller, global workspace) |
| Synapse groups | Up to 48 (20 base; +3 feature, +6 concept, +3 pattern separator, +2 meta-controller, +13 global workspace). 48 active at 1M scale. |
| Total synapses | ~1.31 billion |

**Base regions**: brainstem, reflex arc, sensory cortex, motor cortex, cerebellum, association cortex, predictive layer, working memory.

**Hierarchical regions** (enabled via env vars): feature layer, concept layer (k-WTA), pattern separator (dentate gyrus analog), meta-controller, global workspace (GWT).

Motor cortex has 6 sub-ranges: locomotion (0-30%), manipulation (30-60%), head (60-80%), speech (80-83%), expression (83-85%), cognitive (85-100%).

#### Hierarchical Motor Control Architecture

Motor control follows the biological hierarchy established in vertebrate neuroscience (Grillner & El Manira 2020, Physiol Rev 100:271-320):

1. **Passive joint stiffness (muscle tone)** - PD controllers per joint hold the body in a neutral standing pose. Analogous to spinal stretch reflexes that provide baseline postural tone in neonates (Dominici et al. 2011, Science 334:997-999). Motor cortex output adds to this baseline rather than replacing it.

2. **Central Pattern Generators (CPGs)** - Half-centre oscillator circuits in brainstem produce rhythmic alternating flexion/extension patterns for locomotion. Based on the Graham Brown (1911/1914) half-centre model, validated in lamprey (Grillner 1985, Science 228:143-149) and demonstrated on neuromorphic hardware (Gutierrez-Galan et al. 2020, Neurocomputing 381:10-19; Polykretis & Michmizos 2020, arXiv:2006.04765). Decerebrate cats walk on treadmills without cortex (Whelan 1996, Prog Neurobiol 49:481-515).

3. **Motor cortex modulation** - Cortex does not generate locomotion patterns from scratch. It modulates CPG timing, amplitude, and gait selection via STDP-learned connections. Cortical contribution during locomotion is primarily for modifying the ongoing pattern (obstacle avoidance, precision stepping), not generating the basic rhythm (Drew et al. 2002, Brain Res Rev 40:178-191).

4. **Topographic motor initialization** - Initial sensory-motor synaptic weights are structured so proprioceptive inputs preferentially connect to corresponding motor neuron groups. Reflects the genetically specified cortical "protomap" (Rakic 1988, Science 241:170-176) and somatotopic maps observed in preterm infants (Dall'Orso et al. 2018, Cereb Cortex 28:2507-2515). STDP refines this innate map through experience.

The brain runs as a Docker service, subscribes to `observation.*` NATS subjects for sensory input, and publishes `proposal.new` (motor commands), `speech.output`, `cognitive.query`, and `neuromorphic.metrics`.

### MuJoCo Virtual Body

The MuJoCo virtual body (`neuromorphic/src/neuromorphic/mujoco_body.py`) provides a physics-simulated embodiment for the brain. It closes the sensorimotor loop: brain motor output drives MuJoCo actuators, and proprioceptive/visual feedback flows back as sensory input.

- **Body-agnostic**: Supports any MJCF model via `model_xml` + `channel_actuators` config. Default is a 29-DOF humanoid.
- **Continuous physics loop**: MuJoCo steps at 50 Hz, proprioceptive observations published at 5 Hz, camera frames at 2 Hz, body state at 10 Hz.
- **CPU rendering**: Uses OSMesa for headless 64x64 grayscale camera rendering (no GPU required).
- **Motor feedback adapter**: Routes motor commands to either MuJoCo (simulation) or real hardware, with R-STDP reward signals for motor learning.
- Enabled via `NEURO_MUJOCO_CONTINUOUS=1` env var. Backward compatible (disabled by default).

### Sensory Gateway

The sensory gateway (`sensory-gateway/`) runs on the host machine (not Docker) to access hardware sensors directly (cameras, microphones, serial devices). It extends the SDK's `SensorPlugin` for a consistent interface.

- **Auto-discovery**: Detects cameras, microphones, serial ports, and network sensors on startup.
- **Per-sensor plugins**: `VideoFileSensor` (64x64 gray, CNN 576 features), `AudioFileSensor` (13 MFCC at 10 Hz), plus live camera/mic sensors.
- **Aggregation**: `AggregatingEventBus` reduces ~1,400 observations/sec to ~4/sec (99.7% reduction) before publishing to NATS.
- **Auditory STM**: 107-float feature vectors with 4-second temporal history for cross-modal binding.
- Publishes to `observation.*` NATS subjects. Status available on `sensory.gateway.status`.

### Message Bus: NATS

All components communicate through NATS subjects:

| Subject | Publisher | Subscriber | Payload |
|---------|-----------|------------|---------|
| `observation.{sensor_id}` | Sensors | Planner, Memory, Neuromorphic | `Observation` |
| `proposal.new` | Planner, Neuromorphic | Kernel | `ActionProposal` |
| `decision.{trace_id}` | Kernel | Planner, Actuators | `KernelDecision` |
| `outcome.{trace_id}` | Planner | Actuators, Memory | `Outcome` |
| `motor.outcome.{channel}` | Actuators, Sensors, Simulator | Neuromorphic | `MotorOutcome` |
| `speech.output` | Neuromorphic | Dashboard, TTS | `SpeechToken` |
| `cognitive.query` | Neuromorphic | Cognitive Bridge | `CognitiveQuery` |
| `cognitive.response` | Cognitive Bridge | Neuromorphic | `CognitiveResponse` |
| `neuromorphic.metrics` | Neuromorphic | Dashboard | `BrainMetrics` |
| `sensory.gateway.status` | Gateway | Dashboard | `GatewayStatus` |
| `code.proposal` | Meta-Programmer | Kernel | `CodeProposal` |
| `code.decision.{trace_id}` | Kernel | Meta-Programmer | `KernelDecision` |
| `approval.request` | Any | Dashboard | `ApprovalRequest` |
| `approval.response.{id}` | Dashboard | Requester | `ApprovalResponse` |
| `system.health` | All | Monitor | `HealthStatus` |

#### Motor Outcome Payload (`motor.outcome.{channel}`)

Closes the sensorimotor feedback loop. Published by actuators, sensors (IMU, camera, joints), simulators, or human teachers after a motor command executes.

```json
{
    "channel": "locomotion",
    "success": true,
    "confidence": 0.85,
    "proprioceptive_state": [0.1, -0.02, 0.98],
    "error_magnitude": 0.05
}
```

- **channel**: Which motor sub-range fired (locomotion, manipulation, head)
- **success**: Whether the commanded action achieved its goal
- **confidence**: How certain the feedback source is (0-1)
- **proprioceptive_state**: Optional raw sensor data (IMU angles, joint positions, force readings)
- **error_magnitude**: Optional distance from target state

Sources of `motor.outcome` messages:
- **IMU/gyro**: Balance state after locomotion commands
- **Camera**: Visual change confirming manipulation success
- **Joint encoders**: Target position reached
- **Force sensors**: Grip success/failure
- **Simulator**: Ground truth outcome
- **Human teacher**: Explicit reward/punishment via dashboard

#### Kernel Body Configuration (Future)

Joint limits, velocity limits, and force thresholds are enforced by the **Kernel**, not the brain. Each robot body loads a configuration file:

```yaml
# robot_body.yaml — loaded by Kernel, NOT the brain
joints:
  left_shoulder_pitch:
    min_deg: -90
    max_deg: 180
    max_velocity_deg_s: 120
    max_torque_nm: 2.5
  left_elbow:
    min_deg: 0
    max_deg: 145
    max_velocity_deg_s: 90
    max_torque_nm: 1.5
```

The brain never sees joint limits — it only sees the proprioceptive result of its clamped commands. This mirrors biology: the motor cortex sends "extend arm" and the musculoskeletal system limits how far it extends. Three layers of protection:
1. **Hardware limits** — servo/motor firmware prevents over-rotation
2. **Kernel software limits** — clamps commands per robot body config
3. **Pain reflex** — hardwired reflex arc triggers withdrawal on excessive force

**Why NATS?**
- Single 10MB binary, no dependencies
- 18 million msgs/sec on commodity hardware
- Built-in request/reply pattern
- JetStream for persistence if needed
- Perfect for embedded/robotics

### Embedding Generation

All vector operations (task lookup, LLM cache, override search) require text embeddings. The system uses a dedicated embedding model via Ollama:

```python
# sdk/src/activelearning/embeddings.py
import aiohttp
import hashlib

class EmbeddingService:
    """
    Generates text embeddings using Ollama's embedding model.
    Caches embeddings to avoid regeneration.
    """

    def __init__(
        self,
        ollama_host: str = "http://ollama:11434",
        model: str = "nomic-embed-text"  # 768 dimensions, fast
    ):
        self.ollama_host = ollama_host
        self.model = model
        self._cache: dict[str, list[float]] = {}

    async def embed_text(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        # Check memory cache first
        cache_key = hashlib.sha256(text.encode()).hexdigest()[:16]
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Call Ollama embedding endpoint
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.ollama_host}/api/embeddings",
                json={"model": self.model, "prompt": text}
            ) as response:
                result = await response.json()
                embedding = result["embedding"]

        self._cache[cache_key] = embedding
        return embedding

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        return [await self.embed_text(t) for t in texts]

# Global embedding service instance
embedding_service = EmbeddingService()

async def embed_text(text: str) -> list[float]:
    """Convenience function for embedding text."""
    return await embedding_service.embed_text(text)
```

**Recommended embedding models (via Ollama):**

| Model | Dimensions | Speed | Best For |
|-------|------------|-------|----------|
| `nomic-embed-text` | 768 | ~500 emb/s | General purpose (recommended) |
| `mxbai-embed-large` | 1024 | ~200 emb/s | Higher accuracy |
| `all-minilm` | 384 | ~1000 emb/s | Fastest, lower accuracy |

**Qdrant Collections:**

| Collection | Dimensions | Purpose |
|------------|------------|---------|
| `learned_tasks` | 768 | Task metadata for semantic lookup |
| `llm_cache` | 768 | Cached LLM prompt-response pairs |
| `human_overrides` | 768 | Human override prompts |
| `memory_episodes` | 768 | Episodic memory embeddings |

### Knowledge Gap Detection

The system detects knowledge gaps when it encounters situations it cannot handle with existing knowledge:

```python
# coordinator/knowledge_gap.py
@dataclass
class KnowledgeGap:
    """Represents a detected gap in the system's knowledge."""
    trace_id: str
    description: str
    context: dict
    search_results: list[SearchResult]  # What was found (low confidence)
    confidence: float                    # Highest match confidence
    available_sensors: list[str]         # For demonstration learning
    allows_external_query: bool = True   # Can we ask external APIs?
    source: str = "unknown"              # What triggered the gap

class KnowledgeGapDetector:
    """
    Detects when the system lacks knowledge to handle a situation.
    Triggers Meta-Programmer to fill gaps.
    """

    # Confidence thresholds
    HIGH_CONFIDENCE = 0.85   # Execute existing task
    MEDIUM_CONFIDENCE = 0.6  # Adapt existing task
    LOW_CONFIDENCE = 0.4     # Knowledge gap detected

    def __init__(self, task_index: TaskIndex, memory: MemoryService):
        self.task_index = task_index
        self.memory = memory

    async def check_for_gap(
        self,
        request: str,
        context: dict
    ) -> Optional[KnowledgeGap]:
        """
        Check if we have knowledge to handle this request.
        Returns KnowledgeGap if we don't.
        """
        # Search for matching tasks
        task_match = await self.task_index.find_task(request)

        if task_match and task_match.confidence >= self.HIGH_CONFIDENCE:
            # We have this knowledge - no gap
            return None

        # Search memory for similar experiences
        memory_matches = await self.memory.search_similar(request, limit=5)

        # Check if we've seen this before
        if memory_matches and memory_matches[0].confidence >= self.MEDIUM_CONFIDENCE:
            # We have partial knowledge - might be able to adapt
            return None

        # Knowledge gap detected
        return KnowledgeGap(
            trace_id=generate_trace_id(),
            description=request,
            context=context,
            search_results=task_match.to_list() if task_match else [],
            confidence=task_match.confidence if task_match else 0.0,
            available_sensors=list(context.get("sensors", {}).keys()),
            source=self._determine_source(request, context)
        )

    def _determine_source(self, request: str, context: dict) -> str:
        """Determine what triggered the knowledge gap."""
        if "sensor" in context:
            return "unknown_sensor"
        if "hardware" in request.lower():
            return "hardware_interface"
        if "task" in request.lower():
            return "unknown_task"
        return "general"
```

**Meta-Programmer Trigger Priority:**

The Meta-Programmer is triggered in this priority order:

| Priority | Trigger | Action | Example |
|----------|---------|--------|---------|
| 1 (Highest) | Unknown hardware detected | Generate adapter code | New USB camera plugged in |
| 2 | Human request (if available) | Generate/refactor as requested | "Learn how to wave" |
| 3 | Knowledge gap (< 0.4 confidence) | Generate new task | "Make me coffee" (never learned) |
| 4 (Lowest) | Continuous improvement | Refactor existing code | Edge case handling, optimization |

**Meta-Programmer Capabilities:**

| Capability | When Used |
|------------|-----------|
| **Create** | Only when no related task exists |
| **Refactor** | Improve existing code (edge cases, performance) |
| **Adapt** | Modify existing task for new use case |

> **Key principle**: Meta-Programmer prefers refactoring/adapting existing code over creating new code. Creation only happens when nothing related exists.

**Gap Resolution Flow:**

```
Knowledge Gap Detected
         │
         ▼
┌─────────────────────────────────────────┐
│  Check available resolution methods     │
│                                         │
│  1. Can we learn via demonstration?     │
│     → Check if camera available         │
│                                         │
│  2. Can we query external API?          │
│     → Check allows_external_query       │
│                                         │
│  3. Can we generate from patterns?      │
│     → Check similar tasks exist         │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│           META-PROGRAMMER               │
│                                         │
│  - If sensors available: request demo   │
│  - If external allowed: query API       │
│  - Otherwise: generate best-effort      │
└─────────────────────────────────────────┘
         │
         ▼
     New Task Code
         │
         ▼
   Index in Vector DB
```

### Local LLM: Ollama

```yaml
ollama:
  image: ollama/ollama:latest
  volumes:
    - ollama-models:/root/.ollama
  environment:
    - OLLAMA_HOST=0.0.0.0
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia  # or 'mps' for M2 Mac
            capabilities: [gpu]
```

**Recommended models for code generation:**

| Model | Size | Use Case | M2 Pro Performance |
|-------|------|----------|-------------------|
| `deepseek-coder:6.7b` | 4GB | Fast code completion | ~30 tok/s |
| `deepseek-coder:33b` | 20GB | Complex code generation | ~8 tok/s |
| `codellama:13b` | 8GB | Balanced | ~15 tok/s |
| `qwen2.5-coder:7b` | 4GB | Good reasoning | ~25 tok/s |

**Meta-Programmer integration with Ollama:**

The Meta-Programmer calls Ollama directly via HTTP for code generation:

```python
async with aiohttp.ClientSession() as session:
    async with session.post(
        "http://ollama:11434/api/generate",
        json={"model": "deepseek-coder:6.7b", "prompt": prompt}
    ) as response:
        result = await response.json()
```

### Container Sandbox for Code Testing

When Meta-Programmer generates code, it's tested in ephemeral containers:

```python
# Meta-Programmer spawns sandbox containers via Docker API
import docker

client = docker.from_env()

def run_in_sandbox(code: str, test_code: str) -> SandboxResult:
    """Run generated code in isolated container."""
    container = client.containers.run(
        image="sandbox-python:latest",
        command=f"python -c '{test_code}'",
        environment={"CODE_UNDER_TEST": code},
        network_disabled=True,      # No network
        read_only=True,             # Read-only filesystem
        tmpfs={"/tmp": "size=100M"}, # Only /tmp writable
        mem_limit="512m",           # Memory limit
        cpu_period=100000,
        cpu_quota=50000,            # 50% CPU max
        remove=True,                # Auto-delete after
        timeout=30,                 # 30 second max
    )
    return SandboxResult(
        exit_code=container.exit_code,
        output=container.logs(),
    )
```

### Unified SQLite Database

Single database with schemas instead of multiple files:

```sql
-- /data/sqlite/unified.db

-- Memory service tables
CREATE TABLE memory.episodes (
    id TEXT PRIMARY KEY,
    trace_id TEXT NOT NULL,
    timestamp INTEGER NOT NULL,
    embedding_ref TEXT,
    semantic_tags TEXT,  -- JSON array
    utility_score REAL DEFAULT 1.0
);

-- Approval tables (human-in-the-loop via Dashboard)
CREATE TABLE approvals.requests (
    id TEXT PRIMARY KEY,
    trace_id TEXT,
    tool_name TEXT,
    tool_input TEXT,  -- JSON
    status TEXT,
    created_at INTEGER,
    responded_at INTEGER,
    decision TEXT,
    comment TEXT
);

-- Audit trail (unified)
CREATE TABLE audit.entries (
    id TEXT PRIMARY KEY,
    trace_id TEXT NOT NULL,
    timestamp INTEGER NOT NULL,
    component TEXT,      -- 'kernel', 'meta-programmer', 'planner', etc.
    action TEXT,
    details TEXT,        -- JSON
    code_hash TEXT
);
```

---

## Network Isolation

Docker networks enforce security boundaries:

```yaml
networks:
  # Public network - open components
  public:
    driver: bridge

  # Internal network - closed components only
  internal:
    driver: bridge
    internal: true  # No external access

  # Sandbox network - completely isolated
  sandbox:
    driver: bridge
    internal: true
    driver_opts:
      com.docker.network.bridge.enable_ip_masquerade: "false"
```

**Network access matrix:**

| Component | public | internal | sandbox | internet |
|-----------|--------|----------|---------|----------|
| SDK Runtime | ✅ | ❌ | ❌ | ❌ |
| Planner | ✅ | ❌ | ❌ | ❌ |
| Memory | ✅ | ❌ | ❌ | ❌ |
| Kernel | ✅ | ✅ | ❌ | ❌ |
| Meta-Programmer | ✅ | ✅ | ❌ | ❌ |
| Safety Supervisor | ❌ | ✅ | ❌ | ❌ |
| Sandbox containers | ❌ | ❌ | ✅ | ❌ |
| Ollama | ✅ | ✅ | ❌ | ❌ |
| Dashboard | ✅ | ❌ | ❌ | ❌ |

---

## Safety Supervisor vs Moral Kernel

The Safety Supervisor and Moral Kernel work together but have distinct roles:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    SAFETY ARCHITECTURE                                │
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                   SAFETY SUPERVISOR                          │    │
│  │                   (Analysis Engine)                          │    │
│  │                                                               │    │
│  │  Role: Provides heuristics and risk analysis                 │    │
│  │  - Detects elevated scrutiny triggers                        │    │
│  │  - Analyzes code for dangerous patterns                      │    │
│  │  - Computes risk scores                                      │    │
│  │  - Flags potential issues                                    │    │
│  │                                                               │    │
│  │  Does NOT make decisions - only provides analysis            │    │
│  │                                                               │    │
│  │  Examples:                                                    │    │
│  │  - "This code uses eval() - high risk"                       │    │
│  │  - "This proposal modifies /kernel/* - protected path"       │    │
│  │  - "Servo angle 200° exceeds safe range"                     │    │
│  └───────────────────────────────────┬───────────────────────────┘    │
│                                      │                                │
│                                      │ Risk analysis                  │
│                                      ▼                                │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                    MORAL KERNEL                              │    │
│  │                   (Decision Maker)                           │    │
│  │                                                               │    │
│  │  Role: Makes final ALLOW/TRANSFORM/DENY/DEFER decisions      │    │
│  │  - Receives risk analysis from Safety Supervisor             │    │
│  │  - Applies immutable ethical rules                           │    │
│  │  - Issues decision tokens                                    │    │
│  │  - Has final authority                                       │    │
│  │                                                               │    │
│  │  ONLY component that can authorize actions                   │    │
│  │                                                               │    │
│  │  Examples:                                                    │    │
│  │  - Risk score > 0.8 → DENY                                   │    │
│  │  - Protected path → DENY (always)                            │    │
│  │  - Human safety concern → DEFER to Dashboard (human review)  │    │
│  │  - Normal operation → ALLOW                                  │    │
│  │                                                               │    │
│  │  DEFER is provisional: if human review does not answer       │    │
│  │  before expires_at / DEFER_TTL_MS, consumers MUST treat it   │    │
│  │  as DENY (fail-closed). A NATS partition that prevents the   │    │
│  │  Dashboard from publishing approval.response is therefore   │    │
│  │  safe — silence expires to DENY, never to ALLOW.             │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

**DEFER → human review TTL (fail-closed):** Kernel stamps `expires_at` on
`DEFER` decisions (`defer_ttl_ms`, default 5 minutes). Callers that stage work
for human approval (notably Meta-Programmer `human_review/`) must reject the
staged item when the TTL elapses with no `approval.response` — including when
the Dashboard cannot reach NATS. See [META-PROGRAMMER.md](META-PROGRAMMER.md)
§ Human-in-the-Loop.

**Key Distinctions:**

| Aspect | Safety Supervisor | Moral Kernel |
|--------|-------------------|--------------|
| **Function** | Analysis | Decision |
| **Output** | Risk scores, flags, warnings | ALLOW/TRANSFORM/DENY/DEFER |
| **Authority** | Advisory only | Final authority |
| **Modifiable** | Heuristics can be updated | Rules are immutable |
| **Network** | Internal only | Public + Internal |
| **Containers** | 1 instance | Can be clustered |

**Communication Flow:**

```python
# Safety Supervisor analyzes a proposal
async def analyze_proposal(proposal: ActionProposal) -> RiskAnalysis:
    """Safety Supervisor: Analyze risk, don't decide."""
    analysis = RiskAnalysis(
        trace_id=proposal.trace_id,
        risk_score=0.0,
        flags=[],
        recommendations=[]
    )

    # Check for dangerous patterns
    if "eval(" in proposal.code:
        analysis.risk_score += 0.5
        analysis.flags.append("DYNAMIC_EXECUTION")
        analysis.recommendations.append("Remove eval() usage")

    if proposal.target_path.startswith("/kernel"):
        analysis.risk_score = 1.0  # Maximum risk
        analysis.flags.append("PROTECTED_PATH")

    return analysis  # Does NOT decide - sends to Kernel


# Moral Kernel makes the final decision
async def evaluate(proposal: ActionProposal) -> KernelDecision:
    """Moral Kernel: Make the final decision."""
    # Get analysis from Safety Supervisor
    analysis = await safety_supervisor.analyze(proposal)

    # Apply immutable rules
    if "PROTECTED_PATH" in analysis.flags:
        return KernelDecision(type=DENY, reason="Protected path modification attempted")

    if analysis.risk_score > 0.8:
        return KernelDecision(type=DENY, reason=f"Risk too high: {analysis.risk_score}")

    if analysis.risk_score > 0.5:
        return KernelDecision(type=DEFER, reason="Elevated risk - requires human approval")

    if analysis.flags and analysis.risk_score > 0.3:
        # Transform to safer version
        safe_version = await self._apply_safety_transforms(proposal, analysis)
        return KernelDecision(type=TRANSFORM, transformations=[safe_version])

    return KernelDecision(type=ALLOW)
```

**Why Separate Them?**

1. **Single responsibility**: Supervisor analyzes, Kernel decides
2. **Updatable heuristics**: Can improve detection without changing decision rules
3. **Defense in depth**: Even if Supervisor is compromised, Kernel rules remain
4. **Auditability**: Clear separation between "what was detected" and "what was decided"

---

## Python SDK (Replacing TypeScript)

The SDK is Python-first for consistency across all components:

```python
# sdk/core.py
from enum import Enum
from dataclasses import dataclass
from typing import TypeVar, Generic, Optional, List
import nats

T = TypeVar('T')

class KernelDecisionType(Enum):
    ALLOW = "ALLOW"
    TRANSFORM = "TRANSFORM"
    DENY = "DENY"
    DEFER = "DEFER"  # Provisional; unanswered TTL → DENY (fail-closed)

@dataclass
class Observation(Generic[T]):
    trace_id: str
    provenance: str
    data: T
    timestamp: Optional[int] = None

@dataclass
class ActionProposal(Generic[T]):
    trace_id: str
    provenance: str
    action: T

@dataclass
class KernelDecision(Generic[T]):
    type: KernelDecisionType
    transformations: Optional[List[ActionProposal[T]]] = None
    reason: Optional[str] = None

@dataclass
class Outcome(Generic[T]):
    trace_id: str
    decision: KernelDecision[T]
    original: ActionProposal[T]
    final: Optional[ActionProposal[T]] = None
    reason: Optional[str] = None

# NATS-based event bus
class EventBus:
    def __init__(self, nats_url: str = "nats://localhost:4222"):
        self.nats_url = nats_url
        self._nc = None

    async def connect(self):
        self._nc = await nats.connect(self.nats_url)

    async def publish(self, subject: str, payload: dict):
        await self._nc.publish(subject, json.dumps(payload).encode())

    async def subscribe(self, subject: str, callback):
        await self._nc.subscribe(subject, cb=callback)

# Global bus instance
bus = EventBus()
```

---

## Docker Compose Configuration

```yaml
# docker-compose.yml
version: "3.9"

services:
  # === INFRASTRUCTURE ===

  nats:
    image: nats:2.10-alpine
    ports:
      - "4222:4222"   # Client connections
      - "8222:8222"   # HTTP monitoring
    command: ["--jetstream", "--store_dir", "/data"]
    volumes:
      - nats-data:/data
    networks:
      - public
      - internal
    restart: unless-stopped

  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama-models:/root/.ollama
    networks:
      - public
      - internal
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]
    restart: unless-stopped

  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - qdrant-data:/qdrant/storage
    networks:
      - public
    restart: unless-stopped

  # === OPEN COMPONENTS ===

  sdk-runtime:
    build:
      context: ./sdk
      dockerfile: Dockerfile
    environment:
      - NATS_URL=nats://nats:4222
      - QDRANT_URL=http://qdrant:6333
    volumes:
      - sqlite-data:/data/sqlite
    networks:
      - public
    depends_on:
      - nats
      - qdrant
    restart: unless-stopped

  planner:
    build:
      context: ./planner
      dockerfile: Dockerfile
    environment:
      - NATS_URL=nats://nats:4222
      - OLLAMA_URL=http://ollama:11434
    networks:
      - public
    depends_on:
      - nats
      - ollama
    restart: unless-stopped

  memory:
    build:
      context: ./memory
      dockerfile: Dockerfile
    environment:
      - NATS_URL=nats://nats:4222
      - QDRANT_URL=http://qdrant:6333
      - SQLITE_PATH=/data/sqlite/unified.db
    volumes:
      - sqlite-data:/data/sqlite
    networks:
      - public
    depends_on:
      - nats
      - qdrant
    restart: unless-stopped

  # === CLOSED COMPONENTS ===

  kernel:
    build:
      context: ./kernel
      dockerfile: Dockerfile
    environment:
      - NATS_URL=nats://nats:4222
    networks:
      - public
      - internal
    depends_on:
      - nats
    restart: unless-stopped

  meta-programmer:
    build:
      context: ./meta-programmer
      dockerfile: Dockerfile
    environment:
      - NATS_URL=nats://nats:4222
      - OLLAMA_URL=http://ollama:11434
      - QDRANT_URL=http://qdrant:6333
      - SQLITE_PATH=/data/sqlite/unified.db
      - DOCKER_HOST=unix:///var/run/docker.sock
    volumes:
      - sqlite-data:/data/sqlite
      - staging-data:/data/staging
      - /var/run/docker.sock:/var/run/docker.sock  # For spawning sandboxes
    networks:
      - public
      - internal
    depends_on:
      - nats
      - ollama
      - kernel
    restart: unless-stopped

  safety-supervisor:
    build:
      context: ./safety-supervisor
      dockerfile: Dockerfile
    environment:
      - NATS_URL=nats://nats:4222
    networks:
      - internal  # Only internal access
    depends_on:
      - nats
    restart: unless-stopped

  # === HUMAN INTERFACE ===
  # Human-in-the-loop approval is handled by the Dashboard service
  # (approval.request / approval.response.{id} NATS subjects)

networks:
  public:
    driver: bridge
  internal:
    driver: bridge
    internal: true

volumes:
  nats-data:
  ollama-models:
  qdrant-data:
  sqlite-data:
  staging-data:
```

---

## Testing Pipeline

The Meta-Programmer uses a two-stage testing pipeline to validate generated code before deployment:

### Stage 1: Unit Tests (Ephemeral Sandboxes)

Fast, isolated unit tests for pure Python code:

```
┌─────────────────────────────────────────────────────────────────┐
│                     META-PROGRAMMER                             │
│                           │                                     │
│                           │ Docker API                          │
│                           ▼                                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                      │
│  │Sandbox-1 │  │Sandbox-2 │  │Sandbox-3 │  (Ephemeral)         │
│  │ Test A   │  │ Test B   │  │ Test C   │  ~5 seconds total    │
│  │ 30s max  │  │ 30s max  │  │ 30s max  │                      │
│  └──────────┘  └──────────┘  └──────────┘                      │
│       │             │             │                             │
│       └─────────────┴─────────────┘                             │
│                     │                                           │
│                     ▼                                           │
│              Pass? → Stage 2                                    │
└─────────────────────────────────────────────────────────────────┘
```

**Characteristics:**
- No network access
- Read-only filesystem (except /tmp)
- 512MB memory limit, 50% CPU max
- 30 second timeout per test
- Containers destroyed after use

### Stage 2: Integration Tests (Test Runner Container)

A persistent container with mock sensors/actuators for integration testing:

```
┌─────────────────────────────────────────────────────────────────┐
│                      TEST RUNNER CONTAINER                      │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                 MOCK ENVIRONMENT                         │   │
│  │                                                          │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │   │
│  │  │ Mock Camera │  │ Mock Servo  │  │ Mock GPIO   │      │   │
│  │  │   Sensor    │  │  Actuator   │  │   Sensor    │      │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘      │   │
│  │                                                          │   │
│  │  ┌─────────────────────────────────────────────────┐    │   │
│  │  │              NATS Connection                     │    │   │
│  │  │   (Connected to main NATS for message flow)     │    │   │
│  │  └─────────────────────────────────────────────────┘    │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              ISOLATED EXECUTION ZONE                     │   │
│  │                                                          │   │
│  │   Generated plugin code runs here with:                  │   │
│  │   - Access to mock sensors/actuators                     │   │
│  │   - NATS message flow testing                            │   │
│  │   - SDK integration verification                         │   │
│  │   - No access to real hardware                           │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  Results → Meta-Programmer → Deploy (if pass)                  │
└─────────────────────────────────────────────────────────────────┘
```

**Characteristics:**
- Persistent container (always running)
- Mock sensors/actuators for hardware simulation
- Connected to NATS for message flow testing
- Tests plugin integration with SDK
- ~30 seconds per integration test

### Test Runner Docker Configuration

```yaml
# Added to docker-compose.yml
services:
  test-runner:
    build:
      context: ./test-runner
      dockerfile: Dockerfile
    environment:
      - NATS_URL=nats://nats:4222
      - SQLITE_PATH=/data/sqlite/unified.db
      - MOCK_MODE=true
    volumes:
      - staging-data:/data/staging:ro          # Read-only access to staged code
      - /var/run/docker.sock:/var/run/docker.sock  # For spawning Stage 1 sandboxes
    networks:
      - public
    depends_on:
      - nats
    restart: unless-stopped
```

### Test Runner Implementation

```python
# test-runner/test_runner/runner.py
import asyncio
import docker
from sdk.core import Observation, ActionProposal, Outcome
from sdk.nats_client import EventBus

class MockSensor:
    """Simulates a hardware sensor for testing."""
    def __init__(self, sensor_id: str):
        self.sensor_id = sensor_id
        self.bus = EventBus()

    async def emit_test_observation(self, data: dict):
        """Emit a test observation to verify plugin handles it."""
        obs = Observation(
            trace_id=f"test-{uuid.uuid4()}",
            provenance=self.sensor_id,
            data=data,
            timestamp=int(time.time() * 1000)
        )
        await self.bus.publish(f"observation.{self.sensor_id}", obs.to_dict())
        return obs.trace_id

class MockActuator:
    """Captures actuator commands for verification."""
    def __init__(self, actuator_id: str):
        self.actuator_id = actuator_id
        self.received_outcomes: list[Outcome] = []
        self.bus = EventBus()

    async def start_listening(self):
        """Listen for outcomes and capture them."""
        async def handler(msg):
            outcome = Outcome.from_dict(json.loads(msg.data))
            self.received_outcomes.append(outcome)
        await self.bus.subscribe(f"outcome.*", handler)

    def assert_received(self, trace_id: str, expected_action: dict):
        """Assert that a specific action was received."""
        for outcome in self.received_outcomes:
            if outcome.trace_id == trace_id:
                assert outcome.final.action == expected_action
                return
        raise AssertionError(f"No outcome received for trace_id {trace_id}")

class TestRunner:
    """Orchestrates Stage 1 and Stage 2 tests."""

    def __init__(self):
        self.docker_client = docker.from_env()
        self.mock_sensors: dict[str, MockSensor] = {}
        self.mock_actuators: dict[str, MockActuator] = {}

    async def run_stage1_unit_tests(self, code_path: str) -> TestResult:
        """Run unit tests in ephemeral sandbox."""
        container = self.docker_client.containers.run(
            image="sandbox-python:latest",
            command=["pytest", "/code", "-v"],
            volumes={code_path: {"bind": "/code", "mode": "ro"}},
            network_disabled=True,
            read_only=True,
            mem_limit="512m",
            remove=True,
            timeout=30,
        )
        return TestResult(
            stage="unit",
            passed=container.attrs["State"]["ExitCode"] == 0,
            output=container.logs().decode(),
        )

    async def run_stage2_integration_tests(self, plugin_code: str) -> TestResult:
        """Run integration tests with mock environment."""
        # Load the plugin in isolated execution zone
        exec_globals = {"__builtins__": __builtins__}
        exec(plugin_code, exec_globals)

        # Get the plugin class
        plugin_class = exec_globals.get("Plugin")
        if not plugin_class:
            return TestResult(stage="integration", passed=False, error="No Plugin class found")

        # Instantiate with mock environment
        plugin = plugin_class(
            sensors=self.mock_sensors,
            actuators=self.mock_actuators,
        )

        # Run integration test scenarios
        results = []
        for scenario in self.get_test_scenarios(plugin):
            result = await self.run_scenario(plugin, scenario)
            results.append(result)

        return TestResult(
            stage="integration",
            passed=all(r.passed for r in results),
            scenarios=results,
        )
```

### Testing Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                      META-PROGRAMMER                            │
│                                                                 │
│  1. CodeGen generates plugin code                               │
│                    │                                            │
│                    ▼                                            │
│  2. Submit to Kernel for review                                 │
│                    │                                            │
│         ┌─────────┴─────────┐                                   │
│         │                   │                                   │
│      ALLOW              DENY/DEFER                              │
│         │                   │                                   │
│         ▼                   ▼                                   │
│  3. Stage 1: Unit      Return error                             │
│     (Sandbox)          or escalate                              │
│         │                                                       │
│      Pass?                                                      │
│         │                                                       │
│    ┌────┴────┐                                                  │
│    │         │                                                  │
│   Yes        No → Reject, try alternative                       │
│    │                                                            │
│    ▼                                                            │
│  4. Stage 2: Integration                                        │
│     (Test Runner)                                               │
│         │                                                       │
│      Pass?                                                      │
│         │                                                       │
│    ┌────┴────┐                                                  │
│    │         │                                                  │
│   Yes        No → Reject, try alternative                       │
│    │                                                            │
│    ▼                                                            │
│  5. Deploy to /data/plugins/                                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### NATS Subjects for Testing

| Subject | Publisher | Subscriber | Payload |
|---------|-----------|------------|---------|
| `test.request` | Meta-Programmer | Test Runner | `TestRequest` |
| `test.result.{trace_id}` | Test Runner | Meta-Programmer | `TestResult` |
| `test.stage1.start` | Test Runner | Monitor | `StageStartEvent` |
| `test.stage2.start` | Test Runner | Monitor | `StageStartEvent` |

---

## Future: Isaac Sim Integration (Phase C)

> **Note**: Isaac Sim requires NVIDIA GPU and will be implemented as a cloud-based testing option in the future.

When NVIDIA hardware is available, a third testing stage can be added:

### Stage 3: Robotics Simulation (Isaac Sim)

```
┌─────────────────────────────────────────────────────────────────┐
│                    ISAAC SIM CONTAINER                          │
│                    (Cloud / NVIDIA GPU)                         │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                 SIMULATED ROBOT                          │   │
│  │                                                          │   │
│  │  - Full physics simulation                               │   │
│  │  - Simulated cameras, LiDAR, IMU                        │   │
│  │  - Collision detection                                   │   │
│  │  - Environment interaction                               │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                 ROS 2 BRIDGE                             │   │
│  │                                                          │   │
│  │  - Connects Isaac Sim to NATS                           │   │
│  │  - Translates sensor data to Observations               │   │
│  │  - Translates Outcomes to actuator commands             │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ~5-10 minutes per simulation test                             │
└─────────────────────────────────────────────────────────────────┘
```

**Prerequisites for Phase C:**
- NVIDIA GPU (Jetson, cloud instance, or desktop)
- Isaac Sim license
- ROS 2 Humble
- Robot URDF/USD models
- Simulation environments

**Docker Compose (future, separate file):**

```yaml
# docker-compose.isaac.yml
# Only used when NVIDIA GPU available (cloud or local)
services:
  isaac-sim:
    image: nvcr.io/nvidia/isaac-sim:2023.1.1
    runtime: nvidia
    environment:
      - DISPLAY=${DISPLAY}
      - ACCEPT_EULA=Y
    volumes:
      - ./sim/robots:/isaac-sim/robots
      - ./sim/environments:/isaac-sim/environments
    networks:
      - public
    ports:
      - "8211:8211"
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]

  ros2-bridge:
    image: ros:humble
    environment:
      - ROS_DOMAIN_ID=0
      - NATS_URL=nats://nats:4222
    networks:
      - public
    depends_on:
      - isaac-sim
      - nats
```

This will be implemented when cloud GPU resources are available for testing.

---

## Human Override System

The system distinguishes between **operational parameters** (human-adjustable) and **Moral Kernel rules** (immutable). Humans can override how the robot operates, but never the safety constraints.

### Override Hierarchy

```
┌─────────────────────────────────────────────────────────────────┐
│                     IMMUTABLE (Cannot Override)                  │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                    MORAL KERNEL                          │    │
│  │  - Harm prevention rules                                │    │
│  │  - Safety boundaries                                    │    │
│  │  - Protected paths                                      │    │
│  │  - Core ethical constraints                             │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                              ▲
                              │ Cannot bypass
                              │
┌─────────────────────────────────────────────────────────────────┐
│                   HUMAN OVERRIDABLE (Operational)               │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              OPERATIONAL PARAMETERS                      │    │
│  │  - Motor movement limits ("motor can move X far")       │    │
│  │  - Sensor sensitivity thresholds                        │    │
│  │  - Task priorities and scheduling                       │    │
│  │  - Learning preferences                                 │    │
│  │  - Hardware calibration values                          │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  Even operational changes pass through Kernel for validation    │
│  Kernel ensures override doesn't violate safety rules           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Demonstration Learning Triggers

The system enters demonstration learning mode through several triggers:

```python
# coordinator/demo_triggers.py
class DemoTriggerDetector:
    """
    Detects when a human wants to demonstrate a task.
    Multiple trigger mechanisms supported.
    """

    def __init__(self, sensor_manager: SensorManager, bus: EventBus):
        self.sensors = sensor_manager
        self.bus = bus

    async def start(self):
        """Subscribe to potential demo triggers."""
        # Voice trigger: "Let me show you how to..."
        await self.bus.subscribe("voice.command", self.on_voice_command)
        # Gesture trigger: Raised hand, specific pose
        await self.bus.subscribe("gesture.detected", self.on_gesture)
        # Touch trigger: Human physically guides robot arm
        await self.bus.subscribe("touch.guidance", self.on_touch_guidance)
        # Explicit trigger: Button press or API call
        await self.bus.subscribe("demo.request", self.on_explicit_request)

    async def on_voice_command(self, msg):
        """Detect voice commands that start demos."""
        command = msg.data.get("text", "").lower()
        demo_phrases = [
            "let me show you",
            "watch me",
            "learn this",
            "i'll demonstrate",
            "follow my movements"
        ]
        if any(phrase in command for phrase in demo_phrases):
            task_name = self._extract_task_name(command)
            await self.start_demo(task_name, trigger="voice")

    async def on_gesture(self, msg):
        """Detect gestures that start demos."""
        gesture = msg.data.get("gesture")
        if gesture == "raised_hand_open_palm":
            # Universal "attention" gesture
            await self.start_demo(task_name="unnamed_task", trigger="gesture")

    async def on_touch_guidance(self, msg):
        """Detect when human is physically guiding robot."""
        force = msg.data.get("force", 0)
        if force > 0.5:  # Human is applying force to robot
            # Auto-start demo in physical guidance mode
            await self.start_demo(task_name="guided_task", trigger="touch")

    async def on_explicit_request(self, msg):
        """Handle explicit demo requests (button, API)."""
        task_name = msg.data.get("task_name", "unnamed_task")
        await self.start_demo(task_name, trigger="explicit")

    async def start_demo(self, task_name: str, trigger: str):
        """Start a demonstration session."""
        await self.bus.publish("learning.demo.start", {
            "task_name": task_name,
            "trigger": trigger,
            "timestamp": int(time.time() * 1000),
            "available_sensors": list(self.sensors.available_sensors.keys())
        })
```

**Trigger Methods:**

| Method | Trigger | Example |
|--------|---------|---------|
| **Voice** | Keywords detected | "Watch me pick up this cup" |
| **Gesture** | Specific pose | Raised hand with open palm |
| **Touch** | Physical guidance | Human moves robot's arm |
| **Explicit** | Button/API | "Start Demo" button in UI |
| **Knowledge Gap** | Auto-triggered | System can't find matching task |

### Human Detection (Anti-Spoofing)

Human overrides must come from verified humans, not automated systems:

```python
# Human verification via multi-modal detection
class HumanVerifier:
    """Verifies override requests come from actual humans."""

    def __init__(self, sensors: dict[str, Sensor]):
        self.camera = sensors.get("camera")
        self.microphone = sensors.get("microphone")

    async def verify_human_presence(self) -> HumanVerification:
        """Multi-modal verification that a human is present."""
        checks = []

        # Camera: Face detection, liveness check
        if self.camera:
            face_result = await self.camera.detect_face()
            liveness = await self.camera.check_liveness()  # Anti-photo spoofing
            checks.append(HumanCheck(
                sensor="camera",
                confidence=face_result.confidence * liveness.confidence,
                features={"face_detected": face_result.detected, "is_live": liveness.is_live}
            ))

        # Microphone: Voice activity, speech patterns
        if self.microphone:
            voice_result = await self.microphone.detect_voice()
            checks.append(HumanCheck(
                sensor="microphone",
                confidence=voice_result.confidence,
                features={"voice_detected": voice_result.detected}
            ))

        # Require at least one high-confidence verification
        verified = any(c.confidence > 0.8 for c in checks)

        return HumanVerification(
            verified=verified,
            checks=checks,
            timestamp=int(time.time() * 1000)
        )

    async def get_fallback_verification(self) -> HumanVerification:
        """
        Fallback when no sensors available for verification.
        Options: PIN code, physical button, or reject.
        """
        # Check if physical verification button is available
        if self.physical_button_available():
            await self.prompt_button_press()
            pressed = await self.wait_for_button(timeout=30)
            if pressed:
                return HumanVerification(
                    verified=True,
                    checks=[HumanCheck(sensor="physical_button", confidence=1.0)],
                    fallback_used=True
                )

        # No verification possible - reject override
        logger.warn("No human verification method available - rejecting override")
        return HumanVerification(
            verified=False,
            checks=[],
            fallback_used=True,
            rejection_reason="no_verification_method"
        )
```

**Verification Fallback Hierarchy:**

| Priority | Method | Confidence | Availability |
|----------|--------|------------|--------------|
| 1 | Camera + Microphone | Highest | Requires both sensors |
| 2 | Camera only | High | Face detection |
| 3 | Microphone only | Medium | Voice activity |
| 4 | Physical button | High | Hardware button |
| 5 | Reject | N/A | No override allowed |

> **Note**: PIN codes are intentionally NOT supported as they can be easily automated by other programs.

### Override Storage and Retrieval

Human overrides are stored as prompts in SQLite and indexed in vector DB:

```sql
-- Override prompt storage
CREATE TABLE overrides.human_prompts (
    id TEXT PRIMARY KEY,
    trace_id TEXT NOT NULL,
    timestamp INTEGER NOT NULL,

    -- The override itself
    prompt TEXT NOT NULL,              -- Natural language override instruction
    parameter_path TEXT,               -- e.g., "motor.left_arm.max_range"
    old_value TEXT,                    -- Previous value (JSON)
    new_value TEXT,                    -- New value (JSON)

    -- Verification
    human_verified BOOLEAN DEFAULT FALSE,
    verification_method TEXT,          -- 'camera', 'voice', 'both'
    verification_confidence REAL,

    -- Vector reference for semantic lookup
    embedding_ref TEXT,                -- Points to Qdrant collection

    -- Code generation (if Meta-Programmer creates code)
    generated_code_path TEXT,          -- Path in /data/overrides/
    code_trace_id TEXT,                -- Links to code audit trail

    -- Audit
    kernel_decision TEXT,
    applied_at INTEGER
);
```

**Override Flow:**

```
Human speaks: "The arm motor can move up to 45 degrees"
         │
         ▼
┌─────────────────────────────────────────┐
│          HUMAN VERIFIER                  │
│  - Camera: face detected (0.92)         │
│  - Mic: voice detected (0.88)           │
│  - Result: VERIFIED                      │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│          PROMPT PROCESSOR                │
│  - Parse: parameter=motor.arm.max_range │
│  - Extract: new_value=45°               │
│  - Embed prompt for vector DB           │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│          MORAL KERNEL                    │
│  - Check: Does 45° violate safety?      │
│  - Decision: ALLOW (within safe range)  │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│          STORAGE                         │
│  - Save to SQLite (overrides.human_prompts) │
│  - Index in Qdrant (semantic lookup)    │
│  - If complex: Meta-Programmer generates code │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│          COORDINATOR                     │
│  - Future requests check vector DB      │
│  - Finds override by semantic similarity │
│  - Retrieves code path from filesystem  │
└─────────────────────────────────────────┘
```

### NATS Subjects for Overrides

| Subject | Publisher | Subscriber | Payload |
|---------|-----------|------------|---------|
| `override.request` | Dashboard | Override Processor | `OverrideRequest` |
| `override.verify` | Override Processor | Human Verifier | `VerificationRequest` |
| `override.verified.{trace_id}` | Human Verifier | Override Processor | `VerificationResult` |
| `override.decision.{trace_id}` | Kernel | Override Processor | `KernelDecision` |
| `override.applied.{trace_id}` | Override Processor | All | `OverrideApplied` |

---

## Multi-Sensory Learning System

The system learns from multiple sensors with priority-based fusion and demonstration learning from humans.

### Sensor Priority Hierarchy

```
                    HIGHEST PRIORITY
                          ▲
┌─────────────────────────┴─────────────────────────┐
│                      CAMERA                        │
│  - Richest information density                     │
│  - Human pose estimation for demonstration         │
│  - Object recognition                              │
│  - Priority weight: 1.0                            │
└───────────────────────────────────────────────────┘
                          │
┌─────────────────────────▼─────────────────────────┐
│                      SOUND                         │
│  - Voice commands                                  │
│  - Environmental awareness                         │
│  - Human feedback ("good", "no", "stop")          │
│  - Priority weight: 0.8                            │
└───────────────────────────────────────────────────┘
                          │
┌─────────────────────────▼─────────────────────────┐
│                      TOUCH                         │
│  - Physical guidance (human moving robot's arm)   │
│  - Force feedback for learning pressure           │
│  - Collision detection                            │
│  - Priority weight: 0.6                            │
└───────────────────────────────────────────────────┘
                          │
┌─────────────────────────▼─────────────────────────┐
│                      SMELL                         │
│  - Environmental hazard detection                  │
│  - Context awareness (kitchen vs garage)          │
│  - Priority weight: 0.4                            │
│                                                    │
│  LOWEST PRIORITY                                   │
└───────────────────────────────────────────────────┘
```

### Sensor Fusion (Within Coordinator)

Sensor Fusion is a **component within the Coordinator**, not a separate container. It combines observations from multiple sensors with priority-based weighting:

```python
# coordinator/sensor_fusion.py
class SensorFusion:
    """
    Fuses observations from multiple sensors.
    Lives inside the Coordinator container.
    """

    PRIORITY_WEIGHTS = {
        "camera": 1.0,
        "microphone": 0.8,
        "touch": 0.6,
        "smell": 0.4,
    }

    def __init__(self, sensor_manager: SensorManager):
        self.sensors = sensor_manager
        self.pending_observations: dict[str, list[Observation]] = {}
        self.fusion_window_ms = 100  # Fuse observations within 100ms

    async def fuse_observations(
        self,
        observations: list[Observation]
    ) -> FusedObservation:
        """
        Combine multiple sensor observations into a unified view.
        Uses priority weighting for conflicting information.
        """
        if len(observations) == 1:
            return FusedObservation.from_single(observations[0])

        # Group by data type
        by_type = defaultdict(list)
        for obs in observations:
            by_type[obs.data.get("type")].append(obs)

        # Resolve conflicts using priority
        fused_data = {}
        for data_type, obs_list in by_type.items():
            if len(obs_list) == 1:
                fused_data[data_type] = obs_list[0].data
            else:
                # Multiple sensors report same type - use highest priority
                sorted_obs = sorted(
                    obs_list,
                    key=lambda o: self.PRIORITY_WEIGHTS.get(o.provenance, 0),
                    reverse=True
                )
                fused_data[data_type] = sorted_obs[0].data
                fused_data[f"{data_type}_confidence"] = sorted_obs[0].confidence

        return FusedObservation(
            trace_id=observations[0].trace_id,
            sources=[o.provenance for o in observations],
            data=fused_data,
            timestamp=max(o.timestamp for o in observations)
        )
```

### Distributed Sensor Architecture

Each sensor operates independently but provides feedback to others:

```
┌──────────────────────────────────────────────────────────────────────┐
│                        DISTRIBUTED SENSORS                            │
│                                                                       │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐           │
│  │ Camera  │◄──►│ Sound   │◄──►│ Touch   │◄──►│ Smell   │           │
│  │ Sensor  │    │ Sensor  │    │ Sensor  │    │ Sensor  │           │
│  └────┬────┘    └────┬────┘    └────┬────┘    └────┬────┘           │
│       │              │              │              │                  │
│       │    NATS: sensor.feedback.{from}.{to}      │                  │
│       │              │              │              │                  │
│       └──────────────┴──────────────┴──────────────┘                  │
│                              │                                        │
│                              ▼                                        │
│                    ┌────────────────────┐                            │
│                    │  SENSOR FUSION     │                            │
│                    │  (Weighted by      │                            │
│                    │   priority)        │                            │
│                    └─────────┬──────────┘                            │
│                              │                                        │
│                              ▼                                        │
│                    ┌────────────────────┐                            │
│                    │    COORDINATOR     │                            │
│                    │                    │                            │
│                    │  - Check vector DB │                            │
│                    │  - Find task code  │                            │
│                    │  - Execute or      │                            │
│                    │    generate new    │                            │
│                    └────────────────────┘                            │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘
```

### Sensor Feedback Interaction

Sensors can send feedback to each other via NATS:

```python
# Sensor feedback system
class SensorFeedback:
    """Allows sensors to provide context to each other."""

    def __init__(self, bus: EventBus):
        self.bus = bus

    async def send_feedback(
        self,
        from_sensor: str,
        to_sensor: str,
        feedback: dict
    ):
        """Send feedback from one sensor to another."""
        await self.bus.publish(
            f"sensor.feedback.{from_sensor}.{to_sensor}",
            {
                "from": from_sensor,
                "to": to_sensor,
                "feedback": feedback,
                "timestamp": int(time.time() * 1000)
            }
        )

# Example: Camera tells Sound sensor where to focus
await feedback.send_feedback(
    from_sensor="camera",
    to_sensor="microphone",
    feedback={
        "type": "focus_direction",
        "direction": "left",
        "reason": "human_detected_at_angle_45"
    }
)

# Example: Touch sensor tells Camera about physical contact
await feedback.send_feedback(
    from_sensor="touch",
    to_sensor="camera",
    feedback={
        "type": "contact_alert",
        "location": "left_arm",
        "force": 2.5,
        "reason": "human_guiding_arm"
    }
)
```

### Sensor Detection and Adaptation

The system detects available sensors at startup and adapts:

```python
class SensorManager:
    """Detects and manages available sensors."""

    SENSOR_PRIORITY = {
        "camera": 1.0,
        "microphone": 0.8,
        "touch": 0.6,
        "smell": 0.4,
    }

    def __init__(self):
        self.available_sensors: dict[str, Sensor] = {}
        self.bus = EventBus()

    async def detect_sensors(self) -> dict[str, Sensor]:
        """Probe for available hardware sensors."""
        detected = {}

        # Try each sensor type
        for sensor_type in self.SENSOR_PRIORITY.keys():
            sensor = await self._probe_sensor(sensor_type)
            if sensor:
                detected[sensor_type] = sensor
                logger.info(f"Sensor detected: {sensor_type}")
            else:
                logger.warn(f"Sensor not available: {sensor_type}")

        self.available_sensors = detected
        return detected

    async def adapt_learning_mode(self):
        """Adapt learning capabilities based on available sensors."""
        available = set(self.available_sensors.keys())

        if "camera" in available:
            # Full demonstration learning possible
            self.learning_mode = "full_demonstration"
        elif "touch" in available:
            # Physical guidance only
            self.learning_mode = "physical_guidance"
        elif "microphone" in available:
            # Voice instruction only
            self.learning_mode = "voice_instruction"
        else:
            # No real-time learning, use stored patterns
            self.learning_mode = "pattern_playback"
```

---

## Demonstration Learning Pipeline

The system learns tasks by watching humans and receiving feedback.

### Learning Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                    DEMONSTRATION LEARNING                            │
│                                                                      │
│  PHASE 1: WATCH                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  Human demonstrates task (e.g., picking up cup)             │    │
│  │  Camera: Tracks pose, joint angles, trajectory              │    │
│  │  Touch: Records if human guides robot arm                   │    │
│  │  Sound: Captures verbal explanations                        │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              │                                       │
│                              ▼                                       │
│  PHASE 2: IMITATE                                                    │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  Robot attempts to replicate observed movement              │    │
│  │  Motor commands generated from pose sequence                │    │
│  │  All proposals pass through Kernel                          │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              │                                       │
│                              ▼                                       │
│  PHASE 3: FEEDBACK                                                   │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  Human provides correction feedback                         │    │
│  │  - Voice: "No, slower", "Good", "Higher"                   │    │
│  │  - Touch: Physically adjusts robot's position              │    │
│  │  - Camera: Gestures (thumbs up/down)                       │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              │                                       │
│                              ▼                                       │
│  PHASE 4: REFINE                                                     │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  Adjust movement based on feedback                          │    │
│  │  Loop back to IMITATE until success                        │    │
│  │  Max iterations before escalating to Meta-Programmer        │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              │                                       │
│                              ▼                                       │
│  PHASE 5: SAVE                                                       │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  Save learned task for future use                          │    │
│  │  - Code generated by Meta-Programmer                       │    │
│  │  - Stored in /data/tasks/{task_id}/                        │    │
│  │  - Indexed in vector DB for semantic lookup                │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Task Storage Pattern

Tasks are stored as code files with vector DB references:

```
/data/tasks/
├── pick_up_cup_001/
│   ├── task.py              # Generated by Meta-Programmer
│   ├── parameters.json      # Learned parameters (heights, speeds)
│   ├── trajectory.json      # Recorded joint trajectories
│   ├── metadata.json        # Semantic tags, success rate
│   └── tests/
│       └── test_task.py     # Auto-generated tests
│
├── wave_hand_002/
│   └── ...
│
└── open_door_003/
    └── ...
```

**Vector DB indexes task metadata for semantic lookup:**

```python
# Qdrant collection for task lookup
class TaskIndex:
    """Index learned tasks for semantic search."""

    def __init__(self, qdrant_client: QdrantClient):
        self.client = qdrant_client
        self.collection = "learned_tasks"

    async def index_task(self, task: LearnedTask):
        """Add task to vector index."""
        # Embed task description + metadata
        embedding = await self.embed(
            f"{task.name}: {task.description}. "
            f"Objects: {task.objects}. "
            f"Context: {task.context}"
        )

        await self.client.upsert(
            collection_name=self.collection,
            points=[{
                "id": task.id,
                "vector": embedding,
                "payload": {
                    "name": task.name,
                    "code_path": task.code_path,  # /data/tasks/{task_id}/
                    "success_rate": task.success_rate,
                    "learned_at": task.timestamp,
                    "objects": task.objects,
                    "context": task.context,
                }
            }]
        )

    async def find_task(self, query: str, threshold: float = 0.7) -> Optional[TaskMatch]:
        """Find matching task by semantic similarity."""
        embedding = await self.embed(query)

        results = await self.client.search(
            collection_name=self.collection,
            query_vector=embedding,
            limit=1,
            score_threshold=threshold
        )

        if results:
            return TaskMatch(
                task_id=results[0].id,
                code_path=results[0].payload["code_path"],
                confidence=results[0].score
            )
        return None
```

### Coordinator Task Lookup Flow

The Coordinator follows a strict "update-first" policy: always check for existing tasks and update them rather than creating new ones. Creation only happens when nothing related exists.

```python
class Coordinator:
    """Orchestrates task execution with vector DB lookup."""

    def __init__(self, task_index: TaskIndex, meta_programmer: MetaProgrammer):
        self.task_index = task_index
        self.meta_programmer = meta_programmer

    async def execute_request(self, request: str) -> ExecutionResult:
        """
        Execute a task request.

        Priority:
        1. Execute existing task if high confidence match
        2. Update/adapt existing task if medium confidence match
        3. Create new task ONLY if nothing related exists
        """

        # Step 1: Check vector DB for existing task
        match = await self.task_index.find_task(request)

        if match and match.confidence > 0.85:
            # HIGH CONFIDENCE: Found existing task - execute directly
            logger.info(f"Found cached task: {match.task_id} (conf: {match.confidence})")
            task_code = await self.load_task_code(match.code_path)
            return await self.execute_task(task_code)

        elif match and match.confidence > 0.6:
            # MEDIUM CONFIDENCE: Similar task exists - try to update/adapt it
            logger.info(f"Adapting similar task: {match.task_id}")
            existing_code = await self.load_task_code(match.code_path)

            try:
                # Try to update the existing task
                adapted_code = await self.meta_programmer.adapt_task(
                    existing_code,
                    request,
                    update_in_place=True  # Update the original task
                )
                # Save the updated task back to same location
                await self.save_task_code(match.code_path, adapted_code)
                return await self.execute_task(adapted_code)
            except AdaptationError as e:
                # Adaptation failed - fall through to create new
                logger.warn(f"Adaptation failed: {e}, creating new task")
                # Continue to Step 2

        # Step 2: No match OR adaptation failed - generate new task
        logger.info(f"Generating new task for: {request}")
        new_task = await self.meta_programmer.generate_task(request)
        await self.task_index.index_task(new_task)  # Cache for future
        return await self.execute_task(new_task.code)
```

**Task Adaptation vs Creation Policy:**

| Confidence | Action | Behavior |
|------------|--------|----------|
| > 0.85 (High) | Execute | Use existing task directly |
| 0.6 - 0.85 (Medium) | Update | Try to adapt existing task in-place |
| < 0.6 (Low) | Create | Generate new task (only if adaptation fails or no match) |

> **Key principle**: The system always prefers updating existing tasks over creating new ones. This ensures the knowledge base evolves rather than fragmenting into many similar tasks.

**Adaptation Failure Handling:**

When adaptation fails (e.g., existing task is too different, or changes would break existing behavior):

1. Log the adaptation failure with reason
2. Create a new task as fallback
3. Keep both tasks in the index (the original unchanged, new one for the new use case)
4. Over time, if the new task proves more useful, it may replace the old one

---

## LLM Caching for Autopilot Mode

To avoid repeatedly calling the local LLM for the same queries, results are cached in vector DB:

### LLM Response Cache

```python
class LLMCache:
    """Cache LLM responses in vector DB for 'autopilot' mode."""

    def __init__(self, qdrant: QdrantClient, ollama: Ollama):
        self.qdrant = qdrant
        self.ollama = ollama
        self.collection = "llm_cache"

    async def query(self, prompt: str, use_cache: bool = True) -> LLMResponse:
        """Query LLM with optional cache lookup."""

        if use_cache:
            # Check vector cache first
            cached = await self._find_cached(prompt)
            if cached and cached.confidence > 0.95:
                logger.debug(f"Cache hit for prompt (conf: {cached.confidence})")
                return LLMResponse(
                    text=cached.response,
                    from_cache=True,
                    cache_confidence=cached.confidence
                )

        # Cache miss - call local LLM
        logger.debug("Cache miss, calling Ollama")
        response = await self.ollama.generate(prompt)

        # Cache the response for future
        await self._cache_response(prompt, response)

        return LLMResponse(
            text=response,
            from_cache=False,
            cache_confidence=0.0
        )

    async def _find_cached(self, prompt: str) -> Optional[CacheHit]:
        """Find cached response by prompt similarity."""
        embedding = await self.embed(prompt)

        results = await self.qdrant.search(
            collection_name=self.collection,
            query_vector=embedding,
            limit=1,
            score_threshold=0.9
        )

        if results:
            return CacheHit(
                response=results[0].payload["response"],
                confidence=results[0].score
            )
        return None

    async def _cache_response(self, prompt: str, response: str):
        """Store LLM response in vector cache."""
        embedding = await self.embed(prompt)

        await self.qdrant.upsert(
            collection_name=self.collection,
            points=[{
                "id": str(uuid.uuid4()),
                "vector": embedding,
                "payload": {
                    "prompt": prompt,
                    "response": response,
                    "timestamp": int(time.time() * 1000),
                    "model": self.ollama.model_id
                }
            }]
        )
```

### Cache Invalidation Strategy

The LLM cache needs invalidation when underlying conditions change:

```python
# coordinator/cache_invalidation.py
class CacheInvalidator:
    """
    Invalidates cached LLM responses when conditions change.
    """

    def __init__(self, llm_cache: LLMCache, bus: EventBus):
        self.cache = llm_cache
        self.bus = bus

    async def start(self):
        """Subscribe to events that trigger invalidation."""
        # Code changes invalidate related cache entries
        await self.bus.subscribe("code.deployed", self.on_code_deployed)
        # Human overrides may invalidate cached decisions
        await self.bus.subscribe("override.applied", self.on_override_applied)
        # Task updates invalidate task-related cache
        await self.bus.subscribe("learning.task.saved", self.on_task_saved)

    async def on_code_deployed(self, msg):
        """Invalidate cache when code changes."""
        target_path = msg.data.get("target_path")
        # Find cache entries that reference this code
        await self.cache.invalidate_by_metadata(
            filter={"related_code": target_path}
        )
        logger.info(f"Invalidated cache entries for: {target_path}")

    async def on_override_applied(self, msg):
        """Invalidate cache when human override is applied."""
        parameter_path = msg.data.get("parameter_path")
        # Invalidate entries that use this parameter
        await self.cache.invalidate_by_metadata(
            filter={"parameter_refs": parameter_path}
        )

    async def on_task_saved(self, msg):
        """Invalidate cache when new task is learned."""
        task_name = msg.data.get("name")
        # Invalidate entries about similar tasks (they might now hit cache)
        await self.cache.invalidate_by_similarity(
            query=task_name,
            threshold=0.7  # Invalidate similar queries
        )

    async def periodic_cleanup(self):
        """Periodic cleanup of stale cache entries."""
        while True:
            await asyncio.sleep(3600)  # Every hour

            # Remove entries older than 7 days
            await self.cache.invalidate_by_age(max_age_days=7)

            # Remove entries with no hits in 3 days
            await self.cache.invalidate_unused(unused_days=3)
```

**Invalidation Triggers:**

| Event | What's Invalidated | Why |
|-------|-------------------|-----|
| Code deployed | Cache entries referencing that code | Behavior changed |
| Override applied | Cache entries using that parameter | Constraints changed |
| Task saved | Similar task queries | New task might be better match |
| 7 days elapsed | Old entries | Prevent stale responses |
| 3 days unused | Unused entries | Free up space |

**Manual Invalidation:**

```python
# Force clear specific cache entries
await llm_cache.invalidate_by_prompt_pattern("*servo*")

# Clear entire cache (use sparingly)
await llm_cache.clear_all()
```

### Autopilot Mode

```python
class AutopilotController:
    """
    Autopilot mode uses cached responses whenever possible,
    only falling back to live LLM for novel situations.
    """

    def __init__(self, llm_cache: LLMCache, task_index: TaskIndex):
        self.llm_cache = llm_cache
        self.task_index = task_index
        self.autopilot_enabled = True

    async def handle_observation(self, obs: Observation) -> ActionProposal:
        """Process observation, preferring cached knowledge."""

        # Step 1: Check if we have a cached task for this situation
        task = await self.task_index.find_task(self._describe_situation(obs))

        if task and task.confidence > 0.9:
            # High confidence match - use cached task
            return await self._execute_cached_task(task)

        # Step 2: Check LLM cache for planning
        planning_prompt = self._create_planning_prompt(obs)
        response = await self.llm_cache.query(planning_prompt, use_cache=self.autopilot_enabled)

        if response.from_cache:
            logger.info(f"Autopilot: using cached plan (conf: {response.cache_confidence})")
        else:
            logger.info("Autopilot: generated new plan via LLM")

        return self._parse_plan(response.text)
```

### NATS Subjects for Learning System

| Subject | Publisher | Subscriber | Payload |
|---------|-----------|------------|---------|
| `sensor.observation.{sensor_id}` | Sensors | Coordinator, Memory | `Observation` |
| `sensor.feedback.{from}.{to}` | Sensors | Sensors | `SensorFeedback` |
| `learning.demo.start` | Human Interface | Learning Controller | `DemoStartEvent` |
| `learning.demo.frame` | Camera | Learning Controller | `DemoFrame` |
| `learning.feedback` | Human Interface | Learning Controller | `HumanFeedback` |
| `learning.task.saved` | Learning Controller | Task Index | `LearnedTask` |
| `task.lookup.request` | Coordinator | Task Index | `TaskLookup` |
| `task.lookup.result.{trace_id}` | Task Index | Coordinator | `TaskMatch` |
| `llm.cache.hit` | LLM Cache | Monitor | `CacheHitEvent` |
| `llm.cache.miss` | LLM Cache | Monitor | `CacheMissEvent` |

---

## External API Learning

The system can learn from external APIs while maintaining safety through Kernel validation.

### External Knowledge Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                    EXTERNAL API LEARNING                             │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  1. Knowledge Gap Detected                                   │    │
│  │     "How do I interface with this new sensor?"              │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              │                                       │
│                              ▼                                       │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  2. Check Local Knowledge First                              │    │
│  │     - Vector DB: similar sensors/patterns                    │    │
│  │     - Task Index: related tasks                              │    │
│  │     - Memory: past experiences                               │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              │                                       │
│           ┌──────────────────┴──────────────────┐                   │
│           │                                      │                   │
│        Found                               Not Found                 │
│           │                                      │                   │
│           ▼                                      ▼                   │
│     Use Local                    ┌─────────────────────────────┐    │
│     Knowledge                    │  3. Request Kernel Approval │    │
│                                  │     for External API Query  │    │
│                                  │     - API endpoint          │    │
│                                  │     - Query parameters      │    │
│                                  │     - Data to send          │    │
│                                  └─────────────────────────────┘    │
│                                              │                       │
│                              ┌───────────────┴───────────────┐      │
│                              │                               │      │
│                           ALLOW                           DENY      │
│                              │                               │      │
│                              ▼                               ▼      │
│                  ┌──────────────────┐             Mark gap   │      │
│                  │  4. Call External│             for human  │      │
│                  │     API          │             review     │      │
│                  │  - Claude API    │                        │      │
│                  │  - OpenAI API    │                        │      │
│                  │  - Documentation │                        │      │
│                  └──────────────────┘                        │      │
│                              │                                       │
│                              ▼                                       │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  5. Compare with Local Knowledge                             │    │
│  │     - Conflict detection                                    │    │
│  │     - Confidence scoring                                    │    │
│  │     - Human escalation if contradicts existing beliefs      │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              │                                       │
│                              ▼                                       │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  6. Kernel Validates External Knowledge                      │    │
│  │     - Safety check on learned information                   │    │
│  │     - No dangerous patterns                                 │    │
│  │     - No code injection                                     │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              │                                       │
│                              ▼                                       │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  7. Store Validated Knowledge                                │    │
│  │     - Add to vector DB                                      │    │
│  │     - Update belief graph                                   │    │
│  │     - Cache for future queries                              │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### External API Manager

```python
class ExternalAPIManager:
    """Manages external API queries with Kernel approval."""

    def __init__(self, kernel: Kernel, local_knowledge: LocalKnowledge):
        self.kernel = kernel
        self.local = local_knowledge

    async def query_external(
        self,
        query: str,
        api: str = "claude",
        purpose: str = "knowledge_gap"
    ) -> ExternalQueryResult:
        """Query external API after checking local and getting Kernel approval."""

        # Step 1: Check local knowledge first
        local_result = await self.local.search(query)
        if local_result.confidence > 0.85:
            return ExternalQueryResult(
                source="local",
                data=local_result.data,
                confidence=local_result.confidence
            )

        # Step 2: Request Kernel approval for external query
        proposal = ExternalQueryProposal(
            trace_id=generate_trace_id(),
            api=api,
            query=query,
            purpose=purpose,
            data_to_send=None  # No sensitive data
        )

        decision = await self.kernel.evaluate(proposal)

        if decision.type == KernelDecisionType.DENY:
            logger.warn(f"Kernel denied external query: {decision.reason}")
            return ExternalQueryResult(source="denied", error=decision.reason)

        # Step 3: Execute external query
        external_result = await self._call_api(api, query)

        # Step 4: Compare with local knowledge
        comparison = await self._compare_knowledge(local_result, external_result)

        if comparison.has_conflict:
            # Escalate to human if external contradicts local
            resolution = await self._resolve_conflict(comparison)
            external_result = resolution.chosen_knowledge

        # Step 5: Validate external knowledge through Kernel
        validated = await self.kernel.validate_external_knowledge(external_result)

        if validated.safe:
            # Store in local knowledge base
            await self.local.store(query, external_result, source=api)

        return ExternalQueryResult(
            source=api,
            data=validated.data,
            confidence=validated.confidence
        )

    async def _resolve_conflict(self, comparison: KnowledgeComparison) -> ConflictResolution:
        """
        Resolve conflict between local and external knowledge.
        Human makes final decision; both sources are stored with trust scores.
        """
        # Present conflict to human via Dashboard approval flow
        resolution = await self.dashboard.request_conflict_resolution(
            local_knowledge=comparison.local,
            external_knowledge=comparison.external,
            conflict_type=comparison.conflict_type,
            conflict_description=comparison.description
        )

        # Store BOTH knowledge sources with trust scores
        await self.local.store_with_trust(
            comparison.local,
            trust_score=resolution.local_trust,
            human_validated=True
        )
        await self.local.store_with_trust(
            comparison.external,
            trust_score=resolution.external_trust,
            human_validated=True
        )

        # Return the human's chosen knowledge
        return ConflictResolution(
            chosen_knowledge=resolution.chosen,
            reason=resolution.human_explanation,
            both_stored=True
        )
```

### Conflict Resolution Strategy

When external API knowledge conflicts with local knowledge, the system follows this resolution process:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    KNOWLEDGE CONFLICT DETECTED                        │
│                                                                       │
│  Local: "Servo max angle is 90°"                                     │
│  External (Claude): "Servo max angle is 180°"                        │
│                                                                       │
│                              │                                        │
│                              ▼                                        │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │              HUMAN CONFLICT RESOLUTION                        │    │
│  │                                                               │    │
│  │  Dashboard presents both options to human:                    │    │
│  │  - "Local knowledge says 90° (from demonstration)"            │    │
│  │  - "External API says 180° (from Claude)"                     │    │
│  │  - "Which is correct for this servo?"                        │    │
│  │                                                               │    │
│  │  Human selects: "Local is correct - this specific servo"     │    │
│  │  Human explains: "My servo has a hardware stop at 90°"        │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              │                                        │
│                              ▼                                        │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │              STORE BOTH WITH TRUST SCORES                     │    │
│  │                                                               │    │
│  │  Local: trust=0.95 (human-validated, hardware-specific)      │    │
│  │  External: trust=0.7 (general knowledge, may apply to other) │    │
│  │                                                               │    │
│  │  Both kept for different contexts:                            │    │
│  │  - Local used for "my servo"                                  │    │
│  │  - External used for "typical servos"                         │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

**Resolution Outcomes:**

| Outcome | Action | Trust Scores |
|---------|--------|--------------|
| Local correct | Use local, store both | Local: 0.95, External: 0.5 |
| External correct | Use external, store both | Local: 0.5, External: 0.95 |
| Both valid (context-dependent) | Keep both active | Both: 0.8 with context tags |
| Neither correct | Human provides correct value | Both: 0.3, Human: 1.0 |

**Why Store Both?**

1. **Context matters**: Local knowledge may be specific to your hardware, external may be general
2. **Audit trail**: Track where knowledge came from and why decisions were made
3. **Future learning**: System can learn when each source is more reliable
4. **Reversibility**: If human was wrong, can revisit later

### Unified SQLite Tables for Learning

```sql
-- Learned tasks storage
CREATE TABLE learning.tasks (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    code_path TEXT NOT NULL,           -- /data/tasks/{task_id}/
    trajectory_path TEXT,

    -- Learning metadata
    learned_from TEXT,                  -- 'demonstration', 'external_api', 'meta_programmer'
    demonstration_count INTEGER DEFAULT 0,
    success_count INTEGER DEFAULT 0,
    failure_count INTEGER DEFAULT 0,

    -- Vector reference
    embedding_ref TEXT,

    -- Timestamps
    created_at INTEGER,
    last_used_at INTEGER,
    last_updated_at INTEGER
);

-- LLM response cache
CREATE TABLE cache.llm_responses (
    id TEXT PRIMARY KEY,
    prompt_hash TEXT NOT NULL,          -- SHA-256 of prompt
    prompt TEXT NOT NULL,
    response TEXT NOT NULL,
    model TEXT,
    embedding_ref TEXT,

    -- Cache metadata
    hit_count INTEGER DEFAULT 0,
    created_at INTEGER,
    last_hit_at INTEGER
);

-- External API queries log
CREATE TABLE external.api_queries (
    id TEXT PRIMARY KEY,
    trace_id TEXT NOT NULL,
    api TEXT NOT NULL,                  -- 'claude', 'openai', etc.
    query TEXT NOT NULL,
    response TEXT,

    -- Kernel approval
    kernel_decision TEXT,
    kernel_trace_id TEXT,

    -- Comparison with local
    local_confidence REAL,
    external_confidence REAL,
    conflict_detected BOOLEAN DEFAULT FALSE,

    -- Timestamps
    requested_at INTEGER,
    responded_at INTEGER
);
```

---

## Scaling to Multiple Machines (Future)

When you're ready to scale beyond your M2:

```yaml
# docker-compose.prod.yml (overlay)
services:
  kernel:
    deploy:
      replicas: 3
      placement:
        constraints:
          - node.labels.zone == secure

  meta-programmer:
    deploy:
      replicas: 2

  nats:
    deploy:
      replicas: 3
      mode: global
```

Or migrate to Kubernetes:

```bash
# Convert docker-compose to k8s manifests
kompose convert -f docker-compose.yml -o k8s/
```

---

## Quick Start

```bash
# 1. Clone and enter directory
cd ActiveLearningAI

# 2. Pull Ollama model (one-time)
docker compose run --rm ollama ollama pull deepseek-coder:6.7b

# 3. Start all services
docker compose up -d

# 4. Check health
docker compose ps
curl http://localhost:8222/healthz  # NATS health
curl http://localhost:6333/health   # Qdrant health
curl http://localhost:11434/api/tags # Ollama models

# 5. View logs
docker compose logs -f meta-programmer

# 6. Stop everything
docker compose down
```

---

## File Structure (Updated)

```
/ActiveLearningAI
├── docker-compose.yml          # Main orchestration
├── docker-compose.override.yml # Local dev overrides
├── .env                        # Environment variables
│
├── /neuromorphic               # Spiking neural network core (the brain)
│   ├── Dockerfile
│   ├── pyproject.toml          # Uses uv for deps
│   └── src/neuromorphic/
│       ├── network.py          # SNN network, regions, synapse groups
│       ├── service.py          # NATS service wrapper
│       ├── config.py           # All brain configuration
│       ├── regions.py          # Brain region implementations
│       ├── synapses.py         # SynapseGroup with STDP, eligibility, BCM
│       ├── neuromodulation.py  # 4-channel neuromodulation + critical periods
│       ├── mujoco_body.py      # MuJoCo virtual body (physics sim)
│       ├── motor_feedback_adapter.py  # Motor feedback loop
│       ├── decoding.py         # Speech, cognitive, motor decoders
│       ├── encoding.py         # Sensory encoding pipeline
│       └── ...
│
├── /sdk                        # Python SDK (open)
│   ├── Dockerfile
│   ├── pyproject.toml
│   └── sdk/
│       ├── __init__.py
│       ├── core.py            # Types, EventBus
│       ├── sensors.py         # Sensor plugin base
│       ├── actuators.py       # Actuator plugin base
│       └── nats_client.py     # NATS wrapper
│
├── /sensory-gateway            # Host-side sensor discovery + streaming
│   ├── discovery.py           # Auto-detect cameras, mics, serial
│   ├── aggregating_bus.py     # Aggregation (1400 obs/s -> 4/s)
│   ├── auditory_stm.py        # Auditory short-term memory
│   └── sensors/               # Per-modality sensor plugins
│
├── /dashboard                  # Engram Dashboard (standalone FastAPI + vanilla JS)
│   ├── Dockerfile
│   ├── src/dashboard/api.py   # REST + WebSocket endpoints
│   └── static/                # Frontend (app.js, index.html, style.css)
│
├── /brain-viz                  # 3D brain visualization (Three.js)
│   ├── index.html
│   ├── brain-scene.js
│   └── demos/                 # Interactive demos (walking, reaction, concepts, etc.)
│
├── /planner                    # Planner service (open)
│   ├── Dockerfile
│   └── planner/
│       └── ...
│
├── /memory                     # Memory service (open)
│   ├── Dockerfile
│   └── memory/
│       └── ...
│
├── /beliefs                    # Belief graph (open)
│   ├── Dockerfile
│   └── beliefs/
│       └── ...
│
├── /kernel                     # Moral Kernel (closed)
│   ├── Dockerfile
│   └── kernel/
│       └── ...
│
├── /meta-programmer            # Meta-Programmer (closed)
│   ├── Dockerfile
│   └── meta_programmer/
│       ├── __init__.py
│       ├── orchestrator.py    # Agent orchestration
│       ├── agents/
│       │   ├── codegen.py
│       │   ├── refactor.py
│       │   ├── tester.py
│       │   └── deployer.py
│       ├── sandbox.py         # Container sandbox manager
│       └── tools/
│           ├── kernel_gated.py
│           └── vector_search.py
│
├── /safety-supervisor          # Safety Supervisor (closed)
│   ├── Dockerfile
│   └── ...
│
├── /deploy                     # Deployment configs
│   ├── docker-compose.1m.yml  # 1M neuron deployment overlay
│   └── nats-1m.conf           # NATS config for 1M scale
│
├── /sandbox                    # Sandbox base image (Stage 1)
│   └── Dockerfile
│
├── /test-runner                # Test Runner container (Stage 2)
│   ├── Dockerfile
│   └── test_runner/
│       ├── __init__.py
│       ├── runner.py          # TestRunner class
│       ├── mocks/
│       │   ├── sensors.py     # MockSensor classes
│       │   └── actuators.py   # MockActuator classes
│       └── scenarios/         # Test scenario definitions
│
├── /data                       # Mounted volumes (gitignored)
│   ├── /sqlite                 # Unified database + synapse .npy files
│   ├── /vectors                # Qdrant storage
│   ├── /staging                # Code staging area
│   ├── /models                 # Ollama model cache
│   ├── /tasks                  # Learned task code
│   ├── /plugins                # Deployed plugins
│   └── /overrides              # Human override code
│
└── /docs
    ├── ARCHITECTURE.md        # This file
    ├── ROADMAP.md             # Vision + future phases
    ├── BUSINESS-MODEL.md      # Revenue strategy + investor materials
    ├── META-PROGRAMMER.md     # Meta-Programmer subsystem spec
    ├── SENSORY-GATEWAY.md     # Gateway architecture + sensor types
    └── ...
```
