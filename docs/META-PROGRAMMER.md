# Meta-Programmer: Self-Evolution Agent System

> **Status**: Open source (MIT) — part of the Engram cognitive stack
> **Language**: Python 3.11+
> **Container**: `meta-programmer` (Docker)
> **Dependencies**: Moral Kernel, Memory (Qdrant), Belief Graph, Validator
> **LLM**: Local via Ollama (deepseek-coder, codellama, etc.)
> **Human-in-the-loop**: DEFER decisions routed to Dashboard via NATS

The **Meta-Programmer** is the system's self-evolution capability—an orchestration layer that can write, refactor, test, and deploy code to extend or improve the robot's capabilities. Every action it takes must pass through the **Moral Kernel**, making self-modification safe by design.

**Key principle**: The Meta-Programmer decides how to interface with new hardware (Arduino, Pi, sensors, etc.) by generating the necessary adapter code itself.

> ## ⚠️ Implementation Status (read first)
>
> This document describes the **target design**. Several capabilities below are
> **not yet implemented** as written. As of the latest audit:
>
> **Implemented:** code generation via Ollama, staging, submission to the Kernel
> for approval, and the `SandboxManager` Docker configuration (no-network,
> read-only FS, mem/CPU/pid limits).
>
> **Not yet implemented / aspirational (see [ROADMAP.md](../ROADMAP.md)):**
> - The `activelearning-sandbox` image is built from `sandbox/Dockerfile`
>   (added in P0); making missing-sandbox **fail-closed** and adding
>   `cap_drop=ALL` / `no-new-privileges` is **Phase 1.3**.
> - The decision bus is **not yet authenticated/signed** — the "no path around
>   the Kernel, only through it" guarantee is a design goal hardened in
>   **Phase 1.1–1.2**, not an enforced property today.
> - Only the first ~500 chars of generated code are pattern-scanned; full-artifact
>   AST taint analysis is **Phase 1.5**.
> - The multi-agent refactor pipeline, Stage-2 Test-Runner integration,
>   post-deploy auto-rollback, and the 24h approval-timeout-as-DENY are **not yet
>   built** (Phase 1.9 / 4.1).
> - The Coordinator task-execution path does **not** currently route through the
>   Kernel (Phase 1.6).
>
> Treat the sections below as the destination, and ROADMAP.md Phase 1 as the
> work required to get there. Do not build new autonomy on top of the gate until
> Phase 1 lands.

---

## Core Principle: Inviolable Kernel Constraint

```
┌─────────────────────────────────────────────────────────────────┐
│  Even code written by the Meta-Programmer to bypass the        │
│  Kernel still requires Kernel approval to EXECUTE.             │
│                                                                 │
│  The Kernel is the immutable gatekeeper. There is no path      │
│  around it—only through it.                                    │
└─────────────────────────────────────────────────────────────────┘
```

This creates an unbreakable safety loop:
1. Meta-Programmer proposes code change → Kernel evaluates → ALLOW/TRANSFORM/DENY/DEFER
2. Meta-Programmer proposes to execute new code → Kernel evaluates → ALLOW/TRANSFORM/DENY/DEFER
3. If new code attempts to circumvent Kernel → Kernel evaluates that attempt → DENY

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              DOCKER NETWORK                                 │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                         NATS (Message Bus)                            │ │
│  │                      nats://nats:4222                                 │ │
│  └─────────────────────────────────┬─────────────────────────────────────┘ │
│                                    │                                       │
│         ┌──────────────────────────┼──────────────────────────┐           │
│         │                          │                          │           │
│         ▼                          ▼                          ▼           │
│  ┌─────────────┐           ┌─────────────┐            ┌─────────────┐     │
│  │   Kernel    │           │    Meta-    │            │   Ollama    │     │
│  │ Container   │◄─────────►│ Programmer  │◄──────────►│  (Local     │     │
│  │             │  NATS     │  Container  │  HTTP      │    LLM)     │     │
│  │ - Evaluates │           │             │            │             │     │
│  │ - ALLOW/    │           │ - CodeGen   │            │ deepseek-   │     │
│  │   DENY/     │           │ - Testing   │            │ coder:6.7b  │     │
│  │   TRANSFORM/│           │ - Deploy    │            │             │     │
│  │   DEFER     │           │             │            │             │     │
│  └─────────────┘           └──────┬──────┘            └─────────────┘     │
│                                   │                                       │
│                                   │ Docker API                            │
│                                   ▼                                       │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │                     SANDBOX ZONE (Ephemeral)                       │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐                         │  │
│  │  │Sandbox-1 │  │Sandbox-2 │  │Sandbox-3 │   Created per test run  │  │
│  │  │ No network│  │ No network│  │ No network│   Destroyed after use  │  │
│  │  │ Read-only │  │ Read-only │  │ Read-only │   Resource-limited     │  │
│  │  └──────────┘  └──────────┘  └──────────┘                         │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                                   │                                       │
│                                   │ If tests pass + Kernel ALLOW          │
│                                   ▼                                       │
│                                   │                                       │
│                                   │ DEFER → Dashboard (human review)      │
│                                   ▼                                       │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │                         LIVE VOLUMES                               │  │
│  │  /data/plugins/    - Deployed sensor/actuator code                 │  │
│  │  /data/strategies/ - Planning strategies                           │  │
│  │  /data/adapters/   - Hardware adapters (Arduino, Pi, etc.)         │  │
│  └────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Local LLM Integration

The Meta-Programmer uses **local models via Ollama** for code generation - no external API calls required. It calls the Ollama HTTP API directly:

```python
# meta-programmer/src/meta_programmer/agents.py
class MetaProgrammerAgents:
    def __init__(self):
        self.model_name = "deepseek-coder:6.7b"
        self.ollama_url = "http://ollama:11434"

    async def generate_code(self, trace_id: str, description: str, context: dict) -> dict:
        """Generate code via local LLM, then submit to Kernel for approval."""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.ollama_url}/api/generate",
                json={"model": self.model_name, "prompt": description}
            ) as response:
                result = await response.json()
        return result
```

**Recommended models (self-hosted):**

| Model | VRAM | Speed (M2) | Best For |
|-------|------|------------|----------|
| `deepseek-coder:6.7b` | 4GB | ~30 tok/s | Fast iteration, simple code |
| `deepseek-coder:33b` | 20GB | ~8 tok/s | Complex generation |
| `qwen2.5-coder:7b` | 4GB | ~25 tok/s | Good reasoning |
| `codellama:13b` | 8GB | ~15 tok/s | Balanced |

**When to use external APIs (optional):**
- Complex multi-file refactoring where local model struggles
- Security review of generated code (second opinion)
- Human-escalated DEFER decisions needing explanation

---

## Two-Stage Testing Pipeline

Generated code goes through a two-stage testing pipeline before deployment. See [ARCHITECTURE.md](ARCHITECTURE.md#testing-pipeline) for full details.

### Stage 1: Unit Tests (Ephemeral Sandboxes)

Fast, isolated unit tests in ephemeral containers (~5 seconds):

```python
import docker

client = docker.from_env()

async def run_in_sandbox(code: str, tests: str) -> SandboxResult:
    """Execute generated code in isolated container."""
    container = client.containers.run(
        image="sandbox-python:latest",
        command=["python", "-c", tests],
        environment={"CODE_UNDER_TEST": code},

        # SECURITY: Complete isolation
        network_disabled=True,         # No network access
        read_only=True,                # Read-only root filesystem
        tmpfs={"/tmp": "size=100M"},   # Only /tmp writable

        # SECURITY: Resource limits
        mem_limit="512m",              # 512MB max memory
        cpu_period=100000,
        cpu_quota=50000,               # 50% of one CPU
        pids_limit=100,                # Max 100 processes

        # SECURITY: Drop all capabilities
        cap_drop=["ALL"],
        security_opt=["no-new-privileges:true"],

        # Cleanup
        remove=True,                   # Delete container after
        timeout=30,                    # 30 second max runtime
    )

    return SandboxResult(
        exit_code=container.attrs["State"]["ExitCode"],
        stdout=container.logs(stdout=True, stderr=False),
        stderr=container.logs(stdout=False, stderr=True),
    )
```

### Stage 2: Integration Tests (Test Runner Container)

After Stage 1 passes, code is tested in the Test Runner container with mock sensors/actuators (~30 seconds):

```python
# Test Runner verifies plugin integrates with SDK
class TestRunner:
    def __init__(self):
        self.mock_sensors = {
            "camera": MockSensor("camera"),
            "gpio": MockSensor("gpio"),
        }
        self.mock_actuators = {
            "servo": MockActuator("servo"),
            "led": MockActuator("led"),
        }

    async def run_integration_test(self, plugin_code: str) -> TestResult:
        """Test plugin with mock environment and NATS message flow."""
        # Load plugin in isolated zone
        # Emit test observations via mock sensors
        # Verify actuators receive expected outcomes
        # Check NATS message flow
        ...
```

**Test Runner provides:**
- Mock sensors (camera, GPIO, IMU, etc.)
- Mock actuators (servo, LED, motor, etc.)
- NATS connection for message flow testing
- SDK integration verification

### Why Two Stages?

| Stage 1: Sandbox | Stage 2: Test Runner |
|------------------|---------------------|
| Pure unit tests | Integration tests |
| No dependencies | Mock sensors/actuators |
| No network | NATS connected |
| ~5 seconds | ~30 seconds |
| Catches syntax, logic errors | Catches integration issues |

### Future: Stage 3 (Isaac Sim)

> **Note**: Isaac Sim requires NVIDIA GPU. Will be added as cloud-based option in the future.

When NVIDIA hardware is available, robotics simulation testing can be added:
- Full physics simulation
- Simulated cameras, LiDAR, IMU
- Collision detection
- Real-world behavior validation

---

## Agent Pipeline

The Meta-Programmer follows a sequential pipeline for code generation:

1. **Gap Analysis** - Identify what needs to be generated
2. **Code Generation** - Generate code via local LLM (Ollama)
3. **Kernel Evaluation** - Submit to Kernel for ALLOW/TRANSFORM/DENY/DEFER
4. **Testing** - Run tests in sandboxed containers
5. **Deployment** - Promote to live volumes (if approved)

On TRANSFORM, the pipeline loops back to step 2 (max 3 iterations). On DEFER, the request is routed to the Dashboard for human review.

> **Note**: The agent orchestration is currently a simplified sequential pipeline. Multi-agent team orchestration (with specialized CodeGen, Refactor, Tester, and Deployer agents) is planned but not yet implemented.

---

## NATS Message Flow

All communication happens via NATS pub/sub:

```python
import nats
import json

async def submit_to_kernel(proposal: CodeProposal) -> KernelDecision:
    """Submit code proposal to Kernel via NATS."""
    nc = await nats.connect("nats://nats:4222")

    # Publish proposal
    await nc.publish(
        "code.proposal",
        json.dumps(proposal.to_dict()).encode()
    )

    # Wait for decision on trace-specific subject
    sub = await nc.subscribe(f"code.decision.{proposal.trace_id}")
    msg = await sub.next_msg(timeout=30)

    return KernelDecision.from_dict(json.loads(msg.data))
```

**NATS subjects used by Meta-Programmer:**

| Subject | Direction | Payload |
|---------|-----------|---------|
| `code.proposal` | Publish | `CodeProposal` |
| `code.decision.{trace_id}` | Subscribe | `KernelDecision` |
| `approval.request` | Publish | `ApprovalRequest` |
| `approval.response.{id}` | Subscribe | `ApprovalResponse` |
| `system.health` | Publish | `HealthStatus` |

---

## Human-in-the-Loop (Dashboard)

When Kernel returns `DEFER`, the approval request is published to NATS and presented in the Dashboard for human review:

| Trigger | Action |
|---------|--------|
| Kernel returns `DEFER` | Publishes `approval.request` to NATS, Dashboard shows approval UI |
| Elevated scrutiny patterns detected | Route to human review |
| 3+ consecutive failures | Escalate for guidance |
| Deployment to live volumes | Always requires approval |

The Dashboard subscribes to `approval.request` and publishes `approval.response.{id}` when the human decides.

---

## Staging Flow

```
/data/staging/
├── pending/{trace_id}/       # CodeGen writes here
│   ├── code.py
│   ├── tests.py
│   └── metadata.json
│
├── testing/{trace_id}/       # After Kernel ALLOW
│   └── ...                   # Sandbox tests run here
│
├── human_review/{trace_id}/  # After Kernel DEFER
│   └── ...                   # Awaiting human approval via Dashboard
│
├── approved/{trace_id}/      # Tests passed, ready to deploy
│   └── ...
│
└── rejected/{trace_id}/      # Denied or failed (kept for audit)
    └── ...
```

**Flow:**
1. CodeGen → `/staging/pending/`
2. Kernel ALLOW → `/staging/testing/`
3. Kernel DEFER → `/staging/human_review/` → Dashboard approval
4. Tests pass → `/staging/approved/`
5. Deploy → `/data/plugins/` (live volume)

---

## Protected Paths

These paths are **immutable**—Meta-Programmer cannot modify them even with Kernel approval:

| Path | Reason |
|------|--------|
| `/kernel/*` | Moral Kernel source |
| `/safety-supervisor/*` | Safety heuristics |
| `/meta-programmer/orchestrator.py` | Own coordination logic |
| `/meta-programmer/agents/*.py` | Agent configurations |

**Can modify** (with Kernel approval):
| Path | Purpose |
|------|---------|
| `/data/plugins/*` | Sensor/actuator plugins |
| `/data/strategies/*` | Planning strategies |
| `/data/adapters/*` | Hardware adapters |

---

## Kernel Decision Types

| Decision | Meaning | Action |
|----------|---------|--------|
| `ALLOW` | Approved as-is | Proceed to testing |
| `TRANSFORM` | Approved with modifications | Apply transforms, resubmit |
| `DENY` | Rejected | Move to rejected, log |
| `DEFER` | Needs human review | Route to Dashboard |

### Elevated Scrutiny Triggers (→ DEFER)

- `import os`, `import subprocess`, `import socket`
- `eval()`, `exec()`, `compile()`
- `__import__()`, `importlib`
- File operations outside staging
- Network-related code
- Self-referential code (mentions "meta-programmer", "kernel")

---

## Audit Trail

Every action logged to unified SQLite:

```python
@dataclass
class CodeAuditEntry:
    trace_id: str
    timestamp: int
    agent: str              # 'CodeGen', 'Refactor', 'Tester', 'Deployer'
    action: str             # 'PROPOSE', 'EXECUTE', 'PROMOTE', 'ROLLBACK'
    target_path: str
    kernel_decision: str
    code_hash: str          # SHA-256 of generated code
    parent_hash: str | None # Hash of code being replaced
    test_results: dict | None
    human_required: bool
    human_approved: bool | None
```

---

## Failure Recovery

| Failure | Detection | Recovery |
|---------|-----------|----------|
| Tests fail | TesterAgent reports | Reject, try alternative |
| Runtime error after deploy | Health check | Auto-rollback |
| Generation loop | Max iterations | Break, escalate to human |
| Resource exhaustion | Container limits | Kill sandbox, deny |
| Kernel unavailable | Health check | SAFE_HALT mode |
| Approval timeout | 24h deadline | Treat as DENY |

---

## Guardrails Summary

1. **Kernel approval required** for all code actions
2. **Container sandbox** for all code execution
3. **Protected paths immutable** even with approval
4. **NATS isolation** between components
5. **Local LLM** for code generation (no API dependency)
6. **Human escalation** via Dashboard for DEFER
7. **Full audit trail** in unified SQLite
8. **Auto-rollback** on failures

---

## Task Code Generation and Storage

When the system learns a new task (via demonstration or synthesis), the Meta-Programmer generates executable code and stores it in a structured filesystem with vector DB indexing.

### Task Storage Architecture

```
/data/tasks/
├── pick_up_cup_001/
│   ├── task.py              # Main task implementation
│   ├── parameters.json      # Learned numeric parameters
│   ├── trajectory.json      # Joint trajectory data (if applicable)
│   ├── metadata.json        # Semantic info, tags, success rate
│   └── tests/
│       └── test_task.py     # Auto-generated tests
│
├── wave_hand_002/
│   ├── task.py
│   ├── parameters.json
│   └── ...
│
└── ...
```

### Task Code Template

Meta-Programmer generates task code following this pattern:

```python
# /data/tasks/{task_id}/task.py
from sdk.core import ActionProposal, Outcome
from sdk.actuators import ActuatorPlugin
from dataclasses import dataclass
import json
from pathlib import Path

@dataclass
class TaskParameters:
    """Learned parameters for this task."""
    approach_speed: float = 0.3
    grip_force: float = 1.5
    lift_height: float = 0.15
    # Loaded from parameters.json

class Task:
    """
    Task: Pick up cup
    Learned from: demonstration
    Success rate: 0.87
    """

    def __init__(self, task_dir: Path):
        self.task_dir = task_dir
        self.params = self._load_parameters()

    def _load_parameters(self) -> TaskParameters:
        """Load learned parameters from JSON."""
        params_file = self.task_dir / "parameters.json"
        if params_file.exists():
            with open(params_file) as f:
                return TaskParameters(**json.load(f))
        return TaskParameters()

    async def execute(self, context: dict) -> list[ActionProposal]:
        """
        Execute the task.
        All actions pass through Kernel before actuator execution.
        """
        proposals = []

        # Phase 1: Approach
        proposals.append(ActionProposal(
            trace_id=context["trace_id"],
            provenance="task.pick_up_cup",
            action={
                "type": "move_arm",
                "target": context["object_position"],
                "speed": self.params.approach_speed
            }
        ))

        # Phase 2: Grip
        proposals.append(ActionProposal(
            trace_id=context["trace_id"],
            provenance="task.pick_up_cup",
            action={
                "type": "gripper",
                "command": "close",
                "force": self.params.grip_force
            }
        ))

        # Phase 3: Lift
        proposals.append(ActionProposal(
            trace_id=context["trace_id"],
            provenance="task.pick_up_cup",
            action={
                "type": "move_arm",
                "target_relative": {"z": self.params.lift_height},
                "speed": self.params.approach_speed * 0.5
            }
        ))

        return proposals
```

### Vector DB Indexing

Task metadata is indexed in Qdrant for semantic lookup:

```python
# Meta-Programmer indexes new tasks
async def index_learned_task(task_id: str, task_dir: Path):
    """Index a learned task for semantic retrieval."""

    # Load metadata
    with open(task_dir / "metadata.json") as f:
        metadata = json.load(f)

    # Create semantic embedding
    description = f"""
    Task: {metadata['name']}
    Description: {metadata['description']}
    Objects: {', '.join(metadata.get('objects', []))}
    Context: {metadata.get('context', 'general')}
    Actions: {', '.join(metadata.get('action_types', []))}
    """

    embedding = await embed_text(description)

    # Store in Qdrant
    await qdrant.upsert(
        collection_name="learned_tasks",
        points=[{
            "id": task_id,
            "vector": embedding,
            "payload": {
                "name": metadata["name"],
                "code_path": str(task_dir),
                "success_rate": metadata.get("success_rate", 0.0),
                "learned_at": metadata.get("created_at"),
                "learned_from": metadata.get("learned_from", "unknown"),
                "objects": metadata.get("objects", []),
                "context": metadata.get("context"),
            }
        }]
    )
```

### Coordinator Task Lookup

The Coordinator finds and executes tasks using vector similarity:

```python
class TaskCoordinator:
    """Orchestrates task execution with vector DB lookup."""

    async def handle_request(self, request: str, context: dict) -> ExecutionResult:
        """
        Handle a task request:
        1. Search vector DB for matching task
        2. Load task code from filesystem
        3. Execute (all actions pass through Kernel)
        """

        # Step 1: Semantic search for matching task
        match = await self.vector_search(request)

        if match and match.confidence > 0.85:
            # High confidence - execute existing task
            task = await self.load_task(match.code_path)
            return await self.execute_task(task, context)

        elif match and match.confidence > 0.6:
            # Similar task exists - ask Meta-Programmer to adapt
            existing_task = await self.load_task(match.code_path)
            adapted_task = await self.meta_programmer.adapt_task(
                existing_task,
                request,
                context
            )
            return await self.execute_task(adapted_task, context)

        else:
            # No match - trigger Meta-Programmer to generate new task
            # May involve demonstration learning if sensors available
            knowledge_gap = KnowledgeGap(
                description=request,
                context=context,
                available_sensors=self.sensor_manager.available_sensors
            )
            new_task = await self.meta_programmer.fill_gap(knowledge_gap)
            return await self.execute_task(new_task, context)

    async def vector_search(self, query: str) -> Optional[TaskMatch]:
        """Search for task by semantic similarity."""
        embedding = await embed_text(query)

        results = await self.qdrant.search(
            collection_name="learned_tasks",
            query_vector=embedding,
            limit=1,
            score_threshold=0.5
        )

        if results:
            return TaskMatch(
                task_id=results[0].id,
                code_path=results[0].payload["code_path"],
                confidence=results[0].score,
                metadata=results[0].payload
            )
        return None
```

---

## Human Override Integration

Meta-Programmer respects human overrides stored in the system. When generating or adapting task code, it checks for relevant overrides:

```python
class OverrideAwareCodeGen:
    """CodeGen agent that respects human overrides."""

    async def generate_task(self, spec: TaskSpec, context: dict) -> GeneratedTask:
        """Generate task code, incorporating relevant human overrides."""

        # Step 1: Find relevant overrides from vector DB
        overrides = await self.find_relevant_overrides(spec.description)

        # Step 2: Build constraint list from overrides
        constraints = []
        for override in overrides:
            constraints.append(f"""
            # Human override ({override.timestamp}):
            # "{override.prompt}"
            # Parameter: {override.parameter_path} = {override.new_value}
            """)

        # Step 3: Generate code with override awareness
        prompt = f"""
        Generate Python task code for: {spec.description}

        IMPORTANT: The following human overrides MUST be respected:
        {chr(10).join(constraints)}

        These overrides have been validated by the Moral Kernel and represent
        human-specified operational parameters.
        """

        # Generate via local LLM
        code = await self.llm.generate(prompt)

        return GeneratedTask(
            code=code,
            respected_overrides=[o.id for o in overrides]
        )

    async def find_relevant_overrides(self, description: str) -> list[Override]:
        """Find human overrides relevant to this task."""
        embedding = await embed_text(description)

        results = await self.qdrant.search(
            collection_name="human_overrides",
            query_vector=embedding,
            limit=10,
            score_threshold=0.6
        )

        return [Override.from_payload(r.payload) for r in results]
```

---

## LLM Cache Integration

Meta-Programmer leverages the LLM cache for faster code generation:

```python
class CachedCodeGen:
    """CodeGen with LLM response caching."""

    def __init__(self, llm_cache: LLMCache):
        self.cache = llm_cache

    async def generate_code(self, spec: CodeSpec) -> str:
        """Generate code, using cache when possible."""

        # Build the prompt
        prompt = self._build_prompt(spec)

        # Check cache first (autopilot mode)
        response = await self.cache.query(prompt, use_cache=True)

        if response.from_cache:
            logger.info(f"Code generation cache hit (conf: {response.cache_confidence})")
            # Still validate cached code through Kernel
            return response.text

        # Cache miss - generate fresh
        logger.info("Code generation cache miss, using LLM")
        return response.text

    async def generate_similar_task(self, existing_code: str, new_spec: str) -> str:
        """
        Adapt existing task code for new use case.
        Higher cache hit rate for similar tasks.
        """
        prompt = f"""
        Adapt the following task code for a new use case:

        EXISTING CODE:
        ```python
        {existing_code}
        ```

        NEW REQUIREMENTS:
        {new_spec}

        Generate adapted Python code that reuses the structure
        but modifies parameters and logic as needed.
        """

        return (await self.cache.query(prompt, use_cache=True)).text
```

---

## External API Knowledge Gaps

When local LLM cannot solve a knowledge gap, Meta-Programmer can query external APIs:

```python
class ExternalKnowledgeCodeGen:
    """CodeGen that can leverage external APIs for knowledge gaps."""

    def __init__(self, external_api: ExternalAPIManager, kernel: Kernel):
        self.external = external_api
        self.kernel = kernel

    async def fill_knowledge_gap(self, gap: KnowledgeGap) -> GeneratedCode:
        """Fill a knowledge gap, potentially using external APIs."""

        # Step 1: Try local knowledge first
        local_patterns = await self.search_local_patterns(gap)

        if local_patterns.confidence > 0.7:
            # Can solve with local knowledge
            return await self.generate_from_patterns(local_patterns, gap)

        # Step 2: Check if external query is appropriate
        if gap.allows_external_query:
            # Query external API (requires Kernel approval)
            external_info = await self.external.query_external(
                query=f"How to implement: {gap.description}",
                api="claude",
                purpose="code_knowledge_gap"
            )

            if external_info.source != "denied":
                # Use external knowledge to generate code
                return await self.generate_from_external(external_info, gap)

        # Step 3: Fall back to best-effort local generation
        logger.warn(f"Generating with limited knowledge: {gap.description}")
        return await self.generate_best_effort(gap)
