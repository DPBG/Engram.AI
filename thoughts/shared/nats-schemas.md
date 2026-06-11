# NATS Message Schemas - Engram

> **Last Updated**: 2026-02-03
> **Purpose**: Comprehensive documentation of all NATS subjects and message schemas

---

## Table of Contents

1. [Core Data Flow](#core-data-flow)
2. [Subject Patterns](#subject-patterns)
3. [Message Schemas](#message-schemas)
4. [Service Integration Map](#service-integration-map)

---

## Core Data Flow

```
Sensor → observation.*
       → Planner → proposal.new
                 → Kernel (+ Safety Supervisor)
                         → decision.{trace_id}
                                 → Planner → outcome.{trace_id}
                                          → Actuator
```

---

## Subject Patterns

### Wildcards
- `observation.*` - All sensor observations
- `decision.*` - All kernel decisions
- `safety.analyze.*` - All safety analysis requests
- `demo.*` - All demonstration learning events
- `override.applied.*` - All override notifications

### Dynamic Patterns (use trace_id)
- `decision.{trace_id}` - Specific kernel decision
- `code.decision.{trace_id}` - Code-specific kernel decision
- `outcome.{trace_id}` - Action outcome
- `override.result.{trace_id}` - Override result
- `safety.analysis.action.{trace_id}` - Action analysis result
- `safety.analysis.code.{trace_id}` - Code analysis result
- `knowledge.gap.result.{trace_id}` - Knowledge gap result
- `llm.cache.{hit|miss}` - Cache events

---

## Message Schemas

### 1. Observations (Sensors → Planner)

#### `observation.{sensor_id}`
**Published by**: Sensor plugins
**Subscribed by**: Planner, Memory
**Schema**:
```json
{
  "trace_id": "uuid-v4",
  "provenance": "sensor.camera.front",
  "data": {}, // Sensor-specific data
  "timestamp": 1234567890123,
  "confidence": 0.95,
  "tags": ["motion", "detected"]
}
```
**SDK Type**: `Observation[T]` (sdk/src/activelearning/core.py:74-101)

---

### 2. Proposals (Planner → Kernel)

#### `proposal.new`
**Published by**: Planner, External API, Overrides
**Subscribed by**: Kernel
**Schema**:
```json
{
  "trace_id": "uuid-v4",
  "provenance": "planner.main",
  "action": {
    "type": "alert",
    "message": "...",
    "parameters": {}
  },
  "priority": 5,
  "requires_approval": false,
  "metadata": {}
}
```
**SDK Type**: `ActionProposal[T]` (sdk/src/activelearning/core.py:104-129)

#### `code.proposal`
**Published by**: Meta-Programmer
**Subscribed by**: Kernel
**Schema**:
```json
{
  "trace_id": "uuid-v4",
  "gap_ref": "uuid-v4",
  "proposed_action": "CREATE|MODIFY|REFACTOR|DELETE",
  "target_path": "/data/plugins/my_plugin.py",
  "code_preview": "def my_function():\n    pass",
  "test_plan": "...",
  "rollback_plan": "...",
  "agent": "CodeGen"
}
```
**SDK Type**: `CodeProposal` (sdk/src/activelearning/core.py:315-343)

---

### 3. Decisions (Kernel → Planner)

#### `decision.{trace_id}`
**Published by**: Kernel
**Subscribed by**: Planner
**Schema**:
```json
{
  "trace_id": "uuid-v4",
  "type": "ALLOW|TRANSFORM|DENY|DEFER",
  "reason": "Action approved within safety bounds",
  "transformations": [/* modified proposals */],
  "risk_score": 0.15,
  "issued_at": 1234567890123,
  "expires_at": 1234567890123
}
```
**SDK Type**: `KernelDecision[T]` (sdk/src/activelearning/core.py:132-162)

#### `code.decision.{trace_id}`
**Published by**: Kernel
**Subscribed by**: Meta-Programmer
**Schema**: Same as `decision.{trace_id}` but for code proposals

---

### 4. Outcomes (Planner → Actuators)

#### `outcome.{trace_id}`
**Published by**: Planner
**Subscribed by**: Actuator plugins, Memory
**Schema**:
```json
{
  "trace_id": "uuid-v4",
  "action": {
    "type": "alert",
    "message": "...",
    "parameters": {}
  },
  "decision_type": "ALLOW|DENY",
  "success": true,
  "error": null,
  "timestamp": 1234567890123
}
```
**SDK Type**: `Outcome[T]` (sdk/src/activelearning/core.py:165-184)

---

### 5. Safety & Risk Analysis

#### `safety.analyze.action`
**Published by**: Kernel (request/reply)
**Subscribed by**: Safety Supervisor
**Request Schema**:
```json
{
  "trace_id": "uuid-v4",
  "provenance": "planner.main",
  "action": {},
  "priority": 5,
  "metadata": {}
}
```

#### `safety.analysis.action.{trace_id}`
**Published by**: Safety Supervisor
**Subscribed by**: Kernel
**Response Schema**:
```json
{
  "trace_id": "uuid-v4",
  "risk_score": 0.15,
  "flags": ["system_import_detected"],
  "recommendations": ["Review import safety"],
  "details": {}
}
```
**SDK Type**: `RiskAnalysis` (sdk/src/activelearning/core.py:49-60)

#### `safety.analyze.code`
**Published by**: Kernel (request/reply)
**Subscribed by**: Safety Supervisor
**Request Schema**: Code proposal
**Response Schema**: Same as action analysis

---

### 6. Human Interaction & Approvals

#### `approval.request`
**Published by**: Planner, Meta-Programmer
**Subscribed by**: Dashboard
**Schema**:
```json
{
  "trace_id": "uuid-v4",
  "proposal": {},
  "reason": "Action requires human review",
  "timestamp": 1234567890123
}
```

#### `override.request`
**Published by**: Human interface
**Subscribed by**: Overrides service
**Schema**:
```json
{
  "trace_id": "uuid-v4",
  "override_type": "operational|knowledge",
  "prompt": "Set planner priority threshold to 8",
  "verification_data": {
    "method": "camera|microphone|button",
    "confidence": 0.92,
    "timestamp": 1234567890123
  }
}
```

#### `override.applied.{trace_id}`
**Published by**: Overrides service
**Subscribed by**: Cache invalidator
**Schema**:
```json
{
  "trace_id": "uuid-v4",
  "override_type": "operational",
  "parameter_path": "planner.priority_threshold",
  "value": 8,
  "timestamp": 1234567890123
}
```

---

### 7. Memory & Knowledge

#### `memory.store`
**Published by**: Any service
**Subscribed by**: Memory service
**Schema**:
```json
{
  "trace_id": "uuid-v4",
  "episode_data": {},
  "tags": ["important", "success"],
  "utility_score": 0.8,
  "timestamp": 1234567890123
}
```

#### `memory.query`
**Published by**: Any service (request/reply)
**Subscribed by**: Memory service
**Request Schema**:
```json
{
  "query_type": "semantic|time_window|tag",
  "query": "how to handle motion detection",
  "limit": 10
}
```
**Response Schema**:
```json
{
  "episodes": [
    {
      "trace_id": "uuid-v4",
      "data": {},
      "tags": [],
      "utility_score": 0.8,
      "timestamp": 1234567890123,
      "relevance": 0.92
    }
  ]
}
```

#### `knowledge.gap`
**Published by**: Coordinator, Learning Controller
**Subscribed by**: Meta-Programmer
**Schema**:
```json
{
  "trace_id": "uuid-v4",
  "description": "Need Raspberry Pi camera sensor plugin",
  "context": {
    "hardware": "raspberry_pi_4",
    "sensor_type": "camera"
  },
  "search_results": [],
  "confidence": 0.3,
  "available_sensors": ["camera", "microphone"],
  "allows_external_query": true,
  "source": "coordinator"
}
```
**SDK Type**: `KnowledgeGap` (sdk/src/activelearning/core.py:283-311)

---

### 8. Beliefs Graph

#### `beliefs.add_node`
**Schema**:
```json
{
  "id": "uuid-v4",
  "type": "value|norm|fact",
  "content": "Safety is paramount",
  "confidence": 0.95,
  "source": "human_teaching",
  "metadata": {}
}
```
**SDK Type**: `BeliefNode` (sdk/src/activelearning/core.py:188-216)

#### `beliefs.add_edge`
**Schema**:
```json
{
  "id": "uuid-v4",
  "type": "supports|contradicts|entails|refines",
  "source_id": "uuid-v4",
  "target_id": "uuid-v4",
  "strength": 0.8,
  "evidence": "Observed in 50 cases"
}
```
**SDK Type**: `BeliefEdge` (sdk/src/activelearning/core.py:219-244)

---

### 9. Meta-Programmer & Code Generation

#### `code.deployed`
**Published by**: Meta-Programmer
**Subscribed by**: Cache invalidator
**Schema**:
```json
{
  "trace_id": "uuid-v4",
  "file_path": "/data/plugins/camera_plugin.py",
  "action": "CREATE|MODIFY",
  "timestamp": 1234567890123
}
```

---

### 10. Coordinator & Task Execution

#### `task.request`
**Published by**: External interface, API
**Subscribed by**: Coordinator
**Schema**:
```json
{
  "trace_id": "uuid-v4",
  "task_description": "Pick up the red ball",
  "context": {
    "scene": "table_with_objects",
    "available_actuators": ["arm", "gripper"]
  }
}
```

#### `task.result`
**Published by**: Coordinator
**Subscribed by**: Requestor
**Schema**:
```json
{
  "trace_id": "uuid-v4",
  "success": true,
  "task_id": "pickup_red_ball",
  "execution_time_ms": 5432,
  "confidence": 0.88,
  "error": null
}
```

#### `demo.start`
**Published by**: External interface
**Subscribed by**: Coordinator
**Schema**:
```json
{
  "trace_id": "uuid-v4",
  "task_description": "Watch me pick up this object",
  "available_sensors": ["camera", "touch"]
}
```

#### `demo.finished`
**Published by**: Coordinator
**Subscribed by**: Interested services
**Schema**:
```json
{
  "trace_id": "uuid-v4",
  "task_id": "learned_task_123",
  "success": true,
  "saved_path": "/data/tasks/learned_task_123/",
  "metadata": {
    "duration_ms": 15000,
    "observations_count": 45
  }
}
```

---

### 11. LLM Cache & Autopilot

#### `cache.query`
**Published by**: Any service (request/reply)
**Subscribed by**: Cache service
**Request Schema**:
```json
{
  "prompt": "How do I initialize the camera sensor?",
  "context": {},
  "min_confidence": 0.95
}
```
**Response Schema**:
```json
{
  "hit": true,
  "response": "To initialize the camera...",
  "confidence": 0.97,
  "source": "cached"
}
```

#### `llm.cache.hit` / `llm.cache.miss`
**Published by**: Cache service
**Subscribed by**: Monitoring/metrics
**Schema**:
```json
{
  "prompt_hash": "sha256_hash",
  "confidence": 0.97,
  "timestamp": 1234567890123
}
```

---

### 12. System Control & Status

#### `planner.mode`
**Published by**: Admin/control interface
**Subscribed by**: Planner
**Schema**:
```json
{
  "mode": "EXECUTION|LEARNING|EXPLORATION|SAFE_HALT"
}
```

#### `*.status` (Generic Status Request)
**Pattern**: `{service}.status`
**Examples**: `kernel.status`, `planner.status`, `safety.status`
**Schema**:
```json
{
  "request_id": "uuid-v4"
}
```

#### `*.status.response` (Generic Status Response)
**Schema**:
```json
{
  "status": "running",
  "metrics": {
    "uptime_ms": 123456,
    "requests_processed": 542
  }
}
```

#### `system.shutdown`
**Published by**: Admin/control interface
**Subscribed by**: All services
**Schema**:
```json
{
  "reason": "Scheduled maintenance",
  "grace_period_ms": 30000
}
```

---

## Service Integration Map

### Service: Planner
**Subscribes to**:
- `observation.*` - Receives sensor data
- `decision.*` - Receives kernel decisions
- `planner.mode` - Mode changes
- `planner.status` - Status requests

**Publishes to**:
- `proposal.new` - Action proposals
- `outcome.{trace_id}` - Execution results
- `approval.request` - Human approval requests

---

### Service: Kernel
**Subscribes to**:
- `proposal.new` - Action proposals
- `code.proposal` - Code proposals
- `kernel.status` - Status requests

**Publishes to**:
- `decision.{trace_id}` - Action decisions
- `code.decision.{trace_id}` - Code decisions
- `kernel.status.response` - Status responses

**Requests**:
- `safety.analyze.action` → Safety Supervisor
- `safety.analyze.code` → Safety Supervisor

---

### Service: Safety Supervisor
**Subscribes to**:
- `safety.analyze.action` - Action analysis requests (request/reply)
- `safety.analyze.code` - Code analysis requests (request/reply)
- `safety.status` - Status requests

**Publishes to**:
- `safety.analysis.action.{trace_id}` - Action analysis results
- `safety.analysis.code.{trace_id}` - Code analysis results
- `safety.status.response` - Status responses

---

### Service: Memory
**Subscribes to**:
- `memory.store` - Store episode
- `memory.query` - Query memory (request/reply)
- `memory.recall` - Recall memories

**Publishes to**: Response via NATS reply mechanism

---

### Service: Meta-Programmer
**Subscribes to**:
- `knowledge.gap` - Knowledge gaps
- `code.decision.{trace_id}` - Code decisions
- `metaprogrammer.status` - Status requests

**Publishes to**:
- `code.proposal` - Code proposals
- `approval.request` - Human approval (for DEFER)
- `knowledge.gap.result.{trace_id}` - Gap resolution results
- `code.deployed` - Deployment notifications
- `metaprogrammer.status.response` - Status responses

---

### Service: Coordinator
**Subscribes to**:
- `task.request` - Task requests
- `demo.start` - Demo start
- `demo.observation` - Demo observations
- `demo.finish` - Demo finish
- `coordinator.status` - Status requests

**Publishes to**:
- `task.result` - Task results
- `demo.started` - Demo started
- `demo.finished` - Demo finished
- `demo.failed` - Demo failed
- `knowledge.gap` - Knowledge gaps
- `coordinator.status.result` - Status responses

---

### Service: Overrides
**Subscribes to**:
- `override.request` - Override requests
- `override.status` - Status requests

**Publishes to**:
- `override.result.{trace_id}` - Override results
- `override.applied.{trace_id}` - Override applied
- `proposal.new` - Proposals (for operational overrides)
- `planner.priority_threshold` - Config changes
- `cache.setting` - Cache enable/disable
- `autopilot.setting` - Autopilot enable/disable

---

### Service: Cache
**Subscribes to**:
- `cache.query` - Cache queries (request/reply)
- `cache.setting` - Enable/disable
- `autopilot.setting` - Enable/disable
- `cache.status` - Status requests
- `code.deployed` - Invalidation trigger
- `override.applied.*` - Invalidation trigger
- `task.saved` - Invalidation trigger

**Publishes to**:
- `llm.cache.hit` - Cache hit event
- `llm.cache.miss` - Cache miss event

---

### Service: External API
**Subscribes to**:
- `external.query` - External knowledge queries
- `external.status` - Status requests

**Publishes to**:
- `proposal.new` - Proposals (for knowledge conflicts)

---

### Service: Beliefs
**Subscribes to**:
- `beliefs.add_node` - Add belief node
- `beliefs.add_edge` - Add belief edge
- `beliefs.update` - Update belief
- `beliefs.query` - Query beliefs
- `beliefs.contradictions` - Find contradictions

**Publishes to**: Response via NATS reply mechanism

---

## Request/Reply Patterns

NATS supports two patterns for request/reply:

### 1. Implicit Reply (using msg.reply)
Used by services that subscribe to subjects and respond via the message's reply field.
Example: Safety Supervisor responding to analysis requests

### 2. Explicit Dynamic Subjects
Used for decisions and outcomes where the subject includes the trace_id.
Example: `decision.{trace_id}` for kernel decisions

---

## Consistency Checklist

✅ All dynamic subjects use `{trace_id}` consistently
✅ All dataclasses match SDK types in `sdk/src/activelearning/core.py`
✅ Wildcard patterns documented
✅ Request/reply patterns documented
✅ Service integration map complete
✅ Message schemas include all required fields

---

## Notes

1. **Trace IDs**: All messages that are part of a flow should carry the same `trace_id` for end-to-end tracing
2. **Timestamps**: Unix timestamps in milliseconds
3. **Confidence Scores**: Float between 0.0 and 1.0
4. **Dynamic Subjects**: Use curly braces `{trace_id}` to indicate dynamic parts
5. **Wildcard Subscriptions**: Use `*` for single-level wildcards, `>` for multi-level (NATS standard)

---

## Future Enhancements

- Add JetStream subject mappings
- Document message retention policies
- Add message size limits
- Document rate limiting strategies
