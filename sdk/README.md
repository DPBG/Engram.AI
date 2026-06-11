# activelearning — Engram SDK

The shared Python SDK for Engram services. It provides the common runtime that
every micro-service builds on:

- **`BaseService`** — service lifecycle (connect, setup, run, graceful shutdown).
- **`EventBus`** — NATS client wrapper: pub/sub, request-reply, reconnection,
  dataclass-aware (de)serialization.
- **`Database`** — thin async SQLite (`aiosqlite`) helper.
- **`EmbeddingService`** — Ollama embedding client with an in-memory LRU cache.
- **Plugins** — `SensorPlugin` / `ActuatorPlugin` base classes and registries.
- **Core types** — `Observation`, `ActionProposal`, `KernelDecision`,
  `BeliefNode`, `BeliefEdge`, etc.

## Install

```bash
pip install -e .            # editable, for local development
```

Services normally import it via `PYTHONPATH=sdk/src` (pure-Python launcher) or as
an installed package (Docker images). See the top-level
[CLAUDE.md](../CLAUDE.md) and [RUN-LOCAL.md](../RUN-LOCAL.md).

## Usage

```python
from activelearning import BaseService

class MyService(BaseService):
    def __init__(self):
        super().__init__("my-service", use_database=True, use_event_bus=True)

    async def _setup(self) -> None:
        await self.event_bus.subscribe("my.subject", self._handle)

    async def _handle(self, data: dict) -> None:
        ...
```

Licensed under the MIT License (see the repository [LICENSE](../LICENSE)).
