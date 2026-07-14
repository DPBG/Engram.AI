# Message-Schema Versioning Policy

> **Milestone:** M2 — Reliable Backbone
> **Issue:** [M2.4] Formalize a message-schema versioning policy for NATS subjects (#227)
> **Status:** Canonical — all NATS wire-model changes MUST conform to this document.

This document defines how Engram's NATS message payloads (the pydantic
`WireModel` types in [`sdk/src/activelearning/messages.py`](../sdk/src/activelearning/messages.py))
evolve over time without breaking running services.

---

## 1. The `version` field

Every wire payload carries a single integer `version` field — the **wire-schema
major version** it was produced against. It is defined once on the base model and
inherited by every message type:

```python
WIRE_SCHEMA_VERSION = 1

class WireModel(BaseModel):
    model_config = ConfigDict(extra="allow")
    version: int = WIRE_SCHEMA_VERSION
```

- On **publish**, `EventBus.publish()` validates dict payloads through the wire
  model, which stamps `version = WIRE_SCHEMA_VERSION` when the caller did not set
  it explicitly.
- On **subscribe**, incoming payloads are validated before the handler runs, so a
  handler always sees an explicit `version`.
- The current value is `1`. It is exported from the SDK as
  `activelearning.WIRE_SCHEMA_VERSION` so services and tools can reference it.

### Why a single base-model field (not per-message versions)

The version describes the **wire envelope**, not each individual message type.
Keeping one number keeps the policy simple: a consumer only has to answer "do I
understand major version N?" rather than track a version per subject. Because the
field lives on `WireModel`, adding it required no change to any of the ~25 concrete
message classes.

---

## 2. Compatibility policy

The version follows a **major-version-only** scheme. There is no minor/patch
component on the wire; backward-compatible growth is handled by tolerant parsing
(`extra="allow"` plus optional fields with defaults) instead of a version bump.

### 2.1 Backward-compatible changes — DO NOT bump the version

These changes are safe because old consumers ignore what they don't recognize and
new consumers fall back to defaults for what old producers omit:

- **Adding a new optional field** (with a default, or `X | None = None`).
- **Adding a new message type / subject** to `SUBJECT_SCHEMAS`.
- **Relaxing** a constraint (e.g. widening a type, adding an allowed enum value that
  older consumers already treat as "unknown/ignored").
- **Documentation / comment / field-description** changes.

Requirements for an additive change:

1. The new field MUST have a default so existing producers stay valid.
2. Consumers MUST NOT assume the field is present unless they also require a version
   bump; treat it as "may be absent" until every producer sets it.

### 2.2 Breaking changes — MUST bump `WIRE_SCHEMA_VERSION`

A breaking change is anything an old consumer could **misinterpret** rather than
simply ignore:

- Removing or renaming a field.
- Making a previously optional field required.
- Changing a field's **type** (e.g. `str` → `int`) or its **units/semantics**.
- Changing the meaning of an existing enum value.

When you make one of these:

1. Increment `WIRE_SCHEMA_VERSION` in `messages.py`.
2. Update this document's changelog (§4).
3. Add/adjust tests in `sdk/tests/test_messages.py`.
4. Coordinate rollout (§3) — producers and consumers do not deploy atomically.

---

## 3. Consumer guidance

A consumer that cares about compatibility should branch on `version`:

```python
async def handle(data: dict) -> None:
    v = data.get("version", WIRE_SCHEMA_VERSION)
    if v > WIRE_SCHEMA_VERSION:
        # Message from a newer producer. Fields we know still parse (extra="allow"),
        # but semantics may have changed — log and handle conservatively.
        logger.warning("message version %s newer than supported %s", v, WIRE_SCHEMA_VERSION)
    # ... process known fields ...
```

Rollout order for a breaking (major) bump:

1. **Deploy consumers first** so they can accept both the old and new major
   version during the transition.
2. **Then deploy producers** emitting the new version.
3. Retire the old-version handling once no producer emits it.

Safety-critical subjects (`proposal.new`, `code.proposal`, `decision.*`) are
JetStream-persisted, so a stored message may be delivered **after** a version bump.
Consumers of these subjects MUST keep accepting the previous major version until the
stream has fully drained of old-version messages.

---

## 4. Version changelog

| Version | Date       | Change |
|---------|------------|--------|
| 1       | 2026-07-12 | Initial wire-schema version. `version` field added to `WireModel`; establishes this compatibility policy. |

---

*Document owner: SDK / backbone team.*
