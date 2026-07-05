# Stub / TODO Audit — Issue #141

> Audit of `neuromorphic/src` and `sensory-gateway/gateway.py` for
> `TODO | FIXME | NotImplementedError | XXX:` hits, triaged as "resolved",
> "intentional", or "spin off sub-issue".

## Scan command

```bash
grep -rn "TODO|FIXME|NotImplementedError|XXX:" neuromorphic/src sensory-gateway/gateway.py
```

Run against `dev` HEAD after merge of recent M5/M6 PRs.

---

## Results

### `neuromorphic/src` — 0 hits

All 11 stubs originally flagged in this issue have been resolved by prior PRs
(the commit history in `dev` confirms removal in the M5/M6 wave of work).

No remaining `TODO`, `FIXME`, `XXX:`, or bare `NotImplementedError` raises
exist in any Python file under `neuromorphic/src/`.

**Additional patterns checked for completeness:**

| Location | Pattern | Verdict |
|---|---|---|
| `neuromorphic/src/neuromorphic/tasks.py:63–66` | `@abstractmethod def reset/evaluate(...): ...` | **Intentional** — abstract method stubs on the `Task` base class; concrete subclasses implement them. |
| `neuromorphic/src/neuromorphic/motor_feedback_adapter.py:39,41` | `...` inside docstring code example | **Intentional** — ellipsis is documentation notation, not Python code. |
| `neuromorphic/src/neuromorphic/persistence.py:21` | `...` inside module-level docstring directory listing | **Intentional** — directory structure example, not Python code. |

### `sensory-gateway/gateway.py:1222` — 1 hit

```python
except NotImplementedError:
    pass  # Windows: add_signal_handler unsupported; Ctrl+C still works
```

**Verdict: Intentional.** `asyncio.loop.add_signal_handler()` raises
`NotImplementedError` on Windows (the ProactorEventLoop does not support it).
The `except` clause is a platform-compatibility guard, not a stub. No action needed.

---

## Triage summary

| # | Location | Kind | Resolution |
|---|---|---|---|
| 1–11 | `neuromorphic/src` (various) | `TODO/FIXME/NotImplementedError` | **Already resolved** by prior PRs |
| 12 | `sensory-gateway/gateway.py:1222` | `except NotImplementedError` | **Intentional** — Windows compat guard |

---

## README update

The README `Known limitations` section previously said:

> *"Some stubs remain. A few subsystems contain placeholders (see the TODOs in
> the code). Contributions here are especially welcome."*

This was inaccurate. The line has been replaced with a pointer to the
`good first issue` label backlog (filed separately in issue #146).

---

## Sub-issues filed

No non-trivial stubs were found requiring spin-off sub-issues. The `good first issue`
backlog was populated separately via issue #146 (which identified documentation gaps
and test-coverage gaps rather than code stubs).
