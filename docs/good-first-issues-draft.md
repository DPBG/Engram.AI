# Good First Issues — Audit for Issue #146

> Audit of `neuromorphic/`, `sdk/`, and `sensory-gateway/` for beginner-friendly
> contribution opportunities. File each section below as a separate GitHub issue
> labelled **`good first issue`** and referencing `Closes #146` in the first issue
> or as a group note.
>
> None of these touch the safety-critical areas listed in CLAUDE.md §3
> (`kernel/`, `safety-supervisor/`, `meta-programmer/`, `overrides/`, `beliefs/`).

---

## Issue 1 — Add docstrings to `HomeostaticDriveSystem` state methods

**Title:** `docs(neuromorphic): add docstrings to HomeostaticDriveSystem.get_state / set_state / summary`

**Labels:** `good first issue`, `documentation`

**Body:**

### Context

`HomeostaticDriveSystem` in `neuromorphic/src/neuromorphic/drives.py` manages the
four internal drives (energy, damage, temperature, fatigue) that shape the network's
behaviour. Three of its public methods lack docstrings:

| Method | Line | What it does |
|---|---|---|
| `get_state()` | 133 | Returns a `dict[str, float]` snapshot of all four drive values |
| `set_state(state)` | 141 | Restores drive values from a previously-saved snapshot |
| `summary()` | 156 | Returns a human-readable log string of current drive levels |

### Task

Add a concise docstring to each of the three methods. Each docstring should cover:

- What it returns / what `state` is expected to contain
- Units (all values are `float` in `[0.0, 1.0]`)
- Any side-effects or guarantees

### Acceptance criteria

- [ ] `get_state`, `set_state`, and `summary` in `drives.py` each have a docstring
- [ ] The docstrings are accurate (cross-check against the implementation)
- [ ] Existing `neuromorphic/tests/test_drives.py` still passes

### Good to know

- No logic changes required — documentation only
- `test_drives.py` already exercises all three methods, so you can run
  `cd neuromorphic && uv run --extra dev python -m pytest tests/test_drives.py -v`
  to verify nothing broke

---

## Issue 2 — Add docstrings to `ConceptDifferentiationTracker` public interface

**Title:** `docs(neuromorphic): add docstrings to ConceptDifferentiationTracker.is_differentiated / get_state / set_state`

**Labels:** `good first issue`, `documentation`

**Body:**

### Context

`ConceptDifferentiationTracker` (in `neuromorphic/src/neuromorphic/neuromodulation.py`)
gates the adolescent developmental transition — the network enters the adolescent phase
only when `is_differentiated` becomes `True`. Three public members lack docstrings:

| Member | Line | What it does |
|---|---|---|
| `is_differentiated` (property) | 76 | `True` when enough distinct concept clusters have been learned |
| `get_state()` | 79 | Returns a serialisable dict for checkpoint persistence |
| `set_state(state)` | 87 | Restores from a previously-saved checkpoint dict |

### Task

Add a docstring to `is_differentiated`, `get_state`, and `set_state`. Each should explain:

- What the method returns / what `state` is expected to contain
- For `is_differentiated`: what "differentiated" means in plain language
- For `get_state`/`set_state`: the keys present in the serialised dict

### Acceptance criteria

- [ ] All three members have accurate docstrings
- [ ] `neuromorphic/tests/test_neuromodulation.py` still passes

---

## Issue 3 — Add docstrings to individual benchmark `run()` methods

**Title:** `docs(neuromorphic): add docstrings to BenchmarkSuite benchmark run() methods`

**Labels:** `good first issue`, `documentation`

**Body:**

### Context

`neuromorphic/src/neuromorphic/benchmarks.py` contains four benchmark classes used
to produce quantitative evidence that the network is learning. Each class has a `run()`
method that is missing a docstring:

| Class | Line | What it measures |
|---|---|---|
| `CrossModalRecallBenchmark.run()` | 82 | Visual→auditory and auditory→visual cross-modal recall ratios |
| `NoveltyDetectionBenchmark.run()` | 120 | Prediction-error difference between familiar and novel stimuli |
| `AssociationStrengthBenchmark.run()` | 173 | Weight changes after paired multi-modal training |
| `EnergyEfficiencyBenchmark.run()` | 231 | Energy per learned association vs baseline |

### Task

Add a docstring to each `run()` method covering:

- **What** the benchmark measures, in plain language
- **Parameters**: what each argument controls (training reps, steps, etc.)
- **Returns**: the keys present in the returned `dict` and what they represent

### Acceptance criteria

- [ ] All four `run()` methods have a docstring
- [ ] Return-value keys are documented correctly (verify against the `return` statement)
- [ ] `neuromorphic/tests/test_benchmarks.py` still passes

---

## Issue 4 — Add unit tests for `sdk/src/activelearning/subjects.py`

**Title:** `test(sdk): add unit tests for subjects.py helper functions`

**Labels:** `good first issue`, `testing`

**Body:**

### Context

`sdk/src/activelearning/subjects.py` exports three helper functions that build NATS
subject strings used for message routing across every Engram service:

```python
def decision_subject(trace_id: str) -> str: ...      # "decision.<trace_id>"
def code_decision_subject(trace_id: str) -> str: ... # "code.decision.<trace_id>"
def observation_subject(sensor_id: str) -> str: ...  # "observation.<sensor_id>"
```

There is currently no dedicated test file for this module. Because these subject strings
are the glue between every publisher and subscriber in the system, verifying them with
explicit tests is valuable.

### Task

Create `sdk/tests/test_subjects.py` with parametrised tests covering:

1. **Happy path** — typical UUID `trace_id` and alphanumeric `sensor_id`
2. **Prefix verification** — assert the returned string starts with the correct prefix
3. **No spaces / special chars** — sensor IDs with dots, hyphens (e.g. `"camera.0"`)
4. **`Subjects` class constants** — spot-check a few key constants are the expected literal strings

### Acceptance criteria

- [ ] `sdk/tests/test_subjects.py` created with at least 5 test cases
- [ ] All tests pass: `cd sdk && uv run --extra dev python -m pytest tests/test_subjects.py -v`
- [ ] No new dependencies introduced (stdlib + pytest only)

---

## Issue 5 — Add unit tests for `sdk/src/activelearning/database.py`

**Title:** `test(sdk): add unit tests for Database class pre-init guard and CRUD operations`

**Labels:** `good first issue`, `testing`

**Body:**

### Context

`sdk/src/activelearning/database.py` provides the async SQLite wrapper used by every
Engram service that persists data. There is no dedicated test file for this module.
Two behaviours are worth covering:

1. **Pre-init guard** — `execute()`, `fetchone()`, `fetchall()` raise `RuntimeError`
   when called before `initialize()` is called.
2. **Basic CRUD** — after `initialize()`, `execute`, `insert`, `update`, `fetchone`,
   and `fetchall` work correctly against an in-memory SQLite database.

### Task

Create `sdk/tests/test_database.py` that:

1. Tests the `RuntimeError` guard on `execute`, `fetchone`, `fetchall` before init
2. Uses a real in-memory SQLite (pass `db_path=":memory:"`) to test a round-trip
   insert + fetchone + fetchall
3. Tests `commit()` is idempotent (calling it twice should not raise)

### Acceptance criteria

- [ ] `sdk/tests/test_database.py` created with at least 5 test cases
- [ ] Tests use `asyncio.run()` or `pytest-asyncio` to drive async code
- [ ] All tests pass: `cd sdk && uv run --extra dev python -m pytest tests/test_database.py -v`
- [ ] No new dependencies beyond `pytest` and `pytest-asyncio`

---

## Issue 6 — Create `sensory-gateway/tests/` with smoke tests for `QueueStateManager`

**Title:** `test(sensory-gateway): create tests/ directory with smoke tests for QueueStateManager`

**Labels:** `good first issue`, `testing`

**Body:**

### Context

The entire `sensory-gateway/` module has **zero tests**. `QueueStateManager`
(`sensory-gateway/queue_state.py`) is the most isolated class in the module — it is
pure Python, fully synchronous, has no OpenCV/NATS/asyncio dependencies, and manages
video training queue persistence to a JSON file on disk.

This makes it the ideal first test target for the module.

### Task

1. Create the directory `sensory-gateway/tests/`
2. Create `sensory-gateway/tests/__init__.py` (empty)
3. Create `sensory-gateway/tests/test_queue_state.py` with the following test cases:

   - `test_save_creates_valid_json` — after calling `save()`, the file exists and is
     valid JSON
   - `test_load_roundtrip` — data saved with `save()` is faithfully restored by `load()`
   - `test_load_missing_file_returns_empty` — `load()` when no file exists returns `{}`
     (or equivalent empty state) without raising
   - `test_save_multiple_sessions_ordering` — active session appears first in saved list
   - `test_path_property` — `QueueStateManager.path` returns the path given at construction

Use `tmp_path` (pytest's built-in fixture) so tests never write to the real
`queue_state.json`.

### Acceptance criteria

- [ ] `sensory-gateway/tests/test_queue_state.py` created with at least 5 tests
- [ ] All tests pass: `python -m pytest sensory-gateway/tests/test_queue_state.py -v`
  from the repo root (no special env needed — pure stdlib + pytest)
- [ ] Tests use `tmp_path` and never touch the real `queue_state.json`

---

## Issue 7 — Add docstrings to `VideoFileSensor` progress-tracking properties

**Title:** `docs(sensory-gateway): add docstrings to VideoFileSensor progress properties`

**Labels:** `good first issue`, `documentation`

**Body:**

### Context

`sensory-gateway/sensors/video_file.py` exposes several `@property` accessors that
external code uses to monitor playback progress. Five of them lack docstrings:

| Property | Line | What it returns |
|---|---|---|
| `loop_count` | 83 | Number of times the video has looped from start |
| `frames_emitted` | 87 | Total frames sent to the event bus since `start()` |
| `duration_s` | 91 | Total video duration in seconds (from OpenCV metadata) |
| `elapsed_s` | 95 | Wall-clock seconds since `start()` was called |
| `current_pos_s` | 101 | Current OpenCV capture position in seconds |

(`progress` on line 249 already has a docstring — no change needed there.)

### Task

Add a one-line docstring to each of the five properties listed above. The docstring
should state the units where applicable (seconds, frame count, loop count).

### Acceptance criteria

- [ ] All five properties have a docstring
- [ ] Docstrings are accurate (units match the implementation)
- [ ] No logic changes — documentation only
