# Running Engram Locally (Pure Python, No Docker)

Engram now runs **without Docker**. A pure-Python launcher (`run.py`) replaces
`docker compose`: it downloads and manages a local **NATS** server, then starts
each micro-service as a native Python subprocess with the right environment.

```
python run.py --install     # one-time: install Python dependencies
python run.py --doctor      # (optional) check your machine is set up correctly
python run.py               # start the default "core" profile
```

Then open the dashboard at **http://localhost:8080** and press **Ctrl+C** to stop
everything.

---

## Requirements

- **Python 3.11+** (3.11 matches the services' `requires-python`; newer works too).
- Internet access on first run (to fetch the small `nats-server` binary, ~6 MB).
- That's it for the **core** profile. The **full** profile additionally needs
  Qdrant and/or Ollama (see below).

A virtual environment is recommended:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1      # Windows PowerShell
# source .venv/bin/activate       # macOS/Linux
python run.py --install
```

---

## What runs (profiles)

List everything with `python run.py --list`.

| Profile | Services | Extra infra needed |
|---------|----------|--------------------|
| **core** (default) | kernel, safety-supervisor, beliefs, planner, external-api, neuromorphic, dashboard | none (just NATS, auto-managed) |
| **full** | core **＋** memory, cache, coordinator, cognitive-bridge | Qdrant and/or Ollama |
| **all** | full **＋** overrides (camera/mic), sdk-runtime | + opencv/pyaudio for overrides |

```powershell
python run.py                       # core
python run.py --profile full        # core + Qdrant/Ollama-backed services
python run.py --only kernel,planner # just a subset, by name
```

## Preflight check (`--doctor`)

Not sure your machine is ready? `python run.py --doctor` runs read-only checks
and prints a report **without starting anything**:

```
python run.py --doctor
```

It verifies your Python version, that a usable `nats-server` is present (or will
be downloaded), that the NATS client/monitor ports are free, whether the
optional Qdrant/Ollama services are reachable, that the `data/` directory is
writable, free disk space, and that the service registry is consistent.

Each line is graded `[ OK ]`, `[WARN]`, or `[FAIL]`. Optional infrastructure
being down is only a **warning** (those services are simply skipped on the core
profile), so the command exits `0` whenever the stack can actually start, and
exits non-zero only when a **blocking** problem (`[FAIL]`) must be fixed first —
which makes it handy as a CI/setup gate, too.

Services that need Qdrant/Ollama are **skipped with a warning** if those servers
aren't reachable (override with `--skip-infra-check`).

> **meta-programmer** is *not* runnable without Docker — it needs the Docker
> socket to spawn sandbox containers — so it is excluded from the launcher.

---

## Infrastructure

| Component | Required? | How it's handled |
|-----------|-----------|------------------|
| **NATS** (message bus, :4222) | Yes — every service uses it | **Auto-downloaded & managed** by `run.py`. If a NATS is already listening on 4222 (or `nats-server` is on PATH), it's reused. |
| **SQLite** | Yes | File-based, created automatically under `./data/sqlite/`. No server. |
| **Qdrant** (vector DB, :6333) | Only for `full` profile (memory/cache/coordinator) | Install separately and run it, then use `--profile full`. |
| **Ollama** (local LLM, :11434) | Only for cache + cognitive-bridge | Install from https://ollama.com, then `ollama pull deepseek-coder:6.7b`. |

### Optional: run Qdrant / Ollama for the full profile

- **Qdrant** — download `qdrant` from its
  [GitHub releases](https://github.com/qdrant/qdrant/releases) and run `qdrant`,
  or `pip install qdrant` is *not* enough (the cache/coordinator services use its
  HTTP API, so the server must be running).
- **Ollama** — install the app, `ollama serve` runs automatically; pull a model:
  `ollama pull deepseek-coder:6.7b`.

---

## Command reference

```
python run.py [options]

  --install              Install Python deps from requirements-local.txt, then exit
  --list                 Show all services, their profiles, and what they need
  --profile {core,full,all}   Which set of services to run (default: core)
  --only a,b,c           Run a specific comma-separated subset by name
  --no-nats              Don't download/manage NATS (assume it's already running)
  --skip-infra-check     Start Qdrant/Ollama-dependent services even if undetected
```

---

## Where things live

- `./data/sqlite/unified.db` — shared service database
- `./data/sqlite/neuromorphic.db` — the brain's weights (persists across restarts)
- `./.localrun/nats/` — the downloaded `nats-server` binary
- `./.localrun/nats-data/` — NATS JetStream storage
- `./data/nats.log` — NATS server log

You can delete `./.localrun/` and `./data/` to reset to a clean state.

---

## Tuning the brain (neuromorphic)

The launcher starts a small, laptop-friendly brain (~50K neurons). Override any
`NEURO_*` variable before launching to scale up, e.g. PowerShell:

```powershell
$env:NEURO_SENSORY_N = "50000"
$env:NEURO_ASSOCIATION_N = "50000"
python run.py
```

See `.env.example` for the full list of `NEURO_*` knobs and preset scales.

The brain logs mean synaptic weights periodically. On fast hardware this can be
chatty — throttle or silence it:

```powershell
$env:NEURO_WEIGHT_LOG_INTERVAL = "50"   # default (log ~every 2.5s)
$env:NEURO_WEIGHT_LOG_INTERVAL = "0"    # disable the mean-weights log
```

---

## Feeding the brain (sensory input)

Run the brain alone and you'll see `WATCHDOG CRITICAL: No sensory input` — that's
expected: the brain is healthy but has nothing to sense. It listens on the NATS
subject `observation.>`, and the **sensory-gateway** is what publishes there.

Feed it a looping video (visual input needs only `opencv-python`):

```powershell
pip install opencv-python
python run.py --only sensory-gateway --video path\to\clip.mp4
# or run the brain + gateway together:
python run.py --only neuromorphic,sensory-gateway --video path\to\clip.mp4
```

- The gateway runs with `--no-camera --no-mic`, so it needs a `--video` (or the
  `SENSORY_VIDEO` env var); without one it's skipped with a warning.
- **Audio from the video** is optional and auto-disables if `ffmpeg` +
  `sounddevice` aren't present — you still get visual observations.
- Continuous observations are what drive STDP/learning, so this is the
  recommended way to actually exercise the system (vs. letting it idle).

---

## Troubleshooting

- **`NATS did not become ready`** — first run needs internet to fetch the binary.
  If you're offline, install `nats-server` from https://nats.io/download/, put it
  on PATH, and run with `--no-nats` (start NATS yourself) or let the launcher find
  it on PATH.
- **A service exits with `ModuleNotFoundError`** — run `python run.py --install`
  (or `pip install -r requirements-local.txt`). For `overrides`, also install
  `opencv-python` and `pyaudio` (see the bottom of `requirements-local.txt`).
- **`memory`/`cache`/`coordinator` are skipped** — that's expected unless Qdrant
  (and Ollama for cache) are running. Start them, then `python run.py --profile full`.
- **numpy/scipy fail to install** — use Python 3.11/3.12 where prebuilt wheels are
  available; very new interpreters may lack wheels for `scipy`.
