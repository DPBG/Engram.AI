# SQLite WAL-mode behavior under concurrent service writes

Several Engram services share one SQLite file — `docker-compose.yml`'s
`x-common-env` points `kernel`, `safety-supervisor`, `beliefs`, `planner`,
`external-api`, `memory`, `cache`, and `coordinator` all at
`/data/sqlite/unified.db` (`neuromorphic` uses its own separate
`neuromorphic.db`). `Database.initialize()`
(`sdk/src/activelearning/database.py`) enables `PRAGMA journal_mode=WAL` and
`PRAGMA synchronous=NORMAL` on every connection. This document records what
was actually measured about that setup under concurrency (issue #231), not
just how WAL mode is supposed to behave in theory.

The stress tests backing these numbers live in
`sdk/tests/test_sqlite_wal_concurrency.py`.

## Two different regimes

WAL mode's concurrency guarantees only fully apply **after** a file already
has WAL established. There are two distinct situations, and they behave very
differently:

### 1. Concurrent first-time `initialize()` — the regime that has a real limit

When several service processes start at (or near) the same instant against a
**brand-new** database file — the exact situation at a fresh Docker Compose
`up` or a first-ever `run.py` launch — each one independently calls
`Database.initialize()`, which does its own `PRAGMA journal_mode=WAL` +
`executescript(SCHEMA_SQL)`. Switching a fresh file into WAL mode and running
first-time schema DDL both need an **exclusive** lock; that lock acquisition
is not always retried through SQLite's normal busy handler the way an
ordinary read/write lock is.

**Measured:** with 4 concurrent `Database()` instances calling `initialize()`
against one new file (Python 3.12.3, SQLite 3.50.4, 20 independent trials),
**4/20 trials (20%) hit at least one failure**, for 6 total
`sqlite3.OperationalError` occurrences across 80 initialize attempts (~7.5%
per-attempt). The errors observed were `database is locked` and — more
surprisingly — `disk I/O error`, both raised promptly (not a hang) once
SQLite's internal exclusive-lock wait gave up.

This is a real, reproducible limit, not an artifact of this dev sandbox: it
reproduces identically under the CI-matrix Python (3.12) and under a newer
local interpreter (3.14), on a plain ext4 filesystem.

**Fix applied:** `Database.initialize()` now retries on
`sqlite3.OperationalError` with exponential backoff + jitter (5 attempts,
starting at 50ms) before giving up and re-raising. Re-running the identical
20-trial stress scenario against the fixed code: **0/20 trials failed**
(and 0/40 trials failed at 8 concurrent services). A service starting up
alone — the common case — never enters the retry path at all, so this adds
no latency outside of genuine contention.

**Without the fix**, running many back-to-back concurrent-initialize trials
in a loop measurably degrades — in one run, 15 trials × 4 services took over
two minutes (each trial's contention burns up to the ~5s per-connection
`sqlite3` default timeout) instead of well under a second. This is the
"could silently degrade or block services" scenario the issue was worried
about, made concrete: it doesn't corrupt data or hang forever, but an
unlucky startup race did reliably slow service startup and could raise an
unhandled exception before this fix.

### 2. Concurrent steady-state writes — reliable, no limit found

Once the schema and WAL mode already exist (the state every service is in
after the first few seconds of a deployment's life), many separate
connections writing concurrently behaved reliably in every trial run:

- 12 concurrent `Database` instances each inserting 20 rows (240 total) into
  `audit_entries` — all 240 rows landed, verified by both the in-process
  insert count and an independent read-back connection's `COUNT(*)`.
- Scaling to 32 concurrent writers still produced zero data loss; SQLite's
  single-writer-at-a-time WAL semantics serialize the actual disk writes,
  and each writer's `INSERT ... ; COMMIT` is fast enough that the default
  5-second busy timeout comfortably absorbs the queueing.

## The failure mode is explicit, not silent

`Database.initialize()` now sets `PRAGMA busy_timeout=5000` explicitly
(previously this relied implicitly on `aiosqlite.connect()`'s default
`timeout=5.0` parameter). When contention genuinely exceeds what a writer is
willing to wait, SQLite raises `sqlite3.OperationalError` with a message
containing `"database is locked"` — a well-known, catchable exception, not a
silent write loss and not (with the fix in place) an unbounded hang.
`TestObservedLimit` in the stress-test file reproduces this directly: a
writer holding `BEGIN IMMEDIATE` past another connection's `busy_timeout`
causes the second connection's write to raise exactly that error.

## Recommendation for new code

- Prefer going through `activelearning.database.get_database()` /
  `Database`, not a hand-rolled `aiosqlite.connect()` — the retry-on-init
  and `busy_timeout` are only applied there.
- If a service performs a write outside of `Database` (rare), catch
  `sqlite3.OperationalError` and retry with backoff rather than treating it
  as a hard failure — it usually means "the shared file was busy for a
  moment," not corruption.
