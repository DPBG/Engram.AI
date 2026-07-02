# Sim-to-Real Transfer Plan (MuJoCo → Physical Robot)

> **Status**: Design document (issue #139, Phase 5 — Embodied Autonomy).
> Describes the concrete gaps between Engram's current MuJoCo-only
> embodiment and a physical robot deployment, and a phased plan to close
> them. Nothing in this document is implemented yet unless noted
> "(exists today)".

## 1. Why this matters

Engram's brain currently learns motor behavior entirely inside a
simulated 29-DOF humanoid (`neuromorphic/src/neuromorphic/mujoco_body.py`).
`motor_feedback_adapter.py` already has a routing seam for real hardware
(per-channel heartbeat detection, §2.2 below), but no physical driver has
ever been plugged into it. Before adding IMU, depth-camera, or actuator
drivers (the M5 milestone), we need to be explicit about what MuJoCo is
*not* modeling today, because those gaps are exactly what will bite on
first contact with real hardware.

## 2. Concrete gaps in the current MuJoCo configuration

### 2.1 Physics fidelity

- **Fixed timestep, no domain randomization.** The model
  (`_HUMANOID_MJCF` in `mujoco_body.py:34-46`) runs at
  `timestep="0.002"`, uniform joint damping (`damping="5"
  armature="0.02"`) and isotropic friction (`friction="1.0 0.005
  0.0001"`) for every geom. Real linkages vary in friction, backlash,
  and inertia by tens of percent, and that variance is currently
  **zero** in sim — nothing forces the brain to learn motor policies
  that are robust to it.
- **No MJCF file — the model is an inline Python string.** There is no
  standalone `.xml`/MJCF asset checked into the repo; the model lives at
  `mujoco_body.py:34-194`. Any real chassis (different mass distribution,
  joint ranges, actuator count) requires editing this string directly
  today, with no versioned model file to diff or randomize.
- **Actuators are intensity-clamped, not torque/velocity-limited per
  joint.** `MotorFeedbackConfig.channel_actuators`
  (`neuromorphic/src/neuromorphic/config.py:421-436`) maps 3 broad
  channels (`locomotion`, `manipulation`, `head`) to lists of MJCF
  actuator names; the brain outputs a single 0–1 intensity per channel,
  fanned out to every actuator in that group. Real servos need
  per-joint torque, velocity, and position limits — none of which
  Engram's motor path currently expresses.

### 2.2 Actuation path — the Kernel gate is opt-in, not default

This is the single most important finding for physical deployment.
`SafetyGateConfig.enabled` defaults to **`False`**
(`config.py:451-458`, env `NEURO_SAFETY_GATE`). With the gate off,
`neuromorphic/service.py:1142-1151` is fire-and-forget: the motor
command is published to `proposal.new` for audit *and*, in the same
call, routed straight to `_motor_adapter.handle_motor_command(...)`
— i.e. straight to MuJoCo (or a real actuator) — without ever waiting
for a Kernel verdict. This is fine for a simulator with no physical
consequences. On real hardware it means the Kernel is **not** actually
the gatekeeper CLAUDE.md §3 describes unless `NEURO_SAFETY_GATE=1` is
set explicitly.

When the gate *is* enabled, the design is sound: `_motor_adapter.
handle_motor_command` is only invoked from the `decision_type ==
"ALLOW"` branch of `_await_safety_decision`
(`service.py:1209-1217`), and a Kernel-decision timeout
(`decision_timeout`, default 2.0s) never executes the command either
way — `fail_open` (default `True`, env `NEURO_SAFETY_FAIL_OPEN`) only
controls whether the dropped command is logged as neutral or fed back
as a negative (unsafe-pattern) learning signal, not whether it
actuates. So the fix here isn't a logic change — it's a deployment
requirement: **the safety gate must be mandatory, not opt-in, before a
`motor_feedback_adapter` channel is ever connected to real hardware.**

- **Body profiles don't distinguish sim from real.**
  `beliefs/src/beliefs/profiles.py:60-140` defines `motor_limits` as a
  single `max_intensity` (0–1) per channel with no concept of a
  simulated vs. physical body — the same profile YAML
  (`beliefs/profiles/base.yaml` etc.) would apply identically to
  MuJoCo and a real chassis, even though only one of them can actually
  be damaged.
- **Hardware routing exists; hardware drivers don't.**
  `motor_feedback_adapter.py:45-130` already auto-detects a real
  channel via `actuator.heartbeat.{channel}` and falls back to MuJoCo
  after `heartbeat_timeout_s` (30s default) — the seam is real. But no
  `ActuatorPlugin` (`sdk/src/activelearning/plugins.py:211+`)
  implementation exists for any physical actuator (servo, stepper,
  linear actuator) anywhere in the repo. Today, every channel always
  falls back to MuJoCo because nothing ever sends a heartbeat.

### 2.3 Sensing

- **IMU and contact-force "sensors" are derived from MuJoCo ground
  truth, not modeled as noisy real sensors.** Orientation and angular
  velocity are read directly from the root body's quaternion/`cvel`
  (`mujoco_body.py:691-903`); contact "force" is a penetration-depth
  proxy from `self._data.ncon` (`mujoco_body.py:909-920`). Real IMUs
  drift, have bias and temperature sensitivity; real contact/force
  sensing has noise and a different dynamic range. None of that is
  simulated, so STDP has never seen it.
- **`sensory-gateway/sensors/serial_device.py` is read-only.** It can
  poll a JSON-lines serial device into `observation.{sensor_id}`
  (useful for a real IMU or joint encoder as a *sensor*), but there is
  no equivalent serial/CAN/PWM *write* path for actuator commands —
  confirming §2.2's actuator-driver gap from the sensing side too.
- **Vision is a single trackcom camera, not the robot's own
  viewpoint.** The `"track"` camera (`mujoco_body.py:46`) that feeds
  the dashboard's "Brain's Eye View" is a third-person chase camera,
  not a simulated onboard/head-mounted camera — the visual training
  pipeline in `sensory-gateway/` (which does support real cameras) and
  the MuJoCo body's self-view are two different things that will need
  to be reconciled before a real onboard camera feed is a drop-in
  replacement.

### 2.4 Safety envelope

- **Software E-stop exists; a hardware dead-man's switch doesn't.**
  SAFE_HALT (`kernel/src/kernel/service.py:131-156`) is real: it sets
  the evaluator to deny-all, forces the Planner to `SAFE_HALT`, and
  zeros every motor channel's `max_intensity` via `policy.restrict` —
  but it is entirely NATS/software-mediated. If the process, NATS, or
  network dies, nothing physically removes power from a real actuator.
  A physical build needs a hardware kill switch that does not depend
  on any of Engram's software being alive.
- **Pain/nociception only reacts to joint-limit proximity.**
  `pain_enabled` (`config.py`, default `True` when motor feedback is
  on) triggers within `pain_limit_zone` (outer 20% of joint range) —
  it has no model of impact force, sustained overtorque, or thermal
  load, all of which matter for a real actuator and none of which
  exist in the current MJCF (no torque or temperature sensors are
  defined).
- **The "three layers of protection" described in
  [docs/ARCHITECTURE.md](ARCHITECTURE.md)
  (hardware limits → Kernel software limits → pain reflex) currently
  has only layer 2 partially built.** Layer 1 (servo/motor firmware
  limits) doesn't exist because no hardware driver exists yet; layer 3
  (pain reflex) is joint-limit-only as above.

### 2.5 Timing

- MuJoCo steps at `dt=0.002`s with `mujoco_steps_per_command=500`
  (1 simulated second per command) and a configurable
  `virtual_delay_ms` (default 75ms) standing in for real actuator
  latency. Real serial/CAN/network round-trips (sensor read + command
  write) commonly run 10–100ms+ with jitter that MuJoCo's fixed,
  deterministic delay does not reproduce. `motor_rate_limit_hz`
  (`config.py:396`, 0 = unlimited by default, "set 10–20 for physical
  hardware" per its own comment) is a real, existing knob for this —
  but it is not yet exercised by anything in the repo, because no
  hardware target exists to tune it against.

## 3. Phased validation methodology

Each phase gates entry into the next on an explicit, checkable
condition — no phase is time-boxed by calendar date alone, mirroring
the "experience-dependent, not hardcoded" spirit of CLAUDE.md
Invariant 2.

### Phase 0 — Make the gate mandatory (prerequisite, no hardware needed)

Close §2.2 before any physical wiring happens:
- Default `NEURO_SAFETY_GATE=1` for any deployment profile that isn't
  pure MuJoCo (keep the opt-out for sim-only dev).
- Add a body-profile flag (extending `beliefs/profiles.py`) that
  marks a profile as `embodiment: real`, and have the Kernel/evaluator
  refuse to run a `real` profile with the safety gate disabled —
  fail-closed, per CLAUDE.md §3.
- **Exit condition:** an integration test proves a motor command
  cannot reach `_motor_adapter.handle_motor_command` under a `real`
  profile without a Kernel ALLOW.

### Phase 1 — Sensor-only bring-up (no actuation)

Wire one real sensor (e.g. an IMU or a single joint encoder) as a
`SensorPlugin` (`sdk/src/activelearning/plugins.py:69+`), following the
existing `sensory-gateway/sensors/serial_device.py` pattern for the
transport. The brain does not act on it yet — this phase is purely
about comparing real sensor streams against MuJoCo's derived
proprioception (§2.3) offline, to characterize noise/drift/latency
before it ever touches a learning signal.
- **Exit condition:** logged real-vs-sim proprioception divergence is
  characterized (magnitude + latency) for the target sensor.

### Phase 2 — Single-channel, tethered actuation

Implement one `ActuatorPlugin` for one motor channel (start with
`head`, the lowest-risk/lowest-torque channel in
`channel_actuators`), with hard-coded conservative limits below the
hardware's rated max, physically tethered/restrained, and a human
finger on a hardware kill switch (§2.4) at all times. `heartbeat_timeout_s`
stays low so any silence falls back to MuJoCo immediately.
- **Exit condition:** N consecutive successful ALLOW→execute cycles
  with zero DENY-after-execute events (i.e. the Kernel never approved
  something that turned out unsafe in practice).

### Phase 3 — Domain-randomized sim pretraining

Before expanding beyond one channel, add domain randomization to the
MJCF (mass, friction, joint damping, actuator gain — §2.1) and retrain
in sim. This is cheap (sim-only) and directly reduces how much of the
Phase 2/4 gap has to be closed by real-world trial and error.
- **Exit condition:** policy performance is stable across a
  randomized-parameter sweep in MuJoCo, not just the nominal model.

### Phase 4 — Closed-loop supervised trials, full body

All channels wired via `ActuatorPlugin`/`SensorPlugin`, safety gate
mandatory (Phase 0), body profile `autonomy_level: human_on_loop`,
DEFER decisions routed to the dashboard operator (existing
`approval.request` flow, `neuromorphic/service.py:1245-1258`).
- **Exit condition:** a fixed evaluation task (e.g. stand + reach) is
  completed successfully across repeated trials with human approval
  on every DEFER, and zero hardware-limit violations.

### Phase 5 — Staged autonomy expansion

Only after Phase 4 is stable, consider raising `autonomy_level` /
`max_autonomous_duration_min` in the body profile. Gate this decision
on ALLOW/TRANSFORM/DENY/DEFER trend data for the `neuromorphic` source:
rising DENY/DEFER rates on real motor proposals is exactly the kind of
early-warning signal that should block further autonomy expansion. If
the Kernel decision-rate tracking proposed in issue #143 has landed by
this point, reuse it directly rather than building a parallel metric.

## 4. Staged rollout — M5 driver implementations

| Driver | Plugin base | Starting point | Status |
|---|---|---|---|
| IMU | `SensorPlugin` | `sensory-gateway/sensors/serial_device.py` (JSON-lines transport already works) | Not implemented — Phase 1 |
| Joint encoder | `SensorPlugin` | same as above | Not implemented — Phase 1 |
| Depth camera | `SensorPlugin` | `sensory-gateway/sensors/camera.py` (OpenCV capture works; depth-specific decoding is new) | Not implemented — Phase 1/2 |
| Servo/actuator (single channel) | `ActuatorPlugin` | `sdk/src/activelearning/plugins.py:211` interface; wire into `motor_feedback_adapter.py`'s existing heartbeat routing | Not implemented — Phase 2 |
| Hardware E-stop | N/A (out-of-band) | Independent of NATS/software stack; software SAFE_HALT already exists as the upstream trigger (`kernel/service.py:131`) | Not implemented — Phase 0/2 prerequisite |

No timeline is committed here — each row's exit criterion in §3 is the
gate, not a date.

## 5. Explicitly out of scope

- Isaac Sim (`docs/ARCHITECTURE.md` "Future: Isaac Sim Integration
  (Phase C)") is a *separate* planned system for Meta-Programmer
  code-testing (Stage 3 of the test pipeline), not the brain's
  embodiment loop described here. Worth revisiting once GPU hardware
  is available, but it doesn't substitute for the phases above.
- Choice of specific hardware (which servo protocol, which IMU part
  number, which chassis) is deliberately not decided in this document
  — the plugin interfaces in §4 are hardware-agnostic by design so
  that decision can be made independently.

## 6. References

- `neuromorphic/src/neuromorphic/mujoco_body.py` — MuJoCo body sim
- `neuromorphic/src/neuromorphic/motor_feedback_adapter.py` — sim/real routing
- `neuromorphic/src/neuromorphic/config.py` — `MotorFeedbackConfig`, `SafetyGateConfig`
- `neuromorphic/src/neuromorphic/service.py` — motor proposal → Kernel → actuation path
- `beliefs/src/beliefs/profiles.py`, `beliefs/profiles/*.yaml` — body profiles
- `kernel/src/kernel/service.py` — SAFE_HALT kill switch
- `sdk/src/activelearning/plugins.py` — `SensorPlugin` / `ActuatorPlugin`
- `sensory-gateway/sensors/` — existing sensor plugin implementations
- [docs/META-PROGRAMMER.md](META-PROGRAMMER.md) — Kernel decision types (ALLOW/TRANSFORM/DENY/DEFER), referenced in §3 Phase 5
- [issue #143](https://github.com/DPBG/Engram.AI/issues/143) — Kernel decision-rate governance signal (if landed, reuse for §3 Phase 5)
- [docs/ARCHITECTURE.md](ARCHITECTURE.md) — NATS bus, three-layers-of-protection framing, Isaac Sim (out of scope, §5)
- [CLAUDE.md](../CLAUDE.md) §3 — Kernel as sole decision authority, fail-closed requirement
