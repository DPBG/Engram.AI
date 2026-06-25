# Test Runner

Integration testing framework with mock hardware for Engram.

## Purpose

The Test Runner provides a complete testing environment with:
- Mock sensors (camera, GPIO, IMU)
- Mock actuators (servo, LED, motor)
- NATS message flow testing
- SDK integration verification

## Mock Hardware

### Sensors

- **MockCamera**: Simulates camera with configurable resolution and FPS
- **MockGPIO**: Simulates digital/analog GPIO pins
- **MockIMU**: Simulates accelerometer, gyroscope, magnetometer

### Actuators

- **MockServo**: Simulates servo motor with position control (0-180°)
- **MockLED**: Simulates LED with brightness and RGB color
- **MockMotor**: Simulates DC motor with speed and direction

## Usage

The suite has two tiers:

- **Bus integration tests** (`test_nats_flow.py`, `test_jetstream_durability.py`,
  `test_sdk_integration.py`) run against a live NATS broker only — no other
  services required. These run in CI (the `integration` job) with hard
  assertions.
- **Full-stack flow tests** (`test_core_services.py`) exercise
  observation → planner → kernel → actuator paths and require the actual
  services to be running. They self-skip their assertions when services are
  absent, so they are **not** part of the CI gate.

### Bus integration tests (broker only)

```bash
# Start a JetStream-enabled broker, then run the broker-only subset:
nats-server -js &
NATS_URL=nats://localhost:4222 python -m pytest \
    src/test_runner/tests/test_nats_flow.py \
    src/test_runner/tests/test_jetstream_durability.py \
    src/test_runner/tests/test_sdk_integration.py -v
```

### Full-stack flow tests (requires running services)

Bring up the stack (see the repository `RUN-LOCAL.md` or `docker compose up`),
then point the tests at the same broker:

```bash
NATS_URL=nats://localhost:4222 python -m pytest \
    src/test_runner/tests/test_core_services.py -v
```

## Test Structure

```
test-runner/
├── src/
│   └── test_runner/
│       ├── mocks/          # Mock hardware implementations
│       │   ├── sensors.py
│       │   └── actuators.py
│       └── tests/          # Integration tests
│           ├── conftest.py
│           ├── test_core_services.py
│           ├── test_nats_flow.py
│           └── test_sdk_integration.py
└── Dockerfile
```

## Performance Targets

- **Full integration test suite**: ~30 seconds
- **Individual test**: <5 seconds
- **NATS message flow**: <1 second roundtrip

## Writing Tests

Example integration test:

```python
import pytest
from test_runner.mocks import MockCamera

@pytest.mark.asyncio
async def test_camera_capture():
    camera = MockCamera(width=640, height=480, fps=30)
    await camera.start()

    frame = await camera.capture()
    assert frame["width"] == 640
    assert frame["height"] == 480
    assert "data" in frame

    await camera.stop()
```

## Environment Variables

- `NATS_URL`: NATS server URL (default: `nats://localhost:4222`)
- `QDRANT_URL`: Qdrant server URL (default: `http://localhost:6333`)
- `SQLITE_PATH`: SQLite database path (default: `/data/sqlite/unified.db`)

## Continuous Integration

The Test Runner is used in CI/CD pipelines to verify:
- SDK functionality
- NATS message routing
- Planner → Kernel → Actuator flow
- Memory and belief system integration
- Meta-Programmer code generation (via sandbox)
