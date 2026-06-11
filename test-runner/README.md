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

Run integration tests:

```bash
# Via docker-compose
docker compose --profile testing up test-runner

# Or manually
docker build -t activelearning-test-runner:latest test-runner/
docker run --rm --network activelearning-public \
    -e NATS_URL=nats://nats:4222 \
    activelearning-test-runner:latest
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
