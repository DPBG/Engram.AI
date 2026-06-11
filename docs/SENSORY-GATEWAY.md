# Sensory Gateway

The sensory gateway discovers hardware sensors on the host machine and streams preprocessed data to the neuromorphic brain over NATS. It runs on the host (not Docker) because it needs direct access to camera, microphone, and USB devices.

## Architecture

```
┌─── Host Mac ──────────────────────────────┐     ┌─── Docker ────────────────┐
│                                            │     │                           │
│  sensory-gateway/                         │     │  neuromorphic service      │
│  ├─ discovery.py (find hardware)          │     │  subscribes: observation.* │
│  ├─ gateway.py   (orchestrate)            │     │                           │
│  └─ sensors/                              │     │  encoding pipeline:        │
│     ├─ camera.py     → observation.camera │────→│  visual → 30K neurons     │
│     ├─ microphone.py → observation.mic    │────→│  auditory → 20K neurons   │
│     ├─ speech.py     → observation.voice  │────→│  auditory (text ordinals) │
│     └─ serial_device.py → observation.serial──→│  proprioceptive/tactile   │
│                                            │     │                           │
│  NATS @ localhost:4222                     │     │  DynamicSensoryAllocator  │
│  (exposed by docker-compose)               │     │  auto-rebalances neurons  │
└────────────────────────────────────────────┘     └───────────────────────────┘
```

## Quick Start

```bash
cd sensory-gateway

# Install (uses the SDK from ../sdk)
pip install -e ../sdk
pip install -e .

# Optional: speech-to-text
pip install -e ".[stt]"

# List available sensors
python gateway.py --list

# Run with camera + mic
python gateway.py

# Run with speech-to-text enabled
python gateway.py --stt

# Custom settings
python gateway.py --camera-fps 2 --audio-hz 10 --stt --stt-model base
```

## How Discovery Works

On startup, the gateway probes:

1. **Cameras** — OpenCV probes indices 0-9 via `cv2.VideoCapture(i)`
2. **Microphones** — `sounddevice.query_devices()` lists all PortAudio inputs
3. **Serial devices** — `serial.tools.list_ports.comports()` finds USB serial

Each discovered device is mapped to a `SensorPlugin` subclass from the SDK. The plugin handles its own capture loop and publishes to NATS.

### Network sensors (Raspberry Pi, ESP32)

Remote devices connect to NATS directly. A Pi running a servo encoder would:

```python
# On the Raspberry Pi:
import asyncio, json, nats

async def main():
    nc = await nats.connect("nats://mac-ip:4222")
    while True:
        reading = {"angle": read_servo_angle(), "torque": read_torque()}
        await nc.publish(
            "observation.servo",
            json.dumps({"provenance": "sensor.joint", "data": reading}).encode()
        )
        await asyncio.sleep(0.1)

asyncio.run(main())
```

No gateway changes needed — the neuromorphic service subscribes to `observation.*` and picks up any matching message. The provenance `sensor.joint` maps to the proprioceptive modality.

## Provenance → Brain Modality Mapping

The neuromorphic encoding pipeline maps provenance strings to brain regions:

| Provenance | Modality | Brain Region |
|-----------|----------|-------------|
| `sensor.camera`, `sensor.video`, `sensor.image` | visual | Sensory cortex 0-60% |
| `sensor.microphone`, `sensor.audio`, `sensor.voice` | auditory | Sensory cortex 60-100% |
| `sensor.text`, `observation.text` | auditory | (text as character ordinals) |
| `sensor.touch`, `sensor.pressure`, `sensor.tactile` | tactile | Sensory cortex (when active) |
| `sensor.imu`, `sensor.gyro`, `sensor.joint`, `sensor.motor` | proprioceptive | Sensory cortex (when active) |

Neuron allocation is dynamic — the `DynamicSensoryAllocator` redistributes the 50K sensory neurons based on which modalities are actively receiving data.

## NATS Message Contract

The neuromorphic service expects this JSON format on `observation.*`:

```json
{
  "provenance": "sensor.camera",
  "data": [0.1, 0.5, 0.3, ...]
}
```

The `data` field can be:
- **List of floats** — used directly (camera pixels, MFCC coefficients)
- **Dict with numeric values** — values extracted as feature vector
- **String** — characters converted to ordinals/127 (speech transcription)
- **Nested dict** — looks for `features`, `data`, `values`, or `vector` keys

Queue: max 100 observations buffered, silently drops when full (100ms timeout).

## Writing a New Sensor

To add a new sensor type (e.g., LIDAR, GPS, temperature):

### 1. Create the adapter file

```python
# sensory-gateway/sensors/lidar.py
from activelearning.plugins import SensorPlugin, PluginCapability, RiskClass

class LidarSensor(SensorPlugin[list]):
    def __init__(self, device_path: str):
        super().__init__(
            sensor_id=f"lidar.{device_path.split('/')[-1]}",
            name="LIDAR Scanner",
            rate_limit_hz=10.0,
            risk_class=RiskClass.LOW,
        )
        self._device_path = device_path
        self.add_capability(PluginCapability(
            name="lidar_scan",
            description="360-degree distance scan",
            parameters={"points": "360", "format": "distance_meters"},
        ))

    async def start(self, bus=None):
        # Open device connection
        await super().start(bus)

    async def capture(self) -> list:
        # Read one scan, return as list of floats
        return [0.0] * 360  # placeholder

    async def stop(self):
        await super().stop()
```

### 2. Add discovery (optional)

In `discovery.py`, add detection logic — e.g., check for a known USB VID/PID in `discover_serial_devices()`, or add a new `discover_lidar()` function.

### 3. Add to gateway.py

Map the device type to the sensor class in `create_sensor()`:

```python
elif device.device_type == "lidar":
    from sensors.lidar import LidarSensor
    return LidarSensor(device_path=device.metadata["port"])
```

### 4. Add provenance mapping (if needed)

If the sensor's provenance doesn't match an existing modality, add it to the neuromorphic encoding pipeline:

```python
# neuromorphic/src/neuromorphic/encoding.py, _PROVENANCE_MAP
"sensor.lidar": "visual",  # or create a new modality
```

## Data Flow: Camera Frame to Brain Activity

```
1. OpenCV reads 640x480 RGB frame
2. CameraSensor.capture() → 64x64 grayscale → 4096 floats
3. SensorPlugin._emit_loop() → Observation(provenance="sensor.camera.0", data=[...])
4. EventBus.publish("observation.camera.0", observation) → NATS
5. Neuromorphic service._handle_observation() → queue
6. SpikeEncoder.encode(data, provenance="sensor.camera.0")
   - Provenance resolves to "visual" modality
   - DynamicSensoryAllocator assigns visual → neurons 0-30,000
   - _extract_features() → 4096-element float32 array
   - _rate_encode() → normalize [0,1], scale by 100 Hz gain, add noise
   - Tile 4096 → 30,000 visual neurons
7. Sensory cortex steps with injected current
8. STDP updates sensory→association connections
9. Association cortex binds visual with concurrent auditory/other input
```

## CLI Reference

```
python gateway.py [options]

Sensor Options:
  --nats URL              NATS server URL (default: nats://localhost:4222)
  --list                  List discovered sensors and exit
  --no-camera             Skip camera sensors
  --no-mic                Skip microphone sensors
  --camera-fps N          Camera capture rate in FPS (default: 5.0)
  --audio-hz N            Audio MFCC extraction rate in Hz (default: 30.0)
  --stt                   Enable speech-to-text via Whisper
  --stt-model SIZE        Whisper model: tiny|base|small|medium (default: tiny)

Video Training:
  --video PATH [PATH ...] Video file path(s) or YouTube URL(s) to play as training input
  --video-fps N           Video playback FPS (default: 2.0)
  --video-loop            Loop video playback
  --video-transcript      Enable Whisper transcription of video audio
  --turbo                 Turbo training mode: remove sleep between frames for max throughput
  --batch-size N          Number of video sensors to start per batch (default: 25)
  --batch-delay N         Seconds to wait between sensor startup batches (default: 2.0)

Preprocessing:
  --cnn                   Enable CNN feature extraction (MobileNetV3) for visual sensors
  --no-cnn                Disable CNN feature extraction (use raw 64x64 pixels)
  --sparse-threshold N    Sparse encoding threshold - skip frames below this (default: 0.05)
  --no-aggregate          Disable observation aggregation (send every message individually)
  --aggregate-hz N        Aggregation flush rate in Hz (default: 2.0)

General:
  -v, --verbose           Enable debug logging
```
