# Engram Dashboard

Web-based monitoring dashboard for the Engram system.

## Features

- **Service Status**: Real-time status of all system services
- **NATS Message Feed**: Live view of all NATS pub/sub messages
- **Memory & Beliefs Stats**: Episode and belief counts
- **Task Queue**: Recent learned tasks
- **System Metrics**: CPU, memory, and network stats per container
- **WebSocket Updates**: Real-time updates without page refresh

## Architecture

- **Backend**: FastAPI with WebSocket support
- **Frontend**: Vanilla JavaScript (no frameworks)
- **Styling**: Custom CSS with GitHub dark theme
- **Updates**: WebSocket for real-time data push

## Access

Once running, access the dashboard at:
```
http://localhost:8081
```

The default external port is 8081 (mapped from internal port 8080). This can be configured via the `DASHBOARD_PORT` environment variable in `.env`.

## Development

### Local Development

```bash
cd dashboard
pip install -r requirements.txt
python -m dashboard.api
```

### Docker Build

```bash
docker-compose build dashboard
docker-compose up dashboard
```

## API Endpoints

### Core
- `GET /api/health` - Health check
- `GET /api/system` - System info (OS, CPU, RAM, GPU)
- `GET /api/services` - Service status
- `GET /api/messages` - Recent NATS messages
- `GET /api/metrics` - Container metrics
- `GET /api/insights` - System insights

### Brain
- `GET /api/neuromorphic` - Neuromorphic brain state
- `GET /api/skills` - Skill registry
- `GET /api/skills/log` - Skill execution log
- `GET /api/knowledge` - Knowledge base entries
- `GET /api/flywheel` - Flywheel stats
- `GET /api/benchmark/latest` - Latest benchmark
- `GET /api/benchmark/history` - Benchmark history

### Gateway & Video
- `GET /api/gateway` - Gateway status
- `POST /api/gateway/command` - Send gateway command
- `GET /api/video/sessions` - Video training sessions
- `POST /api/video/submit` - Submit video URLs
- `POST /api/video/stop` - Stop video training
- `POST /api/video/queue` - Queue videos
- `POST /api/video/skip` - Skip current video
- `POST /api/video/clear-queue` - Clear video queue
- `POST /api/video/remove-queued` - Remove queued video
- `POST /api/video/blacklist` - Blacklist a video
- `GET /api/video/blacklist` - List blacklisted videos

### MuJoCo
- `GET /api/mujoco/model` - MuJoCo model info
- `POST /api/mujoco/guide` - Send motor guidance
- `GET /api/mujoco/joints` - Joint states

### Chat & Interaction
- `GET /api/chat/history` - Chat history
- `POST /api/chat` - Send chat message
- `POST /api/observation` - Inject observation
- `POST /api/concept-probe` - Run concept probe
- `GET /api/concept-probe/results` - Get probe results
- `DELETE /api/concept-probe/results` - Clear probe results

### WebSocket
- `WS /ws` - Real-time updates

## WebSocket Messages

The dashboard receives these message types:

- `init` - Initial data when connecting
- `message` - New NATS message
- `service_status` - Service status update
- `heartbeat` - Keep-alive ping

## Container Metrics

The dashboard accesses Docker stats via the Docker socket (`/var/run/docker.sock`) to provide real-time container metrics. This requires the socket to be mounted in docker-compose.yml.

## Dependencies

- fastapi
- uvicorn[standard]
- aiohttp
- websockets
- nats-py
- pydantic
