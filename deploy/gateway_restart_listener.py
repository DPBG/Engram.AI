#!/usr/bin/env python3
"""Gateway restart listener -- runs alongside the gateway tmux session.

Subscribes to `gateway.restart.request` on NATS. When the brain publishes
this subject (after prolonged sensory starvation), this script kills and
restarts the gateway tmux session.

Usage:
    python3 gateway_restart_listener.py [--nats-url nats://127.0.0.1:4222]

Install as a systemd service or run in a separate tmux pane.
"""

import argparse
import asyncio
import json
import logging
import os
import signal
import subprocess
import time

import nats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [gateway-restart-listener] %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

TMUX_SESSION = "gateway"
STARTUP_SCRIPT = "/data/start_gateway.sh"
# Minimum cooldown between restarts to prevent flapping
MIN_RESTART_INTERVAL = 120  # seconds


class GatewayRestartListener:
    def __init__(self, nats_url: str):
        self.nats_url = nats_url
        self._nc = None
        self._last_restart = 0.0
        self._running = True

    async def run(self):
        self._nc = await nats.connect(
            self.nats_url,
            name="gateway-restart-listener",
            max_reconnect_attempts=-1,
        )
        logger.info("Connected to NATS at %s", self.nats_url)

        await self._nc.subscribe("gateway.restart.request", cb=self._handle_restart)
        logger.info("Listening for gateway.restart.request...")

        # Keep running until signaled
        while self._running:
            await asyncio.sleep(1)

        if self._nc and not self._nc.is_closed:
            await self._nc.drain()

    async def _handle_restart(self, msg):
        try:
            data = json.loads(msg.data.decode("utf-8"))
        except Exception:
            data = {}

        now = time.monotonic()
        elapsed = now - self._last_restart
        if elapsed < MIN_RESTART_INTERVAL:
            logger.warning(
                "Restart request from brain (step=%s, gap=%s) -- "
                "cooldown active (%.0fs remaining)",
                data.get("step", "?"),
                data.get("gap", "?"),
                MIN_RESTART_INTERVAL - elapsed,
            )
            return

        logger.warning(
            "Restart request from brain (step=%s, gap=%s) -- restarting gateway",
            data.get("step", "?"),
            data.get("gap", "?"),
        )
        self._last_restart = now
        self._restart_gateway()

    def _restart_gateway(self):
        # Kill existing tmux session
        try:
            subprocess.run(
                ["tmux", "kill-session", "-t", TMUX_SESSION],
                capture_output=True,
                timeout=10,
            )
            logger.info("Killed tmux session '%s'", TMUX_SESSION)
        except Exception as e:
            logger.warning("Could not kill tmux session: %s", e)

        time.sleep(2)

        # Start fresh
        try:
            subprocess.run(
                ["tmux", "new-session", "-d", "-s", TMUX_SESSION, f"bash {STARTUP_SCRIPT}"],
                capture_output=True,
                timeout=10,
            )
            logger.info("Started new tmux session '%s'", TMUX_SESSION)
        except Exception as e:
            logger.error("Failed to start gateway: %s", e)

    def stop(self):
        self._running = False


def main():
    parser = argparse.ArgumentParser(description="Gateway restart listener")
    parser.add_argument(
        "--nats-url",
        default=os.environ.get("NATS_URL", "nats://127.0.0.1:4222"),
    )
    args = parser.parse_args()

    listener = GatewayRestartListener(args.nats_url)

    loop = asyncio.new_event_loop()

    def handle_signal(sig, frame):
        logger.info("Signal %s received, shutting down", sig)
        listener.stop()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        loop.run_until_complete(listener.run())
    except KeyboardInterrupt:
        pass
    finally:
        loop.close()
        logger.info("Shutdown complete")


if __name__ == "__main__":
    main()
