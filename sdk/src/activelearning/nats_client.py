"""
NATS Event Bus client for ActiveLearningAI.

Provides async pub/sub messaging for all components to communicate
through the NATS message bus.
"""

import asyncio
import json
import logging
import os
import time
from dataclasses import asdict, is_dataclass
from typing import Any, Callable, Optional, TypeVar, Awaitable

import nats
from nats.aio.client import Client as NATSClient
from nats.aio.msg import Msg
from nats.js import JetStreamContext

from activelearning.signing import verify_decision

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Type alias for message handlers
MessageHandler = Callable[[dict[str, Any]], Awaitable[None]]
# Handler that can also reply to NATS request-reply messages
ReplyHandler = Callable[[dict[str, Any], Msg], Awaitable[None]]


def serialize_message(data: Any) -> bytes:
    """Serialize a message payload to bytes."""
    if is_dataclass(data) and not isinstance(data, type):
        data = asdict(data)
    return json.dumps(data, default=str).encode("utf-8")


def deserialize_message(data: bytes) -> dict[str, Any]:
    """Deserialize bytes to a message payload."""
    return json.loads(data.decode("utf-8"))


class EventBus:
    """
    NATS-based event bus for component communication.

    All components in ActiveLearningAI communicate through this bus.
    Supports pub/sub patterns, request/reply, and JetStream persistence.

    Usage:
        bus = EventBus()
        await bus.connect()

        # Subscribe to observations
        async def handle_obs(data):
            print(f"Got observation: {data}")

        await bus.subscribe("observation.*", handle_obs)

        # Publish an observation
        await bus.publish("observation.camera", {"frame": "..."})

        # Cleanup
        await bus.close()
    """

    def __init__(
        self,
        nats_url: Optional[str] = None,
        name: str = "activelearning",
    ):
        """
        Initialize the EventBus.

        Args:
            nats_url: NATS server URL (defaults to NATS_URL env var or localhost)
            name: Client name for identification
        """
        self.nats_url = nats_url or os.environ.get("NATS_URL", "nats://localhost:4222")
        self.name = name
        self._nc: Optional[NATSClient] = None
        self._js: Optional[JetStreamContext] = None
        self._subscriptions: dict[str, nats.aio.subscription.Subscription] = {}
        self._handlers: dict[str, MessageHandler] = {}
        self._connected = asyncio.Event()

    async def connect(self) -> None:
        """Connect to the NATS server."""
        if self._nc is not None and self._nc.is_connected:
            logger.debug("Already connected to NATS")
            return

        logger.info(f"Connecting to NATS at {self.nats_url}")

        self._nc = await nats.connect(
            self.nats_url,
            name=self.name,
            error_cb=self._error_callback,
            disconnected_cb=self._disconnected_callback,
            reconnected_cb=self._reconnected_callback,
            max_reconnect_attempts=-1,  # Unlimited reconnection attempts
        )

        # Initialize JetStream for persistence
        self._js = self._nc.jetstream()

        self._connected.set()
        logger.info("Connected to NATS successfully")

    async def close(self) -> None:
        """Close the NATS connection."""
        if self._nc is not None:
            logger.info("Closing NATS connection")
            await self._nc.drain()
            await self._nc.close()
            self._nc = None
            self._js = None
            self._connected.clear()
            self._subscriptions.clear()

    async def publish(self, subject: str, data: Any) -> None:
        """
        Publish a message to a subject.

        Args:
            subject: NATS subject (e.g., "observation.camera")
            data: Message payload (dataclass or dict)
        """
        await self._ensure_connected()
        payload = serialize_message(data)
        await self._nc.publish(subject, payload)
        logger.debug(f"Published to {subject}: {len(payload)} bytes")

    async def subscribe(
        self,
        subject: str,
        handler: MessageHandler,
        queue: Optional[str] = None,
        pending_msgs_limit: int = 65536,
        pending_bytes_limit: int = 128 * 1024 * 1024,
        is_request_handler: bool = False,
    ) -> None:
        """
        Subscribe to a subject with a message handler.

        Args:
            subject: NATS subject pattern (supports wildcards like "observation.*")
            handler: Async function to handle messages
            queue: Optional queue group for load balancing
            pending_msgs_limit: Max pending messages before slow consumer (default 65536)
            pending_bytes_limit: Max pending bytes before slow consumer (default 128 MB)
            is_request_handler: If True, handler receives (data, msg) so it can
                reply to NATS request-reply messages via msg.respond()
        """
        await self._ensure_connected()

        if subject in self._subscriptions:
            logger.warning(f"Already subscribed to {subject}, unsubscribing first")
            await self.unsubscribe(subject)

        async def message_callback(msg: Msg) -> None:
            try:
                data = deserialize_message(msg.data)
                if is_request_handler:
                    await handler(data, msg)
                else:
                    await handler(data)
            except Exception as e:
                logger.error(f"Error handling message on {subject}: {e}")
                # For request-reply handlers, send an error response so the
                # caller doesn't hang until timeout.
                if is_request_handler and msg.reply:
                    try:
                        await msg.respond(serialize_message({
                            "error": str(e),
                            "type": "error",
                        }))
                    except Exception:
                        pass  # best-effort error reply

        sub = await self._nc.subscribe(
            subject,
            cb=message_callback,
            queue=queue or "",
            pending_msgs_limit=pending_msgs_limit,
            pending_bytes_limit=pending_bytes_limit,
        )
        self._subscriptions[subject] = sub
        self._handlers[subject] = handler
        logger.info(f"Subscribed to {subject}")

    async def unsubscribe(self, subject: str) -> None:
        """Unsubscribe from a subject."""
        if subject in self._subscriptions:
            await self._subscriptions[subject].unsubscribe()
            del self._subscriptions[subject]
            if subject in self._handlers:
                del self._handlers[subject]
            logger.info(f"Unsubscribed from {subject}")

    async def request(
        self,
        subject: str,
        data: Any,
        timeout: float = 30.0,
    ) -> dict[str, Any]:
        """
        Send a request and wait for a response.

        Args:
            subject: NATS subject
            data: Request payload
            timeout: Timeout in seconds

        Returns:
            Response data as dict

        Raises:
            asyncio.TimeoutError: If no response within timeout
        """
        await self._ensure_connected()
        payload = serialize_message(data)
        response = await self._nc.request(subject, payload, timeout=timeout)
        return deserialize_message(response.data)

    async def wait_for_decision(
        self,
        trace_id: str,
        timeout: float = 30.0,
    ) -> dict[str, Any]:
        """
        Wait for a Kernel decision on a specific trace_id.

        Args:
            trace_id: The trace ID to wait for
            timeout: Timeout in seconds

        Returns:
            Decision data as dict
        """
        subject = f"decision.{trace_id}"
        decision_received = asyncio.Event()
        result: dict[str, Any] = {}

        async def handler(data: dict[str, Any]) -> None:
            nonlocal result
            # Authenticate the decision. A forged or unsigned decision (when
            # signing is enabled) is ignored, so it cannot satisfy the wait —
            # the caller times out and must fail closed (deny/halt).
            if not verify_decision(data):
                logger.error(
                    "Rejected decision on %s: missing/invalid signature "
                    "(possible forgery) — ignoring and continuing to wait.",
                    subject,
                )
                return
            result = data
            decision_received.set()

        await self.subscribe(subject, handler)
        try:
            await asyncio.wait_for(decision_received.wait(), timeout=timeout)
            return result
        finally:
            await self.unsubscribe(subject)

    async def force_reconnect(self) -> None:
        """Tear down a dead NATS connection and create a fresh one.

        nats-py's built-in reconnection sometimes fails silently -- is_connected
        returns False but no reconnect happens.  This method:
        1. Saves current subscription subjects + handlers
        2. Closes the dead connection (best-effort)
        3. Creates a completely new connection
        4. Re-subscribes all previously registered handlers

        Safe to call even if already connected (becomes a no-op reconnect).
        """
        saved_handlers: dict[str, tuple[MessageHandler, bool]] = {}
        for subject, sub in self._subscriptions.items():
            handler = self._handlers.get(subject)
            if handler is not None:
                # Detect if this was a request handler by checking stored metadata
                # We store the original handler, not the wrapper
                saved_handlers[subject] = (handler, False)

        logger.warning(f"force_reconnect: tearing down NATS connection ({len(saved_handlers)} subs to restore)")

        # Best-effort close of old connection
        if self._nc is not None:
            try:
                await asyncio.wait_for(self._nc.close(), timeout=5.0)
            except Exception as e:
                logger.warning(f"force_reconnect: close failed (expected): {e}")
            self._nc = None
            self._js = None
            self._connected.clear()
            self._subscriptions.clear()

        # Create fresh connection
        await self.connect()

        # Re-subscribe all handlers
        for subject, (handler, is_req) in saved_handlers.items():
            try:
                await self.subscribe(subject, handler, is_request_handler=is_req)
            except Exception as e:
                logger.error(f"force_reconnect: failed to re-subscribe {subject}: {e}")

        logger.info(f"force_reconnect: complete, {len(self._subscriptions)} subs restored")

    @property
    def is_connected(self) -> bool:
        """Check if connected to NATS."""
        return self._nc is not None and self._nc.is_connected

    async def _ensure_connected(self) -> None:
        """Ensure we're connected, waiting briefly for reconnection if needed."""
        if self.is_connected:
            return
        # NATS client may be reconnecting — wait up to 10s before giving up.
        # This prevents transient disconnects from crashing publish loops.
        if self._nc is not None:
            try:
                await asyncio.wait_for(self._connected.wait(), timeout=10.0)
                return
            except asyncio.TimeoutError:
                pass
        raise RuntimeError("Not connected to NATS. Call connect() first.")

    async def _error_callback(self, e: Exception) -> None:
        """Handle NATS errors.

        Several nats-py async errors stringify to an empty message, so always
        include the exception type (and repr) to keep the log diagnosable.
        Repeated identical errors (e.g. reconnect TimeoutError during a NATS
        outage) are throttled to once every 15s to avoid flooding the console.
        """
        etype = type(e).__name__
        now = time.monotonic()
        last_type = getattr(self, "_last_error_type", None)
        last_ts = getattr(self, "_last_error_ts", 0.0)
        if etype == last_type and (now - last_ts) < 15.0:
            return  # same error still recurring — already reported recently
        self._last_error_type = etype
        self._last_error_ts = now
        detail = str(e).strip()
        logger.error("NATS error [%s]: %s", etype, detail or repr(e))

    async def _disconnected_callback(self) -> None:
        """Handle NATS disconnection."""
        self._connected.clear()
        logger.warning("Disconnected from NATS")

    async def _reconnected_callback(self) -> None:
        """Handle NATS reconnection."""
        self._connected.set()
        logger.info("Reconnected to NATS")


# Global event bus instance for convenience
_global_bus: Optional[EventBus] = None


async def get_event_bus() -> EventBus:
    """Get or create the global EventBus instance."""
    global _global_bus
    if _global_bus is None:
        _global_bus = EventBus()
        await _global_bus.connect()
    return _global_bus


async def publish(subject: str, data: Any) -> None:
    """Convenience function to publish via global bus."""
    bus = await get_event_bus()
    await bus.publish(subject, data)


async def subscribe(
    subject: str,
    handler: MessageHandler,
    queue: Optional[str] = None,
) -> None:
    """Convenience function to subscribe via global bus."""
    bus = await get_event_bus()
    await bus.subscribe(subject, handler, queue)
