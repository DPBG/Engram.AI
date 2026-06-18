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
from nats.js.api import ConsumerConfig, DeliverPolicy, StreamConfig

from activelearning.signing import verify_decision

logger = logging.getLogger(__name__)

# JetStream stream that guarantees delivery for every safety-critical subject.
# Proposals and decisions must never be fire-and-forget.
SAFETY_STREAM_NAME = "SAFETY_CRITICAL"
_SAFETY_STREAM_SUBJECTS: list[str] = [
    "proposal.new",
    "code.proposal",
    "decision.>",
    "code.decision.>",
]
# Auto-delete idle waiter consumers after this many nanoseconds of inactivity.
_CONSUMER_INACTIVE_NS: int = 60 * 1_000_000_000  # 60 s

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
        self._js_durables: dict[str, str] = {}  # subject -> durable consumer name
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
        await self._ensure_safety_stream()

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

    async def _ensure_safety_stream(self) -> None:
        """Create or update the durable JetStream stream for safety-critical subjects.

        Idempotent — NATS upserts the stream if it already exists with a
        compatible config. Raises on hard failures (JetStream not enabled, etc.).
        """
        assert self._js is not None
        config = StreamConfig(
            name=SAFETY_STREAM_NAME,
            subjects=_SAFETY_STREAM_SUBJECTS,
        )
        await self._js.add_stream(config)
        logger.info("JetStream stream '%s' ready", SAFETY_STREAM_NAME)

    @staticmethod
    def _is_safety_critical(subject: str) -> bool:
        """Return True when this subject must use JetStream persistence."""
        if subject in ("proposal.new", "code.proposal"):
            return True
        if subject.startswith("decision.") or subject.startswith("code.decision."):
            return True
        return False

    async def publish(self, subject: str, data: Any) -> None:
        """
        Publish a message to a subject.

        Safety-critical subjects (proposal.new, code.proposal, decision.*)
        are published via JetStream so they are persisted and survive a broker
        restart.  All other subjects use core NATS.

        Args:
            subject: NATS subject (e.g., "observation.camera")
            data: Message payload (dataclass or dict)
        """
        await self._ensure_connected()
        payload = serialize_message(data)
        if self._is_safety_critical(subject):
            assert self._js is not None
            ack = await self._js.publish(subject, payload)
            logger.debug("JS-published to %s (seq=%d): %d bytes", subject, ack.seq, len(payload))
        else:
            await self._nc.publish(subject, payload)
            logger.debug("Published to %s: %d bytes", subject, len(payload))

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

    async def js_subscribe(
        self,
        subject: str,
        handler: MessageHandler,
        durable: str,
    ) -> None:
        """Subscribe to a JetStream subject with a named durable push consumer.

        Unlike the core subscribe(), this consumer survives a broker restart —
        any un-ACKed messages are redelivered when the connection is restored.
        The caller's handler receives deserialized dicts as usual; ACK/NAK is
        handled automatically by this wrapper.

        Args:
            subject: JetStream subject (must be covered by SAFETY_STREAM_NAME).
            handler: Async function called with the deserialized payload dict.
            durable: Unique consumer name; stable across restarts.
        """
        await self._ensure_connected()
        assert self._js is not None

        if subject in self._subscriptions:
            logger.warning("Already subscribed to %s, unsubscribing first", subject)
            await self.unsubscribe(subject)

        async def _js_message_callback(msg: Msg) -> None:
            try:
                data = deserialize_message(msg.data)
                await handler(data)
                await msg.ack()
            except Exception as e:
                logger.error("Error handling JS message on %s: %s", subject, e)
                await msg.nak()

        sub = await self._js.subscribe(subject, cb=_js_message_callback, durable=durable)
        self._subscriptions[subject] = sub
        self._handlers[subject] = handler
        self._js_durables[subject] = durable
        logger.info("JS-subscribed to %s (durable=%s)", subject, durable)

    async def unsubscribe(self, subject: str) -> None:
        """Unsubscribe from a subject."""
        if subject in self._subscriptions:
            await self._subscriptions[subject].unsubscribe()
            del self._subscriptions[subject]
            if subject in self._handlers:
                del self._handlers[subject]
            self._js_durables.pop(subject, None)
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

        Uses a JetStream durable consumer with deliver_all so that a decision
        already stored in the stream (e.g. published while this client was
        briefly reconnecting) is still received.  The consumer auto-expires
        after 60 s of inactivity to avoid accumulation.

        Args:
            trace_id: The trace ID to wait for
            timeout: Timeout in seconds

        Returns:
            Decision data as dict
        """
        subject = f"decision.{trace_id}"
        durable = f"waiter-{trace_id}"
        decision_received = asyncio.Event()
        result: dict[str, Any] = {}

        async def _js_handler(msg: Msg) -> None:
            nonlocal result
            data = deserialize_message(msg.data)
            # Authenticate the decision. A forged or unsigned decision (when
            # signing is enabled) is ignored, so it cannot satisfy the wait —
            # the caller times out and must fail closed (deny/halt).
            if not verify_decision(data):
                logger.error(
                    "Rejected decision on %s: missing/invalid signature "
                    "(possible forgery) — ignoring and continuing to wait.",
                    subject,
                )
                await msg.ack()  # remove from stream; retrying won't fix a bad sig
                return
            result = data
            decision_received.set()
            await msg.ack()

        assert self._js is not None
        sub = await self._js.subscribe(
            subject,
            cb=_js_handler,
            durable=durable,
            config=ConsumerConfig(
                deliver_policy=DeliverPolicy.ALL,
                inactive_threshold=_CONSUMER_INACTIVE_NS,
            ),
        )
        try:
            await asyncio.wait_for(decision_received.wait(), timeout=timeout)
            return result
        finally:
            await sub.unsubscribe()

    async def force_reconnect(self) -> None:
        """Tear down a dead NATS connection and create a fresh one.

        nats-py's built-in reconnection sometimes fails silently -- is_connected
        returns False but no reconnect happens.  This method:
        1. Saves current subscription subjects + handlers (core and JS)
        2. Closes the dead connection (best-effort)
        3. Creates a completely new connection (which re-ensures the safety stream)
        4. Re-subscribes all previously registered handlers

        Safe to call even if already connected (becomes a no-op reconnect).
        """
        saved_handlers: dict[str, tuple[MessageHandler, bool]] = {}
        saved_js: dict[str, str] = dict(self._js_durables)  # subject -> durable name
        for subject in self._subscriptions:
            handler = self._handlers.get(subject)
            if handler is not None:
                saved_handlers[subject] = (handler, False)

        logger.warning(
            "force_reconnect: tearing down NATS connection (%d subs to restore)",
            len(saved_handlers),
        )

        # Best-effort close of old connection
        if self._nc is not None:
            try:
                await asyncio.wait_for(self._nc.close(), timeout=5.0)
            except Exception as e:
                logger.warning("force_reconnect: close failed (expected): %s", e)
            self._nc = None
            self._js = None
            self._connected.clear()
            self._subscriptions.clear()
            self._js_durables.clear()

        # Create fresh connection (also re-ensures the safety stream)
        await self.connect()

        # Re-subscribe all handlers, preserving JS vs core distinction
        for subject, (handler, is_req) in saved_handlers.items():
            try:
                if subject in saved_js:
                    await self.js_subscribe(subject, handler, durable=saved_js[subject])
                else:
                    await self.subscribe(subject, handler, is_request_handler=is_req)
            except Exception as e:
                logger.error("force_reconnect: failed to re-subscribe %s: %s", subject, e)

        logger.info("force_reconnect: complete, %d subs restored", len(self._subscriptions))

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
