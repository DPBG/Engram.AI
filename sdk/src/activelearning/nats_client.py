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
from collections.abc import Awaitable, Callable
from dataclasses import asdict, is_dataclass
from typing import Any, TypeVar

import nats
from nats.aio.client import Client as NATSClient
from nats.aio.msg import Msg
from nats.js import JetStreamContext
from nats.js.api import AckPolicy, ConsumerConfig, DeliverPolicy, StreamConfig
from pydantic import BaseModel

from activelearning.messages import (
    KernelDecisionMessage,
    MessageValidationError,
    schema_for_subject,
    validate_payload,
)
from activelearning.signing import verify_decision
from activelearning.subjects import Subjects, code_decision_subject, decision_subject

logger = logging.getLogger(__name__)

# JetStream stream that guarantees delivery for every safety-critical subject.
# Proposals and decisions must never be fire-and-forget.
SAFETY_STREAM_NAME = "SAFETY_CRITICAL"
_SAFETY_STREAM_SUBJECTS: list[str] = [
    Subjects.PROPOSAL_NEW,
    Subjects.CODE_PROPOSAL,
    f"{Subjects.DECISION_PREFIX}>",
    f"{Subjects.CODE_DECISION_PREFIX}>",
]
# Auto-delete idle waiter consumers after this many seconds of inactivity.
_CONSUMER_INACTIVE_THRESHOLD_S: float = 60.0

# Durable-consumer defaults for safety-critical streams (E2.3.3).
# Explicit ack with bounded redelivery: a handler that fails is retried with
# backoff up to MAX_DELIVER times, after which the message is routed to the
# poison/dead-letter path instead of redelivering forever or being dropped.
DEFAULT_ACK_WAIT_SECONDS: float = 30.0
DEFAULT_MAX_DELIVER: int = 5
# Backoff between redeliveries (seconds); length should be MAX_DELIVER - 1 so
# every retry has its own interval. The server uses backoff[0] as the ack-wait.
DEFAULT_REDELIVERY_BACKOFF: list[float] = [1.0, 5.0, 15.0, 30.0]
# Exhausted or unprocessable messages are republished here (core NATS, not the
# JetStream-backed safety stream) so they are observable, never silently dropped.
DLQ_SUBJECT_PREFIX: str = "dlq."


def poison_subject(subject: str) -> str:
    """Dead-letter subject for a poisoned safety-critical message."""
    return f"{DLQ_SUBJECT_PREFIX}{subject}"


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
        nats_url: str | None = None,
        name: str = "activelearning",
        nats_creds: str | None = None,
    ):
        """
        Initialize the EventBus.

        Args:
            nats_url: NATS server URL (defaults to NATS_URL env var or localhost)
            name: Client name for identification
            nats_creds: Path to a NATS .creds file (NKEY/JWT). Falls back to
                the NATS_CREDS env var. When the file is absent at connect time
                the bus logs a warning and connects without credentials (dev
                fallback) rather than crashing.
        """
        self.nats_url = nats_url or os.environ.get("NATS_URL", "nats://localhost:4222")
        self.name = name
        self.nats_creds: str | None = nats_creds or os.environ.get("NATS_CREDS") or None
        self._nc: NATSClient | None = None
        self._js: JetStreamContext | None = None
        self._subscriptions: dict[str, nats.aio.subscription.Subscription] = {}
        self._handlers: dict[str, MessageHandler] = {}
        self._request_handlers: set[str] = set()
        self._js_durables: dict[str, str] = {}  # subject -> durable consumer name
        # subject -> async callback invoked when a message is poisoned (exhausted
        # redelivery or unprocessable). Optional; the DLQ subject is always used.
        self._poison_handlers: dict[str, MessageHandler] = {}
        self._connected = asyncio.Event()

    async def connect(self) -> None:
        """Connect to the NATS server."""
        if self._nc is not None and self._nc.is_connected:
            logger.debug("Already connected to NATS")
            return

        connect_kwargs: dict[str, Any] = {
            "name": self.name,
            "error_cb": self._error_callback,
            "disconnected_cb": self._disconnected_callback,
            "reconnected_cb": self._reconnected_callback,
            "max_reconnect_attempts": -1,
        }

        if self.nats_creds:
            if os.path.isfile(self.nats_creds):
                connect_kwargs["user_credentials"] = self.nats_creds
                logger.info("Connecting to NATS at %s with credentials %s", self.nats_url, self.nats_creds)
            else:
                # Credentials configured but file absent — dev fallback, never
                # crash so local runs work before gen-creds.sh has been run.
                logger.warning(
                    "NATS_CREDS=%s not found — connecting without per-service "
                    "credentials (dev fallback). Run deploy/scripts/gen-creds.sh "
                    "to enable per-service identities.",
                    self.nats_creds,
                )
                logger.info("Connecting to NATS at %s (no credentials)", self.nats_url)
        else:
            logger.info("Connecting to NATS at %s (no credentials)", self.nats_url)

        self._nc = await nats.connect(self.nats_url, **connect_kwargs)

        # Initialize JetStream for persistence
        self._js = self._nc.jetstream()
        await self._ensure_safety_stream()

        self._connected.set()
        logger.info("Connected to NATS successfully (name=%s)", self.name)

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
            self._handlers.clear()
            self._request_handlers.clear()
            self._js_durables.clear()
            self._poison_handlers.clear()

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
        if subject in (Subjects.PROPOSAL_NEW, Subjects.CODE_PROPOSAL):
            return True
        if subject.startswith(Subjects.DECISION_PREFIX) or subject.startswith(
            Subjects.CODE_DECISION_PREFIX,
        ):
            return True
        return False

    async def publish(
        self,
        subject: str,
        data: Any,
        *,
        message_model: type[BaseModel] | None = None,
    ) -> None:
        """
        Publish a message to a subject.

        Safety-critical subjects (proposal.new, code.proposal, decision.*)
        are published via JetStream so they are persisted and survive a broker
        restart.  All other subjects use core NATS.

        Args:
            subject: NATS subject (e.g., "observation.camera")
            data: Message payload (dataclass or dict)
            message_model: Optional pydantic model to validate dict payloads
        """
        await self._ensure_connected()
        if isinstance(data, dict):
            data = validate_payload(subject, data, message_model)  # type: ignore[arg-type]
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
        queue: str | None = None,
        pending_msgs_limit: int = 65536,
        pending_bytes_limit: int = 128 * 1024 * 1024,
        is_request_handler: bool = False,
        message_model: type[BaseModel] | None = None,
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
            message_model: Optional pydantic model; defaults to registry lookup
        """
        await self._ensure_connected()

        if subject in self._subscriptions:
            logger.warning(f"Already subscribed to {subject}, unsubscribing first")
            await self.unsubscribe(subject)

        wire_model = message_model or schema_for_subject(subject)

        async def message_callback(msg: Msg) -> None:
            try:
                data = deserialize_message(msg.data)
                if not isinstance(data, dict):
                    raise MessageValidationError(
                        subject,
                        f"expected JSON object, got {type(data).__name__}",
                    )
                data = validate_payload(subject, data, wire_model)  # type: ignore[arg-type]
                if is_request_handler:
                    await handler(data, msg)
                else:
                    await handler(data)
            except MessageValidationError as e:
                logger.error(str(e))
                if is_request_handler and msg.reply:
                    try:
                        await msg.respond(serialize_message({
                            "error": "validation_failed",
                            "detail": e.detail,
                            "type": "error",
                        }))
                    except Exception:
                        pass
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
        if is_request_handler:
            self._request_handlers.add(subject)
        else:
            self._request_handlers.discard(subject)
        logger.info(f"Subscribed to {subject}")

    async def js_subscribe(
        self,
        subject: str,
        handler: MessageHandler,
        durable: str,
        message_model: type[BaseModel] | None = None,
        *,
        ack_wait: float = DEFAULT_ACK_WAIT_SECONDS,
        max_deliver: int = DEFAULT_MAX_DELIVER,
        backoff: list[float] | None = None,
        poison_handler: MessageHandler | None = None,
    ) -> None:
        """Subscribe to a JetStream subject with a durable, explicit-ack consumer.

        Reliability semantics for safety-critical streams (E2.3.3):

        - **Explicit ack on success** — the message is acked exactly once only
          after the handler returns. A consumer killed mid-processing never acks,
          so the broker redelivers; the message is not lost.
        - **Bounded redelivery with backoff** — a handler that raises causes a
          ``nak``; the server redelivers after ``backoff`` intervals, up to
          ``max_deliver`` total attempts.
        - **Poison path** — once attempts reach ``max_deliver`` (or the message
          is structurally unprocessable), it is routed to the dead-letter subject
          ``dlq.<subject>`` (and an optional ``poison_handler``) and ``term``-ed,
          so it stops redelivering but is never silently dropped.

        Args:
            subject: JetStream subject (must be covered by SAFETY_STREAM_NAME).
            handler: Async function called with the deserialized payload dict.
            durable: Unique consumer name; stable across restarts.
            message_model: Optional pydantic model; defaults to registry lookup.
            ack_wait: Seconds the server waits for an ack before redelivering
                (used when no explicit backoff is given).
            max_deliver: Maximum delivery attempts before poisoning.
            backoff: Per-retry delays in seconds; the server uses ``backoff[0]``
                as the ack-wait. Defaults to DEFAULT_REDELIVERY_BACKOFF.
            poison_handler: Optional async callback invoked with the poison
                envelope when a message is dead-lettered.
        """
        await self._ensure_connected()
        assert self._js is not None

        if subject in self._subscriptions:
            logger.warning("Already subscribed to %s, unsubscribing first", subject)
            await self.unsubscribe(subject)

        wire_model = message_model or schema_for_subject(subject)
        backoff = backoff if backoff is not None else DEFAULT_REDELIVERY_BACKOFF
        if poison_handler is not None:
            self._poison_handlers[subject] = poison_handler

        async def _js_message_callback(msg: Msg) -> None:
            # Structurally bad messages can never succeed on retry → poison now.
            try:
                data = deserialize_message(msg.data)
                if not isinstance(data, dict):
                    raise MessageValidationError(
                        subject,
                        f"expected JSON object, got {type(data).__name__}",
                    )
                data = validate_payload(subject, data, wire_model)  # type: ignore[arg-type]
            except MessageValidationError as e:
                logger.error("Poisoning unprocessable message on %s: %s", subject, e)
                await self._route_to_poison(subject, msg, f"validation_error: {e}")
                await msg.term()
                return

            try:
                await handler(data)
                await msg.ack()  # exactly-once ack, only after success
            except Exception as e:
                delivered = self._num_delivered(msg)
                if delivered >= max_deliver:
                    # Exhausted bounded redelivery → dead-letter, then term so it
                    # stops redelivering (never an infinite loop, never dropped).
                    logger.error(
                        "Poisoning %s after %d/%d deliveries: %s",
                        subject, delivered, max_deliver, e,
                    )
                    await self._route_to_poison(
                        subject, msg, f"max_deliver_exhausted({delivered}): {e}"
                    )
                    await msg.term()
                else:
                    logger.warning(
                        "Handler failed on %s (delivery %d/%d), will redeliver: %s",
                        subject, delivered, max_deliver, e,
                    )
                    await msg.nak()

        config = ConsumerConfig(
            durable_name=durable,
            ack_policy=AckPolicy.EXPLICIT,
            max_deliver=max_deliver,
            # When backoff is set the server derives ack-wait from backoff[0];
            # passing both can be rejected, so set only one.
            **({"backoff": [float(b) for b in backoff]} if backoff else {"ack_wait": float(ack_wait)}),
        )
        sub = await self._js.subscribe(
            subject,
            cb=_js_message_callback,
            durable=durable,
            manual_ack=True,
            config=config,
        )
        self._subscriptions[subject] = sub
        self._handlers[subject] = handler
        self._js_durables[subject] = durable
        logger.info(
            "JS-subscribed to %s (durable=%s, max_deliver=%d)", subject, durable, max_deliver
        )

    @staticmethod
    def _num_delivered(msg: Msg) -> int:
        """Delivery attempt count for a JetStream message (1 on first delivery)."""
        try:
            return int(msg.metadata.num_delivered)
        except Exception:
            return 1

    async def _route_to_poison(self, subject: str, msg: Msg, reason: str) -> None:
        """Dead-letter a message: republish to dlq.<subject> + call any handler.

        Best-effort and never raises — a failure here must not crash the consumer
        callback (which still needs to term/nak the original message).
        """
        envelope = {
            "original_subject": subject,
            "reason": reason,
            "num_delivered": self._num_delivered(msg),
            "payload": msg.data.decode("utf-8", errors="replace"),
        }
        dlq = poison_subject(subject)
        try:
            assert self._nc is not None
            await self._nc.publish(dlq, serialize_message(envelope))
            logger.error("Dead-lettered message from %s to %s: %s", subject, dlq, reason)
        except Exception as e:  # noqa: BLE001 — poison routing is best-effort
            logger.error("Failed to publish poison message to %s: %s", dlq, e)
        ph = self._poison_handlers.get(subject)
        if ph is not None:
            try:
                await ph(envelope)
            except Exception as e:  # noqa: BLE001 — handler must not crash callback
                logger.error("Poison handler for %s raised: %s", subject, e)

    async def unsubscribe(self, subject: str) -> None:
        """Unsubscribe from a subject."""
        if subject in self._subscriptions:
            await self._subscriptions[subject].unsubscribe()
            del self._subscriptions[subject]
            if subject in self._handlers:
                del self._handlers[subject]
            self._request_handlers.discard(subject)
            self._js_durables.pop(subject, None)
            self._poison_handlers.pop(subject, None)
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
        *,
        code: bool = False,
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
            code: When True, wait on ``code.decision.{trace_id}`` instead of
                ``decision.{trace_id}`` (Kernel code-proposal gate).

        Returns:
            Decision data as dict
        """
        subject = code_decision_subject(trace_id) if code else decision_subject(trace_id)
        durable = f"waiter-{'code' if code else 'action'}-{trace_id}"
        decision_received = asyncio.Event()
        result: dict[str, Any] = {}

        async def _js_handler(msg: Msg) -> None:
            nonlocal result
            try:
                data = deserialize_message(msg.data)
                if not isinstance(data, dict):
                    raise MessageValidationError(
                        subject,
                        f"expected JSON object, got {type(data).__name__}",
                    )
                data = validate_payload(subject, data, KernelDecisionMessage)
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
            except MessageValidationError as e:
                logger.error(str(e))
                await msg.ack()

        assert self._js is not None
        sub = await self._js.subscribe(
            subject,
            cb=_js_handler,
            durable=durable,
            config=ConsumerConfig(
                deliver_policy=DeliverPolicy.ALL,
                inactive_threshold=_CONSUMER_INACTIVE_THRESHOLD_S,
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
        saved_poison: dict[str, MessageHandler] = dict(self._poison_handlers)
        for subject, handler in self._handlers.items():
            saved_handlers[subject] = (handler, subject in self._request_handlers)

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
            self._handlers.clear()
            self._request_handlers.clear()
            self._js_durables.clear()
            self._poison_handlers.clear()

        # Create fresh connection (preserves nats_creds; also re-ensures the safety stream)
        await self.connect()

        # Re-subscribe all handlers, preserving JS vs core distinction
        for subject, (handler, is_req) in saved_handlers.items():
            try:
                if subject in saved_js:
                    await self.js_subscribe(
                        subject, handler, durable=saved_js[subject],
                        poison_handler=saved_poison.get(subject),
                    )
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
            except TimeoutError:
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
_global_bus: EventBus | None = None


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
    queue: str | None = None,
) -> None:
    """Convenience function to subscribe via global bus."""
    bus = await get_event_bus()
    await bus.subscribe(subject, handler, queue)
