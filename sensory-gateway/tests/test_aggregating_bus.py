"""Load test for AggregatingEventBus's latest-only backpressure policy (issue #230).

aggregating_bus.py's own docstring claims it drops ~1,400 msgs/sec of raw
sensor observations down to ~4 NATS msgs/sec by keeping only the latest
observation per modality between flushes. That claim was never exercised by
a test — this file simulates high-frequency sensor input end-to-end (through
the real publish() -> internal buffer -> _flush_once() path, with only the
outbound NATS call itself replaced by an in-memory recorder) and asserts:

  1. Flooding publish() with many rapid observations does not grow the
     internal buffer past one entry per modality (no unbounded buffering).
  2. Only the LAST observation submitted before a flush is what actually
     reaches NATS — older ones are dropped, not queued.
  3. The realized NATS message rate stays pinned near flush_hz regardless of
     how much faster the input rate is, reproducing the docstring's
     1,400 msgs/sec -> ~4 msgs/sec reduction claim with real numbers.
  4. Audio is deliberately excluded from "latest-only" (it temporally
     aggregates instead, by design -- see the module docstring) -- a
     regression test for that exclusion boundary is included so a future
     refactor can't silently start dropping audio too.
"""

from __future__ import annotations

import asyncio
import os
import sys
import types as _types

# ---------------------------------------------------------------------------
# Path setup: sensory-gateway uses a flat src layout (module == "gateway"),
# same trick as test_depth_camera.py -- stub the activelearning package's
# __init__ (which eagerly pulls in base_service/database) while pointing its
# __path__ at the real sdk/src/activelearning so submodule imports like
# `from activelearning.nats_client import EventBus` still load the genuine
# EventBus class. AggregatingEventBus subclasses EventBus directly, so a
# fully fake stand-in isn't an option here -- the real base class is needed.
# ---------------------------------------------------------------------------
_GW_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _GW_ROOT not in sys.path:
    sys.path.insert(0, _GW_ROOT)

_SDK_SRC = os.path.abspath(os.path.join(_GW_ROOT, "..", "sdk", "src"))
if "activelearning" not in sys.modules:
    _al_pkg = _types.ModuleType("activelearning")
    _al_pkg.__path__ = [os.path.join(_SDK_SRC, "activelearning")]  # type: ignore[attr-defined]
    _al_pkg.__package__ = "activelearning"
    sys.modules["activelearning"] = _al_pkg
if _SDK_SRC not in sys.path:
    sys.path.insert(0, _SDK_SRC)

from activelearning.nats_client import EventBus  # noqa: E402

from aggregating_bus import AggregatingEventBus  # noqa: E402


def _run(coro):
    return asyncio.run(coro)


class _RecordingBus(AggregatingEventBus):
    """AggregatingEventBus with connection state faked so no real broker is
    needed. publish() (the inbound path under test -- modality routing,
    _latest/_audio_buffers, latest-only overwrite semantics) is untouched;
    tests that need to observe outbound NATS traffic monkeypatch
    EventBus.publish (the real superclass method _flush_once() calls via
    super()) separately with _fake_super_publish, recording into `.sent`.
    """

    def __init__(self, **kwargs):
        super().__init__(nats_url="nats://unused:4222", **kwargs)
        self.sent: list[tuple[str, object]] = []

    @property
    def is_connected(self) -> bool:  # noqa: D102 - see class docstring
        return True


async def _fake_super_publish(self, subject: str, data) -> None:
    self.sent.append((subject, data))


def _make_bus(**kwargs) -> _RecordingBus:
    bus = _RecordingBus(**kwargs)
    return bus


def _video_obs(frame_id: int) -> dict:
    return {"provenance": "sensor.videofile.cam0", "data": {"frame": frame_id}}


class TestLatestOnlyDropsOlderObservations:
    """Core policy: between flushes, only the newest observation survives."""

    def test_buffer_holds_one_entry_per_modality_regardless_of_publish_count(self):
        async def scenario():
            bus = _make_bus()
            for i in range(5000):
                await bus.publish("observation.videofile.cam0", _video_obs(i))
            # No flush has run yet -- confirm the buffer did NOT grow with
            # the publish count. This is the "no unbounded buffering" claim.
            assert len(bus._latest) == 1
            assert bus._latest["visual"] == _video_obs(4999)

        _run(scenario())

    def test_flush_sends_only_the_latest_value(self, monkeypatch):
        async def scenario():
            bus = _make_bus()
            monkeypatch.setattr(EventBus, "publish", _fake_super_publish)
            for i in range(1000):
                await bus.publish("observation.videofile.cam0", _video_obs(i))
            await bus._flush_once()

            assert len(bus.sent) == 1
            subject, data = bus.sent[0]
            assert subject == "observation.videofile.cam0"
            assert data == _video_obs(999)  # last one in, nothing earlier

        _run(scenario())

    def test_intermediate_values_are_unrecoverable_after_being_overwritten(self):
        """Older observations aren't just unsent -- they're gone. A latent
        bug that queued them "for later" instead of dropping them would
        still pass a naive "last value published" check but fail this one."""

        async def scenario():
            bus = _make_bus()
            for i in range(100):
                await bus.publish("observation.videofile.cam0", _video_obs(i))
            # The only trace of the stream is the single latest entry.
            assert bus._latest["visual"]["data"]["frame"] == 99
            assert len(bus._latest) == 1
            # No hidden per-message queue anywhere on the instance.
            assert not hasattr(bus, "_queue")
            assert not hasattr(bus, "_pending")

        _run(scenario())

    def test_multiple_modalities_each_keep_only_their_own_latest(self):
        async def scenario():
            bus = _make_bus()
            for i in range(50):
                await bus.publish("observation.videofile.cam0", _video_obs(i))
                await bus.publish(
                    "observation.serial.imu0",
                    {"provenance": "sensor.imu0", "data": {"reading": i}},
                )
            assert len(bus._latest) == 2
            assert bus._latest["visual"]["data"]["frame"] == 49
            assert bus._latest["proprioceptive"]["data"]["reading"] == 49

        _run(scenario())


class TestHighFrequencyEndToEndReduction:
    """End-to-end: flood the real publish()->flush_loop path at a rate far
    above flush_hz and confirm the realized NATS message rate collapses to
    ~flush_hz, reproducing the module docstring's reduction claim."""

    def test_flush_loop_caps_message_rate_under_sustained_flood(self, monkeypatch):
        async def scenario():
            # 20 Hz flush -> over a ~0.5s flood window we expect on the order
            # of ~10 flushes, independent of how many thousands of raw
            # observations were published in that window.
            bus = _make_bus(flush_hz=20.0)
            monkeypatch.setattr(EventBus, "publish", _fake_super_publish)
            bus._flush_task = asyncio.create_task(bus._flush_loop())
            bus._heartbeat_task = asyncio.create_task(asyncio.sleep(3600))
            try:
                stop_at = asyncio.get_event_loop().time() + 0.5
                i = 0
                # Flood far faster than flush_hz -- publish() itself never
                # awaits the network, so this reproduces the docstring's
                # ~1,400 msgs/sec sensor rate against a 20 Hz flush easily.
                while asyncio.get_event_loop().time() < stop_at:
                    await bus.publish("observation.videofile.cam0", _video_obs(i))
                    i += 1
                    if i % 50 == 0:
                        await asyncio.sleep(0)  # yield so the flush loop runs
                await asyncio.sleep(bus._flush_interval * 1.5)  # let a final flush land
            finally:
                bus._flush_task.cancel()
                bus._heartbeat_task.cancel()
                for t in (bus._flush_task, bus._heartbeat_task):
                    try:
                        await t
                    except asyncio.CancelledError:
                        pass

            raw_published = i
            nats_messages = len(bus.sent)
            # The whole point of the policy: NATS traffic must not scale
            # with the raw publish rate.
            assert raw_published > 500, "flood didn't actually flood -- test is too weak"
            assert nats_messages < raw_published / 10
            # And it should roughly track flush_hz over the window, not 0
            # (i.e. flushing is actually happening, not just suppressed).
            assert 1 <= nats_messages <= 20

        _run(scenario())


class TestAudioIsIntentionallyExemptFromLatestOnly:
    """Audio uses temporal accumulation, not latest-only, by design (see
    module docstring). This pins that boundary so a refactor that
    accidentally routes audio through the latest-only path (destroying
    temporal structure STDP needs) fails loudly."""

    def test_mfcc_chunks_accumulate_instead_of_overwriting(self):
        async def scenario():
            bus = _make_bus()
            for i in range(10):
                chunk = [float(i)] * 13  # exactly 13 floats = MFCC chunk
                await bus.publish(
                    "observation.audiofile.mic0",
                    {"provenance": "sensor.audiofile.mic0", "data": chunk},
                )
            assert "auditory" not in bus._latest  # not routed through latest-only
            assert len(bus._audio_buffers["observation.audiofile.mic0"]) == 10

        _run(scenario())
