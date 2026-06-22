"""beliefs norm-query returns the norms (and risk-boost metadata) the Kernel
consumes (E1.8.1).

The Kernel calls BELIEFS_QUERY_REQUEST with {"type": "norms", "threshold": 0.8}
and derives per-action risk boosts from each norm's metadata (e.g.
norm.gradual_motor.max_intensity_delta, norm.force_limit.motor_channels). This
test asserts the service reply preserves that contract.

The handler is exercised directly with a fake request message, bypassing
BaseService.__init__ (no NATS/DB needed).
"""

import asyncio
import os
import sys

_SDK = os.path.join(os.path.dirname(__file__), "..", "..", "sdk", "src")
_SRC = os.path.join(os.path.dirname(__file__), "..", "src")
for _p in (_SDK, _SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from activelearning.nats_client import deserialize_message  # noqa: E402

from beliefs.graph import BeliefGraph  # noqa: E402
from beliefs.service import BeliefsService  # noqa: E402


class _FakeMsg:
    def __init__(self):
        self.reply = "_INBOX.test"
        self.response = None

    async def respond(self, data):
        self.response = data


class _Logger:
    def info(self, *a, **k):
        pass

    def warning(self, *a, **k):
        pass

    def error(self, *a, **k):
        pass


def _service() -> BeliefsService:
    svc = BeliefsService.__new__(BeliefsService)
    svc.logger = _Logger()
    svc._graph = BeliefGraph()
    svc._graph.seed_constitutional_beliefs()
    return svc


def _query(svc, payload) -> dict:
    msg = _FakeMsg()
    asyncio.run(svc._handle_query_request(payload, msg))
    assert msg.response is not None, "handler must reply"
    return deserialize_message(msg.response)


def test_norms_query_returns_high_confidence_norms():
    svc = _service()
    resp = _query(svc, {"type": "norms", "threshold": 0.8})
    assert resp["query_type"] == "norms"
    norm_ids = {n["id"] for n in resp["result"]}
    # Constitutional norms at/above the Kernel's 0.8 threshold.
    assert {"norm.force_limit", "norm.gradual_motor", "norm.no_self_modify"} <= norm_ids
    assert all(n["confidence"] >= 0.8 for n in resp["result"])


def test_norms_query_preserves_risk_boost_metadata():
    svc = _service()
    resp = _query(svc, {"type": "norms", "threshold": 0.8})
    by_id = {n["id"]: n for n in resp["result"]}

    # The exact metadata fields the Kernel reads to compute risk boosts.
    assert by_id["norm.gradual_motor"]["metadata"]["max_intensity_delta"] == 0.3
    assert "manipulation" in by_id["norm.force_limit"]["metadata"]["motor_channels"]


def test_norms_query_threshold_filters_below():
    svc = _service()
    # Threshold above every seeded norm confidence → empty result.
    resp = _query(svc, {"type": "norms", "threshold": 0.999})
    assert resp["result"] == []
