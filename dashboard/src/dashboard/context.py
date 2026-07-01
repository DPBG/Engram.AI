"""
Dependency container for the dashboard.

``DashboardContext`` bundles the shared state and the behavioural components
(NATS, chat, metrics) so router factories receive a single object to wire
against instead of a long argument list. It deliberately holds no behaviour of
its own — it is the composition root's hand-off to the routers, and is
web-stack-free so it can be faked in tests.
"""

from dataclasses import dataclass

from dashboard.chat import ChatEngine
from dashboard.metrics import MetricsMonitor
from dashboard.nats_stream import NatsStreamManager
from dashboard.state import DashboardState


@dataclass
class DashboardContext:
    """Shared state + components passed to every router factory."""

    state: DashboardState
    nats: NatsStreamManager
    chat: ChatEngine
    metrics: MetricsMonitor
