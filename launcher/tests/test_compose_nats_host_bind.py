"""Default Compose must not expose unauthenticated NATS on all interfaces.

The stock ``nats:2.10-alpine`` image used by ``docker-compose.yml`` has no
auth, and the default stack does not set ``ENGRAM_DECISION_KEY``. Publishing
ports as ``4222:4222`` (0.0.0.0) lets any network peer connect anonymously
and forge ``decision.*`` ALLOW messages. Bind to loopback by default.
"""

from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_COMPOSE = _REPO_ROOT / "docker-compose.yml"

# Matches Compose host:container port mappings for the NATS client/monitor ports.
_NATS_PORT_LINE = re.compile(
    r"""["'](?P<host>[^"']+):(?P<published>4222|8222):(?P<target>4222|8222)["']"""
)


def test_default_compose_binds_nats_to_loopback() -> None:
    text = _COMPOSE.read_text(encoding="utf-8")
    matches = list(_NATS_PORT_LINE.finditer(text))
    assert matches, "expected NATS 4222/8222 host port mappings in docker-compose.yml"

    published = {m.group("published") for m in matches}
    assert published == {"4222", "8222"}

    for m in matches:
        assert m.group("published") == m.group("target")
        host = m.group("host")
        assert "127.0.0.1" in host, (
            f"NATS port {m.group('published')} host binding must default to "
            f"127.0.0.1 (got {host!r}); unauthenticated broker must not be "
            f"reachable from other network interfaces"
        )
        # Bare all-interfaces bind is forbidden; optional override stays explicit.
        assert not host.strip().startswith("0.0.0.0")
        assert host.strip() not in {"4222", "8222"}
