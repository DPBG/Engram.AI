"""NATS server bootstrap — download and run the nats-server binary.

NATS has no pure-Python server, so the launcher fetches the small official
binary on first run (using only the Python stdlib) and spawns it as a local
subprocess. If a `nats-server` is already on PATH or already listening on the
client port, that one is reused instead.
"""

from __future__ import annotations

import hashlib
import logging
import os
import platform
import shutil
import socket
import subprocess
import tarfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from .registry import ROOT

logger = logging.getLogger("launcher.nats")

NATS_VERSION = "v2.10.22"
CLIENT_PORT = 4222
MONITOR_PORT = 8222

# Known-good SHA256 digests for the NATS release archives, keyed by asset
# filename. When an entry exists (or NATS_SHA256 is set in the environment) the
# download is verified and a mismatch aborts startup — a supply-chain guard.
# Populate from the official release `SHA256SUMS` to enable strict verification;
# until then the launcher logs the observed digest so it can be pinned.
KNOWN_SHA256: dict[str, str] = {}

_DOWNLOAD_RETRIES = 3

# Where downloaded binaries and runtime data live (git-ignored).
LOCALRUN = ROOT / ".localrun"
NATS_DIR = LOCALRUN / "nats"
NATS_DATA = LOCALRUN / "nats-data"


def _platform_asset() -> tuple[str, str]:
    """Return (asset_filename, archive_kind) for the current OS/arch."""
    system = platform.system().lower()  # windows / linux / darwin
    machine = platform.machine().lower()
    if machine in ("amd64", "x86_64", "x64"):
        arch = "amd64"
    elif machine in ("arm64", "aarch64"):
        arch = "arm64"
    else:
        raise RuntimeError(f"Unsupported CPU architecture for NATS: {machine!r}")

    if system == "windows":
        return f"nats-server-{NATS_VERSION}-windows-{arch}.zip", "zip"
    if system == "darwin":
        return f"nats-server-{NATS_VERSION}-darwin-{arch}.zip", "zip"
    if system == "linux":
        return f"nats-server-{NATS_VERSION}-linux-{arch}.tar.gz", "tar"
    raise RuntimeError(f"Unsupported OS for NATS bootstrap: {system!r}")


def _binary_name() -> str:
    return "nats-server.exe" if platform.system().lower() == "windows" else "nats-server"


def port_open(host: str, port: int, timeout: float = 0.5) -> bool:
    """True if something is already listening on host:port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        return sock.connect_ex((host, port)) == 0


def _find_existing_binary() -> Path | None:
    """Prefer a previously downloaded binary, then one on PATH."""
    local = NATS_DIR / _binary_name()
    if local.exists():
        return local
    on_path = shutil.which("nats-server")
    return Path(on_path) if on_path else None


def _verify_checksum(archive: Path, asset: str) -> None:
    """Verify the downloaded archive's SHA256 against a known/expected digest.

    Strict (raises on mismatch) when an expected digest is configured via the
    ``NATS_SHA256`` env var or the ``KNOWN_SHA256`` table; otherwise logs the
    observed digest so it can be pinned later.
    """
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    expected = os.environ.get("NATS_SHA256") or KNOWN_SHA256.get(asset)
    if expected:
        if digest.lower() != expected.lower():
            archive.unlink(missing_ok=True)
            raise RuntimeError(
                f"NATS download checksum mismatch for {asset}:\n"
                f"  expected {expected}\n  got      {digest}\n"
                "Refusing to use a binary that does not match the expected hash."
            )
        logger.info("nats-server checksum verified (sha256=%s)", digest)
    else:
        logger.info(
            "nats-server sha256=%s (no expected digest configured; set "
            "NATS_SHA256 or KNOWN_SHA256 to enforce verification)",
            digest,
        )


def _download_binary() -> Path:
    """Download and extract the nats-server binary into NATS_DIR."""
    asset, kind = _platform_asset()
    url = (
        "https://github.com/nats-io/nats-server/releases/download/"
        f"{NATS_VERSION}/{asset}"
    )
    NATS_DIR.mkdir(parents=True, exist_ok=True)
    archive = NATS_DIR / asset

    logger.info("Downloading nats-server %s ...", NATS_VERSION)
    logger.info("  %s", url)
    last_exc: Exception | None = None
    for attempt in range(1, _DOWNLOAD_RETRIES + 1):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp, open(archive, "wb") as f:
                shutil.copyfileobj(resp, f)
            break
        except urllib.error.URLError as exc:  # pragma: no cover - network dependent
            last_exc = exc
            logger.warning(
                "nats-server download attempt %d/%d failed: %s",
                attempt, _DOWNLOAD_RETRIES, exc,
            )
            if attempt < _DOWNLOAD_RETRIES:
                time.sleep(2 * attempt)
    else:
        raise RuntimeError(
            f"Failed to download nats-server from {url} after "
            f"{_DOWNLOAD_RETRIES} attempts: {last_exc}\n"
            "Check your internet connection, or install nats-server manually "
            "and put it on PATH (https://nats.io/download/)."
        ) from last_exc

    # Supply-chain guard: verify the archive hash before extracting/executing.
    _verify_checksum(archive, asset)

    binary_name = _binary_name()
    extracted: Path | None = None
    if kind == "zip":
        with zipfile.ZipFile(archive) as zf:
            for member in zf.namelist():
                if member.endswith(binary_name):
                    zf.extract(member, NATS_DIR)
                    extracted = NATS_DIR / member
                    break
    else:  # tar.gz
        with tarfile.open(archive, "r:gz") as tf:
            for member in tf.getmembers():
                if member.name.endswith(binary_name):
                    tf.extract(member, NATS_DIR)
                    extracted = NATS_DIR / member.name
                    break

    archive.unlink(missing_ok=True)
    if extracted is None:
        raise RuntimeError(f"nats-server binary not found inside {asset}")

    final = NATS_DIR / binary_name
    if extracted != final:
        extracted.replace(final)
        # Clean up the now-empty extracted sub-directory if present.
        parent = extracted.parent
        if parent != NATS_DIR and parent.exists():
            shutil.rmtree(parent, ignore_errors=True)

    if platform.system().lower() != "windows":
        final.chmod(0o755)
    logger.info("nats-server ready at %s", final)
    return final


def ensure_binary() -> Path:
    """Return a path to a usable nats-server, downloading if necessary."""
    existing = _find_existing_binary()
    if existing:
        return existing
    return _download_binary()


def start(log_file: Path | None = None) -> subprocess.Popen | None:
    """Start nats-server (with JetStream) unless one is already running.

    Returns the Popen handle, or None if an external NATS is reused.
    """
    if port_open("127.0.0.1", CLIENT_PORT):
        logger.info(
            "NATS already listening on port %d — reusing it.", CLIENT_PORT
        )
        return None

    binary = ensure_binary()
    NATS_DATA.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(binary),
        "--jetstream",
        "--store_dir",
        str(NATS_DATA),
        "-m",
        str(MONITOR_PORT),
        "-p",
        str(CLIENT_PORT),
    ]
    logger.info("Starting NATS: %s", " ".join(cmd))
    out = open(log_file, "ab") if log_file else subprocess.DEVNULL
    proc = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT)
    return proc


def wait_ready(timeout: float = 20.0) -> bool:
    """Poll the client port until NATS accepts connections (or timeout)."""
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if port_open("127.0.0.1", CLIENT_PORT):
            return True
        time.sleep(0.3)
    return False
