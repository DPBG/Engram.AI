"""Smoke fixture: a trivially PASSING test set.

Run inside activelearning-sandbox:latest to prove the image can collect and
execute generated tests and report success (exit code 0). Imports only what the
sandbox image guarantees (stdlib + numpy), since the container has no network
and a read-only root filesystem.
"""

import numpy as np


def test_arithmetic():
    assert 1 + 1 == 2


def test_numpy_available():
    # numpy is preinstalled in the image; generated drivers commonly import it.
    assert np.array([1, 2, 3]).sum() == 6


def test_tmp_is_writable():
    # /tmp is the only writable path (tmpfs); everything else is read-only.
    with open("/tmp/smoke.txt", "w") as fh:
        fh.write("ok")
    with open("/tmp/smoke.txt") as fh:
        assert fh.read() == "ok"
