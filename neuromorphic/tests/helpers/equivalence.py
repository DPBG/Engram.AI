"""Shared helpers for compiled-kernel equivalence tests (issue #438).

Extracted from test_compiled_stdp.py's original pattern (issue #161, CLAUDE.md
Invariant 1: "Any change to those intervals MUST preserve equivalence and be
covered by an equivalence test."). A compiled/vectorized kernel must be
proven numerically equivalent to its NumPy reference implementation to
float32 precision before it's trusted as a drop-in replacement -- this
generalizes that one-off pattern so new compiled-kernel work (e.g. a
CSC-gather spike for issue #435, or any future kernel needing the same
proof) reuses the same tolerance and skip-mark conventions instead of each
test file re-deriving its own.
"""

from __future__ import annotations

import numpy as np
import pytest

# float32 precision tolerances used throughout the original STDP equivalence
# suite; centralized here so every compiled-kernel equivalence test uses the
# same bar rather than each file picking its own ad hoc rtol/atol.
DEFAULT_RTOL = 1e-5
DEFAULT_ATOL = 1e-7


def compiled_only_mark(enabled: bool, *, reason: str) -> pytest.MarkDecorator:
    """skipif mark for tests that require a compiled kernel path to be active.

    `enabled` should be the live module flag (e.g. COMPILED_STDP_ENABLED) read
    at collection time -- pass it through rather than importing a flag here,
    since availability is an optional-dependency check that varies per
    environment and each kernel module owns its own flag.
    """
    return pytest.mark.skipif(not enabled, reason=reason)


def assert_kernel_equivalent(
    compiled_output: np.ndarray,
    reference_output: np.ndarray,
    *,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
    err_msg: str = "",
) -> None:
    """assert a compiled kernel's output matches its NumPy reference to float32 precision.

    Thin wrapper over np.testing.assert_allclose with this suite's standard
    tolerances -- every equivalence test states its intent
    (assert_kernel_equivalent) rather than reaching for a generic numeric
    comparison with hand-picked tolerance constants repeated at each call site.
    """
    np.testing.assert_allclose(
        compiled_output, reference_output, rtol=rtol, atol=atol, err_msg=err_msg,
    )
