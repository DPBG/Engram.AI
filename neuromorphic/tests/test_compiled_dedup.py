"""Equivalence tests for the compiled active-synapse dedup kernel (issue #435).

Invariant 1 (CLAUDE.md §2): equivalence-tested per PR #161's pattern, now via
the shared harness (issue #438, tests/helpers/equivalence.py).

dedup_indices() has a different equivalence contract than the STDP kernels:
it is NOT required to match np.unique's exact (sorted) output array -- only
its *set* of values (each appearing exactly once). Both call sites in
synapses.py use the result purely as a fancy-index array, never relying on
order, so this is the correct equivalence notion to test (see ADR 0002 and
compiled_kernels.py's dedup_indices docstring).

Tests skip when Numba is not installed or NEURO_COMPILED_DEDUP=0.
"""

from __future__ import annotations

import numpy as np
import pytest
from helpers.equivalence import compiled_only_mark

import neuromorphic.compiled_kernels as _ck_mod
from neuromorphic.compiled_kernels import COMPILED_DEDUP_ENABLED, dedup_indices

_COMPILED_ONLY = compiled_only_mark(
    COMPILED_DEDUP_ENABLED,
    reason="Numba not installed or NEURO_COMPILED_DEDUP=0 — compiled dedup tests skipped",
)

_RNG = np.random.default_rng(0)


def _assert_same_set(got: np.ndarray, expected_sorted_unique: np.ndarray, err_msg: str) -> None:
    """dedup_indices' contract: same set of values, no order guarantee."""
    np.testing.assert_array_equal(np.sort(got), expected_sorted_unique, err_msg=err_msg)
    assert len(np.unique(got)) == len(got), f"{err_msg}: output contains a duplicate"


@_COMPILED_ONLY
class TestCompiledDedupIndices:
    """Compiled dedup_indices must match np.unique's *set* of values."""

    def test_empty_input(self):
        arr = np.array([], dtype=np.intp)
        got = dedup_indices(arr)
        assert len(got) == 0

    def test_single_element(self):
        arr = np.array([5], dtype=np.intp)
        got = dedup_indices(arr)
        _assert_same_set(got, np.array([5]), "single element")

    def test_all_duplicates(self):
        arr = np.full(200, 7, dtype=np.intp)
        got = dedup_indices(arr)
        _assert_same_set(got, np.array([7]), "all duplicates")

    def test_no_duplicates(self):
        arr = np.arange(500, dtype=np.intp)
        got = dedup_indices(arr)
        _assert_same_set(got, np.arange(500), "no duplicates")

    @pytest.mark.parametrize("n,high", [(1, 10), (10, 5), (500, 200), (5_000, 10_000)])
    def test_randomized_small_to_medium(self, n, high):
        arr = _RNG.integers(0, high, size=n).astype(np.intp)
        got = dedup_indices(arr)
        _assert_same_set(got, np.unique(arr), f"n={n}, high={high}")

    @pytest.mark.parametrize(
        "n,high",
        [(50_000, 2_000_000), (200_000, 5_000_000), (500_000, 9_000_000)],
    )
    def test_randomized_large_scale(self, n, high):
        """Realistic 'large matrix' path scale (ADR 0002: 100K < nnz < 10M)."""
        arr = _RNG.integers(0, high, size=n).astype(np.intp)
        got = dedup_indices(arr)
        _assert_same_set(got, np.unique(arr), f"n={n}, high={high}")

    def test_no_duplicate_in_output_never(self):
        """Direct check on the property that actually matters to callers:
        eligibility[active_idx] += dw silently under-counts if active_idx
        has a repeat (NumPy fancy-indexing += is not additive on repeats)."""
        for _ in range(50):
            n = _RNG.integers(1, 1000)
            high = _RNG.integers(1, 500)
            arr = _RNG.integers(0, high, size=n).astype(np.intp)
            got = dedup_indices(arr)
            assert len(np.unique(got)) == len(got)

    def test_values_bounded_by_input_range(self):
        """No spurious/garbage values -- output is a subset of the input."""
        arr = _RNG.integers(0, 1000, size=2000).astype(np.intp)
        got = dedup_indices(arr)
        assert set(got.tolist()).issubset(set(arr.tolist()))


class TestDedupIndicesNumPyFallback:
    """NumPy fallback (no compiled path) must exactly match np.unique,
    including its sorted-output guarantee -- the fallback IS np.unique."""

    def test_fallback_matches_unique_exactly(self, monkeypatch):
        monkeypatch.setattr(_ck_mod, "COMPILED_DEDUP_ENABLED", False)
        arr = np.array([5, 3, 5, 1, 3, 9, 1], dtype=np.intp)
        got = dedup_indices(arr)
        np.testing.assert_array_equal(got, np.unique(arr))

    def test_fallback_empty(self, monkeypatch):
        monkeypatch.setattr(_ck_mod, "COMPILED_DEDUP_ENABLED", False)
        arr = np.array([], dtype=np.intp)
        got = dedup_indices(arr)
        assert len(got) == 0
