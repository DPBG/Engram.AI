"""Ground-truth correlated-stimulus fixtures for cross-modal binding benchmarks.

Provides synchronized visual+auditory pulse pairs with known pair IDs, plus
decoy (uncorrelated) combinations for precision/recall evaluation.
"""
from __future__ import annotations

from typing import Any

import numpy as np

# Minimum metrics for CI regression (small benchmark network, seed=42).
# Values measured on upstream main — update only when binding improves.
BINDING_ACCURACY_REGRESSION_THRESHOLDS: dict[str, float] = {
    "binding_strength_min": 0.01,
    "matched_coupling_ratio_min": 1.0,
    "precision_min": 0.0,
    "recall_min": 0.0,
    "f1_min": 0.0,
    "n_cross_modal_min": 1.0,
}


def generate_correlated_stimulus_fixtures(
    n_pairs: int = 4,
    seed: int = 42,
    visual_size: int = 64,
    auditory_size: int = 13,
) -> dict[str, Any]:
    """Build synchronized pulse pairs and decoy mismatches.

    Each correlated pair uses a distinct visual gaussian blob and auditory
    signature peak so ground-truth bindings are identifiable.  Decoys pair
    each visual stimulus with an auditory signature from a different pair.
    """
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:visual_size, 0:visual_size]

    correlated_pairs: list[dict[str, Any]] = []
    for i in range(n_pairs):
        cx = (i * 17 + 10) % visual_size
        cy = (i * 23 + 5) % visual_size
        dist2 = (x - cx).astype(np.float32) ** 2 + (y - cy).astype(np.float32) ** 2
        visual = np.exp(-dist2 / 20.0, dtype=np.float32).flatten()
        visual /= visual.max() + 1e-8

        auditory = np.zeros(auditory_size, dtype=np.float32)
        auditory[i % auditory_size] = 1.0
        auditory[(i + 5) % auditory_size] = 0.3

        correlated_pairs.append({
            "pair_id": f"pair_{i:03d}",
            "visual": visual.tolist(),
            "auditory": auditory.tolist(),
            "label": f"synchronized_pulse_{i:03d}",
        })

    decoy_pairs: list[dict[str, Any]] = []
    for i in range(n_pairs):
        j = (i + 1) % n_pairs
        decoy_pairs.append({
            "pair_id": f"decoy_{i:03d}",
            "visual_pair_id": correlated_pairs[i]["pair_id"],
            "auditory_pair_id": correlated_pairs[j]["pair_id"],
            "visual": correlated_pairs[i]["visual"],
            "auditory": correlated_pairs[j]["auditory"],
            "label": f"decoy_{i:03d}",
        })

    return {
        "correlated_pairs": correlated_pairs,
        "decoy_pairs": decoy_pairs,
        "seed": seed,
        "n_pairs": n_pairs,
    }
