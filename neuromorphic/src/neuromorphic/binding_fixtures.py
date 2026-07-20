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
    "precision_min": 0.1,
    "recall_min": 0.1,
    "f1_min": 0.1,
    "n_cross_modal_min": 1.0,
}


def generate_correlated_stimulus_fixtures(
    n_pairs: int = 4,
    seed: int = 42,
    visual_size: int = 64,
    auditory_size: int = 13,
    hard_decoys: bool = False,
) -> dict[str, Any]:
    """Build synchronized pulse pairs and decoy mismatches.

    Each correlated pair uses a distinct visual gaussian blob and auditory
    signature peak so ground-truth bindings are identifiable.

    Args:
        hard_decoys: When False (default) each decoy pairs visual_i with the
            auditory from pair_{i+1} — a clearly different signature whose hot
            bit sits at a completely different index (cosine similarity ≈ 0).
            When True each decoy uses the same visual_i paired with a
            noise-perturbed copy of auditory_i (σ=0.2, clipped to [0, 1]).
            The hard decoy cosine-similarity to the true match is ≥ 0.7, so
            the network must detect exact synchrony rather than gross modal
            mismatch.  Use hard_decoys=True to verify that the benchmark
            measures genuine binding discrimination and that
            matched_to_decoy_ratio does not trivially saturate.
    """
    if n_pairs < 2:
        raise ValueError("n_pairs must be at least 2 to create decoy mismatches")
    if visual_size <= 0 or auditory_size <= 0:
        raise ValueError("visual_size and auditory_size must be positive")
    if n_pairs > auditory_size:
        raise ValueError("n_pairs must not exceed auditory_size for distinct signatures")

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

        correlated_pairs.append(
            {
                "pair_id": f"pair_{i:03d}",
                "visual": visual.tolist(),
                "auditory": auditory.tolist(),
                "label": f"synchronized_pulse_{i:03d}",
            }
        )

    decoy_pairs: list[dict[str, Any]] = []
    for i in range(n_pairs):
        if hard_decoys:
            # Near-miss: same visual, auditory is a noise-perturbed copy of the
            # matched auditory (σ=0.2 Gaussian noise, clipped to [0, 1]).
            # Cosine similarity to the true match is high (≥ 0.7), so the
            # benchmark cannot be trivially saturated by gross modal mismatch.
            matched_aud = np.array(correlated_pairs[i]["auditory"], dtype=np.float32)
            noise = rng.normal(0.0, 0.2, size=matched_aud.shape).astype(np.float32)
            noisy_aud = np.clip(matched_aud + noise, 0.0, 1.0)
            decoy_pairs.append(
                {
                    "pair_id": f"hard_decoy_{i:03d}",
                    "visual_pair_id": correlated_pairs[i]["pair_id"],
                    "auditory_pair_id": f"noisy_version_of_{correlated_pairs[i]['pair_id']}",
                    "visual": correlated_pairs[i]["visual"],
                    "auditory": noisy_aud.tolist(),
                    "label": f"hard_decoy_{i:03d}",
                }
            )
        else:
            # Easy decoy: pair i's visual with pair (i+1)'s auditory.
            # Hot bit sits at a completely different index — cosine similarity ≈ 0.
            j = (i + 1) % n_pairs
            decoy_pairs.append(
                {
                    "pair_id": f"decoy_{i:03d}",
                    "visual_pair_id": correlated_pairs[i]["pair_id"],
                    "auditory_pair_id": correlated_pairs[j]["pair_id"],
                    "visual": correlated_pairs[i]["visual"],
                    "auditory": correlated_pairs[j]["auditory"],
                    "label": f"decoy_{i:03d}",
                }
            )

    return {
        "correlated_pairs": correlated_pairs,
        "decoy_pairs": decoy_pairs,
        "seed": seed,
        "n_pairs": n_pairs,
        "hard_decoys": hard_decoys,
    }
