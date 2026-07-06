"""Phase 3.5 cross-body motor skill projection via semantic tags.

Pure functions that, given:
  - a saved motor weights blob from one body (source),
  - the source body's per-actuator manifest (name + tags per channel),
  - the target body's per-actuator manifest,
  - the brain's motor cortex sub-range row layout (channel -> (start, end)
    in the motor synapse group's post axis),
  - the canonical set of motor synapse group names,

return a new blob with weights rearranged so that:
  - target actuators with body-part tags overlapping a source actuator
    inherit (a resampled copy of) that source actuator's weight rows.
  - target actuators with no matching source actuator keep the rows
    they came in with -- i.e. start fresh on this body.

Iron rule: this function never interprets tag strings. It does string-set
intersection only. Adding new body part tags (rotor, propulsion, wheel,
fin, thruster) requires no change here.

Source-of-truth note: this module does NOT define "what is motor" -- the
caller passes ``motor_group_names`` (derived from
``HomeostasisScheduler.motor_groups`` in network.py). It does NOT define
how synapse weights are serialized -- it operates on the flat blob shape
already produced by ``NeuromorphicNetwork.get_motor_skill_blob``. It does
NOT define motor cortex sub-range layout -- the caller passes
``channel_neuron_ranges``. Single-responsibility: tag-overlap matching +
nearest-neighbor row resampling.

Algorithm (per channel present in both manifests):
  1. Slice the channel's row range [r_start, r_end) into K_src equal
     bands (one per source actuator) and K_tgt equal bands (one per
     target actuator). This mirrors the PopulationVectorDecoder's
     equal-slice convention.
  2. For each target actuator T_i (band [t0, t1)):
     - tag set T = set(target_manifest[channel][i].tags)
     - find source actuator with maximum |T ∩ S_j| across j; require
       at least one shared body-part tag (configurable via
       ``min_required_tags``) to project; otherwise leave T_i band
       untouched.
     - source band [s0, s1) -> nearest-neighbor resample to length
       (t1 - t0), copy from source weights into target weights for
       every motor synapse group's blob row range.

Body-part tags vs spatial tags:
  ``left`` and ``right`` overlap across channels (a humanoid hip and a
  humanoid shoulder are both on the same body side). To avoid spurious
  cross-channel projection, callers may pass ``body_part_required_tags``
  (default: {"leg", "arm", "hip", "knee", "ankle", "shoulder", "elbow",
  "wrist", "neck", "waist", "core", "head"}) -- at least one of these
  must overlap for a projection match to count. Pass an empty set to
  allow projection on spatial overlap alone (looser, more risk of
  cross-channel contamination).
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


_DEFAULT_BODY_PART_TAGS: frozenset[str] = frozenset(
    {
        "leg",
        "arm",
        "hip",
        "knee",
        "ankle",
        "shoulder",
        "elbow",
        "wrist",
        "neck",
        "waist",
        "core",
        "head",
        "rotor",
        "wheel",
        "fin",
        "thruster",
        "propulsion",
    }
)

# Single source of truth for the motor-skill blob key separator. Used by
# both the projection helper here and NeuromorphicNetwork.get/set_motor_skill_blob
# in network.py. Network imports this constant so both sides can never drift.
MOTOR_SKILL_SEP = "::"
_MOTOR_SKILL_SEP = MOTOR_SKILL_SEP  # backward compat for existing callers


# --------------------------------------------------------------------------
# A3 (2026-06-21): per-actuator dopamine weighting for Phase 3 motor.outcome
# receive. Pure functions; no network state, no body assumptions.
#
# When a cross-body motor.outcome arrives carrying source-channel actuator
# tags, we want the DA application to be PROPORTIONAL to how relevant the
# source skill is to each target actuator -- not a flat all-or-nothing gate.
# A walker walk_short outcome (tags ~ {leg, pelvis}) should land DA most
# strongly on the humanoid's leg-tagged actuators and least on
# torso/arm-tagged ones.
#
# The current scalar-DA path in network.py is preserved -- we collapse the
# per-actuator weight vector to a single [0, 1] aggregate scale that
# multiplies the existing _outcome_da_multiplier. A future phase can push
# the full weight vector into network.py for true per-column gating; until
# then this is the cleanest body-agnostic refinement of the existing gate.
# --------------------------------------------------------------------------


def compute_per_actuator_da_weights(
    src_tags: Iterable[str],
    target_actuators: list[dict],
) -> list[float]:
    """For each target actuator in the channel, compute a normalized [0, 1]
    DA weight based on how much of the target actuator's tag set is covered
    by the source tag set.

    Per-actuator raw score = ``|src ∩ target.tags| / |target.tags|``
    (containment of target in source). 1.0 when every tag the target
    actuator carries is also present on the source side, 0.0 on no overlap,
    fractional otherwise. Containment (not Jaccard) so that a "broader"
    source skill that mentions tags the target does not carry still scores
    1.0 on actuators it fully covers -- the natural same-body identity.

    Then normalized so the best-matched target actuator scores 1.0,
    preserving full DA on the most relevant column. Returns all zeros if
    either side is empty or no actuator overlaps at all.

    Body-agnostic: brain code never interprets the tag strings; we do pure
    string-set intersection math.
    """
    src = set(src_tags)
    raw: list[float] = []
    for tgt in target_actuators:
        tgt_tags = set(tgt.get("tags") or [])
        if not src or not tgt_tags:
            raw.append(0.0)
            continue
        # |tgt_tags| can never be zero here because the branch above
        # caught empty targets.
        raw.append(len(src & tgt_tags) / len(tgt_tags))
    m = max(raw) if raw else 0.0
    if m <= 0.0:
        return [0.0] * len(raw)
    return [w / m for w in raw]


def aggregate_actuator_weights_to_scale(weights: list[float]) -> float:
    """Aggregate per-actuator weights into a single [0, 1] scale factor for
    the existing scalar-DA path.

    Uses the arithmetic mean (not max) so a small overlap into a single
    actuator does not yield full DA across the whole channel sub-range --
    the brain learns proportional to how much of the channel the source
    skill actually applies to. Same-body transfer (specialist walker ->
    walker 1M) has every actuator at weight 1.0 so the mean is 1.0 and
    the existing DA path is unchanged.

    Returns 0.0 on empty input.
    """
    if not weights:
        return 0.0
    return float(sum(weights) / len(weights))


@dataclass
class _Match:
    """One projection decision: target actuator at band [t0, t1) inherits
    from source band [s0, s1) with weight overlap-strength."""

    channel: str
    target_index: int
    target_band: tuple[int, int]
    source_index: int
    source_band: tuple[int, int]
    overlap: frozenset[str]


def _equal_bands(r_start: int, r_end: int, k: int) -> list[tuple[int, int]]:
    """Split [r_start, r_end) into k equal-ish integer bands. Mirrors the
    PopulationVectorDecoder layout."""
    if k <= 0:
        return []
    total = r_end - r_start
    bands = []
    for i in range(k):
        b0 = r_start + (i * total) // k
        b1 = r_start + ((i + 1) * total) // k
        bands.append((b0, b1))
    return bands


def _nn_index_map(src_size: int, tgt_size: int) -> np.ndarray:
    """Nearest-neighbor index array: for each of ``tgt_size`` target
    rows, return the index (0..src_size-1) of the closest source row."""
    if src_size <= 0 or tgt_size <= 0:
        return np.empty(0, dtype=np.int64)
    return np.minimum(
        (np.arange(tgt_size) * src_size) // tgt_size,
        src_size - 1,
    )


def _channel_matches(
    channel: str,
    source_actuators: list[dict],
    target_actuators: list[dict],
    r_start: int,
    r_end: int,
    body_part_required_tags: frozenset[str],
) -> list[_Match]:
    """For one channel, decide which target actuators get which source
    actuator's rows projected. Returns a list of accepted matches."""
    src_bands = _equal_bands(r_start, r_end, len(source_actuators))
    tgt_bands = _equal_bands(r_start, r_end, len(target_actuators))
    matches: list[_Match] = []
    for i_tgt, tgt in enumerate(target_actuators):
        tgt_tags = set(tgt.get("tags") or [])
        if not tgt_tags:
            continue
        best_j = -1
        best_overlap: set[str] = set()
        for j_src, src in enumerate(source_actuators):
            src_tags = set(src.get("tags") or [])
            ov = tgt_tags & src_tags
            if not ov:
                continue
            if body_part_required_tags and not (ov & body_part_required_tags):
                continue
            # Prefer the source with the largest body-part overlap, then
            # the largest total overlap.
            ov_body = ov & body_part_required_tags if body_part_required_tags else ov
            best_ov_body = (
                best_overlap & body_part_required_tags if body_part_required_tags else best_overlap
            )
            if len(ov_body) > len(best_ov_body) or (
                len(ov_body) == len(best_ov_body) and len(ov) > len(best_overlap)
            ):
                best_j = j_src
                best_overlap = ov
        if best_j >= 0:
            matches.append(
                _Match(
                    channel=channel,
                    target_index=i_tgt,
                    target_band=tgt_bands[i_tgt],
                    source_index=best_j,
                    source_band=src_bands[best_j],
                    overlap=frozenset(best_overlap),
                )
            )
    return matches


def project_motor_skill_via_tags(
    *,
    source_blob: dict[str, np.ndarray],
    source_actuator_manifest: dict[str, list[dict]],
    target_actuator_manifest: dict[str, list[dict]],
    channel_neuron_ranges: dict[str, tuple[int, int]],
    motor_group_names: Iterable[str],
    body_part_required_tags: frozenset[str] = _DEFAULT_BODY_PART_TAGS,
) -> tuple[dict[str, np.ndarray], list[_Match]]:
    """Project a source-body motor blob to the target body's actuator
    layout. Returns ``(projected_blob, applied_matches)``.

    ``projected_blob`` starts as a copy of ``source_blob`` and has the
    target-actuator row bands overwritten in place where a source match
    exists. Rows outside motor groups (e.g. eligibility, BCM theta) are
    NOT row-remapped; we only touch the weights_data of motor synapse
    groups via their CSR ``weights_indptr`` -> row mapping.

    For Phase 3.5 MVP, we operate on dense weight rows by expanding the
    relevant CSR group on the fly. Sparse-aware projection is a follow-up.

    Iron rule: never reads body-specific identifiers, never branches on
    channel name. The only set comparison is body-part tag overlap.
    """
    motor_set = set(motor_group_names)
    projected: dict[str, np.ndarray] = {k: v.copy() for k, v in source_blob.items()}
    all_matches: list[_Match] = []
    for channel, target_actuators in target_actuator_manifest.items():
        source_actuators = source_actuator_manifest.get(channel) or []
        if not source_actuators or not target_actuators:
            continue
        if channel not in channel_neuron_ranges:
            continue
        r_start, r_end = channel_neuron_ranges[channel]
        if r_end <= r_start:
            continue
        matches = _channel_matches(
            channel,
            source_actuators,
            target_actuators,
            r_start,
            r_end,
            body_part_required_tags,
        )
        all_matches.extend(matches)

        # Apply matches to each motor synapse group's weights. Read from
        # the FROZEN source blob (source_blob) so multiple matches within
        # the same group don't interfere; write to ``projected``.
        for group_name in motor_set:
            indptr_key = f"{group_name}{_MOTOR_SKILL_SEP}weights_indptr"
            data_key = f"{group_name}{_MOTOR_SKILL_SEP}weights_data"
            indices_key = f"{group_name}{_MOTOR_SKILL_SEP}weights_indices"
            if not (
                indptr_key in source_blob and data_key in source_blob and indices_key in source_blob
            ):
                continue
            src_indptr = source_blob[indptr_key]
            src_data = source_blob[data_key]
            src_indices = source_blob[indices_key]
            n_post = len(src_indptr) - 1
            # Build a per-row mapping: for every post-row in the output,
            # decide which source row it inherits from. Default = identity
            # (no projection -> stays as source).
            row_source = np.arange(n_post, dtype=np.int64)
            for m in matches:
                s0, s1 = m.source_band
                t0, t1 = m.target_band
                src_size = s1 - s0
                tgt_size = t1 - t0
                if src_size <= 0 or tgt_size <= 0:
                    continue
                if s1 > n_post or t1 > n_post:
                    continue
                nn = _nn_index_map(src_size, tgt_size)
                for offset in range(tgt_size):
                    row_source[t0 + offset] = s0 + int(nn[offset])
            # Now rebuild the CSR matrix in one pass.
            new_lengths = np.empty(n_post, dtype=np.int64)
            new_data_parts: list[np.ndarray] = []
            new_indices_parts: list[np.ndarray] = []
            for r in range(n_post):
                src_r = int(row_source[r])
                p0, p1 = int(src_indptr[src_r]), int(src_indptr[src_r + 1])
                new_lengths[r] = p1 - p0
                new_data_parts.append(src_data[p0:p1])
                new_indices_parts.append(src_indices[p0:p1])
            new_indptr = np.empty(n_post + 1, dtype=src_indptr.dtype)
            new_indptr[0] = 0
            np.cumsum(new_lengths, out=new_indptr[1:])
            projected[indptr_key] = new_indptr
            projected[data_key] = (
                np.concatenate(new_data_parts).astype(src_data.dtype)
                if new_data_parts
                else src_data.copy()
            )
            projected[indices_key] = (
                np.concatenate(new_indices_parts).astype(src_indices.dtype)
                if new_indices_parts
                else src_indices.copy()
            )
        # Log channel-level summary.
        if matches:
            logger.info(
                "Phase 3.5 projection: channel=%s projected %d target " "actuators via tag overlap",
                channel,
                len(matches),
            )
    return projected, all_matches


__all__ = [
    "project_motor_skill_via_tags",
    "MOTOR_SKILL_SEP",
    "_DEFAULT_BODY_PART_TAGS",
]
