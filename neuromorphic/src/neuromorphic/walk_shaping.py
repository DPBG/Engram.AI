"""Phase 2.13b (2026-06-16): dense reward shaping for walk-family tasks.

Two metrics, both purely body-derived:

A1 -- gait-frequency band reward. Foot-contact transitions per
second in the recent history; reward 1.0 if in a healthy walking
band, linear falloff outside. Body-only: if the brain stops driving
locomotion (e.g. silences its CPG), foot contacts stop alternating
and A1 falls to 0 by construction. No silence-threshold knob needed.

A2 -- postural stability + forward motion. Reward 1.0 when COM_xy
projects within a slab around the line segment between the two
feet AND forward velocity is positive. Stationary balance is not
rewarded -- this is the walk-family shaping signal, not the stand
signal.

Both are dimensionless, in [0, 1], and combined as their mean
multiplied by ``NEURO_WALK_SHAPING_GAIN`` (default 0.2) before
being added to the task reward. One env var, hardcoded thresholds
elsewhere -- if tuning data demands per-knob control later, split
then.
"""

from __future__ import annotations

from collections import deque

import numpy as np


class FootContactHistory:
    """Rolling history of (left_contact, right_contact) booleans.

    Provides the foot-transition rate used by A1. Sized by the number
    of evaluation ticks expected within the window; the adapter
    pushes one tick per task-evaluation cycle (~1 Hz today).
    """

    def __init__(self, window_seconds: float, tick_period_seconds: float) -> None:
        # Round up so we never under-sample. Minimum 2 to detect any
        # transition at all.
        capacity = max(2, int(np.ceil(window_seconds / max(tick_period_seconds, 1e-6))))
        self._buf: deque[tuple[bool, bool]] = deque(maxlen=capacity)
        self._window_seconds = window_seconds
        self._tick_period_seconds = tick_period_seconds

    def push(self, left: bool, right: bool) -> None:
        self._buf.append((bool(left), bool(right)))

    def reset(self) -> None:
        self._buf.clear()

    def transitions_per_second(self) -> float:
        """Count left-only-stance <-> right-only-stance transitions.

        Double-support (both feet down) and flight (neither down) do
        not count as either stance; they are passed through and the
        transition is detected on the next single-support state.

        Returns 0.0 if the buffer is empty or no transitions are
        observed.
        """
        if len(self._buf) < 2:
            return 0.0
        last_stance: int | None = None  # 0 = left-only, 1 = right-only
        transitions = 0
        for left, right in self._buf:
            if left and not right:
                stance = 0
            elif right and not left:
                stance = 1
            else:
                continue
            if last_stance is not None and stance != last_stance:
                transitions += 1
            last_stance = stance
        elapsed = (len(self._buf) - 1) * self._tick_period_seconds
        if elapsed <= 0:
            return 0.0
        return transitions / elapsed


def gait_frequency_reward(
    transitions_per_second: float,
    band_min_hz: float,
    band_max_hz: float,
) -> float:
    """A1 reward function.

    Inside ``[band_min_hz, band_max_hz]`` -> 1.0. Linear falloff to
    0.0 outside the band; reaches 0 at half the band_min below and
    twice the band_max above. Clamped to [0, 1].
    """
    if band_max_hz <= band_min_hz:
        return 0.0
    if band_min_hz <= transitions_per_second <= band_max_hz:
        return 1.0
    if transitions_per_second < band_min_hz:
        # Linear from 0 at (band_min/2) to 1 at band_min.
        lower_edge = band_min_hz * 0.5
        if transitions_per_second <= lower_edge:
            return 0.0
        return float((transitions_per_second - lower_edge) / (band_min_hz - lower_edge))
    # Above band_max: linear from 1 at band_max to 0 at 2 * band_max.
    upper_edge = band_max_hz * 2.0
    if transitions_per_second >= upper_edge:
        return 0.0
    return float((upper_edge - transitions_per_second) / (upper_edge - band_max_hz))


def postural_forward_reward(
    com_xy: tuple[float, float],
    l_foot_xy: tuple[float, float],
    r_foot_xy: tuple[float, float],
    forward_velocity: float,
    slab_half_width: float,
) -> float:
    """A2 reward function.

    Returns 1.0 if BOTH:
      - COM_xy projects within ``slab_half_width`` of the line segment
        between the two foot positions (postural stability), AND
      - forward_velocity > 0 (active forward motion).
    Else 0.0.

    The support polygon for a biped is the convex hull of foot
    contact areas. With two foot points the polygon is the line
    segment between them; "over the polygon" is "within a slab" of
    that segment in the ground plane.
    """
    if forward_velocity <= 0.0:
        return 0.0
    cx, cy = com_xy
    lx, ly = l_foot_xy
    rx, ry = r_foot_xy
    seg_dx = rx - lx
    seg_dy = ry - ly
    seg_len_sq = seg_dx * seg_dx + seg_dy * seg_dy
    if seg_len_sq < 1e-9:
        # Feet at identical xy -- degenerate, fall back to distance.
        dist = float(np.hypot(cx - lx, cy - ly))
        return 1.0 if dist <= slab_half_width else 0.0
    # Project COM onto segment, clamp t to [0,1], compute perpendicular distance.
    t = ((cx - lx) * seg_dx + (cy - ly) * seg_dy) / seg_len_sq
    t = max(0.0, min(1.0, t))
    proj_x = lx + t * seg_dx
    proj_y = ly + t * seg_dy
    dist = float(np.hypot(cx - proj_x, cy - proj_y))
    return 1.0 if dist <= slab_half_width else 0.0


def combined_shaping_reward(a1: float, a2: float, gain: float) -> float:
    """Combine A1 and A2 into a single additive shaping term.

    Mean of the two components scaled by gain. Mean (not sum) keeps
    the maximum shaping bonus at ``gain * 1.0`` so a brain that
    succeeds on the original task is not drowned out by shaping.
    """
    return float(gain * 0.5 * (a1 + a2))


# ---------------------------------------------------------------------------
# Phase 2.17 Block B (2026-06-20): A3 ghost-tracking reward.
#
# Make watching the ghost demonstration a first-class reward signal, not
# just hoping cross-modal binding catches the alignment via STDP. The
# brain receives ghost joint angles on its own sensory sub-range AND
# now gets DA when its own joint angles match the ghost.
# ---------------------------------------------------------------------------


def ghost_tracking_reward(
    current_joint_angles: dict[str, float],
    ghost_joint_angles: dict[str, float],
) -> float:
    """A3 reward function: similarity between current pose and ghost pose.

    Cosine similarity over the joints present in BOTH dicts, mapped
    from [-1, 1] to [0, 1]. Returns 0.0 if no joints overlap or if
    either side has zero-norm vector.

    Pure function. Gain and clamping live in the combined reward.
    """
    shared = sorted(set(current_joint_angles) & set(ghost_joint_angles))
    if not shared:
        return 0.0
    cur = np.array([current_joint_angles[j] for j in shared], dtype=np.float32)
    gst = np.array([ghost_joint_angles[j] for j in shared], dtype=np.float32)
    cur_norm = float(np.linalg.norm(cur))
    gst_norm = float(np.linalg.norm(gst))
    if cur_norm < 1e-6 or gst_norm < 1e-6:
        return 0.0
    cos = float(np.dot(cur, gst) / (cur_norm * gst_norm))
    cos = max(-1.0, min(1.0, cos))
    return 0.5 * (cos + 1.0)


class GhostTrackingBuffer:
    """Rolling buffer of per-tick A3 scores; provides the mean over a window.

    Sized like ``FootContactHistory``: capacity rounded up from
    (window_seconds / tick_period_seconds). Empty buffer returns 0.0.
    """

    def __init__(self, window_seconds: float, tick_period_seconds: float) -> None:
        capacity = max(2, int(np.ceil(window_seconds / max(tick_period_seconds, 1e-6))))
        self._buf: deque[float] = deque(maxlen=capacity)

    def push(self, score: float) -> None:
        self._buf.append(float(score))

    def reset(self) -> None:
        self._buf.clear()

    def mean(self) -> float:
        if not self._buf:
            return 0.0
        return float(sum(self._buf) / len(self._buf))


def combined_shaping_reward_with_ghost(
    a1: float,
    a2: float,
    a3: float,
    gain: float,
) -> float:
    """Three-component shaping: gait + posture + ghost-tracking.

    Mean of (A1, A2, A3) scaled by gain. Mean keeps the cap at
    ``gain * 1.0`` so success-on-task still wins. When the ghost-
    tracking pipeline is disabled, callers pass A3=0 and use the
    two-component ``combined_shaping_reward`` instead -- this path
    is only taken when ghost-tracking is on, so A3 is always a
    real measurement.
    """
    return float(gain * (a1 + a2 + a3) / 3.0)
