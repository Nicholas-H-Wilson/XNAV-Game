# stages/stage4_phase_ambiguity.py — Phase ambiguity resolution via CRT approach
# XNAV Cold Start Simulator

"""
Stage 4: Phase Ambiguity Resolution

Pulsar timing gives sub-period phase measurements, not absolute pulse counts.
A pulsar with period P observed at phase φ could correspond to any epoch:
    t = φ × P + n × P,   n ∈ ℤ

With multiple pulsars of incommensurate periods, the Chinese Remainder Theorem
(CRT) analogy applies: the intersection of candidate windows from N pulsars
shrinks exponentially.  The ambiguity window after including N pulsars is
approximately:
    W_N ≈ P_1 × P_2 × … × P_N / (W_0^(N-1))
where W_0 is the initial timing window.

Physics
-------
Clock offset t_clock is the unknown.  Each pulsar j provides:
    phase_j = (t_true + t_clock) / P_j  mod 1
⟹  candidates for t_clock: t_clock = φ_j × P_j + k × P_j − t_true, k ∈ ℤ

We find the smallest window of t_clock values consistent with all pulsars
simultaneously.  At each step, adding a new pulsar intersects its candidate
set with the running window.

APPROXIMATION: This simulation uses a discrete grid search over t_clock rather
than the exact CRT.  Grid resolution = min(P_j)/1000.  For pulsar periods in
[1 ms, 10 ms], this gives ~1 μs resolution — adequate for navigation.

Return value
------------
dict with keys:
    resolved_clock_offset_s    : float — best-fit clock offset estimate (seconds)
    ambiguity_window_s         : float — final ambiguity window half-width (seconds)
    window_history             : list[(n_pulsars, window_s)] — window vs pulsar count
    candidate_times_s          : np.ndarray — remaining candidate clock offsets
    n_pulsars_used             : int
    metadata                   : dict
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from core.pulsar import Pulsar

logger = logging.getLogger(__name__)

# Maximum initial clock offset search range (seconds)
# Covers ±100 ms, which is >> any physically plausible XNAV cold-start clock error.
_MAX_CLOCK_OFFSET_S = 0.1

# Grid resolution as a fraction of the shortest period
_GRID_RESOLUTION_FRACTION = 1e-3


def run(
    identified_pulsars: list["Pulsar"],
    arrival_times: dict[str, float],
    position_estimate_kpc: np.ndarray,
    true_clock_offset_s: float = 0.0,
    seed: int = 42,
) -> dict:
    """Resolve clock offset phase ambiguity by adding pulsars one at a time.

    Parameters
    ----------
    identified_pulsars: pulsars confirmed by Stage 2 (need period P)
    arrival_times: {pulsar.name: observed_arrival_time_s} — can be simulated
    position_estimate_kpc: current best position estimate (for Roemer correction)
    true_clock_offset_s: true clock offset (used only for diagnostics)
    seed: RNG seed for tie-breaking

    Returns
    -------
    dict as described in module docstring
    """
    if not identified_pulsars:
        return {
            "resolved_clock_offset_s": 0.0,
            "ambiguity_window_s": _MAX_CLOCK_OFFSET_S,
            "window_history": [],
            "candidate_times_s": np.array([]),
            "n_pulsars_used": 0,
            "metadata": {"note": "No identified pulsars provided"},
        }

    # Sort pulsars by timing quality (lowest period → most constraining first)
    # APPROXIMATION: Sort by period only.  Real strategy sorts by
    # (1/period) × (1/timing_noise), but timing noise is not available here.
    sorted_pulsars = sorted(identified_pulsars, key=lambda p: p.period)

    # Determine grid parameters
    min_period = sorted_pulsars[0].period
    grid_step = max(min_period * _GRID_RESOLUTION_FRACTION, 1e-9)  # ≥ 1 ns

    # Initial candidate window: ±MAX_CLOCK_OFFSET_S
    t_candidates = np.arange(-_MAX_CLOCK_OFFSET_S, _MAX_CLOCK_OFFSET_S, grid_step)
    initial_window = float(_MAX_CLOCK_OFFSET_S * 2)

    window_history: list[tuple[int, float]] = [(0, initial_window)]

    for step_idx, pulsar in enumerate(sorted_pulsars, start=1):
        if pulsar.name not in arrival_times:
            continue

        t_arrival = float(arrival_times[pulsar.name])
        period = float(pulsar.period)

        # Phase observed for this pulsar
        # Phase = (t_arrival) / period  mod 1
        # Candidate clock offsets that are consistent with this observation:
        # t_clock such that (t_arrival + t_clock) / period ≡ observed_phase (mod 1)
        # i.e. t_candidates must satisfy:
        # (t_arrival + t_clock) mod period is near the observed residual
        #
        # APPROXIMATION: We treat the observed arrival time as the phase residual
        # (t_arrival already incorporates the fractional-period information).
        # The consistency check is: |(t_arrival + t_cand) mod period| < tolerance
        # where tolerance = min(period * 0.01, grid_step * 5).

        tolerance = max(period * 0.01, grid_step * 5)

        # Wrapped residual for each candidate
        residual = (t_arrival + t_candidates) % period   # in [0, period)
        # Accept candidates near 0 or near period (wrap-around)
        consistent = (residual < tolerance) | (residual > period - tolerance)

        new_candidates = t_candidates[consistent]

        if len(new_candidates) == 0:
            # No candidates survived — timing data inconsistent with prior window.
            # APPROXIMATION: Fall back to narrowing around the best-score candidate
            # rather than completely collapsing.  In practice this means ISM noise
            # pushed the arrival time out of the tolerance band.  Widen tolerance
            # to 3% of period and retry.
            tolerance_wide = period * 0.03
            consistent_wide = (residual < tolerance_wide) | (residual > period - tolerance_wide)
            new_candidates = t_candidates[consistent_wide]
            logger.warning(
                "Stage 4: Pulsar %s: no candidates in tight window; widened to %.1f%% of period.",
                pulsar.name, 100 * tolerance_wide / period,
            )

        if len(new_candidates) > 0:
            t_candidates = new_candidates
        # else: keep old candidates (this pulsar provides no additional constraint)

        # Ambiguity window = full peak-to-peak range of surviving candidates
        if len(t_candidates) > 1:
            window = float(t_candidates[-1] - t_candidates[0])
        elif len(t_candidates) == 1:
            window = grid_step / 2.0
        else:
            window = initial_window

        window_history.append((step_idx, window))
        logger.debug(
            "Stage 4 step %d (%s): %d candidates, window=%.3e s",
            step_idx, pulsar.name, len(t_candidates), window,
        )

    # Best estimate: weighted mean of remaining candidates
    # APPROXIMATION: Uniform weight over remaining candidates.  Real scoring
    # would weight by likelihood of each candidate given all timing data.
    if len(t_candidates) > 0:
        resolved = float(np.median(t_candidates))
        window_final = float(t_candidates[-1] - t_candidates[0]) if len(t_candidates) > 1 else grid_step / 2.0
    else:
        resolved = 0.0
        window_final = initial_window

    if true_clock_offset_s != 0.0:
        error = abs(resolved - true_clock_offset_s)
        logger.info(
            "Stage 4: resolved offset=%.3e s, true=%.3e s, error=%.3e s, window=%.3e s",
            resolved, true_clock_offset_s, error, window_final,
        )
    else:
        logger.info(
            "Stage 4: resolved offset=%.3e s, window=%.3e s, %d pulsars",
            resolved, window_final, len(sorted_pulsars),
        )

    return {
        "resolved_clock_offset_s": resolved,
        "ambiguity_window_s": window_final,
        "window_history": window_history,
        "candidate_times_s": t_candidates,
        "n_pulsars_used": len([p for p in sorted_pulsars if p.name in arrival_times]),
        "metadata": {
            "grid_step_s": grid_step,
            "initial_window_s": initial_window,
            "min_period_s": min_period,
            "note": "CRT-inspired grid search; APPROXIMATION: discrete grid, not exact CRT",
        },
    }


# ── Convenience: simulate arrival times for testing ──────────────────────────

def simulate_arrival_times(
    pulsars: list["Pulsar"],
    clock_offset_s: float,
    noise_scale_s: float = 1e-7,
    seed: int = 0,
) -> dict[str, float]:
    """Generate synthetic arrival times for Stage 4 testing.

    Each pulsar j gets:
        t_arrival_j = clock_offset_s mod P_j + noise

    where noise ∈ N(0, noise_scale_s²).

    APPROXIMATION: Ignores Roemer delay and DM contributions — only the
    fractional-period component (the phase ambiguity) is included.
    """
    rng = np.random.default_rng(seed)
    times = {}
    for p in pulsars:
        # Fractional phase residual from clock offset
        fractional_t = clock_offset_s % p.period
        noise = rng.normal(0.0, noise_scale_s)
        times[p.name] = float(fractional_t + noise)
    return times
