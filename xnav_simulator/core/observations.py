# core/observations.py — Synthetic observation generator (filter-consistent)
"""Build synthetic pulsar timing observations for the simulation loop.

This is the single source of truth for the observation forward model, shared
by app.py and tests/run_convergence_study.py.  Keeping one implementation
guarantees the observations and the particle filter likelihood stay consistent.

CRITICAL (Appendix D.2): the LOS convention is origin→pulsar
(los = pulsar_pos / |pulsar_pos|), matching the particle filter kernel, so the
true particle has exactly zero Roemer residual.  Using the spacecraft→pulsar
convention (TimingModel.compute_arrival_time) would create a ~10^11 s Roemer
mismatch and prevent convergence.
"""

from __future__ import annotations

import numpy as np

from config import C_LIGHT, K_DM, KPC_TO_M, SUN_POS_KPC


def build_observations(
    pulsars: list,
    sc_pos: np.ndarray,
    ism,
    frequency_mhz: float = 1400.0,
    rng: np.random.Generator | None = None,
    timing_noise_scale: float = 1.0,
    ism_turb_scale: float = 1.0,
    integration_time_s: float = 1000.0,
) -> tuple[dict, dict]:
    """Generate synthetic arrival times using los = pulsar_pos / |pulsar_pos|.

    Returns
    -------
    observed_timings: dict {name: {"total": float, "geometric": float,
                                   "dispersive": float, "roemer_s": float,
                                   "dispersive_s": float, "timing_noise_s": float}}
    dm_values:        dict {name: observed DM (pc cm⁻³)}
    """
    if rng is None:
        rng = np.random.default_rng()

    observed_timings: dict = {}
    dm_values: dict = {}

    for p in pulsars:
        # LOS: origin → pulsar (filter-consistent convention)
        norm_p = np.linalg.norm(p.position_kpc)
        if norm_p < 1e-10:
            los = np.array([1.0, 0.0, 0.0])
        else:
            los = p.position_kpc / norm_p

        # Roemer delay: projection of sc_pos onto LOS direction (seconds)
        roemer = -float(np.dot(sc_pos, los)) * KPC_TO_M / C_LIGHT

        # DM from ISM model at midpoint between SC and pulsar
        mid = (sc_pos + p.position_kpc) / 2.0
        if ism is not None and ism.grid_loaded():
            dm_at_mid = float(ism.batch_lookup(mid.reshape(1, 3))[0])
        else:
            dm_at_mid = p.dm

        # Scale DM by path length relative to heliocentric distance
        helio_dist = max(float(np.linalg.norm(mid - SUN_POS_KPC)), 0.1)
        path_kpc = float(np.linalg.norm(p.position_kpc - sc_pos))
        dm_obs = max(dm_at_mid / helio_dist * path_kpc, 0.5)

        # DM turbulence noise (ISM floor, 15%)
        if ism_turb_scale > 0:
            dm_turb = rng.normal(0.0, dm_obs * 0.15 * ism_turb_scale)
            dm_obs += dm_turb

        dispersive = K_DM * dm_obs / (frequency_mhz ** 2)

        # Timing noise scales with sqrt(integration time) — radiometer equation
        # sigma_t = sigma_baseline * sqrt(1 s / T_int)
        sigma_t = (p.timing_noise_ns * 1e-9 * timing_noise_scale
                   / np.sqrt(max(integration_time_s, 1.0)))
        noise_t = rng.normal(0.0, sigma_t)

        total = roemer + dispersive + noise_t

        observed_timings[p.name] = {
            "total": total,
            "geometric": roemer,
            "dispersive": dispersive,
            "roemer_s": roemer,
            "dispersive_s": dispersive,
            "timing_noise_s": noise_t,
        }
        dm_values[p.name] = dm_obs

    return observed_timings, dm_values
