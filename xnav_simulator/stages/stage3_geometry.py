# stages/stage3_geometry.py — Geometric triangulation and GDOP computation
# XNAV Cold Start Simulator

"""
Stage 3: Geometric Triangulation

Given confirmed pulsar identifications and their known galactic positions,
this stage:
1. Updates particle filter weights based on geometric line-of-sight consistency
2. Computes GDOP (Geometric Dilution of Precision) from the pulsar sky distribution

Physics summary
---------------
Each identified pulsar at known position r⃗_j provides a line-of-sight constraint:
the spacecraft lies along the ray from r⃗_j in the direction of the observed
pulsar bearing.  The bearing is estimated from the spacecraft's Doppler shift and
the observed pulse-period (period sets the expected Doppler; comparison with
catalogue gives bearing).

In this simulation the bearing is derived directly from the known pulsar direction
(identified via Stage 2 profile matching).  The line-of-sight constraint is applied
by computing, for each particle, the perpendicular distance to the line-of-sight
ray from the pulsar — high-weight particles should be near the ray.

GDOP
----
GDOP (Geometric Dilution of Precision) measures how pulsar sky coverage amplifies
position uncertainty.  It is the condition number of the direction cosine matrix:

    A = [cos(l₁)cos(b₁), sin(l₁)cos(b₁), sin(b₁);
         cos(l₂)cos(b₂), ...               ; ...]

GDOP = sqrt(trace((AᵀA)⁻¹)) × sqrt(3)

Lower GDOP = better geometry.  GDOP < 2 is excellent; GDOP > 5 is poor.

Return value
------------
dict with keys:
    updated_filter      : the particle filter after weight update (same object)
    gdop                : float — geometric dilution of precision
    gdop_label          : str — human-readable interpretation
    n_pulsars_used      : int
    los_weights         : np.ndarray (N,) — the per-particle geometric weight update
    baseline_kpc        : float — characteristic spread of pulsar directions (kpc)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from core.estimator import ParticleFilter
    from core.pulsar import Pulsar

logger = logging.getLogger(__name__)

# Maximum perpendicular distance used to compute LOS soft constraint (kpc)
_LOS_SIGMA_KPC = 2.0


def run(
    identified_pulsars: list["Pulsar"],
    particle_filter: "ParticleFilter",
    los_sigma_kpc: float = _LOS_SIGMA_KPC,
) -> dict:
    """Apply geometric line-of-sight constraints and compute GDOP.

    Parameters
    ----------
    identified_pulsars: pulsars confirmed by Stage 2
    particle_filter: initialised ParticleFilter; weights are updated in-place
    los_sigma_kpc: 1-sigma width of the line-of-sight Gaussian constraint (kpc)

    Returns
    -------
    dict as described in module docstring
    """
    if not identified_pulsars:
        return {
            "updated_filter": particle_filter,
            "gdop": np.inf,
            "gdop_label": "No pulsars identified — GDOP undefined",
            "n_pulsars_used": 0,
            "los_weights": np.ones(particle_filter.n_particles),
            "baseline_kpc": 0.0,
        }

    particle_positions = particle_filter.particles[:, :3]   # (N, 3) kpc

    # ── 1. Compute geometric weight update ───────────────────────────────────
    # For each identified pulsar, compute perpendicular distance of each particle
    # from the pulsar's line-of-sight ray.  Accumulate as a log-weight update.
    #
    # APPROXIMATION: LOS ray is taken as a straight line from the pulsar in the
    # direction of the Sun (heliocentric reference).  Real angular uncertainty from
    # profile matching is ~1 arcsec, giving a transverse position error of
    # ~0.005 kpc at 1 kpc distance — negligible vs the DM-based uncertainty.

    log_w_update = np.zeros(particle_filter.n_particles, dtype=np.float64)

    for pulsar in identified_pulsars:
        psr_pos = np.asarray(pulsar.position_kpc)   # (3,)

        # Direction from pulsar toward Sun (normalised) — the LOS direction
        sun_dir = -psr_pos / (np.linalg.norm(psr_pos) + 1e-12)   # (3,)

        # Perpendicular distance of each particle from the ray through psr_pos
        # along sun_dir:
        #   d_perp = ||(particle - psr_pos) - [(particle - psr_pos)·sun_dir] sun_dir||
        rel = particle_positions - psr_pos[np.newaxis, :]           # (N, 3)
        proj_scalar = (rel * sun_dir[np.newaxis, :]).sum(axis=1)    # (N,)
        proj_vec = proj_scalar[:, np.newaxis] * sun_dir[np.newaxis, :]  # (N, 3)
        perp_vec = rel - proj_vec                                    # (N, 3)
        d_perp = np.linalg.norm(perp_vec, axis=1)                   # (N,)

        log_w_update += -0.5 * (d_perp / los_sigma_kpc) ** 2

    # Stabilise and convert to multiplicative weight update
    log_w_update -= log_w_update.max()
    los_weights = np.exp(log_w_update)   # (N,)

    # Multiply into existing weights (unnormalised; normalise after all updates)
    combined = particle_filter.weights * los_weights
    total = combined.sum()
    if total > 0 and np.isfinite(total):
        particle_filter.weights = combined / total
    else:
        logger.warning("Stage 3: geometric weight update collapsed; retaining prior weights.")

    # ── 2. Compute GDOP ──────────────────────────────────────────────────────
    gdop, gdop_label = _compute_gdop(identified_pulsars)

    logger.info(
        "Stage 3: GDOP=%.2f (%s), %d pulsars used",
        gdop, gdop_label, len(identified_pulsars),
    )

    # Characteristic spread: std dev of pulsar distances from weighted centroid
    psr_pos_arr = np.array([p.position_kpc for p in identified_pulsars])
    baseline_kpc = float(np.std(np.linalg.norm(psr_pos_arr, axis=1)))

    return {
        "updated_filter": particle_filter,
        "gdop": gdop,
        "gdop_label": gdop_label,
        "n_pulsars_used": len(identified_pulsars),
        "los_weights": los_weights,
        "baseline_kpc": baseline_kpc,
    }


# ── GDOP computation ──────────────────────────────────────────────────────────

def _compute_gdop(pulsars: list["Pulsar"]) -> tuple[float, str]:
    """Compute GDOP from pulsar direction cosine matrix.

    Returns (gdop, human_readable_label).
    """
    if len(pulsars) < 4:
        # Need at least 4 non-coplanar directions for 3D fix
        return float("inf"), f"Need ≥4 pulsars (have {len(pulsars)})"

    # Build direction cosine matrix A: rows are (cos_l·cos_b, sin_l·cos_b, sin_b)
    # APPROXIMATION: Uses each pulsar's galactic coordinates as seen from the origin.
    # Real angular coordinates as seen from the spacecraft differ by < 0.1° for
    # pulsars > 0.1 kpc away, which is negligible for GDOP.
    rows = []
    for p in pulsars:
        gl_r = np.radians(p.gl)
        gb_r = np.radians(p.gb)
        row = np.array([
            np.cos(gb_r) * np.cos(gl_r),
            np.cos(gb_r) * np.sin(gl_r),
            np.sin(gb_r),
        ])
        rows.append(row)

    A = np.array(rows)   # (P, 3)
    ATA = A.T @ A        # (3, 3)

    # GDOP = sqrt(trace(cov)) = sqrt(trace((AᵀA)⁻¹)) × sqrt(3)
    # The sqrt(3) normalises for 3D.
    try:
        ATA_inv = np.linalg.inv(ATA)
        gdop = float(np.sqrt(np.trace(ATA_inv)) * np.sqrt(3))
    except np.linalg.LinAlgError:
        return float("inf"), "Degenerate geometry (pulsars coplanar)"

    if not np.isfinite(gdop):
        return float("inf"), "Degenerate geometry"
    elif gdop < 2.0:
        label = "Excellent"
    elif gdop < 4.0:
        label = "Good"
    elif gdop < 6.0:
        label = "Moderate"
    else:
        label = "Poor"

    return gdop, label
