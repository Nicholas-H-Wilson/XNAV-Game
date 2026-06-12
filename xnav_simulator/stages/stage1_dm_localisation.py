# stages/stage1_dm_localisation.py — Coarse galactic localisation from DM pattern
# XNAV Cold Start Simulator

"""
Stage 1: DM Localisation

Runs BEFORE the particle filter initialises.  Given a set of pulsars and their
observed DM values from the spacecraft, this stage sweeps a coarse galactic grid
and accumulates a log-likelihood for each grid point.  The resulting probability
map is used to sample the initial particle distribution — replacing the dangerous
uniform galactic prior.

Physics summary
---------------
The spacecraft measures the dispersion measure (DM) to each pulsar.  DM depends
on the total electron column density along the line of sight.  For a spacecraft
at position r⃗, the DM to pulsar j at r⃗_j differs from the Earth-referenced
catalogue DM because the spacecraft is observing through a different sightline.

At kpc scales the dominant signal is: spacecraft displacement along the
pulsar line-of-sight changes the path length through the galactic disk, changing
the DM by roughly 8 pc cm⁻³/kpc (typical warm-ISM density gradient).

The likelihood at grid point r⃗ is:
    log L(r⃗) = Σ_j  −0.5 × (DM_observed_j − DM_predicted(r⃗, j))² / σ_DM_j²

where σ_DM_j = ISM_turbulence × DM_predicted = 0.15 × DM_predicted
(15% log-normal turbulence baked into the grid).

Return value
------------
dict with keys:
    probability_map   : 3D float64 array (n_x, n_y, n_z) — normalised to sum=1
    x_arr, y_arr, z_arr: coordinate axes in galactocentric kpc
    best_region       : dict(centre_kpc, radius_kpc) — peak-probability region
    dm_residuals      : dict(pulsar_name → observed − expected at best point)
    log_likelihood_map: 3D float64 — raw log-likelihoods (for diagnostics)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import numpy as np

from config import (
    GALAXY_RADIUS_KPC,
    GALAXY_THICKNESS_KPC,
    K_DM,
    SUN_POS_KPC,
)

if TYPE_CHECKING:
    from core.interstellar_medium import InterstellarMedium
    from core.pulsar import Pulsar

logger = logging.getLogger(__name__)

# Default fractional DM uncertainty (log-normal sigma baked into ISM grid)
_ISM_TURB_SIGMA = 0.15


def run(
    pulsars: list["Pulsar"],
    observed_dm_values: dict[str, float],
    ism_model: Optional["InterstellarMedium"] = None,
    grid_resolution_pc: float = 500.0,
    spacecraft_position_kpc: Optional[np.ndarray] = None,
    frequency_mhz: float = 1400.0,
) -> dict:
    """Compute a coarse probability map over galactic volume from DM observations.

    Parameters
    ----------
    pulsars: list of Pulsar objects (all must appear in observed_dm_values)
    observed_dm_values: {pulsar.name: observed_DM_pc_cm3}
    ism_model: InterstellarMedium with loaded DM grid (optional — uses catalogue
               DM values if None or grid not loaded)
    grid_resolution_pc: coarse grid cell size in parsecs
    spacecraft_position_kpc: true position (only used for diagnostic logging)
    frequency_mhz: observing frequency (used for computing DM sigma via K_DM)

    Returns
    -------
    dict as described in module docstring
    """
    # ── Build coarse grid ─────────────────────────────────────────────────────
    step_kpc = grid_resolution_pc / 1000.0

    x_arr = np.arange(-GALAXY_RADIUS_KPC, GALAXY_RADIUS_KPC + step_kpc, step_kpc)
    y_arr = np.arange(-GALAXY_RADIUS_KPC, GALAXY_RADIUS_KPC + step_kpc, step_kpc)
    z_arr = np.arange(
        -GALAXY_THICKNESS_KPC / 2.0,
        GALAXY_THICKNESS_KPC / 2.0 + step_kpc,
        step_kpc,
    )

    nx, ny, nz = len(x_arr), len(y_arr), len(z_arr)
    log_likelihood_map = np.zeros((nx, ny, nz), dtype=np.float64)

    # Filter to pulsars that appear in observed_dm_values
    active_pulsars = [p for p in pulsars if p.name in observed_dm_values]
    if not active_pulsars:
        raise ValueError(
            "No pulsars in observed_dm_values match the provided pulsar list."
        )

    # Pre-collect pulsar positions and observed DMs
    pulsar_positions = np.array([p.position_kpc for p in active_pulsars])   # (P, 3)
    obs_dms = np.array([observed_dm_values[p.name] for p in active_pulsars])  # (P,)

    use_grid = ism_model is not None and ism_model.grid_loaded()

    logger.info(
        "Stage 1: grid %dx%dx%d at %.0f pc, %d pulsars, ISM grid=%s",
        nx, ny, nz, grid_resolution_pc, len(active_pulsars), use_grid,
    )

    # ── Vectorised likelihood accumulation ───────────────────────────────────
    # Build (nx*ny*nz, 3) array of all grid positions, then compute in one pass.
    #
    # APPROXIMATION: The DM prediction uses the ISM grid value at the midpoint
    # between the spacecraft and each pulsar, scaled by path length.  This is
    # the same midpoint approximation used in the particle filter — consistent
    # with how the particle filter will evaluate likelihoods later.

    xx, yy, zz = np.meshgrid(x_arr, y_arr, z_arr, indexing="ij")   # each (nx,ny,nz)
    grid_pts = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)  # (N,3)
    n_grid = grid_pts.shape[0]

    # For each (grid_point, pulsar) compute predicted DM
    # grid_pts (N,3), pulsar_positions (P,3)
    # midpoints: (N,P,3)
    gp_exp = grid_pts[:, np.newaxis, :]          # (N,1,3)
    pp_exp = pulsar_positions[np.newaxis, :, :]  # (1,P,3)
    midpoints_flat = ((gp_exp + pp_exp) / 2.0).reshape(-1, 3)   # (N*P, 3)

    if use_grid:
        dm_at_mid = ism_model.batch_lookup(midpoints_flat)   # (N*P,)
        helio_vecs = midpoints_flat - SUN_POS_KPC
        helio_dists = np.maximum(np.linalg.norm(helio_vecs, axis=1), 0.1)
        mean_density = dm_at_mid / helio_dists                               # pc cm⁻³/kpc
        diff = pp_exp - gp_exp                                               # (N,P,3)
        path_lengths = np.linalg.norm(diff, axis=2).ravel()                 # (N*P,)
        predicted_dm = np.maximum(mean_density * path_lengths, 0.5)         # (N*P,)
    else:
        # Without the ISM grid, catalogue DMs are position-invariant: every
        # grid point would get identical likelihood and the "localisation"
        # map would be uniform noise dressed up as a result.  Fail loudly —
        # the caller can fall back to a region prior and surface the failure.
        raise RuntimeError(
            "Stage 1 DM localisation requires a loaded ISM DM grid; "
            "catalogue DMs carry no position signal. Ensure "
            "data/ne2001_grid.npz exists or call ism.precompute_grid()."
        )

    predicted_dm = predicted_dm.reshape(n_grid, len(active_pulsars))   # (N,P)

    # Uncertainty: 15% ISM turbulence fraction of the predicted DM
    # APPROXIMATION: sigma_DM = ISM_TURB_SIGMA * max(DM_predicted, 1).
    # Real DM uncertainty also includes instrumental noise and solar wind.
    # Dominant term at kpc scales is ISM turbulence (~15%).
    sigma_dm = np.maximum(_ISM_TURB_SIGMA * predicted_dm, 0.5)        # (N,P)

    # Log-likelihood: sum over pulsars of Gaussian log-prob
    obs_exp = obs_dms[np.newaxis, :]   # (1,P)
    residuals = obs_exp - predicted_dm                                 # (N,P)
    log_l = -0.5 * np.sum((residuals / sigma_dm) ** 2, axis=1)        # (N,)

    log_likelihood_map = log_l.reshape(nx, ny, nz)

    # ── Normalise to probability map ──────────────────────────────────────────
    # Shift by max for numerical stability before exp
    log_l_stable = log_likelihood_map - log_likelihood_map.max()
    raw_prob = np.exp(log_l_stable)
    prob_sum = raw_prob.sum()
    probability_map = raw_prob / prob_sum

    # ── Find best region (peak of probability map) ────────────────────────────
    peak_idx = np.unravel_index(np.argmax(probability_map), (nx, ny, nz))
    best_centre = np.array([
        x_arr[peak_idx[0]],
        y_arr[peak_idx[1]],
        z_arr[peak_idx[2]],
    ])

    # Estimate a "region radius" from the 68th-percentile probability contour.
    # APPROXIMATION: Use the weighted std dev of the distribution as the radius.
    flat_prob = probability_map.ravel()
    weighted_mean = np.array([
        np.sum(flat_prob * grid_pts[:, 0]),
        np.sum(flat_prob * grid_pts[:, 1]),
        np.sum(flat_prob * grid_pts[:, 2]),
    ])
    diffs = grid_pts - weighted_mean[np.newaxis, :]
    dist2 = np.sum(diffs ** 2, axis=1)
    region_radius = float(np.sqrt(np.sum(flat_prob * dist2)))

    # ── DM residuals at the best-estimate point ────────────────────────────────
    if use_grid:
        # Predict DM at best_centre → each pulsar
        bc_exp = best_centre[np.newaxis, np.newaxis, :]  # (1,1,3)
        pp_bc = pulsar_positions[np.newaxis, :, :]       # (1,P,3)
        mid_bc = ((bc_exp + pp_bc) / 2.0).reshape(-1, 3)
        dm_bc = ism_model.batch_lookup(mid_bc)
        helio_bc = np.linalg.norm(mid_bc - SUN_POS_KPC, axis=1)
        helio_bc = np.maximum(helio_bc, 0.1)
        density_bc = dm_bc / helio_bc
        path_bc = np.linalg.norm(pulsar_positions - best_centre[np.newaxis, :], axis=1)
        pred_at_best = np.maximum(density_bc * path_bc, 0.5)
    else:
        pred_at_best = np.array([p.dm for p in active_pulsars])

    dm_residuals = {
        p.name: float(observed_dm_values[p.name]) - float(pred_at_best[j])
        for j, p in enumerate(active_pulsars)
    }

    # Diagnostic logging
    if spacecraft_position_kpc is not None:
        true_pos = np.asarray(spacecraft_position_kpc)
        localisation_error = float(np.linalg.norm(best_centre - true_pos))
        logger.info(
            "Stage 1: best-region centre=%.2f,%.2f,%.2f kpc, "
            "true pos=%.2f,%.2f,%.2f kpc, error=%.2f kpc, radius=%.2f kpc",
            *best_centre, *true_pos, localisation_error, region_radius,
        )
    else:
        logger.info(
            "Stage 1: best-region centre=%.2f,%.2f,%.2f kpc, radius=%.2f kpc",
            *best_centre, region_radius,
        )

    return {
        "probability_map": probability_map,
        "x_arr": x_arr,
        "y_arr": y_arr,
        "z_arr": z_arr,
        "best_region": {
            "centre_kpc": best_centre,
            "radius_kpc": max(region_radius, step_kpc),
        },
        "dm_residuals": dm_residuals,
        "log_likelihood_map": log_likelihood_map,
        "n_pulsars_used": len(active_pulsars),
    }
