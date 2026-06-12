# core/estimator.py — Liu-West particle filter for XNAV position estimation
# XNAV Cold Start Simulator

"""
ParticleFilter: Liu-West regularised particle filter for galactic cold-start
position estimation using pulsar timing residuals.

Architecture
------------
Particles are 6D state vectors (x, y, z, vx, vy, vz) in kpc / km s⁻¹.

DM table construction (vectorised, outside Numba):
    Broadcasting over (n_particles × n_pulsars) midpoints gives a single
    RegularGridInterpolator call instead of a Python-level double loop.
    At Balanced tier (5000 particles × 40 pulsars) this replaces 200,000
    Python-level interpolator calls with one batched NumPy call.

Numba JIT inner loop:
    The log-likelihood kernel (_log_likelihoods_jit) is decorated with
    @numba.jit(nopython=True).  It receives only plain float64 arrays —
    no Python objects enter nopython mode.

Liu-West kernel:
    Uses a = sqrt(1 − h²) per Liu & West (2001) §3.
    The build brief formula a = (3h−1)/(2h) is mathematically wrong at
    h ∈ (0,1) — at h=0.1 it yields a = −3.5, causing divergence.

Likelihood sigma:
    sigma_total² = sigma_timing² + sigma_dm_turb²
    where sigma_dm_turb = K_DM × 0.15 × dm_catalogue / f_MHz²
    (15% ISM grid turbulence, fixed at grid precompute time).
    Omitting this floor causes weight collapse even at the true position
    because ISM turbulence (~9.5–16 μs) dominates timing noise (~100 ns).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, NamedTuple, Optional

import numpy as np

import config
from config import (
    C_LIGHT,
    COV_NUGGET,
    ESS_REINJECT_THRESHOLD,
    ESS_RESAMPLE_THRESHOLD,
    K_DM,
    KPC_TO_M,
    LIU_WEST_H,
    LIU_WEST_MIN_KERNEL_KPC,
    REINJECT_FRACTION,
    SUN_POS_KPC,
    TEMPER_TARGET_ESS,
)

from core.galaxy import Galaxy   # top-level import — no circular risk (galaxy.py has no estimator import)

if TYPE_CHECKING:
    from core.interstellar_medium import InterstellarMedium
    from core.pulsar import Pulsar

logger = logging.getLogger(__name__)


# ── Numba JIT log-likelihood kernel ──────────────────────────────────────────
# Only pure float64 arrays are passed — no Python objects enter nopython mode.

try:
    import numba

    @numba.jit(nopython=True, cache=True)
    def _log_likelihoods_jit(
        particle_positions: np.ndarray,   # (N, 3) float64 — kpc
        los_dirs: np.ndarray,             # (P, 3) float64 — unit vectors
        dm_table: np.ndarray,             # (N, P) float64 — pc cm⁻³
        observed_timings: np.ndarray,     # (P,) float64 — seconds
        sigma_totals: np.ndarray,         # (P,) float64 — seconds
        kpc_to_m: float,
        c_light: float,
        k_dm: float,
        frequency_mhz: float,
    ) -> np.ndarray:                      # (N,) float64
        """Compute log-likelihood for each particle (Numba nopython kernel).

        For each particle i and pulsar j:
          predicted_j = Roemer(r_i, los_j) + Dispersive(dm_table[i,j], f)
          residual_j  = observed_timings[j] − predicted_j
          log_l_i    += −0.5 × (residual_j / sigma_totals[j])²

        Only position-dependent terms are computed here.  Non-position-dependent
        terms (Doppler, Shapiro, gravitational redshift) are absorbed into the
        sigma_totals which must be wide enough to cover those contributions.
        """
        n_particles = particle_positions.shape[0]
        n_pulsars = los_dirs.shape[0]
        log_l = np.zeros(n_particles, dtype=np.float64)

        for i in range(n_particles):
            ll = 0.0
            px = particle_positions[i, 0]
            py = particle_positions[i, 1]
            pz = particle_positions[i, 2]
            for j in range(n_pulsars):
                # Roemer delay: −(r⃗_particle · n̂_j) × (kpc→m) / c
                # TEMPO2 sign: displaced toward pulsar → pulse arrives earlier → negative
                roemer = (
                    -(px * los_dirs[j, 0] + py * los_dirs[j, 1] + pz * los_dirs[j, 2])
                    * kpc_to_m
                    / c_light
                )
                # Dispersive delay from ISM grid DM at particle's LOS midpoint
                dispersive = k_dm * dm_table[i, j] / (frequency_mhz * frequency_mhz)
                predicted = roemer + dispersive
                residual = observed_timings[j] - predicted
                sigma = sigma_totals[j]
                ll += -0.5 * (residual / sigma) * (residual / sigma)
            log_l[i] = ll

        return log_l

    _NUMBA_AVAILABLE = True
    logger.info("Numba JIT compiled particle filter kernel is available.")

except ImportError:
    _NUMBA_AVAILABLE = False
    logger.warning(
        "numba not available; particle filter will use pure NumPy fallback (slower)."
    )

    def _log_likelihoods_jit(  # type: ignore[misc]
        particle_positions: np.ndarray,
        los_dirs: np.ndarray,
        dm_table: np.ndarray,
        observed_timings: np.ndarray,
        sigma_totals: np.ndarray,
        kpc_to_m: float,
        c_light: float,
        k_dm: float,
        frequency_mhz: float,
    ) -> np.ndarray:
        """Pure NumPy fallback when Numba is unavailable."""
        # Roemer: (N, 3) @ (3, P) = (N, P)
        roemer = -(particle_positions @ los_dirs.T) * kpc_to_m / c_light
        dispersive = k_dm * dm_table / (frequency_mhz ** 2)
        predicted = roemer + dispersive
        residuals = observed_timings[np.newaxis, :] - predicted   # (N, P)
        return -0.5 * np.sum(
            (residuals / sigma_totals[np.newaxis, :]) ** 2, axis=1
        )


# ── Named tuple for per-iteration history ────────────────────────────────────

class FilterState(NamedTuple):
    """Snapshot of filter state at one iteration."""
    step: int
    position_estimate_kpc: np.ndarray   # (3,)
    velocity_estimate_kms: np.ndarray   # (3,)
    position_cov: np.ndarray            # (3, 3)
    velocity_cov: np.ndarray            # (3, 3)
    ess: float                          # effective sample size (absolute)
    ess_fraction: float                 # ess / n_particles


# ── ParticleFilter class ──────────────────────────────────────────────────────

class ParticleFilter:
    """Liu-West regularised particle filter for galactic cold-start XNAV.

    Usage
    -----
    1. Construct: pf = ParticleFilter(n_particles=5000, seed=42)
    2. Initialise from Stage 1 coarse map OR from a bounding region:
           pf.initialise_from_stage1(coarse_likelihood_map)
       or: pf.initialise_from_region(center_kpc, radius_kpc)
    3. Per observation epoch: pf.update(pulsars, observed_timings_dict, ism, ...)
    4. Read estimate: pf.get_estimate()
    5. Recovery: pf.reset()
    """

    def __init__(
        self,
        n_particles: int = 5000,
        tier_config: Optional[dict] = None,
        seed: int = 42,
    ) -> None:
        """Allocate particle arrays.  Does NOT initialise positions — call
        initialise_from_stage1() or initialise_from_region() before update().

        Parameters
        ----------
        n_particles: number of particles
        tier_config: optional ACCURACY_TIERS entry dict (currently reserved)
        seed: base RNG seed for reproducible runs
        """
        self.n_particles: int = n_particles
        self._tier_config: Optional[dict] = tier_config
        self._init_seed: int = seed
        self._rng: np.random.Generator = np.random.default_rng(seed)

        # 6D state: [x, y, z (kpc), vx, vy, vz (km/s)]
        self.particles: np.ndarray = np.empty((n_particles, 6), dtype=np.float64)
        self.weights: np.ndarray = np.full(n_particles, 1.0 / n_particles,
                                           dtype=np.float64)

        self._initialised: bool = False
        self._step: int = 0
        self._consecutive_low_ess: int = 0
        self._diverged: bool = False
        self._history: list[FilterState] = []
        # Tempering exponent applied at the most recent update (0 = no update
        # yet, 1 = observations fully assimilated).  See TEMPER_TARGET_ESS.
        self.last_beta: float = 0.0

        logger.info(
            "ParticleFilter allocated: %d particles, seed=%d", n_particles, seed
        )

    # ── Initialisation ────────────────────────────────────────────────────────

    def initialise_from_region(
        self,
        center_kpc: np.ndarray,
        radius_kpc: float,
        velocity_scale_kms: float = 200.0,
    ) -> None:
        """Initialise particles uniformly within a spherical region.

        Use this when Stage 1 has not been run or for testing.
        Positions are drawn uniformly from the ball of radius_kpc around
        center_kpc.  Velocities are drawn from N(0, velocity_scale_kms²).

        Parameters
        ----------
        center_kpc: (3,) galactocentric centre of the initialisation region
        radius_kpc: radius of the uniform ball
        velocity_scale_kms: std dev of each velocity component (km/s)
        """
        center = np.asarray(center_kpc, dtype=np.float64)

        # Uniform distribution over a ball: use rejection sampling with a cube
        positions = np.empty((self.n_particles, 3), dtype=np.float64)
        filled = 0
        while filled < self.n_particles:
            need = self.n_particles - filled
            samples = self._rng.uniform(-radius_kpc, radius_kpc, (need * 2, 3))
            in_ball = np.linalg.norm(samples, axis=1) <= radius_kpc
            accepted = samples[in_ball]
            take = min(len(accepted), need)
            positions[filled:filled + take] = accepted[:take]
            filled += take

        velocities = self._rng.normal(0.0, velocity_scale_kms, (self.n_particles, 3))

        self.particles[:, :3] = positions + center
        self.particles[:, 3:] = velocities
        self.weights[:] = 1.0 / self.n_particles

        self._initialised = True
        logger.info(
            "ParticleFilter initialised from region: centre=%s, radius=%.2f kpc",
            center, radius_kpc,
        )

    def initialise_from_stage1(
        self,
        coarse_likelihood_map: dict,
        uniform_fraction: float = 0.25,
    ) -> None:
        """Sample particle positions from the Stage 1 coarse likelihood map.

        A fraction of particles (uniform_fraction) is sampled uniformly over
        the whole galactic disk instead of from the map.  The DM-only Stage 1
        likelihood is multi-modal and can be confidently wrong — placing every
        particle in a single wrong basin leaves the filter nothing to recover
        with.  A thin uniform floor guarantees a few particles near the truth;
        the timing likelihood then amplifies them (a lone good particle takes
        a few percent of the tempered posterior mass on the first update and
        compounds geometrically after resampling).

        Parameters
        ----------
        coarse_likelihood_map: dict with keys:
            probability_map: 3D float64 array (non-negative, need not be normalised)
            x_arr, y_arr, z_arr: coordinate axes (kpc, galactocentric)
            velocity_scale_kms: optional float (default 200.0)
        uniform_fraction: fraction of particles seeded uniformly over the disk
        """
        prob = np.asarray(coarse_likelihood_map["probability_map"], dtype=np.float64)
        x_arr = np.asarray(coarse_likelihood_map["x_arr"])
        y_arr = np.asarray(coarse_likelihood_map["y_arr"])
        z_arr = np.asarray(coarse_likelihood_map["z_arr"])
        v_scale = float(coarse_likelihood_map.get("velocity_scale_kms", 200.0))

        # Flatten and normalise probability map
        flat_prob = prob.ravel()
        flat_prob = np.maximum(flat_prob, 0.0)
        total = flat_prob.sum()
        if total <= 0:
            raise ValueError("Stage 1 probability map sums to zero — cannot initialise.")
        flat_prob /= total

        # Split: map-sampled particles vs uniform disk floor
        n_uniform = int(round(np.clip(uniform_fraction, 0.0, 1.0)
                              * self.n_particles))
        n_map = self.n_particles - n_uniform

        # Sample voxel indices proportional to probability
        n_voxels = flat_prob.size
        indices = self._rng.choice(n_voxels, size=n_map, p=flat_prob)

        nx, ny, nz = prob.shape
        ix = indices // (ny * nz)
        iy = (indices % (ny * nz)) // nz
        iz = indices % nz

        # Jitter within voxel to avoid discrete artefacts
        dx = (x_arr[1] - x_arr[0]) if len(x_arr) > 1 else 0.1
        dy = (y_arr[1] - y_arr[0]) if len(y_arr) > 1 else 0.1
        dz = (z_arr[1] - z_arr[0]) if len(z_arr) > 1 else 0.1

        self.particles[:n_map, 0] = x_arr[ix] + self._rng.uniform(-0.5 * dx, 0.5 * dx, n_map)
        self.particles[:n_map, 1] = y_arr[iy] + self._rng.uniform(-0.5 * dy, 0.5 * dy, n_map)
        self.particles[:n_map, 2] = z_arr[iz] + self._rng.uniform(-0.5 * dz, 0.5 * dz, n_map)

        if n_uniform > 0:
            self.particles[n_map:, :3] = Galaxy.sample_uniform(n_uniform, self._rng)

        self.particles[:, 3:] = self._rng.normal(0.0, v_scale, (self.n_particles, 3))
        self.weights[:] = 1.0 / self.n_particles

        self._initialised = True
        logger.info("ParticleFilter initialised from Stage 1 likelihood map.")

    # ── Core update step ──────────────────────────────────────────────────────

    def update(
        self,
        pulsars: list["Pulsar"],
        observed_timings_dict: dict[str, dict[str, float]],
        ism: Optional["InterstellarMedium"] = None,
        frequency_mhz: float = 1400.0,
        integration_time_s: float = 1000.0,
    ) -> FilterState:
        """Update particle weights given new pulsar timing observations.

        Parameters
        ----------
        pulsars: list of Pulsar objects (must include all keys in observed_timings_dict)
        observed_timings_dict: {pulsar.name: timing_dict} from TimingModel.compute_all_pulsars()
            The "total" field is used as the observed timing.
        ism: InterstellarMedium instance with precomputed grid (None → use catalogue DMs)
        frequency_mhz: observing frequency for dispersion correction
        integration_time_s: integration time per observation (affects timing noise sigma)

        Returns
        -------
        FilterState snapshot for this step.

        Raises
        ------
        RuntimeError if filter has diverged (3 consecutive low-ESS iterations).
        """
        if not self._initialised:
            raise RuntimeError(
                "ParticleFilter has not been initialised. "
                "Call initialise_from_region() or initialise_from_stage1() first."
            )
        if self._diverged:
            raise RuntimeError(
                "Filter has diverged. Call reset() to recover."
            )

        # ── Step 1: Enforce consistent pulsar ordering ─────────────────────────
        # Sort by pulsar name to guarantee dict→array alignment is deterministic
        # regardless of insertion order.
        sorted_keys = sorted(observed_timings_dict.keys())
        pulsar_map = {p.name: p for p in pulsars}
        active_pulsars = [pulsar_map[k] for k in sorted_keys
                          if k in pulsar_map]
        if not active_pulsars:
            raise ValueError(
                "No pulsars in observed_timings_dict match the provided pulsar list."
            )

        n_pulsars = len(active_pulsars)
        particle_positions = self.particles[:, :3].copy()   # (N, 3)

        # ── Step 2: Extract observed timings (total) ───────────────────────────
        observed_timings = np.array(
            [observed_timings_dict[p.name]["total"] for p in active_pulsars],
            dtype=np.float64,
        )

        # ── Step 3: Build LOS unit vectors ────────────────────────────────────
        pulsar_positions = np.array(
            [p.position_kpc for p in active_pulsars], dtype=np.float64
        )   # (P, 3)

        # LOS from spacecraft (approximate with particle mean) to pulsar
        # For the likelihood, we use fixed LOS dirs (the pulsar direction from
        # the origin) since the angular correction across kpc-scale particle
        # clouds is negligible compared to the timing noise floor.
        # APPROXIMATION: LOS direction computed from the galactic origin (not
        # each particle position). Error < 1 arcsec for pulsar distances > 1 kpc.
        los_dirs = np.empty((n_pulsars, 3), dtype=np.float64)
        for j, p in enumerate(active_pulsars):
            norm = np.linalg.norm(p.position_kpc)
            los_dirs[j] = p.position_kpc / norm if norm > 1e-6 else np.array([1., 0., 0.])

        # ── Step 4: Build DM table (vectorised) ───────────────────────────────
        dm_table = self._build_dm_table(
            particle_positions, pulsar_positions, active_pulsars, ism
        )   # (N, P)

        # ── Step 5: Compute per-pulsar sigma_total ────────────────────────────
        # sigma_total² = sigma_timing² + sigma_dm_turb²
        # sigma_dm_turb = K_DM × 0.15 × DM_catalogue / f²
        # The 15% factor matches the log-normal turbulence (sigma=0.15) baked
        # into the ISM grid at precompute time.
        #
        # APPROXIMATION: sigma_dm_turb uses the catalogue DM (Sun→pulsar) as a
        # representative path DM.  The actual DM from a given particle varies,
        # but the sigma should represent expected uncertainty, not per-particle
        # variation (which is the signal we want to detect).
        sigma_totals = np.empty(n_pulsars, dtype=np.float64)
        integration_t = max(integration_time_s, 1.0)
        for j, p in enumerate(active_pulsars):
            sigma_timing = p.timing_noise_ns * 1e-9 / np.sqrt(integration_t)
            sigma_dm_turb = K_DM * 0.15 * max(p.dm, 1.0) / (frequency_mhz ** 2)
            sigma_totals[j] = np.sqrt(sigma_timing ** 2 + sigma_dm_turb ** 2)

        # ── Step 6: Compute log-likelihoods (Numba JIT or NumPy) ──────────────
        log_w = _log_likelihoods_jit(
            particle_positions,
            los_dirs,
            dm_table,
            observed_timings,
            sigma_totals,
            float(KPC_TO_M),
            float(C_LIGHT),
            float(K_DM),
            float(frequency_mhz),
        )

        # ── Step 7: Adaptive tempering → normalise weights ─────────────────────
        # Subtract max before exp() to prevent underflow.  This preserves the
        # relative probabilities exactly since the normalisation constant cancels.
        #
        # The raw likelihood is astronomically peaked relative to a kpc-scale
        # cloud (Roemer residual differences ~10¹¹ s vs sigma ~ms), so applying
        # it at full strength is an argmax: all weight lands on the single
        # nearest particle, resampling duplicates it, and the Liu-West kernel
        # (proportional to the now-zero covariance) can never restore diversity
        # — the filter freezes after one update.  Tempering raises the
        # likelihood to a power beta ∈ (0, 1] chosen so the post-update ESS
        # stays near TEMPER_TARGET_ESS; each iteration then contracts the cloud
        # by a controlled amount and beta climbs to 1 as the cloud tightens.
        log_w -= log_w.max()
        beta = self._adaptive_beta(log_w)
        self.last_beta = beta
        raw_weights = np.exp(beta * log_w)
        weight_sum = raw_weights.sum()
        if weight_sum <= 0 or not np.isfinite(weight_sum):
            logger.warning(
                "Step %d: weight sum is %g — filter near collapse; forcing uniform.",
                self._step, weight_sum,
            )
            raw_weights = np.ones(self.n_particles, dtype=np.float64)
            weight_sum = float(self.n_particles)

        self.weights = raw_weights / weight_sum   # normalise

        # ── Step 8: ESS check (computed after normalisation) ──────────────────
        ess = float(1.0 / max(np.sum(self.weights ** 2), 1e-12))
        ess_frac = ess / self.n_particles

        # Resample whenever the update was tempered (beta < 1 means the cloud
        # is still contracting and needs fresh diversity for the next step) or
        # the ESS dropped below threshold.
        if beta < 1.0 or ess_frac < ESS_RESAMPLE_THRESHOLD:
            logger.info(
                "Step %d: ESS=%.0f (%.1f%%), beta=%.3g — resampling.",
                self._step, ess, 100 * ess_frac, beta,
            )
            self._liu_west_resample()
            ess = float(1.0 / max(np.sum(self.weights ** 2), 1e-12))
            ess_frac = ess / self.n_particles

        if ess_frac < ESS_REINJECT_THRESHOLD:
            n_reinject = int(self.n_particles * REINJECT_FRACTION)
            logger.warning(
                "Step %d: ESS=%.0f (%.1f%%) critically low — reinjecting %d particles.",
                self._step, ess, 100 * ess_frac, n_reinject,
            )
            self._reinject_diversity(n_reinject)
            ess = float(1.0 / max(np.sum(self.weights ** 2), 1e-12))
            ess_frac = ess / self.n_particles
            self._consecutive_low_ess += 1
        else:
            self._consecutive_low_ess = 0

        # ── Step 9: Divergence guard ──────────────────────────────────────────
        if self._consecutive_low_ess >= 3:
            self._diverged = True
            logger.error(
                "Step %d: Filter diverged — ESS critically low for 3 consecutive "
                "iterations. Call reset() to recover.",
                self._step,
            )
            raise RuntimeError(
                "Filter diverged — try increasing particle count or reducing noise levels."
            )

        # ── Step 10: Snapshot and advance step ────────────────────────────────
        state = self._snapshot(ess, ess_frac)
        self._history.append(state)
        self._step += 1

        logger.debug(
            "Step %d complete. ESS=%.0f (%.1f%%), est=%.3f kpc",
            self._step - 1, ess, 100 * ess_frac,
            np.linalg.norm(state.position_estimate_kpc),
        )

        return state

    # ── Adaptive tempering ─────────────────────────────────────────────────────

    def _adaptive_beta(self, log_w: np.ndarray) -> float:
        """Choose tempering exponent beta ∈ (0, 1] by bisection on the ESS.

        Finds the largest beta ≤ 1 such that the effective sample size of
        weights ∝ exp(beta · log_w) is at least TEMPER_TARGET_ESS × N.
        ESS(beta) is monotonically decreasing in beta, so bisection converges.

        log_w must already be max-subtracted (max(log_w) == 0).
        """
        def ess_frac_at(beta: float) -> float:
            w = np.exp(beta * log_w)
            s = w.sum()
            if s <= 0 or not np.isfinite(s):
                return 0.0
            w /= s
            return float(1.0 / np.sum(w ** 2)) / self.n_particles

        if ess_frac_at(1.0) >= TEMPER_TARGET_ESS:
            return 1.0   # likelihood already gentle — assimilate fully

        # Log-weight spreads can reach ~10²⁶ (Roemer residuals across a kpc
        # cloud vs ms-scale sigma), so beta must be searched in log space —
        # linear bisection over [0, 1] cannot resolve beta ~ 10⁻²⁶.
        lo_exp, hi_exp = -40.0, 0.0   # beta = 10^exp
        if ess_frac_at(10.0 ** lo_exp) < TEMPER_TARGET_ESS:
            # Pathologically peaked even at beta=1e-40; apply the minimum.
            return 10.0 ** lo_exp
        for _ in range(80):   # resolves exponent to ~5e-13 decades
            mid_exp = 0.5 * (lo_exp + hi_exp)
            if ess_frac_at(10.0 ** mid_exp) >= TEMPER_TARGET_ESS:
                lo_exp = mid_exp
            else:
                hi_exp = mid_exp
        # Largest tested beta still meeting the ESS target.
        return 10.0 ** lo_exp

    # ── DM table construction (vectorised) ────────────────────────────────────

    def _build_dm_table(
        self,
        particle_positions: np.ndarray,   # (N, 3) kpc
        pulsar_positions: np.ndarray,     # (P, 3) kpc
        active_pulsars: list["Pulsar"],
        ism: Optional["InterstellarMedium"],
    ) -> np.ndarray:                      # (N, P) pc cm⁻³
        """Build DM table for all (particle, pulsar) pairs via vectorised interpolation.

        Uses ISM grid if available and loaded; falls back to catalogue DM values
        (constant across particles) when the grid is unavailable.

        The heliocentric denominator fix is applied here: the ISM grid stores
        cumulative DM from the Sun, so the mean-density normalisation uses
        the heliocentric distance to the midpoint, not the galactocentric distance.
        """
        n_particles = particle_positions.shape[0]
        n_pulsars = pulsar_positions.shape[0]

        if ism is not None and ism.grid_loaded():
            # ── Vectorised grid lookup (single interpolator call) ──────────────
            # Midpoints: (N, P, 3)
            midpoints = (
                particle_positions[:, np.newaxis, :]      # (N, 1, 3)
                + pulsar_positions[np.newaxis, :, :]       # (1, P, 3)
            ) / 2.0

            flat_midpoints = midpoints.reshape(-1, 3)      # (N*P, 3)

            # Single vectorised call via the public batch interface
            dm_at_midpoints = ism.batch_lookup(flat_midpoints)    # (N*P,)

            # Heliocentric distance to each midpoint for density normalisation.
            # APPROXIMATION: Uses SUN_POS_KPC = [-8.178, 0, 0] kpc.
            # Real Sun has a small offset above the midplane (~17 pc) which is
            # negligible compared to the grid resolution.
            helio_vecs = flat_midpoints - SUN_POS_KPC     # (N*P, 3)
            helio_dists = np.linalg.norm(helio_vecs, axis=1)    # (N*P,)
            helio_dists = np.maximum(helio_dists, 0.1)           # guard: 100 pc floor

            # Mean electron density along Sun→midpoint line of sight
            mean_density = dm_at_midpoints / helio_dists          # (N*P,) pc cm⁻³/kpc

            # Path lengths from each particle to each pulsar
            # diff: (N, P, 3) → path_lengths: (N, P)
            diff = (pulsar_positions[np.newaxis, :, :]
                    - particle_positions[:, np.newaxis, :])
            path_lengths = np.linalg.norm(diff, axis=2).ravel()   # (N*P,)

            dm_flat = np.maximum(mean_density * path_lengths, 0.5)
            dm_table = dm_flat.reshape(n_particles, n_pulsars)

        else:
            # ── Fallback: use catalogue DM values (same for all particles) ──────
            # APPROXIMATION: Catalogue DM (Sun→pulsar) is used when the ISM grid
            # is not loaded.  This value is position-invariant so it contributes
            # zero navigation signal via the dispersive term; only the Roemer term
            # then distinguishes particle positions.  This is acceptable for
            # testing and offline use without the precomputed grid.
            dm_catalogue = np.array(
                [p.dm for p in active_pulsars], dtype=np.float64
            )   # (P,)
            dm_table = np.tile(dm_catalogue, (n_particles, 1))   # (N, P)

        return dm_table

    # ── Liu-West resampling ────────────────────────────────────────────────────

    def _liu_west_resample(self) -> None:
        """Liu-West kernel-smoothed resampling.

        Resamples particles proportional to weights, then perturbs each
        resampled particle with a Gaussian kernel to maintain diversity.

        Liu & West (2001) §3:
            a = sqrt(1 − h²)          ← kernel shrinkage toward mean
            h² × var(particles)       ← kernel variance (h = LIU_WEST_H)

        APPROXIMATION: Fixed bandwidth h = LIU_WEST_H (config.py) for all
        iterations.  Adaptive bandwidth would be more accurate but is out of
        scope.  Fixed h=0.1 is standard in the particle filter literature and
        gives adequate performance for position uncertainties of order kpc.

        NOTE: The build brief formula a = (3h−1)/(2h) is mathematically wrong —
        at h=0.1 it gives a = −3.5, which produces negative scaling and particle
        divergence.  The correct formula is a = sqrt(1 − h²) per Liu & West (2001).
        """
        h = LIU_WEST_H
        # Correct Liu-West shrinkage parameter (Liu & West 2001, eq. 3)
        a = np.sqrt(1.0 - h ** 2)   # ≈ 0.995 at h=0.1

        # Weighted mean of particle state
        mean = np.average(self.particles, weights=self.weights, axis=0)  # (6,)

        # Kernel variance: h² × weighted variance of each state dimension
        diff = self.particles - mean   # (N, 6)
        weighted_var = np.average(diff ** 2, weights=self.weights, axis=0)   # (6,)
        kernel_std = h * np.sqrt(np.maximum(weighted_var, 1e-12))            # (6,)
        # Roughening floor on position jitter: if the weighted covariance
        # underflows (near-duplicate particles), the kernel would add nothing
        # and the cloud freezes permanently.  1 pc keeps duplicates apart
        # without affecting kpc-scale convergence.
        kernel_std[:3] = np.maximum(kernel_std[:3], LIU_WEST_MIN_KERNEL_KPC)

        # Systematic resampling (lower variance than multinomial)
        indices = _systematic_resample(self.weights, self._rng)

        # Shrink toward mean, then perturb
        resampled = self.particles[indices]   # (N, 6)
        shrunk = a * resampled + (1.0 - a) * mean
        perturbation = self._rng.normal(0.0, 1.0, resampled.shape) * kernel_std
        new_particles = shrunk + perturbation

        # Clip particles that escaped the galactic disk after perturbation.
        # The RegularGridInterpolator returns fill_value=1.0 outside the grid,
        # so escaped particles would get wrong DM predictions.  Clipping here
        # is cheaper than letting them propagate through the filter.
        for i in range(self.n_particles):
            if not Galaxy.in_galaxy(new_particles[i, :3]):
                new_particles[i, :3] = Galaxy.clip_to_galaxy(new_particles[i, :3])

        self.particles = new_particles
        self.weights[:] = 1.0 / self.n_particles

    def _reinject_diversity(self, n_reinject: int) -> None:
        """Replace the lowest-weight particles with new samples near the current best estimate.

        Called when ESS drops below ESS_REINJECT_THRESHOLD.  The reinjected
        particles are sampled from a Gaussian centred on the current weighted
        mean with std dev = 2 × current position std dev.
        """
        # Identify lowest-weight particles to replace
        sorted_idx = np.argsort(self.weights)
        replace_idx = sorted_idx[:n_reinject]

        # Current weighted mean and std dev
        mean_pos = np.average(self.particles[:, :3], weights=self.weights, axis=0)
        diff = self.particles[:, :3] - mean_pos
        var_pos = np.average(diff ** 2, weights=self.weights, axis=0)
        std_pos = 2.0 * np.sqrt(np.maximum(var_pos, 0.01))   # floor: 100 pc

        new_pos = self._rng.normal(mean_pos, std_pos, (n_reinject, 3))
        new_vel = self._rng.normal(0.0, 200.0, (n_reinject, 3))

        self.particles[replace_idx, :3] = new_pos
        self.particles[replace_idx, 3:] = new_vel
        self.weights[replace_idx] = 1.0 / self.n_particles
        w_sum = self.weights.sum()
        if w_sum > 0:
            self.weights /= w_sum   # renormalise

    # ── Estimate extraction ────────────────────────────────────────────────────

    def get_estimate(self) -> dict:
        """Return weighted mean position/velocity and separate 3×3 covariances.

        Returns
        -------
        dict with keys:
            position_kpc: (3,) weighted mean galactocentric position
            velocity_kms: (3,) weighted mean velocity
            position_cov: (3, 3) weighted covariance of position particles
            velocity_cov: (3, 3) weighted covariance of velocity particles
            position_std_kpc: (3,) marginal std devs (sqrt of diagonal)
            velocity_std_kms: (3,) marginal std devs
        """
        w = self.weights
        pos = self.particles[:, :3]
        vel = self.particles[:, 3:]

        mean_pos = np.average(pos, weights=w, axis=0)
        mean_vel = np.average(vel, weights=w, axis=0)

        # Weighted covariance — separate 3×3 blocks avoids 6×6 singularity risk
        dp = pos - mean_pos
        dv = vel - mean_vel

        pos_cov = np.einsum("n,ni,nj->ij", w, dp, dp)
        vel_cov = np.einsum("n,ni,nj->ij", w, dv, dv)

        # Diagonal nugget prevents near-singular matrices
        pos_cov += COV_NUGGET * np.eye(3)
        vel_cov += COV_NUGGET * np.eye(3)

        return {
            "position_kpc": mean_pos,
            "velocity_kms": mean_vel,
            "position_cov": pos_cov,
            "velocity_cov": vel_cov,
            "position_std_kpc": np.sqrt(np.diag(pos_cov)),
            "velocity_std_kms": np.sqrt(np.diag(vel_cov)),
        }

    def get_ess(self) -> float:
        """Return current ESS as a fraction of n_particles (0.0–1.0).

        Per build brief spec: "Returns current effective sample size as
        fraction of n_particles."  A value near 1.0 means healthy diversity;
        near 0.0 means near-collapse.

        ESS = 1 / sum(w²).  At uniform weights: ESS = N → fraction = 1.0.
        At all weight on one particle: ESS = 1 → fraction = 1/N.
        """
        ess_absolute = float(1.0 / max(np.sum(self.weights ** 2), 1e-12))
        return ess_absolute / self.n_particles

    def get_ess_absolute(self) -> float:
        """Return current ESS as an absolute count (0–n_particles)."""
        return float(1.0 / max(np.sum(self.weights ** 2), 1e-12))

    # ── History and status ────────────────────────────────────────────────────

    @property
    def history(self) -> list[FilterState]:
        """List of FilterState snapshots, one per completed update() call."""
        return list(self._history)

    @property
    def step(self) -> int:
        """Number of completed update() steps."""
        return self._step

    @property
    def diverged(self) -> bool:
        """True if the filter has diverged and reset() is needed."""
        return self._diverged

    @property
    def initialised(self) -> bool:
        """True if initialise_from_region() or initialise_from_stage1() has been called."""
        return self._initialised

    def _snapshot(self, ess: float, ess_frac: float) -> FilterState:
        # ess: absolute ESS count; ess_frac: fraction — both pre-computed in update()
        """Create a FilterState snapshot from current filter state."""
        est = self.get_estimate()
        return FilterState(
            step=self._step,
            position_estimate_kpc=est["position_kpc"].copy(),
            velocity_estimate_kms=est["velocity_kms"].copy(),
            position_cov=est["position_cov"].copy(),
            velocity_cov=est["velocity_cov"].copy(),
            ess=ess,
            ess_fraction=ess_frac,
        )

    # ── Reset ──────────────────────────────────────────────────────────────────

    def reset(self, seed: Optional[int] = None) -> None:
        """Reinitialise all mutable filter state.

        Preserves n_particles and tier_config.  Resets particles, weights,
        step counter, history, and divergence flag.

        Parameters
        ----------
        seed: if provided, use this seed; otherwise reuse the original init seed.
        """
        _seed = seed if seed is not None else self._init_seed
        self._rng = np.random.default_rng(_seed)
        self.particles = np.empty((self.n_particles, 6), dtype=np.float64)
        self.weights = np.full(self.n_particles, 1.0 / self.n_particles, dtype=np.float64)
        self._initialised = False
        self._step = 0
        self._consecutive_low_ess = 0
        self._diverged = False
        self._history = []
        self.last_beta = 0.0
        logger.info("ParticleFilter reset (seed=%d).", _seed)


# ── Systematic resampling (low-variance) ─────────────────────────────────────

def _systematic_resample(
    weights: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Systematic (low-variance) resampling.

    Returns an index array of length N drawn proportional to weights.
    Systematic resampling has lower variance than multinomial resampling
    (O(1) spread vs O(N) for multinomial) and runs in O(N) time.
    """
    N = len(weights)
    cumsum = np.cumsum(weights)
    cumsum[-1] = 1.0   # guard against floating-point drift
    u = (rng.uniform(0.0, 1.0 / N) + np.arange(N, dtype=np.float64) / N)
    indices = np.searchsorted(cumsum, u)
    return np.clip(indices, 0, N - 1)
