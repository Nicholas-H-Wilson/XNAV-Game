# core/timing.py — Pulse arrival time simulation
# XNAV Cold Start Simulator

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional

import numpy as np

from config import C_LIGHT, G_NEWTON, M_SUN, KPC_TO_M

if TYPE_CHECKING:
    from core.pulsar import Pulsar
    from core.spacecraft import Spacecraft


class TimingModel:
    """Computes synthetic pulse arrival times including all relevant effects.

    The full timing model is:
        t_obs = t_geom + t_doppler + t_shapiro + t_grav_redshift + t_dispersion + t_noise

    Each contribution is computed and returned separately so the UI can display
    a breakdown of what each effect contributes.
    """

    @staticmethod
    def compute_arrival_time(
        pulsar: "Pulsar",
        spacecraft: "Spacecraft",
        observation_time_s: float,
        frequency_mhz: float = 1400.0,
        integration_time_s: float = 1000.0,
        include_noise: bool = True,
        noise_seed: int = 0,
        model_dm: Optional[float] = None,
    ) -> dict[str, float]:
        """Compute all timing components for one pulsar observation.

        Returns a dict with keys:
            geometric      — Roemer delay (light travel time correction, s)
            doppler        — Doppler shift from spacecraft velocity (s)
            shapiro        — Shapiro delay through galactic potential (s)
            gravitational  — Gravitational redshift of local clock (s)
            dispersive     — Dispersive delay through ISM (s)
            noise          — Total timing noise (s)
            total          — Sum of all contributions (s)

        All values are in seconds.

        Parameters:
            observation_time_s: barycentric observation epoch (s from reference)
            frequency_mhz: observing frequency for DM correction
            integration_time_s: duration of folded profile (affects noise level)
            include_noise: if False, return noise=0 (for tests / convergence study)
            noise_seed: base seed for reproducible noise draws
            model_dm: if provided, use this DM (pc cm⁻³) for the dispersive delay
                instead of pulsar.dm.  Pass the spacecraft-to-pulsar column density
                from the ISM grid here so the dispersive term varies with particle
                position in Stage 1.  If None, falls back to the catalogue value
                pulsar.dm (Sun-to-pulsar), which is only correct at the SSB.
        """
        # ── Geometric Roemer delay ────────────────────────────────────────────
        # The Roemer delay is the light-travel-time difference between the
        # spacecraft position and the Solar System Barycentre (SSB).
        #
        # COORDINATE NOTE: We use galactocentric Cartesian positions (not
        # SSB-relative).  The Sun's galactocentric position creates a constant
        # term (−R₀·n̂/c) that is IDENTICAL for all particles, so it cancels
        # exactly in the likelihood computation of the particle filter:
        #   Δt_residual = (Δt_obs − Δt_particle) = −(r_sc−r_particle)·n̂/c
        # The solar term cancels.  The Roemer differential between particles
        # is therefore numerically identical to the SSB-relative formulation.
        #
        # APPROXIMATION: First-order Roemer delay; annual parallax and proper
        # motion corrections are < 1 μs at kpc distances — negligible.
        pulsar_pos_kpc = pulsar.position_kpc
        sc_pos_kpc = spacecraft.position_kpc

        # Unit vector from spacecraft toward pulsar
        los_kpc = pulsar_pos_kpc - sc_pos_kpc
        los_dist_kpc = np.linalg.norm(los_kpc)
        if los_dist_kpc < 1e-10:
            los_dir = np.array([1.0, 0.0, 0.0])
        else:
            los_dir = los_kpc / los_dist_kpc

        # Projection of spacecraft position along the pulsar direction.
        # Sign convention (TEMPO2): Δ_R = −r⃗·n̂/c.  A spacecraft displaced
        # toward the pulsar (positive projection) has a shorter path → the
        # pulse arrives EARLIER → the residual is NEGATIVE.
        roemer_delay_s = -np.dot(sc_pos_kpc, los_dir) * KPC_TO_M / C_LIGHT

        # ── Doppler delay ──────────────────────────────────────────────────────
        # The spacecraft's radial velocity toward the pulsar shifts the pulse
        # frequency and effectively adds a linear phase drift.
        # Over integration_time_s, this causes a timing offset:
        # Δt_doppler ≈ (v_r / c) × observation_time_s
        #
        # APPROXIMATION: First-order Doppler (β = v/c ≪ 1).  Relativistic
        # transverse Doppler (second order) is of order (v/c)² × T ≈ 10⁻¹⁰ T
        # for typical spacecraft velocities.  Negligible.
        v_kms = spacecraft.velocity_kms
        v_radial_ms = np.dot(v_kms * 1000.0, los_dir)   # m/s
        # First-order Doppler: accumulated phase offset over observation_time_s
        # is (v_r/c) × T.  At a single epoch (T=0) this is zero; it grows
        # as the spacecraft moves toward/away from the pulsar over successive epochs.
        doppler_delay_s = float(v_radial_ms / C_LIGHT * observation_time_s)

        # ── Shapiro delay ──────────────────────────────────────────────────────
        # Photons travelling through a gravitational potential well are delayed.
        # For the galactic potential, the Shapiro delay along the LOS is:
        # Δt_shapiro = -(2/c³) ∫ Φ dl
        #
        # APPROXIMATION: Integrates the Miyamoto-Nagai + bulge + halo potential
        # at the midpoint of the LOS.  Full integration would require numerical
        # quadrature along every particle's LOS at every iteration — too slow.
        # Instead we sample at 5 equally-spaced points and use the trapezoidal rule.
        # Error: ~5% of the galactic Shapiro delay (~10–100 ns total error).
        # The galactic Shapiro delay is itself ~1–10 μs, so this contributes
        # ~50–500 ps — below current pulsar timing sensitivity.  The dominant
        # Shapiro delay from the Sun is not included since the spacecraft is
        # assumed to be in deep space far from massive bodies on the LOS.
        from core.gravity import Gravity

        n_shapiro = 5
        shapiro_sum = 0.0
        for i in range(n_shapiro):
            frac = (i + 0.5) / n_shapiro
            mid_kpc = sc_pos_kpc + frac * los_kpc
            phi = Gravity.gravitational_potential(
                mid_kpc,
                central_body_mass_kg=0.0,
                central_body_radius_m=0.0,
                include_galactic=True,
            )
            shapiro_sum += phi

        avg_phi = shapiro_sum / n_shapiro
        # Shapiro delay = -2 × Φ_avg × L / c³  (weak field, linearised GR)
        # APPROXIMATION: Weak-field linearised general relativity.
        # Valid when Φ/c² ≪ 1 (always true in Milky Way, Φ/c² ~ 10⁻⁶).
        los_dist_m = los_dist_kpc * KPC_TO_M
        shapiro_delay_s = float(-2.0 * avg_phi * los_dist_m / C_LIGHT**3)

        # ── Gravitational redshift of local clock ──────────────────────────────
        # A clock in a gravitational well runs slow.  Over integration_time_s,
        # this introduces a timing offset:
        # Δt_grav = (Φ / c²) × T_int
        # This is the monopolar signal: identical for all pulsars, independent
        # of direction.
        phi_sc = spacecraft.gravitational_potential(include_galactic=True)
        # APPROXIMATION: We compute the per-spin-cycle gravitational slowing:
        # Δt = (Φ_sc / c²) × P_spin
        # The standard Einstein delay accumulates as (Φ/c²) × T_elapsed, which
        # at T=0 is zero; the per-spin formulation gives a non-zero epoch-invariant
        # contribution representing the fraction-of-period offset per pulse.
        # This scales with pulsar period (not truly monopolar), but the variation
        # across MSPs (1.5–10 ms periods) contributes < 2 ns, well below the
        # timing noise floor for all but the very best millisecond pulsars.
        # For cold-start navigation accuracy (> 10 pc), this approximation is
        # entirely negligible.
        grav_redshift_s = float(Gravity.clock_slowing_factor(phi_sc) * pulsar.period)

        # ── Dispersive delay ───────────────────────────────────────────────────
        # Free electrons in the ISM delay lower-frequency photons.
        # Use model_dm if provided (spacecraft-to-pulsar column density from
        # the ISM grid — varies with particle position in Stage 1).
        # Fall back to catalogue pulsar.dm (Sun-to-pulsar) if not provided.
        # IMPORTANT: pulsar.dm is invariant with spacecraft position, so it
        # contributes zero navigation signal; only model_dm gives useful
        # positional information for the particle filter.
        from core.dispersion import Dispersion

        dm_to_use = model_dm if model_dm is not None else pulsar.dm
        dispersive_delay_s = Dispersion.compute_dispersive_delay(dm_to_use, frequency_mhz)

        # ── Timing noise ───────────────────────────────────────────────────────
        from core.noise import NoiseModel

        if include_noise:
            noise_s = NoiseModel.timing_noise(
                pulsar,
                integration_time_s=integration_time_s,
                seed=noise_seed,
            )
        else:
            noise_s = 0.0

        # ── Assemble result ────────────────────────────────────────────────────
        # Use math.fsum() for exact floating-point summation.  The Roemer delay
        # at galactic distances (~10¹¹ s) is ~10¹² × larger than the
        # gravitational redshift term (~10⁻⁹ s); naive addition would lose the
        # smaller terms entirely due to float64 ULP cancellation.
        total_s = math.fsum([roemer_delay_s, doppler_delay_s, shapiro_delay_s,
                              grav_redshift_s, dispersive_delay_s, noise_s])

        return {
            "geometric":     roemer_delay_s,
            "doppler":       doppler_delay_s,
            "shapiro":       shapiro_delay_s,
            "gravitational": grav_redshift_s,
            "dispersive":    dispersive_delay_s,
            "noise":         noise_s,
            "total":         total_s,
        }

    @staticmethod
    def compute_all_pulsars(
        pulsars: list,
        spacecraft: "Spacecraft",
        observation_time_s: float = 0.0,
        frequency_mhz: float = 1400.0,
        integration_time_s: float = 1000.0,
        include_noise: bool = True,
        base_seed: int = 0,
        model_dms: Optional[dict] = None,
    ) -> dict[str, dict[str, float]]:
        """Compute arrival times for a list of pulsars.

        Returns {pulsar.name: arrival_dict} for all pulsars.
        Seeds are offset by pulsar index so each pulsar gets independent noise.

        model_dms: optional dict mapping pulsar.name → DM (pc cm⁻³) from the ISM
            grid at the current spacecraft position.  Passed through to
            compute_arrival_time so Stage 1 dispersive delays are position-dependent.
        """
        results = {}
        for i, pulsar in enumerate(pulsars):
            dm = model_dms.get(pulsar.name) if model_dms is not None else None
            results[pulsar.name] = TimingModel.compute_arrival_time(
                pulsar,
                spacecraft,
                observation_time_s=observation_time_s,
                frequency_mhz=frequency_mhz,
                integration_time_s=integration_time_s,
                include_noise=include_noise,
                noise_seed=base_seed + i,
                model_dm=dm,
            )
        return results
