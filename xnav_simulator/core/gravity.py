# core/gravity.py — Gravitational potential and well depth estimation
# XNAV Cold Start Simulator

from __future__ import annotations

import numpy as np

from config import C_LIGHT, G_NEWTON, M_SUN, KPC_TO_M


class Gravity:
    """Gravitational potential models and timing-residual extractors.

    The key observable for gravity well estimation is the clock slowing factor
    Δf/f = Φ/c² — the same potential that causes gravitational redshift in
    atomic clocks.  This appears as a monopolar (direction-independent) offset
    in all pulsar timing residuals simultaneously.
    """

    # ── Potential models ──────────────────────────────────────────────────────

    @staticmethod
    def gravitational_potential(
        position_kpc: np.ndarray,
        central_body_mass_kg: float = 0.0,
        central_body_radius_m: float = 0.0,
        include_galactic: bool = True,
        orbit_radius_m: float = 0.0,
    ) -> float:
        """Return total gravitational potential Φ at position (m²/s²).

        Combines the central body potential and the galactic background.

        APPROXIMATION: Central body treated as a point mass outside its radius.
        Error < 0.01% at distances > 10 × body radius.

        APPROXIMATION: Galactic potential modelled as Miyamoto-Nagai disk plus
        a Hernquist bulge and a logarithmic halo.  Real Galaxy has complex
        spiral arm structure, bar, and time evolution.  Error: ~20% in
        absolute potential value; relative variations well-reproduced.

        orbit_radius_m: if > 0, use this as the distance from the central body
            instead of inferring it from position_kpc.  Pass the actual orbital
            radius (e.g. 1 AU = 1.496e11 m) so the central body potential is
            computed correctly regardless of where in the Galaxy the star is.
            Using np.linalg.norm(position_kpc) * KPC_TO_M would give the
            galactocentric distance (kpc-scale), not the orbital radius (AU-scale).
        """
        phi = 0.0

        # Central body: point mass outside radius
        if central_body_mass_kg > 0 and central_body_radius_m > 0:
            if orbit_radius_m > 0:
                r_eff = max(orbit_radius_m, central_body_radius_m)
            else:
                # Fallback: infer from galactocentric position — only valid
                # if position_kpc is measured relative to the central body,
                # not relative to the galactic centre.
                r_eff = max(np.linalg.norm(position_kpc) * KPC_TO_M,
                            central_body_radius_m)
            phi += -G_NEWTON * central_body_mass_kg / r_eff

        if include_galactic:
            phi += Gravity._miyamoto_nagai_potential(position_kpc)
            phi += Gravity._hernquist_bulge_potential(position_kpc)
            phi += Gravity._logarithmic_halo_potential(position_kpc)

        return phi

    @staticmethod
    def _miyamoto_nagai_potential(position_kpc: np.ndarray) -> float:
        """Miyamoto-Nagai disk potential (m²/s²).

        Parameters tuned to Milky Way disk (Bovy 2015 / MWPotential2014):
          Mass M_d = 6.8e10 M_sun, scale radius a = 3.0 kpc, scale height b = 0.28 kpc

        APPROXIMATION: Thin-disk approximation with smooth density profile.
        Real disk has exponential radial and sech² vertical structure, spiral
        arm overdensities, and a central bar.  Error: ~30% in the disk plane.
        """
        M_d = 6.8e10 * M_SUN          # disk mass
        a_kpc = 3.0                    # scale radius (kpc)
        b_kpc = 0.28                   # scale height (kpc)

        x, y, z = position_kpc
        R_kpc = np.sqrt(x**2 + y**2)

        R_m = R_kpc * KPC_TO_M
        z_m = z * KPC_TO_M
        a_m = a_kpc * KPC_TO_M
        b_m = b_kpc * KPC_TO_M

        denom = np.sqrt(R_m**2 + (a_m + np.sqrt(z_m**2 + b_m**2))**2)
        return -G_NEWTON * M_d / denom

    @staticmethod
    def _hernquist_bulge_potential(position_kpc: np.ndarray) -> float:
        """Hernquist bulge potential (m²/s²).

        Parameters: M_b = 5e9 M_sun, scale radius r_s = 0.5 kpc
        (Bovy 2015 MWPotential2014 bulge component).

        APPROXIMATION: Spherically symmetric.  Real bulge is bar-shaped.
        Error: ~50% in central kpc; < 5% beyond 3 kpc.
        """
        M_b = 5.0e9 * M_SUN
        r_s_m = 0.5 * KPC_TO_M

        r_m = np.linalg.norm(position_kpc) * KPC_TO_M
        return -G_NEWTON * M_b / (r_m + r_s_m)

    @staticmethod
    def _logarithmic_halo_potential(position_kpc: np.ndarray) -> float:
        """Logarithmic dark matter halo potential (m²/s²).

        Φ_halo = v_c² × ln(r² + r_c²) / 2
        where v_c = 220 km/s (circular velocity) and r_c = 12 kpc (core radius).

        APPROXIMATION: Spherical NFW or logarithmic halo.  Real dark matter
        halo is triaxial and possibly has substructure.  Error: ~20%.
        We use the logarithmic form rather than NFW because it gives a flat
        rotation curve by construction (matches observations well at 5–15 kpc).
        """
        v_c = 220e3   # m/s (circular velocity at solar radius)
        r_c_m = 12.0 * KPC_TO_M  # core radius

        r_m = np.linalg.norm(position_kpc) * KPC_TO_M
        # Logarithmic halo: Φ = (v_c² / 2) × ln(r² + r_c²) + constant
        # We subtract the value at the solar radius to give Φ=0 at r=8.178 kpc
        # (SOLAR_GALACTOCENTRIC_KPC, GRAVITY Collaboration 2019).
        # Single log of the ratio — the difference-of-logs form cancels
        # catastrophically when r ≈ r_sun (both logs ≈ 95, difference ≈ 1e-2).
        from config import SOLAR_GALACTOCENTRIC_KPC
        r_sun_m = SOLAR_GALACTOCENTRIC_KPC * KPC_TO_M
        phi = 0.5 * v_c**2 * np.log(
            (r_m**2 + r_c_m**2) / (r_sun_m**2 + r_c_m**2)
        )
        return float(phi)

    # ── Clock slowing ─────────────────────────────────────────────────────────

    @staticmethod
    def clock_slowing_factor(potential_m2s2: float) -> float:
        """Return the fractional clock rate change Δf/f = Φ/c².

        A clock deeper in a gravitational well runs slower (Δf/f < 0 for Φ < 0).

        APPROXIMATION: Weak-field limit of general relativity.  The full GR
        expression is (1 + Φ/c²)/sqrt(1 - 2Φ/c²) - 1, which differs from
        Φ/c² by order (Φ/c²)².  For the galactic potential at the solar
        radius, Φ/c² ≈ −7e−7, so the correction is ~5×10⁻¹³ — completely
        negligible for pulsar timing at nanosecond precision.
        """
        return potential_m2s2 / (C_LIGHT ** 2)

    # ── Timing residual from gravitational redshift ───────────────────────────

    @staticmethod
    def timing_residual_from_potential(potential_m2s2: float) -> float:
        """Return the monopolar timing residual (seconds) due to clock slowing.

        This offset appears identically in all pulsar observations regardless
        of viewing direction — it is the "gravity signal" the spacecraft would
        need to extract to measure its gravitational potential.

        Over an observation time T, the accumulated phase error is:
        Δt = (Φ/c²) × T

        For T = 1000 s and Φ/c² = −7×10⁻⁷:
        Δt ≈ 0.7 μs — comparable to timing noise for the best MSPs.

        APPROXIMATION: Treats the potential as constant over the observation
        window.  For an orbiting spacecraft, the potential varies by ~0.01%
        over a few hours, introducing a ~1 ns drift.  Negligible.
        """
        return Gravity.clock_slowing_factor(potential_m2s2)   # dimensionless × T=1 s

    @staticmethod
    def extract_monopolar_residual(
        timing_residuals_per_pulsar: dict[str, float],
    ) -> tuple[float, float]:
        """Extract the common (monopolar) component from per-pulsar residuals.

        The gravity signal is the mean of all residuals; noise averages down
        as 1/sqrt(N_pulsars).  Returns (estimate, uncertainty_1sigma) in seconds.

        APPROXIMATION: Assumes each pulsar's residual is an independent
        Gaussian draw with the same noise level.  Real timing residuals are
        correlated through shared DM fluctuations (chromatic) and gravitational
        wave backgrounds (common red noise in pulsar timing arrays).
        Error: DM-correlated component is achromatic and partially cancels in
        multi-pulsar average; GW background is orders of magnitude below
        current timing precision.
        """
        residuals = np.array(list(timing_residuals_per_pulsar.values()),
                             dtype=np.float64)
        if len(residuals) == 0:
            return 0.0, np.inf

        estimate = float(np.mean(residuals))
        # Standard error of the mean
        if len(residuals) > 1:
            uncertainty = float(np.std(residuals, ddof=1) / np.sqrt(len(residuals)))
        else:
            uncertainty = np.inf

        return estimate, uncertainty

    # ── Gravity well depth ────────────────────────────────────────────────────

    @staticmethod
    def gravity_well_depth(
        potential_m2s2: float,
        central_body_mass_kg: float,
        central_body_radius_m: float,
    ) -> float:
        """Return gravity well depth from body surface to current altitude (J/kg).

        Well depth = Φ_current − Φ_surface  (positive = current position is
        higher in the well, i.e., less negative potential than the surface).

        Example: at 1 AU from Sun, Φ_current ≈ −8.87×10⁸ J/kg,
        Φ_surface ≈ −1.91×10¹¹ J/kg.  Depth ≈ +1.90×10¹¹ J/kg (positive ✓).

        APPROXIMATION: Point mass model for the central body.  Accurate for
        spherical bodies and positions >> body radius.  Error < 0.1% for
        uniform-density spheres at > 2 body radii.
        """
        if central_body_mass_kg <= 0 or central_body_radius_m <= 0:
            return 0.0

        phi_surface = -G_NEWTON * central_body_mass_kg / central_body_radius_m
        # Depth = how much higher (less negative) the current position is vs surface
        return float(potential_m2s2 - phi_surface)
