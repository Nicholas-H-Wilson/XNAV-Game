# core/dispersion.py — Dispersion measure computation and chromatic correction
# XNAV Cold Start Simulator

from __future__ import annotations

import numpy as np

from config import C_LIGHT, K_DM


class Dispersion:
    """Dispersive delay computation and multi-frequency DM correction.

    The dispersion measure (DM) causes pulse arrival times to be delayed at
    lower frequencies relative to higher frequencies:
        Δt = K_DM × DM / f²

    By observing simultaneously at multiple frequencies, the DM can be
    recovered and the arrival times corrected to infinite frequency.
    """

    # ── Single-frequency dispersive delay ─────────────────────────────────────

    @staticmethod
    def compute_dispersive_delay(dm: float, frequency_mhz: float) -> float:
        """Return dispersive delay in seconds for a given DM and frequency.

        Δt = K_DM × DM / f²

        where K_DM = 4.148e3 MHz² pc⁻¹ cm³ s  (IAU 2016 constant).

        APPROXIMATION: Uses the cold-plasma dispersion relation in the limit
        f >> f_plasma (plasma frequency ~10 kHz for warm ISM).  The correction
        term (f_plasma/f)² is of order 10⁻¹⁴ at GHz frequencies — completely
        negligible.  Error < 1 ns for any observing frequency above 100 MHz.
        """
        return K_DM * dm / (frequency_mhz ** 2)

    # ── Multi-frequency dispersive sweep ──────────────────────────────────────

    @staticmethod
    def simulate_multifreq_arrival(
        dm: float,
        n_channels: int,
        freq_low_mhz: float,
        freq_high_mhz: float,
        true_arrival_s: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return dispersive sweep across a frequency band.

        Simulates what a broadband receiver would see: the pulse arrives
        later at lower frequencies.

        Returns:
            frequencies_mhz: array of channel centre frequencies, shape (n_channels,)
            arrival_times_s:  observed arrival time per channel, shape (n_channels,)
                              relative to the infinite-frequency arrival time.
        """
        frequencies_mhz = np.linspace(freq_low_mhz, freq_high_mhz, n_channels)
        delays_s = K_DM * dm / (frequencies_mhz ** 2)
        arrival_times_s = true_arrival_s + delays_s
        return frequencies_mhz, arrival_times_s

    # ── Chromatic DM correction ───────────────────────────────────────────────

    @staticmethod
    def correct_dm_chromatic(
        arrival_times_s: np.ndarray,
        frequencies_mhz: np.ndarray,
    ) -> tuple[float, float, float]:
        """Fit for DM from multi-frequency arrival times and correct to ∞ frequency.

        Solves: t_i = t_∞ + K_DM × DM / f_i²
        as a linear least-squares problem in (t_∞, DM).

        Returns:
            t_corrected_s:  estimated arrival time at infinite frequency (seconds)
            dm_recovered:   best-fit DM (pc cm⁻³)
            dm_uncertainty: 1σ uncertainty in DM from the fit (pc cm⁻³)

        APPROXIMATION: Treats each frequency channel as an independent
        measurement with equal weight.  Real dedispersion uses matched
        filtering weighted by the radiometer noise in each channel.
        Error: ~10% in DM uncertainty estimate for unequal S/N across band.
        """
        # Design matrix: t_i = t_∞ × 1 + DM × (K_DM / f_i²)
        inv_f2 = K_DM / (frequencies_mhz ** 2)
        A = np.column_stack([np.ones_like(frequencies_mhz), inv_f2])

        # Weighted least squares (equal weights here)
        result, residuals, rank, sv = np.linalg.lstsq(A, arrival_times_s, rcond=None)
        t_corrected_s = float(result[0])
        dm_recovered = float(result[1])

        # Estimate DM uncertainty from residual RMS
        n = len(frequencies_mhz)
        if n > 2:
            predicted = A @ result
            rms_residual = np.sqrt(np.sum((arrival_times_s - predicted) ** 2) / (n - 2))
            # Propagate to DM uncertainty via the Fisher matrix diagonal
            AtA_inv = np.linalg.inv(A.T @ A)
            dm_uncertainty = float(rms_residual * np.sqrt(AtA_inv[1, 1]))
        else:
            dm_uncertainty = np.nan

        return t_corrected_s, dm_recovered, dm_uncertainty

    # ── Solar wind / interplanetary DM contribution ───────────────────────────

    @staticmethod
    def interplanetary_dm_contribution(
        spacecraft_pos_au: np.ndarray,
        pulsar_gl: float,
        pulsar_gb: float,
    ) -> float:
        """Estimate solar wind electron column DM contribution (pc cm⁻³).

        APPROXIMATION: Uses simple radial solar wind model (Parker 1958).
        Real IPM is highly structured and time-variable with solar activity.
        Solar wind electron density n_e ~ n_e0 × (r_0/r)² where n_e0 ≈ 7 cm⁻³
        at r_0 = 1 AU (solar minimum).

        The integral along the LOS is approximated assuming the spacecraft is
        far from the Sun (> 1 AU) and the sightline avoids near-solar regions.
        Returns the DM contribution in pc cm⁻³.
        Error: factor ~3 for sightlines near solar conjunction.

        APPROXIMATION: Ignores ecliptic latitude anisotropy of the solar wind.
        Real solar wind has equatorial enhancement and polar holes.
        Error: ~50% for polar sightlines.
        """
        from utils.coordinates import direction_vector

        r_sc = np.linalg.norm(spacecraft_pos_au)
        if r_sc < 0.01:
            return 0.0

        # Solar wind electron density at 1 AU: n_e0 ≈ 7 cm⁻³
        n_e0_per_cm3 = 7.0
        r0_au = 1.0

        # Pulsar direction unit vector in heliocentric galactic coordinates
        # (ignoring difference between heliocentric and galactocentric at this scale)
        pulsar_dir = direction_vector(pulsar_gl, pulsar_gb)

        # Solar elongation angle
        sc_dir = spacecraft_pos_au / r_sc
        cos_elng = np.clip(np.dot(sc_dir, pulsar_dir), -1.0, 1.0)
        sin_elng = np.sqrt(max(1.0 - cos_elng ** 2, 1e-12))

        # Analytical integral of n_e(r) = n_e0 × (r0/r)² along the LOS
        # For a sightline at impact parameter b = r_sc × sin(ε):
        # DM ≈ n_e0 × r0² × π / (2 × b)  (in cm⁻³ × AU)
        # Convert: 1 AU = 3.086e18 cm / (3.086e16 cm/pc) = 100 pc... wait, units:
        # 1 pc = 3.086e18 cm, 1 AU = 1.496e13 cm = 1.496e13 / 3.086e18 pc = 4.848e-6 pc
        AU_TO_PC = 4.848e-6   # 1 AU in parsecs

        impact_param_au = r_sc * sin_elng
        if impact_param_au < 0.1:
            impact_param_au = 0.1   # avoid divergence very close to Sun

        dm_ipm = (n_e0_per_cm3 * r0_au ** 2 * np.pi / 2.0 / impact_param_au) * AU_TO_PC

        return float(max(dm_ipm, 0.0))
