# core/noise.py — All noise source models for pulsar timing
# XNAV Cold Start Simulator

from __future__ import annotations

import numpy as np

from config import C_LIGHT, K_DM


class NoiseModel:
    """Collection of noise source models.

    Every method returns a timing residual in seconds.
    Every method accepts a seed parameter so Monte Carlo runs are reproducible.
    """

    # ── Timing noise ──────────────────────────────────────────────────────────

    @staticmethod
    def timing_noise(
        pulsar,
        integration_time_s: float,
        amplitude_multiplier: float = 1.0,
        seed: int = 0,
    ) -> float:
        """Return a timing residual (seconds) from pulsar intrinsic timing noise.

        White noise component scales as 1/sqrt(T_int) — averaging down with
        longer observations.  A red noise component (power law index ≈ −2 in
        frequency) is added at a low level representative of MSP spin noise.

        APPROXIMATION: White noise floor is pulsar.timing_noise_ns / sqrt(T).
        Real timing noise arises from pulse phase jitter, radiometer noise,
        calibration errors, and interstellar scintillation.  The white-noise
        approximation is valid for integration times much shorter than the
        spin-noise correlation time (~years for MSPs).
        Error: ~30% for individual observations; averages correctly over many.

        APPROXIMATION: Red noise modelled as a single power-law component with
        fixed spectral index −2 and amplitude 10% of the white noise floor.
        Real MSP red noise spans a range of spectral indices (−1 to −6) and
        amplitudes that vary by orders of magnitude between pulsars.
        Error: Dominant for integration times > 10^4 s; negligible below.
        """
        rng = np.random.default_rng(seed)

        t = max(integration_time_s, 1.0)   # guard against zero integration time

        # White noise: scales as 1/sqrt(T)
        sigma_white_s = (pulsar.timing_noise_ns * 1e-9 * amplitude_multiplier
                         / np.sqrt(t))
        white = rng.normal(0.0, sigma_white_s)

        # Red noise: simple random walk component (power-law index −2)
        # Amplitude is 10% of the white noise level at T=1000 s integration
        sigma_red_s = 0.10 * pulsar.timing_noise_ns * 1e-9 * amplitude_multiplier
        # Red noise variance grows as T^1 (random walk), normalised to T=1000 s
        red = rng.normal(0.0, sigma_red_s * np.sqrt(t / 1000.0))

        return float(white + red)

    # ── Photon counting noise ─────────────────────────────────────────────────

    @staticmethod
    def photon_noise(
        pulsar,
        distance_kpc: float,
        collecting_area_m2: float,
        integration_time_s: float,
        seed: int = 0,
    ) -> float:
        """Return timing residual (seconds) from Poisson photon counting statistics.

        Fewer photons → worse profile SNR → worse peak localisation → larger
        timing uncertainty.  Phase uncertainty scales as 1/sqrt(N_photons).

        APPROXIMATION: Uses 1400 MHz radio flux (s1400, mJy) as a proxy for
        X-ray flux.  Real XNAV uses X-ray pulsars observed in 1–10 keV band.
        The correlation between radio and X-ray flux is loose; this
        approximation preserves the relative ranking of pulsars but not the
        absolute photon rates.  Error: factor ~10 in absolute photon counts
        for individual pulsars.

        APPROXIMATION: Flux scales as 1/distance^2 from catalogue value.
        This ignores interstellar absorption (important at X-ray energies) and
        assumes the distance in the catalogue is correct.  Error: factor ~2
        for high-DM pulsars with significant X-ray absorption.

        APPROXIMATION: Profile peak localisation uncertainty modelled as
        W50 / (pulse_period × sqrt(N_photons)).  The exact coefficient depends
        on the profile shape and SNR; this is the leading-order approximation.
        Error: ~50% in timing uncertainty for typical profiles.
        """
        rng = np.random.default_rng(seed)

        # Flux at spacecraft (Jy → photon counts: very rough conversion)
        # 1 mJy at 1 GHz over 1 m² for 1 s ≈ 1.5e−3 photons (radio)
        # For simulation purposes: use s1400 as a dimensionless brightness proxy
        reference_distance_kpc = 1.0
        flux_ratio = (reference_distance_kpc / max(distance_kpc, 0.01)) ** 2
        flux_proxy = pulsar.s1400 * flux_ratio   # mJy at spacecraft distance

        # Effective photon count: proportional to flux × area × time
        n_photons = max(flux_proxy * collecting_area_m2 * integration_time_s * 0.01, 1.0)
        # The factor 0.01 converts the proxy to a plausible photon count order of magnitude

        # Poisson fluctuation in photon count
        n_observed = float(rng.poisson(n_photons))
        if n_observed < 1:
            n_observed = 1.0

        # Phase uncertainty from profile peak localisation
        # σ_phase ≈ W50 / (period × sqrt(N))
        w50_phase = pulsar.w50 / (1e6 * pulsar.period)   # fraction of period
        sigma_phase = w50_phase / np.sqrt(n_observed)     # phase uncertainty (dimensionless)
        sigma_time_s = sigma_phase * pulsar.period        # convert to seconds

        return float(rng.normal(0.0, sigma_time_s))

    # ── DM turbulence ─────────────────────────────────────────────────────────

    @staticmethod
    def dm_turbulence(
        dm: float,
        baseline_kpc: float,
        frequency_mhz: float = 1400.0,
        amplitude_multiplier: float = 1.0,
        seed: int = 0,
    ) -> float:
        """Return timing residual (seconds) from unmodelled ISM DM fluctuations.

        APPROXIMATION: DM uncertainty scales as sqrt(DM × baseline_kpc) × 0.01.
        This is motivated by structure function analyses of ISM turbulence:
        ΔDM ~ DM^0.5 × (path length)^0.5 for Kolmogorov turbulence.
        The coefficient 0.01 is tuned to match observed DM variations in
        pulsar timing array data (~0.001–0.01 pc cm⁻³ yr⁻¹ for high-DM MSPs).
        Error: order of magnitude; individual pulsars vary enormously.

        Returns the residual timing delay after DM correction, in seconds.
        The chromatic correction is assumed to remove the mean DM; this
        models the uncorrected fluctuation component.
        """
        rng = np.random.default_rng(seed)

        # DM fluctuation (pc cm⁻³)
        dm_sigma = amplitude_multiplier * 0.01 * np.sqrt(max(dm, 1.0) * max(baseline_kpc, 0.1))
        delta_dm = rng.normal(0.0, dm_sigma)

        # Convert DM fluctuation to timing residual at the observation frequency
        # Δt = K_DM × ΔDM / f²  (in seconds if K_DM in MHz² pc⁻¹ cm³ s)
        delta_t = K_DM * delta_dm / (frequency_mhz ** 2)

        return float(delta_t)

    # ── Solar wind noise ──────────────────────────────────────────────────────

    @staticmethod
    def solar_wind_noise(
        spacecraft_pos_au: np.ndarray,
        pulsar_direction: np.ndarray,
        solar_activity: float = 0.3,
        seed: int = 0,
    ) -> float:
        """Return timing residual (seconds) from interplanetary medium (IPM).

        The solar wind adds a variable electron column density between the
        spacecraft and any pulsar whose line of sight passes close to the Sun.

        APPROXIMATION: Solar wind electron density modelled as n_e ∝ 1/r^2
        (Parker solar wind). Real IPM is highly structured with corotating
        interaction regions, coronal mass ejections, and solar cycle dependence.
        Error: factor ~3 in DM contribution near solar conjunction.

        APPROXIMATION: Integral is approximated as DM_IPM ≈ A_sw / (r_sc × sin(ε))
        where r_sc is spacecraft distance from Sun in AU and ε is the
        solar elongation of the pulsar.  This is the weak-elongation approximation
        valid for ε > 10°.  Error: factor ~2 at ε < 20°.

        solar_activity in [0, 1]: 0 = solar minimum, 1 = solar maximum.

        A_sw is derived from the Parker 1/r² model:
            DM_IPM = n_e0 × r₀² × π/2 × (1 pc / 3.086e16 m) × r₀_AU/sin(ε)
        where n_e0 is the electron density at 1 AU (cm⁻³) and AU_to_pc
        converts the integration path length to pc.
            Solar min: n_e0 ≈ 7 cm⁻³  → A_sw ≈ 5.33×10⁻⁵ pc cm⁻³ AU
            Solar max: n_e0 ≈ 20 cm⁻³ → A_sw ≈ 1.52×10⁻⁴ pc cm⁻³ AU
        These are consistent with observed interplanetary DM contributions of
        ~10⁻⁵–10⁻⁴ pc cm⁻³ (Madison et al. 2019, Tiburzi et al. 2021).
        """
        rng = np.random.default_rng(seed)

        # Solar elongation: angle between spacecraft–Sun direction and pulsar direction
        r_sc = np.linalg.norm(spacecraft_pos_au)
        if r_sc < 1e-6:
            r_sc = 1.0   # if at Sun, no solar wind contribution modelled

        sc_direction = spacecraft_pos_au / r_sc
        cos_elng = np.clip(np.dot(sc_direction, pulsar_direction), -1.0, 1.0)
        sin_elng = np.sqrt(max(1.0 - cos_elng ** 2, 1e-12))

        # Solar wind DM amplitude (pc cm⁻³ AU) from Parker 1/r² model.
        # A_sw = n_e0 × r₀² × π/2 × AU_to_pc
        # Solar min (n_e0=7 cm⁻³): A_sw = 5.33e-5 pc cm⁻³ AU
        # Solar max (n_e0=20 cm⁻³): A_sw = 1.52e-4 pc cm⁻³ AU
        A_SW_MIN = 5.33e-5   # pc cm⁻³ AU at solar minimum
        A_SW_MAX = 1.52e-4   # pc cm⁻³ AU at solar maximum
        A_sw = A_SW_MIN + (A_SW_MAX - A_SW_MIN) * solar_activity

        # DM contribution from solar wind
        # APPROXIMATION: DM_IPM ≈ A_sw / (r_sc × sin_elng) for ε > 10°
        if sin_elng > 0.17:   # > 10° from Sun
            dm_ipm = A_sw / (r_sc * sin_elng)
        else:
            dm_ipm = A_sw / (r_sc * 0.17)   # cap at ε = 10° to avoid divergence

        # Variable component: solar wind is turbulent, add ~30% random fluctuation.
        # Clamp to zero — DM is a column density and cannot be negative.
        dm_ipm_fluctuation = dm_ipm * rng.normal(0.0, 0.3)
        dm_total = max(dm_ipm + dm_ipm_fluctuation, 0.0)

        # Convert to timing residual at 1400 MHz
        # Use a characteristic low frequency since solar wind is dispersive
        frequency_mhz = 1400.0
        delta_t = K_DM * dm_total / (frequency_mhz ** 2)

        return float(delta_t)
