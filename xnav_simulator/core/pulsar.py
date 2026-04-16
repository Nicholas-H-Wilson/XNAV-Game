# core/pulsar.py — Pulsar data model
# XNAV Cold Start Simulator

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Pulsar:
    """Single pulsar data record, loaded from ATNF catalogue.

    All timing quantities are in SI base units unless otherwise noted.
    Profile arrays are generated deterministically at construction from
    pulsar name so they are reproducible across filter iterations.
    """

    name: str                   # PSR J-name (e.g. "J1713+0747")
    period: float               # Spin period P0 in seconds
    period_dot: float           # Period derivative P1 (dimensionless, s/s)
    dm: float                   # Dispersion Measure in pc cm^-3
    gl: float                   # Galactic longitude in degrees [0, 360)
    gb: float                   # Galactic latitude in degrees [-90, 90]
    distance_kpc: float         # Estimated distance in kpc
    w50: float                  # Pulse width at 50% peak in microseconds
    s1400: float                # Flux density at 1400 MHz in mJy
    timing_noise_ns: float      # RMS timing residual in nanoseconds

    # Generated at construction — do not set manually
    profile: np.ndarray = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.profile is None:
            self.profile = self.generate_profile()

    # ── Profile generation ────────────────────────────────────────────────────

    def generate_profile(self, n_bins: int = 128) -> np.ndarray:
        """Return a normalised folded pulse profile array of length n_bins.

        The profile is a von Mises distribution whose concentration parameter
        kappa is set from w50 (pulse width at 50% maximum).

        Seeded deterministically from the pulsar name hash so the same pulsar
        always produces the same profile, regardless of call order or session.

        APPROXIMATION: Real pulse profiles are multi-component with varying
        spectral indices, polarisation structure, and interstellar scattering
        tails. The von Mises model captures width correctly but not fine
        structure. For cross-correlation matching (Stage 2) this is sufficient
        since the matching is within-simulation and self-consistent.
        """
        # Deterministic seed from name hash
        name_hash = int(hashlib.sha256(self.name.encode()).hexdigest(), 16)
        rng = np.random.default_rng(name_hash % (2**32))

        # Pulse phase axis [0, 1)
        phase = np.linspace(0, 1, n_bins, endpoint=False)

        # Convert w50 (μs) to phase fraction
        # APPROXIMATION: Uses on-pulse fraction = w50 / (1e6 * period).
        # Ignores scattering broadening which is frequency-dependent and
        # can dominate at low frequencies. Error: ~10-50% in profile width
        # for DM > 100 pc/cm^3. Acceptable for illustrative profile matching.
        w50_phase = min(self.w50 / (1e6 * self.period), 0.45)

        # von Mises concentration from the half-maximum condition:
        #   exp(κ cos(π w50_phase)) = ½ exp(κ)
        # For narrow pulses (large κ) the small-angle approximation gives:
        #   κ ≈ 2 ln(2) / (π w50_phase)²
        # This is 39% larger than the naive 1/(π w50_phase)², which produced
        # profiles 16% too broad for the catalogued W50.
        kappa = max(2.0 * np.log(2) / (np.pi * w50_phase) ** 2, 0.5)
        # No upper cap — very large kappa is handled numerically below.

        # Random phase centre for this pulsar
        mu = rng.uniform(0.2, 0.8)

        # Evaluate von Mises distribution using the log-sum-exp trick:
        #   shift log_profile by its maximum before exponentiating.
        # This is mathematically identical to the unshifted form — the constant
        # cancels during normalisation — but avoids float64 overflow for narrow
        # pulsars such as J1939+2134 (w50≈16μs, period≈1.56ms → kappa≈957,
        # exp(957) >> float64 max ≈ exp(709)).
        log_profile = kappa * np.cos(2 * np.pi * (phase - mu))
        log_profile -= log_profile.max()   # shift peak to 0; all values ≤ 0
        profile = np.exp(log_profile)      # all values in (0, 1]; no overflow

        # Some pulsars have a secondary component (interpulse)
        # 30% chance, with amplitude 10-40% of main peak
        if (name_hash % 10) < 3:
            mu2 = (mu + 0.5) % 1.0
            kappa2 = kappa * rng.uniform(0.3, 0.7)
            amp2 = rng.uniform(0.1, 0.4)
            log_ip = kappa2 * np.cos(2 * np.pi * (phase - mu2))
            log_ip -= log_ip.max()
            profile += amp2 * np.exp(log_ip)

        # Normalise to unit sum
        profile -= profile.min()
        total = profile.sum()
        if total > 0:
            profile /= total
        else:
            profile = np.ones(n_bins) / n_bins

        return profile

    # ── Quality scoring ───────────────────────────────────────────────────────

    def timing_quality_score(self) -> float:
        """Return a scalar quality score for ranking pulsars.

        Higher score = better for navigation (lower noise, higher flux).

        Score = log10(s1400 / timing_noise_ns)
        This penalises dim pulsars (poor photon statistics) and noisy pulsars
        (poor phase measurements) equally in log space.

        APPROXIMATION: Ignores spectral index variation across pulsars and
        assumes 1400 MHz flux is representative of X-ray band flux.
        Real XNAV uses X-ray pulsars; the ranking order is broadly preserved
        since MSP brightness and timing quality correlate across bands.
        """
        flux = max(self.s1400, 1e-6)       # guard against zero/negative
        noise = max(self.timing_noise_ns, 1e-3)
        return np.log10(flux / noise)

    # ── Convenience properties ────────────────────────────────────────────────

    @property
    def frequency_hz(self) -> float:
        """Spin frequency in Hz."""
        return 1.0 / self.period

    @property
    def characteristic_age_yr(self) -> float:
        """Characteristic spin-down age in years: P / (2 * P_dot)."""
        if self.period_dot <= 0:
            return np.inf
        return (self.period / (2 * self.period_dot)) / (365.25 * 86400)

    @property
    def position_kpc(self) -> np.ndarray:
        """Galactocentric Cartesian position of this pulsar (kpc)."""
        from utils.coordinates import galactic_to_cartesian
        return galactic_to_cartesian(self.gl, self.gb, self.distance_kpc)

    def __repr__(self) -> str:
        return (
            f"Pulsar({self.name!r}, P={self.period*1e3:.3f} ms, "
            f"DM={self.dm:.1f}, d={self.distance_kpc:.2f} kpc, "
            f"noise={self.timing_noise_ns:.0f} ns)"
        )
