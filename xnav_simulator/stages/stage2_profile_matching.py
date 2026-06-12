# stages/stage2_profile_matching.py — Pulsar identification from folded profiles
# XNAV Cold Start Simulator

"""
Stage 2: Profile Matching

Given a set of observed folded pulse profiles (one per detected source), identify
which catalogued pulsar each profile corresponds to using cross-correlation.

LABEL: This stage is ILLUSTRATIVE. Real XNAV profile matching would use X-ray
pulse profiles from the NICER/eXTP databases, not the synthetic von Mises profiles
used here. The correlation scoring is physically motivated but the profiles themselves
are synthetic.

Cross-correlation approach
--------------------------
Each catalogue pulsar has a deterministic synthetic profile generated from its
w50 pulse width (via von Mises distribution, seeded by pulsar name). The observed
profile is cross-correlated with every candidate in a windowed region of the sky
(candidate_pulsars), and the best match is determined by maximum normalised
cross-correlation coefficient (ZNCC).

Return value
------------
dict with keys:
    identifications: list of dicts, one per observed profile:
        {
            "observed_index": int,
            "best_match": Pulsar or None (if max_confidence < min_confidence threshold),
            "confidence": float in [0, 1],
            "candidates": list of (Pulsar, confidence) sorted descending,
            "phase_offset": float — best-fit rotational phase shift (bins)
        }
    n_identified: int — number of profiles matched above confidence threshold
    n_observed: int — total number of observed profiles provided
    metadata: dict — diagnostics (min/max confidence, method label)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from scipy.signal import correlate

if TYPE_CHECKING:
    from core.pulsar import Pulsar

logger = logging.getLogger(__name__)

# Minimum ZNCC confidence to accept an identification
_MIN_CONFIDENCE = 0.4


def run(
    observed_profiles: list[np.ndarray],
    pulsar_catalogue: list["Pulsar"],
    min_confidence: float = _MIN_CONFIDENCE,
    noise_sigma: float = 0.05,
) -> dict:
    """Match observed profiles to catalogued pulsars via cross-correlation.

    Parameters
    ----------
    observed_profiles: list of 1-D arrays (each normalised to sum=1, length n_bins)
    pulsar_catalogue: list of Pulsar objects to match against
    min_confidence: minimum ZNCC score to accept an identification (default 0.4)
    noise_sigma: profile noise level for confidence scaling

    Returns
    -------
    dict as described in module docstring
    """
    if not observed_profiles:
        return {
            "identifications": [],
            "n_identified": 0,
            "n_observed": 0,
            "metadata": {"method": "ZNCC cross-correlation (illustrative)"},
        }

    identifications = []

    for obs_idx, obs_profile in enumerate(observed_profiles):
        obs_arr = np.asarray(obs_profile, dtype=np.float64)
        candidates = _match_single_profile(
            obs_arr, pulsar_catalogue, min_confidence=0.0   # score all; threshold later
        )

        best_pulsar, best_conf, best_offset = candidates[0] if candidates else (None, 0.0, 0)

        accepted = best_conf >= min_confidence

        identifications.append({
            "observed_index": obs_idx,
            "best_match": best_pulsar if accepted else None,
            "confidence": float(best_conf),
            "candidates": [(p, float(c)) for p, c, _ in candidates[:5]],
            "phase_offset": float(best_offset),
        })

        logger.debug(
            "Profile %d: best=%s conf=%.3f %s",
            obs_idx,
            best_pulsar.name if best_pulsar else "none",
            best_conf,
            "(accepted)" if accepted else "(below threshold)",
        )

    n_identified = sum(1 for r in identifications if r["best_match"] is not None)

    logger.info(
        "Stage 2: %d/%d profiles identified (threshold=%.2f)",
        n_identified, len(observed_profiles), min_confidence,
    )

    return {
        "identifications": identifications,
        "n_identified": n_identified,
        "n_observed": len(observed_profiles),
        "metadata": {
            "method": "ZNCC cross-correlation (illustrative — not real X-ray profiles)",
            "min_confidence": min_confidence,
        },
    }


# ── Matching helpers ──────────────────────────────────────────────────────────

def _zncc(a: np.ndarray, b: np.ndarray) -> tuple[float, int]:
    """Zero-normalised cross-correlation between two 1-D arrays.

    Returns (max_zncc, lag_at_max).  ZNCC is in [-1, 1]; 1 = perfect match.

    APPROXIMATION: Circular cross-correlation (assumes periodic profile).
    Real pulse profiles have sharp leading edges that break circularity for
    very wide pulses (w50 > 50% period). Error < 5% for w50 < 30% of period.
    """
    a = a - a.mean()
    b = b - b.mean()
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-12 or norm_b < 1e-12:
        logger.warning(
            "ZNCC received a zero-norm (flat) profile — returning zero "
            "correlation; the pulsar cannot be identified from this profile."
        )
        return 0.0, 0

    # Circular correlation via FFT — O(n log n)
    n = len(a)
    fa = np.fft.rfft(a, n=n)
    fb = np.fft.rfft(b, n=n)
    corr = np.fft.irfft(fa * np.conj(fb), n=n)
    corr /= norm_a * norm_b

    lag = int(np.argmax(corr))
    return float(corr[lag]), lag


def _match_single_profile(
    obs_profile: np.ndarray,
    catalogue: list["Pulsar"],
    min_confidence: float = 0.0,
) -> list[tuple["Pulsar", float, int]]:
    """Return list of (Pulsar, confidence, phase_offset) sorted by confidence descending.

    For each catalogue pulsar, generate/use its stored profile and compute ZNCC.
    """
    n_bins = len(obs_profile)
    results = []

    for pulsar in catalogue:
        # Get catalogue profile (regenerate at matching n_bins if needed)
        cat_profile = pulsar.profile
        if cat_profile is None or len(cat_profile) != n_bins:
            cat_profile = pulsar.generate_profile(n_bins=n_bins)

        zncc_val, lag = _zncc(obs_profile, cat_profile)

        # Map ZNCC ∈ [-1, 1] to confidence ∈ [0, 1]
        # APPROXIMATION: Confidence = max(0, zncc) since negative correlation
        # is non-physical for pulse profiles (which are always positive).
        confidence = float(max(0.0, zncc_val))
        results.append((pulsar, confidence, lag))

    results.sort(key=lambda x: x[1], reverse=True)
    return results


# ── Convenience: generate noisy observed profiles from catalogue ──────────────

def simulate_observed_profiles(
    pulsars: list["Pulsar"],
    noise_sigma: float = 0.05,
    seed: int = 0,
    n_bins: int = 128,
) -> list[np.ndarray]:
    """Generate synthetic 'observed' profiles by adding Gaussian noise to catalogue profiles.

    Used for testing Stage 2 without a full simulation pipeline.

    APPROXIMATION: Noise is additive Gaussian (white) across bins.  Real X-ray
    profiles have Poisson counting noise which is signal-dependent (higher variance
    in brighter bins). For profile SNR > 10, Gaussian approximation is valid.
    """
    rng = np.random.default_rng(seed)
    profiles = []
    for p in pulsars:
        profile = p.generate_profile(n_bins=n_bins).copy()
        noise = rng.normal(0.0, noise_sigma, n_bins)
        noisy = np.clip(profile + noise, 0.0, None)
        total = noisy.sum()
        if total > 0:
            noisy /= total
        profiles.append(noisy)
    return profiles
