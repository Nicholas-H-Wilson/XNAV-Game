#!/usr/bin/env python3
# tests/test_phase1.py — Phase 1 Foundation Tests
# XNAV Cold Start Simulator
#
# Run with: python tests/test_phase1.py  (from xnav_simulator/ directory)
# or:       python -m tests.test_phase1

from __future__ import annotations

import sys
import os
import traceback

# Ensure the xnav_simulator package root is on the path
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np

# ── Test registry ─────────────────────────────────────────────────────────────

_TESTS: list = []   # registered test functions


def _test(name: str):
    """Decorator — register a test by name."""
    def decorator(fn):
        def wrapper(results: list) -> None:
            try:
                detail = fn()
                results.append((name, True, detail or ""))
                print(f"  PASS  {name}" + (f"  — {detail}" if detail else ""))
            except AssertionError as exc:
                results.append((name, False, str(exc)))
                print(f"  FAIL  {name}  — {exc}")
            except Exception as exc:
                tb = traceback.format_exc()
                results.append((name, False, f"Exception: {exc}"))
                print(f"  FAIL  {name}  — Exception: {exc}")
                print("        " + tb.replace("\n", "\n        "))
        wrapper.__name__ = fn.__name__
        _TESTS.append(wrapper)
        return wrapper
    return decorator


# ── Test 1: config constants ──────────────────────────────────────────────────

@_test("config.py constants are physically reasonable")
def test_config_constants():
    import config

    # Speed of light within 0.1% of 2.998e8 m/s
    assert abs(config.C_LIGHT - 2.998e8) / 2.998e8 < 0.001, (
        f"C_LIGHT = {config.C_LIGHT} differs from 2.998e8 by more than 0.1%"
    )

    # G_NEWTON within 1% of NIST value 6.674e-11
    assert abs(config.G_NEWTON - 6.674e-11) / 6.674e-11 < 0.01, (
        f"G_NEWTON = {config.G_NEWTON}"
    )

    # K_DM within 1% of standard 4.148e3
    assert abs(config.K_DM - 4.148e3) / 4.148e3 < 0.01, (
        f"K_DM = {config.K_DM}"
    )

    # Galactic radius is plausible (between 10 and 25 kpc)
    assert 10.0 < config.GALAXY_RADIUS_KPC < 25.0, (
        f"GALAXY_RADIUS_KPC = {config.GALAXY_RADIUS_KPC}"
    )

    # Solar galactocentric distance is plausible (7–10 kpc)
    assert 7.0 < config.SOLAR_GALACTOCENTRIC_KPC < 10.0, (
        f"SOLAR_GALACTOCENTRIC_KPC = {config.SOLAR_GALACTOCENTRIC_KPC}"
    )

    # Accuracy tiers — must have exactly 5, and cover expected key
    assert len(config.ACCURACY_TIERS) == 5, (
        f"Expected 5 accuracy tiers, got {len(config.ACCURACY_TIERS)}"
    )
    assert config.DEFAULT_TIER in config.ACCURACY_TIERS, (
        f"DEFAULT_TIER {config.DEFAULT_TIER!r} not in ACCURACY_TIERS"
    )

    # All tiers have required keys with sensible values
    for tier_name, tier in config.ACCURACY_TIERS.items():
        for key in ("n_pulsars", "n_particles", "grid_resolution_pc", "expected_runtime_seconds"):
            assert key in tier, f"Tier {tier_name!r} missing key {key!r}"
            assert tier[key] > 0, f"Tier {tier_name!r}[{key!r}] = {tier[key]} is not positive"

    return f"C_LIGHT={config.C_LIGHT:.4e}, K_DM={config.K_DM:.3e}"


# ── Test 2: coordinate conversions ────────────────────────────────────────────

@_test("coordinates.py galactic ↔ cartesian round-trip")
def test_coordinate_roundtrip():
    from utils.coordinates import galactic_to_cartesian, cartesian_to_galactic

    test_cases = [
        (45.0, 30.0, 5.0),
        (0.0, 0.0, 8.0),
        (180.0, -45.0, 2.5),
        (270.0, 60.0, 10.0),
        (90.0, -20.0, 1.0),
    ]

    max_err = 0.0
    for gl, gb, d in test_cases:
        xyz = galactic_to_cartesian(gl, gb, d)
        gl2, gb2, d2 = cartesian_to_galactic(xyz)

        dgl = min(abs(gl - gl2), 360 - abs(gl - gl2))
        dgb = abs(gb - gb2)
        dd = abs(d - d2)
        max_err = max(max_err, dd)

        assert dd < 0.001, (
            f"Round-trip distance error {dd:.6f} kpc > 0.001 for (gl={gl}, gb={gb}, d={d})"
        )
        assert dgl < 0.01, (
            f"Round-trip GL error {dgl:.4f}° > 0.01 for (gl={gl}, gb={gb}, d={d})"
        )
        assert dgb < 0.01, (
            f"Round-trip GB error {dgb:.4f}° > 0.01 for (gl={gl}, gb={gb}, d={d})"
        )

    return f"Max distance round-trip error: {max_err:.2e} kpc ({len(test_cases)} cases)"


@_test("coordinates.py direction_vector is unit length")
def test_direction_vector():
    from utils.coordinates import direction_vector

    test_cases = [(0, 0), (90, 30), (180, -45), (270, 80), (45, -10)]
    for gl, gb in test_cases:
        v = direction_vector(gl, gb)
        mag = np.linalg.norm(v)
        assert abs(mag - 1.0) < 1e-12, f"direction_vector({gl},{gb}) magnitude={mag}"

    return f"All {len(test_cases)} direction vectors have unit magnitude"


@_test("coordinates.py angular_separation_deg — known values")
def test_angular_separation():
    from utils.coordinates import angular_separation_deg

    # Same point → 0°
    sep = angular_separation_deg(45.0, 30.0, 45.0, 30.0)
    assert abs(sep) < 1e-10, f"Same point separation = {sep}, expected 0"

    # North and south poles → 180°
    sep = angular_separation_deg(0.0, 90.0, 0.0, -90.0)
    assert abs(sep - 180.0) < 1e-10, f"Pole-to-pole separation = {sep}, expected 180"

    # Equatorial points separated by 90° in longitude → 90°
    sep = angular_separation_deg(0.0, 0.0, 90.0, 0.0)
    assert abs(sep - 90.0) < 1e-10, f"90° equatorial separation = {sep}"

    # North pole to equator → 90°
    sep = angular_separation_deg(0.0, 90.0, 180.0, 0.0)
    assert abs(sep - 90.0) < 1e-10, f"Pole-to-equator separation = {sep}"

    return "0°, 90°, 180° cases all correct"


@_test("coordinates.py galactic ↔ ICRF round-trip")
def test_icrf_roundtrip():
    from utils.coordinates import galactic_to_icrf, icrf_to_galactic

    # Use several well-separated galactic positions
    test_cases = [
        (0.0, 0.0),    # Galactic centre direction
        (90.0, 30.0),  # Above galactic plane
        (180.0, -10.0),
        (270.0, 45.0),
        (45.0, -60.0),
    ]
    max_gl_err = 0.0
    max_gb_err = 0.0
    for gl, gb in test_cases:
        ra, dec = galactic_to_icrf(gl, gb)
        gl2, gb2 = icrf_to_galactic(ra, dec)
        dgl = min(abs(gl - gl2), 360 - abs(gl - gl2))
        dgb = abs(gb - gb2)
        max_gl_err = max(max_gl_err, dgl)
        max_gb_err = max(max_gb_err, dgb)
        assert dgl < 0.01, f"ICRF round-trip GL error {dgl:.4f}° for gl={gl}, gb={gb}"
        assert dgb < 0.01, f"ICRF round-trip GB error {dgb:.4f}° for gl={gl}, gb={gb}"

    return f"Max errors: GL={max_gl_err:.2e}°, GB={max_gb_err:.2e}°"


# ── Test 3: plotting module ───────────────────────────────────────────────────

@_test("plotting.py make_empty_galaxy_figure returns a Plotly Figure")
def test_plotting_galaxy_figure():
    import plotly.graph_objects as go
    from utils.plotting import make_empty_galaxy_figure

    fig = make_empty_galaxy_figure()

    assert isinstance(fig, go.Figure), (
        f"Expected plotly.graph_objects.Figure, got {type(fig)}"
    )
    assert len(fig.data) > 0, "Galaxy figure has no traces"

    trace_names = [t.name for t in fig.data]
    assert any("Sun" in (n or "") for n in trace_names), (
        f"Sun marker not found in traces: {trace_names}"
    )

    return f"Figure has {len(fig.data)} traces"


@_test("plotting.py colours are single-sourced from config")
def test_plotting_colours_from_config():
    import config
    import utils.plotting as plt_mod

    # The three shared colours must be identical between config and plotting
    for attr in ("COLOUR_ACCENT", "COLOUR_BG", "COLOUR_GRID"):
        cfg_val = getattr(config, attr)
        plt_val = getattr(plt_mod, attr)
        assert cfg_val == plt_val, (
            f"{attr}: config={cfg_val!r} != plotting={plt_val!r} — colours have diverged"
        )

    return "COLOUR_ACCENT, COLOUR_BG, COLOUR_GRID are identical in config and plotting"


@_test("plotting.py make_bar_figure returns a Plotly Figure")
def test_plotting_bar_figure():
    import plotly.graph_objects as go
    from utils.plotting import make_bar_figure

    labels = ["A", "B", "C"]
    values = [1.0, 10.0, 100.0]
    colours = ["#00D4FF", "#FF6B35", "#FF3355"]

    fig = make_bar_figure(labels, values, colours, title="Test", log_scale=True)
    assert isinstance(fig, go.Figure), f"Expected Figure, got {type(fig)}"
    assert len(fig.data) == 1, f"Expected 1 bar trace, got {len(fig.data)}"

    return "Bar figure created with log scale"


# ── Test 4: pulsar data model ─────────────────────────────────────────────────

@_test("pulsar.py profile generation is deterministic")
def test_pulsar_profile_deterministic():
    from core.pulsar import Pulsar

    p = Pulsar(
        name="J1713+0747", period=0.004570, period_dot=8.52e-21,
        dm=15.9918, gl=28.75, gb=25.22, distance_kpc=1.05,
        w50=68.0, s1400=10.2, timing_noise_ns=30.0,
    )

    profile1 = p.generate_profile(n_bins=128)
    profile2 = p.generate_profile(n_bins=128)

    assert np.allclose(profile1, profile2), "Profile is not deterministic across two calls"

    total = profile1.sum()
    assert abs(total - 1.0) < 1e-10, f"Profile does not sum to 1.0 (sum={total:.8f})"
    assert np.all(profile1 >= 0), "Profile has negative values"
    assert len(profile1) == 128, f"Profile length {len(profile1)} ≠ 128"

    return f"Profile deterministic, sum={total:.8f}, max={profile1.max():.4f}"


@_test("pulsar.py narrow-pulse profile (J1939+2134) handles large kappa without overflow")
def test_pulsar_narrow_pulse():
    """J1939+2134: w50=16μs, period=1.558ms → w50_phase≈0.0103 → kappa≈957.
    exp(957) overflows float64. The log-sum-exp fix must keep profile finite and valid.
    """
    from core.pulsar import Pulsar

    p = Pulsar(
        name="J1939+2134", period=0.001558, period_dot=1.05e-19,
        dm=71.0228, gl=57.51, gb=-0.29, distance_kpc=3.55,
        w50=16.0, s1400=13.2, timing_noise_ns=50.0,
    )

    profile = p.profile
    assert np.all(np.isfinite(profile)), "Profile contains NaN or Inf (kappa overflow not fixed)"
    assert np.all(profile >= 0), "Profile has negative values"
    assert abs(profile.sum() - 1.0) < 1e-10, f"Profile sum = {profile.sum():.8f}, expected 1.0"

    # Profile should be narrow — most of the weight in a small fraction of bins
    # For a pulse this narrow (1% of period), top 5% of bins should hold > 80% of weight
    n_bins = len(profile)
    top_bins = int(np.ceil(0.05 * n_bins))   # top 5% = ~6 bins out of 128
    sorted_vals = np.sort(profile)[::-1]
    top_fraction = sorted_vals[:top_bins].sum()
    assert top_fraction > 0.80, (
        f"Narrow pulse: top 5% of bins hold only {top_fraction:.1%} of weight "
        f"(expected >80%% — profile may be too broad due to kappa capping)"
    )

    # Verify kappa is not capped: the w50_phase for this pulsar
    w50_phase = p.w50 / (1e6 * p.period)  # 16 / 1558 ≈ 0.01027
    kappa_uncapped = 1.0 / (np.pi * w50_phase) ** 2
    assert kappa_uncapped > 500, (
        f"Test setup error: expected kappa > 500, got {kappa_uncapped:.1f}"
    )

    return (
        f"kappa≈{kappa_uncapped:.0f}, profile finite, "
        f"top 5% bins hold {top_fraction:.1%} of weight"
    )


@_test("pulsar.py different pulsars produce different profiles")
def test_pulsar_profiles_differ():
    from core.pulsar import Pulsar

    def make_pulsar(name):
        return Pulsar(
            name=name, period=0.005, period_dot=1e-20,
            dm=20.0, gl=45.0, gb=10.0, distance_kpc=1.0,
            w50=200.0, s1400=2.0, timing_noise_ns=100.0,
        )

    p1 = make_pulsar("J1713+0747")
    p2 = make_pulsar("J1909-3744")
    p3 = make_pulsar("J0437-4715")

    assert not np.allclose(p1.profile, p2.profile), "J1713 and J1909 profiles are identical"
    assert not np.allclose(p1.profile, p3.profile), "J1713 and J0437 profiles are identical"

    return "Three pulsars have distinct profiles"


@_test("pulsar.py timing_quality_score ranks correctly")
def test_pulsar_quality_ranking():
    from core.pulsar import Pulsar

    def make_pulsar(name, noise, flux):
        return Pulsar(
            name=name, period=0.005, period_dot=1e-20,
            dm=20.0, gl=45.0, gb=10.0, distance_kpc=1.0,
            w50=200.0, s1400=flux, timing_noise_ns=noise,
        )

    p_quiet = make_pulsar("quiet", noise=50.0, flux=5.0)
    p_noisy = make_pulsar("noisy", noise=500.0, flux=5.0)
    assert p_quiet.timing_quality_score() > p_noisy.timing_quality_score(), (
        "Quieter pulsar should have higher quality score"
    )

    p_bright = make_pulsar("bright", noise=100.0, flux=20.0)
    p_dim = make_pulsar("dim", noise=100.0, flux=0.2)
    assert p_bright.timing_quality_score() > p_dim.timing_quality_score(), (
        "Brighter pulsar should have higher quality score"
    )

    return f"quiet={p_quiet.timing_quality_score():.2f} > noisy={p_noisy.timing_quality_score():.2f}"


# ── Test 5: catalogue loading ─────────────────────────────────────────────────

@_test("catalogue.py loads ATNF cache successfully")
def test_catalogue_loads():
    from core.catalogue import Catalogue

    cat = Catalogue()
    pulsars = cat.all_pulsars

    assert len(pulsars) >= 100, (
        f"Expected ≥100 pulsars for Maximum tier, got {len(pulsars)}"
    )

    bad_period = [p.name for p in pulsars if p.period <= 0]
    assert not bad_period, f"Pulsars with non-positive period: {bad_period}"

    bad_dm = [p.name for p in pulsars if p.dm <= 0]
    assert not bad_dm, f"Pulsars with non-positive DM: {bad_dm}"

    bad_dist = [p.name for p in pulsars if p.distance_kpc <= 0]
    assert not bad_dist, f"Pulsars with non-positive distance: {bad_dist}"

    # All periods must be < 30ms (MSP definition consistent with refresh filter)
    non_msp = [p.name for p in pulsars if p.period >= 0.030]
    assert not non_msp, (
        f"Non-MSP pulsars (P≥30ms) in catalogue: {non_msp}"
    )

    scores = [p.timing_quality_score() for p in pulsars]
    assert scores == sorted(scores, reverse=True), (
        "Pulsars are not sorted by timing quality descending"
    )

    best = pulsars[0]
    detail = (
        f"Loaded {len(pulsars)} pulsars. "
        f"Best timing quality: {best.name} at {best.timing_noise_ns:.0f} ns"
    )
    print(f"        {detail}")
    return detail


@_test("catalogue.py no duplicate pulsar names")
def test_catalogue_no_duplicates():
    from core.catalogue import Catalogue

    cat = Catalogue()
    names = [p.name for p in cat.all_pulsars]
    seen = {}
    duplicates = []
    for i, name in enumerate(names):
        if name in seen:
            duplicates.append(f"{name} (ranks {seen[name]} and {i})")
        else:
            seen[name] = i

    assert not duplicates, (
        f"Duplicate pulsars found: {duplicates}"
    )
    return f"All {len(names)} pulsar names are unique"


@_test("catalogue.py get_top_n returns correct count and ordering")
def test_catalogue_get_top_n():
    from core.catalogue import Catalogue

    cat = Catalogue()

    for n in [5, 20, 40, 100]:
        subset = cat.get_top_n(n)
        assert len(subset) == n, f"get_top_n({n}) returned {len(subset)}"

        all_sorted = cat.all_pulsars
        for i, p in enumerate(subset):
            assert p.name == all_sorted[i].name, (
                f"Pulsar at rank {i} is {p.name}, expected {all_sorted[i].name}"
            )

    return "get_top_n works for n=5, 20, 40, 100"


@_test("catalogue.py all top-100 pulsars have valid profiles")
def test_catalogue_profiles():
    from core.catalogue import Catalogue

    cat = Catalogue()
    pulsars = cat.get_top_n(100)  # exercise the full Maximum tier set

    for p in pulsars:
        assert p.profile is not None, f"{p.name} has no profile"
        assert len(p.profile) == 128, f"{p.name} profile has wrong length"
        assert np.all(np.isfinite(p.profile)), (
            f"{p.name} profile contains NaN or Inf"
        )
        assert abs(p.profile.sum() - 1.0) < 1e-9, (
            f"{p.name} profile does not sum to 1 (sum={p.profile.sum():.8f})"
        )
        assert np.all(p.profile >= 0), f"{p.name} profile has negative values"

    return "All top-100 pulsars have valid finite normalised profiles"


# ── Runner ────────────────────────────────────────────────────────────────────

def run() -> bool:
    """Run all Phase 1 tests. Returns True if all pass."""
    results: list[tuple[str, bool, str]] = []  # fresh list each call — no accumulation

    print()
    print("═" * 55)
    print("  XNAV SIMULATOR — Phase 1 Foundation Tests")
    print("═" * 55)

    for test_fn in _TESTS:
        test_fn(results)

    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    failed = total - passed

    print()
    print("═" * 55)
    print(f"  Phase 1 result: {passed}/{total} passed", end="")
    if failed:
        print(f"  ({failed} FAILED)")
    else:
        print("  — ALL PASS ✓")
    print("═" * 55)
    print()

    return failed == 0


if __name__ == "__main__":
    ok = run()
    sys.exit(0 if ok else 1)
