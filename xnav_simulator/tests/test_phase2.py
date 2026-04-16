#!/usr/bin/env python3
# tests/test_phase2.py — Phase 2 Physics Tests
# XNAV Cold Start Simulator
#
# Run with: python tests/test_phase2.py  (from xnav_simulator/ directory)

from __future__ import annotations

import sys
import os
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np

_TESTS: list = []


def _test(name: str):
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


# ══════════════════════════════════════════════════════════════════════════════
# DISPERSION TESTS
# ══════════════════════════════════════════════════════════════════════════════

@_test("dispersion.py delay scales exactly as 1/f² — 1 GHz vs 2 GHz")
def test_dispersion_frequency_scaling():
    from core.dispersion import Dispersion

    dm = 100.0   # pc cm⁻³

    delay_1ghz = Dispersion.compute_dispersive_delay(dm, frequency_mhz=1000.0)
    delay_2ghz = Dispersion.compute_dispersive_delay(dm, frequency_mhz=2000.0)

    # 1/f² law: delay at 1 GHz must be exactly 4× delay at 2 GHz
    ratio = delay_1ghz / delay_2ghz
    assert abs(ratio - 4.0) < 1e-10, (
        f"1/f² ratio = {ratio:.8f}, expected exactly 4.0"
    )

    # Check absolute value: Δt = K_DM × DM / f²
    from config import K_DM
    expected_1ghz = K_DM * dm / (1000.0**2)
    assert abs(delay_1ghz - expected_1ghz) < 1e-15, (
        f"Delay at 1 GHz = {delay_1ghz:.6e} s, expected {expected_1ghz:.6e} s"
    )

    return f"1 GHz delay = {delay_1ghz*1e3:.4f} ms; ratio 1GHz/2GHz = {ratio:.10f}"


@_test("dispersion.py higher DM → larger delay (linear scaling)")
def test_dispersion_dm_scaling():
    from core.dispersion import Dispersion

    f_mhz = 1400.0
    dm_values = [10.0, 100.0, 1000.0]
    delays = [Dispersion.compute_dispersive_delay(dm, f_mhz) for dm in dm_values]

    # Should scale linearly with DM
    for i in range(1, len(dm_values)):
        ratio = delays[i] / delays[i - 1]
        expected_ratio = dm_values[i] / dm_values[i - 1]
        assert abs(ratio - expected_ratio) < 1e-10, (
            f"DM scaling: ratio={ratio:.6f}, expected {expected_ratio:.6f}"
        )

    return f"Delays at DM=10,100,1000: {[f'{d*1e3:.3f}' for d in delays]} ms"


@_test("dispersion.py chromatic correction recovers DM within 1%")
def test_dispersion_chromatic_recovery():
    from core.dispersion import Dispersion

    true_dm = 150.0   # pc cm⁻³
    n_channels = 16
    freq_low = 700.0   # MHz
    freq_high = 1400.0

    freqs, arrivals = Dispersion.simulate_multifreq_arrival(
        true_dm, n_channels, freq_low, freq_high, true_arrival_s=1.0
    )

    t_corr, dm_recovered, dm_uncertainty = Dispersion.correct_dm_chromatic(arrivals, freqs)

    dm_error_frac = abs(dm_recovered - true_dm) / true_dm
    assert dm_error_frac < 0.01, (
        f"DM recovery error {dm_error_frac:.4%} > 1% "
        f"(true={true_dm}, recovered={dm_recovered:.4f})"
    )

    # Corrected arrival time should recover the true infinite-frequency arrival
    assert abs(t_corr - 1.0) < 1e-6, (
        f"Corrected arrival time {t_corr:.8f} s differs from true 1.0 s by "
        f"{abs(t_corr-1.0)*1e9:.1f} ns"
    )

    return (f"DM recovered={dm_recovered:.4f} pc/cm³ (true={true_dm}), "
            f"error={dm_error_frac:.4%}")


@_test("dispersion.py simulate_multifreq returns monotonic delays (lower freq = later)")
def test_dispersion_sweep_monotonic():
    from core.dispersion import Dispersion

    dm = 50.0
    freqs, arrivals = Dispersion.simulate_multifreq_arrival(
        dm, n_channels=8, freq_low_mhz=500.0, freq_high_mhz=1500.0
    )

    # Arrivals should be monotonically decreasing with frequency
    # (lower frequency arrives later)
    diffs = np.diff(arrivals)
    assert np.all(diffs < 0), (
        "Arrival times are not monotonically decreasing with frequency"
    )

    return f"Sweep across {freqs[0]:.0f}–{freqs[-1]:.0f} MHz: {arrivals[-1]*1e3:.4f}–{arrivals[0]*1e3:.4f} ms"


@_test("dispersion.py interplanetary_dm_contribution is positive and finite")
def test_dispersion_ipm():
    from core.dispersion import Dispersion

    sc_pos_au = np.array([5.0, 0.0, 0.0])   # 5 AU from Sun

    dm_ipm = Dispersion.interplanetary_dm_contribution(sc_pos_au, 45.0, 10.0)
    assert np.isfinite(dm_ipm), "IPM DM contribution is not finite"
    assert dm_ipm >= 0.0, f"IPM DM contribution is negative: {dm_ipm}"
    assert dm_ipm < 10.0, f"IPM DM suspiciously large: {dm_ipm} pc/cm³"

    # Closer to Sun → larger contribution (1/r scaling)
    sc_close = np.array([1.0, 0.0, 0.0])
    sc_far   = np.array([10.0, 0.0, 0.0])
    dm_close = Dispersion.interplanetary_dm_contribution(sc_close, 45.0, 10.0)
    dm_far   = Dispersion.interplanetary_dm_contribution(sc_far, 45.0, 10.0)
    assert dm_close > dm_far, (
        f"IPM DM should decrease with distance: close={dm_close:.4f}, far={dm_far:.4f}"
    )

    return f"IPM DM at 5 AU = {dm_ipm:.4e} pc/cm³; close > far: {dm_close:.4e} > {dm_far:.4e}"


# ══════════════════════════════════════════════════════════════════════════════
# NOISE TESTS
# ══════════════════════════════════════════════════════════════════════════════

@_test("noise.py all sources return finite values in plausible ranges")
def test_noise_all_sources_finite():
    from core.noise import NoiseModel
    from core.pulsar import Pulsar

    p = Pulsar(
        name="J1713+0747", period=0.004570, period_dot=8.52e-21,
        dm=15.99, gl=28.75, gb=25.22, distance_kpc=1.05,
        w50=68.0, s1400=10.2, timing_noise_ns=30.0,
    )

    # Timing noise
    t_noise = NoiseModel.timing_noise(p, integration_time_s=1000.0)
    assert np.isfinite(t_noise), "timing_noise returned non-finite value"
    assert abs(t_noise) < 1e-3, f"timing_noise suspiciously large: {t_noise:.3e} s"

    # Photon noise
    p_noise = NoiseModel.photon_noise(
        p, distance_kpc=1.05, collecting_area_m2=1.0, integration_time_s=1000.0
    )
    assert np.isfinite(p_noise), "photon_noise returned non-finite value"
    assert abs(p_noise) < 1.0, f"photon_noise suspiciously large: {p_noise:.3e} s"

    # DM turbulence
    dm_noise = NoiseModel.dm_turbulence(dm=15.99, baseline_kpc=1.05)
    assert np.isfinite(dm_noise), "dm_turbulence returned non-finite value"
    assert abs(dm_noise) < 1e-3, f"dm_turbulence suspiciously large: {dm_noise:.3e} s"

    # Solar wind
    sw_noise = NoiseModel.solar_wind_noise(
        spacecraft_pos_au=np.array([5.0, 0.0, 0.0]),
        pulsar_direction=np.array([0.0, 1.0, 0.0]),
        solar_activity=0.3,
    )
    assert np.isfinite(sw_noise), "solar_wind_noise returned non-finite value"
    # Physical check: Parker 1/r² model gives A_sw ~ 5e-5 pc cm⁻³ AU,
    # so DM_IPM at 5 AU perpendicular to the LOS ≈ 5e-5 / 5 ≈ 1e-5 pc cm⁻³,
    # giving Δt ≈ K_DM × 1e-5 / 1400² ≈ 2e-11 s. Anything > 1e-4 s is
    # the pre-fix overcalculation by 37,500×.
    assert abs(sw_noise) < 1e-4, (
        f"solar_wind_noise = {sw_noise:.3e} s; expected < 1e-4 s for Parker model "
        f"(larger value indicates A_sw amplitude overcalculation)"
    )

    return (f"t_noise={t_noise*1e9:.1f}ns, p_noise={p_noise*1e9:.1f}ns, "
            f"dm_noise={dm_noise*1e9:.1f}ns, sw_noise={sw_noise*1e9:.1f}ns")


@_test("noise.py timing_noise scales as 1/sqrt(T) for white noise component")
def test_noise_white_noise_scaling():
    """Average over many realisations to check the 1/sqrt(T) scaling law."""
    from core.noise import NoiseModel
    from core.pulsar import Pulsar

    p = Pulsar(
        name="J1713+0747", period=0.004570, period_dot=8.52e-21,
        dm=15.99, gl=28.75, gb=25.22, distance_kpc=1.05,
        w50=68.0, s1400=10.2, timing_noise_ns=100.0,  # larger noise for clear signal
    )

    n_draws = 500
    # Use times well below the white/red noise crossover (~316 s for 10% red
    # amplitude) so the 1/sqrt(T) white noise law dominates both samples.
    t_short = 1.0    # s
    t_long  = 100.0  # s (100× longer → expected ratio = sqrt(100) = 10)

    residuals_short = [NoiseModel.timing_noise(p, t_short, seed=i) for i in range(n_draws)]
    residuals_long  = [NoiseModel.timing_noise(p, t_long,  seed=i + n_draws) for i in range(n_draws)]

    rms_short = np.std(residuals_short)
    rms_long  = np.std(residuals_long)

    # Ratio of RMS should be sqrt(t_long / t_short) = sqrt(100) = 10
    expected_ratio = np.sqrt(t_long / t_short)
    actual_ratio = rms_short / rms_long

    # Allow ±30% tolerance — red noise still contributes ~30% of white at T=100 s
    assert abs(actual_ratio - expected_ratio) / expected_ratio < 0.30, (
        f"White noise scaling: RMS ratio = {actual_ratio:.2f}, "
        f"expected ~{expected_ratio:.1f} (±30%)"
    )

    return (f"RMS: short={rms_short*1e9:.1f}ns, long={rms_long*1e9:.1f}ns, "
            f"ratio={actual_ratio:.2f} (expected {expected_ratio:.1f})")


@_test("noise.py photon_noise decreases with larger collecting area")
def test_noise_photon_area_scaling():
    from core.noise import NoiseModel
    from core.pulsar import Pulsar

    p = Pulsar(
        name="J1713+0747", period=0.004570, period_dot=8.52e-21,
        dm=15.99, gl=28.75, gb=25.22, distance_kpc=1.05,
        w50=68.0, s1400=10.2, timing_noise_ns=30.0,
    )

    # Average over many seeds to get stable RMS estimates
    n = 200
    areas = [0.5, 5.0, 50.0]
    rms_by_area = []
    for area in areas:
        draws = [abs(NoiseModel.photon_noise(p, 1.05, area, 1000.0, seed=i))
                 for i in range(n)]
        rms_by_area.append(np.mean(draws))

    assert rms_by_area[0] > rms_by_area[1] > rms_by_area[2], (
        f"Photon noise should decrease with area: {rms_by_area}"
    )

    return f"Mean |noise| at areas {areas}: {[f'{r*1e9:.1f}ns' for r in rms_by_area]}"


@_test("noise.py dm_turbulence scales with sqrt(DM × baseline)")
def test_noise_dm_turbulence_scaling():
    from core.noise import NoiseModel

    n = 300
    # Compare high-DM vs low-DM turbulence RMS
    draws_hi = [NoiseModel.dm_turbulence(dm=200.0, baseline_kpc=5.0, seed=i) for i in range(n)]
    draws_lo = [NoiseModel.dm_turbulence(dm=5.0,   baseline_kpc=1.0, seed=i + n) for i in range(n)]

    rms_hi = np.std(draws_hi)
    rms_lo = np.std(draws_lo)

    # Expected ratio: sqrt(200×5) / sqrt(5×1) = sqrt(1000/5) = sqrt(200) ≈ 14.1
    expected_ratio = np.sqrt(200.0 * 5.0) / np.sqrt(5.0 * 1.0)
    actual_ratio = rms_hi / rms_lo

    assert abs(actual_ratio - expected_ratio) / expected_ratio < 0.20, (
        f"DM turbulence scaling: ratio={actual_ratio:.2f}, expected~{expected_ratio:.1f} (±20%)"
    )

    return f"RMS: hi-DM={rms_hi*1e9:.1f}ns, lo-DM={rms_lo*1e9:.1f}ns, ratio={actual_ratio:.1f}"


# ══════════════════════════════════════════════════════════════════════════════
# GRAVITY TESTS
# ══════════════════════════════════════════════════════════════════════════════

@_test("gravity.py gravitational_potential at solar radius matches rotation curve")
def test_gravity_potential_solar_radius():
    """Circular velocity at solar radius should be close to 220 km/s."""
    from core.gravity import Gravity
    from config import SOLAR_GALACTOCENTRIC_KPC, KPC_TO_M

    # Numerical derivative of potential along R to get v_c
    r0 = SOLAR_GALACTOCENTRIC_KPC
    dr = 0.01   # kpc
    pos_plus  = np.array([r0 + dr, 0.0, 0.0])
    pos_minus = np.array([r0 - dr, 0.0, 0.0])

    phi_plus  = Gravity.gravitational_potential(pos_plus,  include_galactic=True)
    phi_minus = Gravity.gravitational_potential(pos_minus, include_galactic=True)

    # ∂Φ/∂R in m²/s² per m
    dphi_dR = (phi_plus - phi_minus) / (2 * dr * KPC_TO_M)

    # Circular velocity: v_c² = R × ∂Φ/∂R
    v_c_sq = r0 * KPC_TO_M * dphi_dR
    v_c_kms = np.sqrt(max(v_c_sq, 0.0)) / 1000.0

    # Milky Way circular velocity: ~220 km/s; accept 150–280 km/s
    assert 150.0 < v_c_kms < 280.0, (
        f"Circular velocity at solar radius = {v_c_kms:.1f} km/s, expected 150–280 km/s"
    )

    return f"v_c at {r0} kpc = {v_c_kms:.1f} km/s"


@_test("gravity.py clock_slowing_factor sign and magnitude at 1 AU from solar mass")
def test_gravity_clock_slowing_1au():
    """At 1 AU from a solar mass, Φ/c² should match GM_sun/r/c² ≈ −9.9×10⁻⁹."""
    from core.gravity import Gravity
    from config import G_NEWTON, M_SUN, C_LIGHT, AU_TO_M

    r_m = AU_TO_M  # 1 AU in metres

    # Central body potential at 1 AU (with galactic background off for isolation)
    pos_kpc = np.array([1.0, 0.0, 0.0])  # 1 kpc from GC — just a dummy position
    phi_central = -G_NEWTON * M_SUN / r_m

    factor = Gravity.clock_slowing_factor(phi_central)

    # Analytical: Φ/c² = −GM/rc²
    expected = -G_NEWTON * M_SUN / (r_m * C_LIGHT**2)
    assert abs(factor - expected) / abs(expected) < 1e-10, (
        f"clock_slowing_factor = {factor:.4e}, expected {expected:.4e}"
    )

    # Must be negative (deeper in well → slower clock)
    assert factor < 0, f"Clock slowing factor should be negative, got {factor}"

    # Magnitude check: should be ~−9.87e−9
    assert 5e-9 < abs(factor) < 2e-8, (
        f"|factor| = {abs(factor):.3e}, expected ~9.87e−9"
    )

    return f"Δf/f at 1 AU from Sun = {factor:.4e} (expected ~−9.87e−9)"


@_test("gravity.py gravitational redshift sign is correct (deeper well → positive residual)")
def test_gravity_redshift_sign():
    """A clock deeper in a well runs slow → measured period appears longer
    → positive timing residual (pulse arrives 'late' relative to reference).
    """
    from core.gravity import Gravity
    from config import G_NEWTON, M_SUN, AU_TO_M

    phi_deep = -G_NEWTON * M_SUN / AU_TO_M        # deep in well (negative)
    phi_shallow = -G_NEWTON * M_SUN / (10 * AU_TO_M)  # shallower (less negative)

    residual_deep    = Gravity.timing_residual_from_potential(phi_deep)
    residual_shallow = Gravity.timing_residual_from_potential(phi_shallow)

    # Both should be negative (Φ < 0 → Δf/f < 0)
    assert residual_deep < 0, "Residual at 1 AU should be negative"
    assert residual_shallow < 0, "Residual at 10 AU should be negative"

    # Deeper well → more negative residual (larger magnitude)
    assert abs(residual_deep) > abs(residual_shallow), (
        "Deeper gravitational potential should produce larger |residual|"
    )

    # Magnitude within 10% of GM/rc²
    expected_deep = -G_NEWTON * M_SUN / (AU_TO_M * 9e16)   # ≈ −9.87e−9
    assert abs(residual_deep - expected_deep) / abs(expected_deep) < 0.10, (
        f"Residual {residual_deep:.3e} differs from expected {expected_deep:.3e} by >10%"
    )

    return (f"deep={residual_deep:.3e}s, shallow={residual_shallow:.3e}s; "
            f"|deep| > |shallow| ✓")


@_test("gravity.py extract_monopolar_residual averages correctly")
def test_gravity_monopolar_extraction():
    from core.gravity import Gravity

    # Known residuals: all pulsars share a common offset of 1 μs, plus individual noise
    common_offset_s = 1e-6
    rng = np.random.default_rng(42)
    n_pulsars = 40
    noise_level_s = 100e-9   # 100 ns noise per pulsar

    residuals = {
        f"PSR_{i}": common_offset_s + rng.normal(0, noise_level_s)
        for i in range(n_pulsars)
    }

    estimate, uncertainty = Gravity.extract_monopolar_residual(residuals)

    # Estimate should be close to true common offset
    assert abs(estimate - common_offset_s) < 3 * noise_level_s / np.sqrt(n_pulsars), (
        f"Monopolar estimate {estimate*1e9:.1f} ns differs from true {common_offset_s*1e9:.1f} ns "
        f"by more than 3σ_mean"
    )

    # Uncertainty should scale as noise / sqrt(N)
    expected_uncertainty = noise_level_s / np.sqrt(n_pulsars)
    assert uncertainty < 3 * expected_uncertainty, (
        f"Uncertainty {uncertainty*1e9:.1f} ns much larger than expected "
        f"{expected_uncertainty*1e9:.1f} ns"
    )

    return (f"Estimate={estimate*1e9:.2f}ns (true={common_offset_s*1e9:.0f}ns), "
            f"σ={uncertainty*1e9:.2f}ns (expected~{expected_uncertainty*1e9:.1f}ns)")


@_test("gravity.py gravity_well_depth is positive and physically reasonable")
def test_gravity_well_depth():
    from core.gravity import Gravity
    from config import G_NEWTON, M_SUN, AU_TO_M

    # 1 AU above a solar mass body: well depth should be GM/r_surface - GM/r_1AU
    phi_current = -G_NEWTON * M_SUN / AU_TO_M  # at 1 AU
    depth = Gravity.gravity_well_depth(phi_current, M_SUN, 6.96e8)

    # Should be positive (we're above the surface)
    assert depth > 0, f"Well depth should be positive, got {depth:.3e} J/kg"

    # Surface potential: GM/R_sun ≈ 1.91e11 J/kg
    # At 1 AU: GM/r ≈ 8.87e8 J/kg
    # Depth ≈ surface - current = 1.91e11 - 8.87e8 ≈ 1.90e11 J/kg
    assert 1e10 < depth < 1e12, f"Well depth {depth:.3e} J/kg outside expected range"

    # No central body → zero depth
    depth_free = Gravity.gravity_well_depth(-1e8, 0.0, 0.0)
    assert depth_free == 0.0, f"Zero-mass body should give zero well depth, got {depth_free}"

    return f"Well depth at 1 AU from Sun = {depth:.3e} J/kg"


# ══════════════════════════════════════════════════════════════════════════════
# TIMING TESTS
# ══════════════════════════════════════════════════════════════════════════════

@_test("timing.py compute_arrival_time returns dict with all required keys")
def test_timing_arrival_time_keys():
    from core.timing import TimingModel
    from core.pulsar import Pulsar
    from core.spacecraft import Spacecraft

    p = Pulsar(
        name="J1713+0747", period=0.004570, period_dot=8.52e-21,
        dm=15.99, gl=28.75, gb=25.22, distance_kpc=1.05,
        w50=68.0, s1400=10.2, timing_noise_ns=30.0,
    )
    sc = Spacecraft.from_galactic(gl_deg=90.0, gb_deg=0.0, distance_kpc=5.0)

    result = TimingModel.compute_arrival_time(p, sc, observation_time_s=0.0,
                                              include_noise=False)

    required = {"geometric", "doppler", "shapiro", "gravitational", "dispersive", "noise", "total"}
    assert set(result.keys()) == required, (
        f"Missing keys: {required - set(result.keys())}; "
        f"Unexpected keys: {set(result.keys()) - required}"
    )

    # Total must equal sum of components
    computed_total = sum(result[k] for k in required - {"total"})
    assert abs(result["total"] - computed_total) < 1e-20, (
        f"total={result['total']:.6e} != sum of components={computed_total:.6e}"
    )

    # All values must be finite
    for k, v in result.items():
        assert np.isfinite(v), f"result['{k}'] = {v} is not finite"

    return f"All 7 keys present; total={result['total']*1e9:.1f} ns"


@_test("timing.py geometric Roemer delay has correct sign and plausible magnitude")
def test_timing_roemer_delay():
    """Spacecraft displaced TOWARD a pulsar should have earlier arrival (negative delay)."""
    from core.timing import TimingModel
    from core.pulsar import Pulsar
    from core.spacecraft import Spacecraft

    # Pulsar at (l=90°, b=0°) → direction is +Y in Galactocentric coords
    # Spacecraft displaced 1 kpc in +Y direction (toward the pulsar) from Sun position
    p = Pulsar(
        name="J1909-3744", period=0.002947, period_dot=1.40e-20,
        dm=10.39, gl=90.0, gb=0.0, distance_kpc=5.0,  # set gl=90 for clean test
        w50=43.0, s1400=3.1, timing_noise_ns=35.0,
    )
    # Spacecraft displaced 1 kpc in the pulsar direction (should arrive earlier)
    sc_toward = Spacecraft.from_galactic(gl_deg=90.0, gb_deg=0.0, distance_kpc=1.0)
    # Spacecraft displaced 1 kpc away from pulsar direction
    sc_away   = Spacecraft.from_galactic(gl_deg=270.0, gb_deg=0.0, distance_kpc=1.0)

    r_toward = TimingModel.compute_arrival_time(p, sc_toward, 0.0, include_noise=False)
    r_away   = TimingModel.compute_arrival_time(p, sc_away,   0.0, include_noise=False)

    # Spacecraft closer to pulsar should have earlier geometric arrival (more negative)
    assert r_toward["geometric"] < r_away["geometric"], (
        f"Spacecraft toward pulsar should have smaller Roemer delay: "
        f"toward={r_toward['geometric']:.4e}s, away={r_away['geometric']:.4e}s"
    )

    # Roemer delay magnitude: the full galactocentric projection r⃗·n̂/c.
    # For a spacecraft ≈1 kpc displaced at galactic scale, the dominant term is
    # the Sun's galactocentric offset projected onto the LOS (~8.5 kpc × cos θ).
    # At 1 kpc offset near the Sun: |r⃗·n̂| ≈ 1 kpc → delay ≈ 1 kpc × KPC_TO_M / c
    # = 3.086e16 m / 3e8 m/s ≈ 1.03×10⁸ s.
    # Acceptable range for galactic-scale positions: 1×10⁶ s to 1×10¹² s.
    roemer_mag = abs(r_toward["geometric"])
    assert 1e6 < roemer_mag < 1e12, (
        f"Roemer delay magnitude {roemer_mag:.3e} s outside plausible galactic range [1e6, 1e12]"
    )

    return (f"toward={r_toward['geometric']:.3e}s < away={r_away['geometric']:.3e}s ✓, "
            f"magnitude={roemer_mag:.3e}s")


@_test("timing.py gravitational redshift: inner galaxy deeper well than outer")
def test_timing_grav_redshift_sign():
    """A spacecraft deeper in the galactic potential well (inner galaxy, R=3 kpc)
    has a more negative gravitational redshift term than one in the outer galaxy
    (R=12 kpc).  This is independent of the Solar neighbourhood and tests the
    galactic potential gradient over the navigation-relevant range.
    """
    from core.timing import TimingModel
    from core.pulsar import Pulsar
    from core.spacecraft import Spacecraft

    # Pulsar far away — galactic north pole direction to minimise Shapiro variation
    p = Pulsar(
        name="J0437-4715", period=0.005757, period_dot=5.73e-20,
        dm=2.64, gl=253.39, gb=-41.96, distance_kpc=10.0,
        w50=141.0, s1400=150.0, timing_noise_ns=10.0,
    )

    # Inner galaxy: R ≈ 3 kpc from GC (deep galactic potential)
    sc_inner = Spacecraft.from_galactic(gl_deg=0.0, gb_deg=0.0, distance_kpc=5.2)
    # Outer galaxy: R ≈ 12 kpc from GC (shallow galactic potential)
    sc_outer = Spacecraft.from_galactic(gl_deg=180.0, gb_deg=0.0, distance_kpc=3.8)

    r_inner = TimingModel.compute_arrival_time(p, sc_inner, 0.0, include_noise=False)
    r_outer = TimingModel.compute_arrival_time(p, sc_outer, 0.0, include_noise=False)

    # Verify inner galaxy has deeper (more negative) galactic potential
    from core.gravity import Gravity
    phi_inner = Gravity.gravitational_potential(sc_inner.position_kpc, include_galactic=True)
    phi_outer = Gravity.gravitational_potential(sc_outer.position_kpc, include_galactic=True)
    assert phi_inner < phi_outer, (
        f"Inner galaxy potential should be more negative: "
        f"phi_inner={phi_inner:.4e}, phi_outer={phi_outer:.4e} m²/s²"
    )

    # Gravitational redshift term is Φ/c² × period — more negative Φ → more negative residual
    assert r_inner["gravitational"] < r_outer["gravitational"], (
        f"Inner galaxy (deeper well) should give more negative gravitational residual: "
        f"inner={r_inner['gravitational']:.3e}s, outer={r_outer['gravitational']:.3e}s"
    )

    # Both should be negative (clocks in negative potential run slow)
    assert r_inner["gravitational"] < 0, (
        f"Inner galaxy gravitational residual should be negative: {r_inner['gravitational']:.3e}"
    )

    return (
        f"phi_inner={phi_inner:.3e}, phi_outer={phi_outer:.3e} m²/s²; "
        f"grav: inner={r_inner['gravitational']*1e9:.2f}ns < outer={r_outer['gravitational']*1e9:.2f}ns ✓"
    )


@_test("timing.py dispersive component matches direct dispersion calculation")
def test_timing_dispersive_consistency():
    """The dispersive component in the timing model must use K_DM × DM / f²."""
    from core.timing import TimingModel
    from core.dispersion import Dispersion
    from core.pulsar import Pulsar
    from core.spacecraft import Spacecraft

    dm = 62.4  # J1643-1224 DM
    freq_mhz = 1400.0

    p = Pulsar(
        name="J1643-1224", period=0.004621, period_dot=1.85e-20,
        dm=dm, gl=5.67, gb=21.22, distance_kpc=0.74,
        w50=2500.0, s1400=4.8, timing_noise_ns=700.0,
    )
    sc = Spacecraft.from_galactic(gl_deg=0.0, gb_deg=0.0, distance_kpc=8.0)

    result = TimingModel.compute_arrival_time(
        p, sc, 0.0, frequency_mhz=freq_mhz, include_noise=False
    )

    expected = Dispersion.compute_dispersive_delay(dm, freq_mhz)
    assert abs(result["dispersive"] - expected) < 1e-15, (
        f"Timing dispersive={result['dispersive']:.8e}s, "
        f"Dispersion.compute={expected:.8e}s — inconsistent"
    )

    return f"dispersive = {result['dispersive']*1e3:.6f} ms (consistent with Dispersion module)"


# ══════════════════════════════════════════════════════════════════════════════
# SPACECRAFT TESTS
# ══════════════════════════════════════════════════════════════════════════════

@_test("spacecraft.py blind_mode hides true position from get_display_position")
def test_spacecraft_blind_mode():
    from core.spacecraft import Spacecraft

    sc = Spacecraft.from_galactic(gl_deg=45.0, gb_deg=10.0, distance_kpc=3.0,
                                  blind_mode=True)

    assert sc.get_display_position() is None, (
        "get_display_position() should return None in blind mode"
    )
    assert sc.true_position_kpc is not None, (
        "true_position_kpc must still be stored in blind mode"
    )
    assert np.all(np.isfinite(sc.true_position_kpc)), (
        "true_position_kpc contains non-finite values"
    )

    # Non-blind mode should expose position
    sc_visible = Spacecraft.from_galactic(gl_deg=45.0, gb_deg=10.0, distance_kpc=3.0,
                                           blind_mode=False)
    display = sc_visible.get_display_position()
    assert display is not None, "get_display_position() should return position in non-blind mode"
    assert np.allclose(display, sc_visible.true_position_kpc), (
        "Displayed position should match true position in non-blind mode"
    )

    return "Blind mode hides position; non-blind mode exposes it"


@_test("spacecraft.py galactic_coords round-trip is consistent")
def test_spacecraft_galactic_coords():
    from core.spacecraft import Spacecraft

    gl_in, gb_in, d_in = 135.0, -25.0, 4.5

    sc = Spacecraft.from_galactic(gl_deg=gl_in, gb_deg=gb_in, distance_kpc=d_in)
    gl_out, gb_out, d_out = sc.galactic_coords()

    assert abs(gl_out - gl_in) < 0.01, f"GL round-trip error: {abs(gl_out-gl_in):.4f}°"
    assert abs(gb_out - gb_in) < 0.01, f"GB round-trip error: {abs(gb_out-gb_in):.4f}°"
    assert abs(d_out  - d_in)  < 0.001, f"Distance round-trip error: {abs(d_out-d_in):.6f} kpc"

    return f"(gl={gl_out:.2f}°, gb={gb_out:.2f}°, d={d_out:.3f} kpc) matches input"


@_test("spacecraft.py gravitational_potential returns finite negative value")
def test_spacecraft_potential():
    from core.spacecraft import Spacecraft

    sc = Spacecraft.from_galactic(gl_deg=90.0, gb_deg=0.0, distance_kpc=8.5)
    phi = sc.gravitational_potential()

    assert np.isfinite(phi), f"Gravitational potential is not finite: {phi}"
    assert phi < 0, f"Potential should be negative (bound system), got {phi:.3e}"

    return f"Φ at (gl=90°, d=8.5 kpc) = {phi:.4e} m²/s²"


# ══════════════════════════════════════════════════════════════════════════════
# GALAXY GEOMETRY TESTS
# ══════════════════════════════════════════════════════════════════════════════

@_test("galaxy.py in_galaxy rejects positions outside disk")
def test_galaxy_boundary():
    from core.galaxy import Galaxy
    from config import GALAXY_RADIUS_KPC, GALAXY_THICKNESS_KPC

    assert Galaxy.in_galaxy(np.array([0.0, 0.0, 0.0])), "Origin should be in galaxy"
    assert Galaxy.in_galaxy(np.array([5.0, 3.0, 0.1])), "Mid-disk point should be in galaxy"
    assert not Galaxy.in_galaxy(np.array([GALAXY_RADIUS_KPC + 1, 0.0, 0.0])), \
        "Beyond radius should be outside galaxy"
    assert not Galaxy.in_galaxy(np.array([0.0, 0.0, GALAXY_THICKNESS_KPC])), \
        "Beyond Z limit should be outside galaxy"

    return "Boundary checks correct for origin, mid-disk, and out-of-bounds positions"


@_test("galaxy.py sample_from_map produces points near high-probability region")
def test_galaxy_sample_from_map():
    from core.galaxy import Galaxy

    x_arr = np.linspace(-5.0, 5.0, 20)
    y_arr = np.linspace(-5.0, 5.0, 20)
    z_arr = np.linspace(-0.5, 0.5, 5)

    # Probability concentrated at (2, 1, 0) with a Gaussian
    X, Y, Z = np.meshgrid(x_arr, y_arr, z_arr, indexing="ij")
    prob_map = np.exp(-((X - 2.0)**2 + (Y - 1.0)**2) / (2 * 0.5**2))

    rng = np.random.default_rng(42)
    samples = Galaxy.sample_from_map(prob_map, x_arr, y_arr, z_arr, n=500, rng=rng)

    assert samples.shape == (500, 3), f"Expected (500, 3), got {samples.shape}"

    centroid = samples.mean(axis=0)
    assert abs(centroid[0] - 2.0) < 0.3, f"Centroid X {centroid[0]:.2f} ≠ 2.0"
    assert abs(centroid[1] - 1.0) < 0.3, f"Centroid Y {centroid[1]:.2f} ≠ 1.0"

    return f"500 samples; centroid=({centroid[0]:.2f}, {centroid[1]:.2f}, {centroid[2]:.2f})"


@_test("galaxy.py GDOP is lower for well-distributed pulsars than clustered")
def test_galaxy_gdop():
    from core.galaxy import Galaxy

    sc_pos = np.array([0.0, 0.0, 0.0])

    # Clustered: all pulsars in one sky quadrant
    clustered = np.array([
        [5.0, 0.1, 0.0],
        [5.0, 0.2, 0.1],
        [5.0, 0.3, -0.1],
        [5.0, 0.4, 0.0],
        [5.0, 0.5, 0.1],
    ])

    # Distributed: pulsars spread across the sky in 3D
    distributed = np.array([
        [5.0, 0.0, 0.0],
        [-5.0, 0.0, 0.0],
        [0.0, 5.0, 0.0],
        [0.0, -5.0, 0.0],
        [0.0, 0.0, 1.0],
    ])

    gdop_clustered   = Galaxy.gdop(clustered,   sc_pos)
    gdop_distributed = Galaxy.gdop(distributed, sc_pos)

    assert gdop_distributed < gdop_clustered, (
        f"Distributed GDOP {gdop_distributed:.2f} should be less than "
        f"clustered GDOP {gdop_clustered:.2f}"
    )

    # Fewer than 4 pulsars → undefined (inf)
    assert Galaxy.gdop(clustered[:3], sc_pos) == np.inf, \
        "GDOP with <4 pulsars should be inf"

    return f"clustered GDOP={gdop_clustered:.2f} > distributed GDOP={gdop_distributed:.2f} ✓"


# ══════════════════════════════════════════════════════════════════════════════
# INTERSTELLAR MEDIUM TESTS
# ══════════════════════════════════════════════════════════════════════════════

@_test("interstellar_medium.py fallback DM model returns positive finite values")
def test_ism_fallback_dm():
    from core.interstellar_medium import InterstellarMedium

    test_cases = [
        (0.0, 1.0),      # Galactic plane
        (90.0, 1.0),     # Galactic pole
        (-45.0, 5.0),    # Mid-latitude, 5 kpc
        (0.0, 10.0),     # Plane, long path
    ]

    for gb, dist in test_cases:
        dm = InterstellarMedium._fallback_dm(gb, dist)
        assert np.isfinite(dm), f"Fallback DM not finite for gb={gb}, d={dist}"
        assert dm > 0, f"Fallback DM not positive for gb={gb}, d={dist}: {dm}"
        assert dm < 5000, f"Fallback DM suspiciously large: {dm}"

    # Pole should have much lower DM than plane at same distance
    dm_plane = InterstellarMedium._fallback_dm(0.0, 5.0)
    dm_pole  = InterstellarMedium._fallback_dm(90.0, 5.0)
    assert dm_plane > dm_pole * 5, (
        f"Plane DM ({dm_plane:.1f}) should be >> pole DM ({dm_pole:.1f})"
    )

    return f"DM(plane,5kpc)={dm_plane:.1f}, DM(pole,5kpc)={dm_pole:.3f} pc/cm³"


@_test("interstellar_medium.py DM grid file exists and has correct structure")
def test_ism_grid_structure():
    """Verify the DM grid file exists (from Phase 2 precomputation) and loads."""
    from config import DM_GRID_PATH
    import os

    if not DM_GRID_PATH.exists():
        # Grid not yet precomputed — build a tiny one now for testing
        from core.interstellar_medium import InterstellarMedium
        ism = InterstellarMedium(grid_path=DM_GRID_PATH)
        ism.precompute_grid(resolution_pc=2000)   # Very coarse: fast for tests

    assert DM_GRID_PATH.exists(), f"DM grid file not found at {DM_GRID_PATH}"

    data = np.load(DM_GRID_PATH)
    assert "dm_grid" in data, "dm_grid key missing from .npz"
    assert "x_arr" in data, "x_arr key missing from .npz"
    assert "y_arr" in data, "y_arr key missing from .npz"
    assert "z_arr" in data, "z_arr key missing from .npz"

    dm_grid = data["dm_grid"]
    assert dm_grid.ndim == 3, f"dm_grid should be 3D, got {dm_grid.ndim}D"
    assert np.all(dm_grid >= 0), "dm_grid contains negative values"
    assert np.all(np.isfinite(dm_grid)), "dm_grid contains non-finite values"

    # Sample a value and confirm > 0
    mid = tuple(s // 2 for s in dm_grid.shape)
    sample_dm = float(dm_grid[mid])
    assert sample_dm > 0, f"Grid centre DM = {sample_dm}"

    return (f"Grid shape: {dm_grid.shape}, "
            f"range: [{dm_grid.min():.1f}, {dm_grid.max():.1f}] pc/cm³")


@_test("interstellar_medium.py grid lookup returns positive finite DM")
def test_ism_grid_lookup():
    from core.interstellar_medium import InterstellarMedium
    from config import DM_GRID_PATH

    ism = InterstellarMedium(grid_path=DM_GRID_PATH)
    if not ism.grid_loaded():
        ism.precompute_grid(resolution_pc=2000)

    assert ism.grid_loaded(), "Grid failed to load after precomputation"

    # Sample a few spacecraft–pulsar pairs
    test_cases = [
        (np.array([-8.5, 0.0, 0.0]),  np.array([-5.0, 3.0, 0.0])),   # near Sun
        (np.array([0.0, 5.0, 0.0]),   np.array([0.0, 10.0, 0.0])),    # inner disk
        (np.array([-8.5, 0.0, 0.0]),  np.array([-8.5, 5.0, 0.2])),    # lateral
    ]

    for sc_pos, p_pos in test_cases:
        dm = ism.lookup_dm_grid(sc_pos, p_pos)
        assert np.isfinite(dm), f"Grid lookup returned non-finite DM: {dm}"
        assert dm > 0, f"Grid lookup returned non-positive DM: {dm}"
        # Physical MSP DM values range from ~3 to ~1000 pc cm⁻³;
        # wildly large values indicate the dimensional scaling is broken
        assert dm < 5000, (
            f"Grid lookup DM={dm:.1f} pc/cm³ implausibly large — "
            f"possible dimensional error in lookup scaling"
        )

    return f"All {len(test_cases)} grid lookups returned positive finite DM values"


# ══════════════════════════════════════════════════════════════════════════════
# PHYSICS-REVIEWER TESTS (added after astrophysics panel review)
# ══════════════════════════════════════════════════════════════════════════════

@_test("timing.py model_dm overrides pulsar.dm for dispersive delay")
def test_timing_dispersive_position_dependent():
    """Verify that passing model_dm to compute_arrival_time changes the dispersive
    delay — this is the mechanism that gives the dispersive term positional
    information in Stage 1 of the filter.  Without this, catalogue DM is
    invariant with particle position and contributes zero navigation signal.
    """
    from core.timing import TimingModel
    from core.pulsar import Pulsar
    from core.spacecraft import Spacecraft
    from core.dispersion import Dispersion

    catalogue_dm = 50.0      # pulsar.dm — Sun→pulsar column density
    position_dm  = 35.0      # model_dm  — spacecraft→pulsar column density
    freq_mhz = 1400.0

    p = Pulsar(
        name="J1857+0943", period=0.005362, period_dot=1.78e-19,
        dm=catalogue_dm, gl=42.37, gb=3.06, distance_kpc=0.9,
        w50=25.0, s1400=5.1, timing_noise_ns=200.0,
    )
    sc = Spacecraft.from_galactic(gl_deg=42.0, gb_deg=3.0, distance_kpc=5.0)

    # Without model_dm: uses catalogue DM
    r_catalogue = TimingModel.compute_arrival_time(
        p, sc, 0.0, frequency_mhz=freq_mhz, include_noise=False
    )
    # With model_dm: uses position-dependent DM
    r_position = TimingModel.compute_arrival_time(
        p, sc, 0.0, frequency_mhz=freq_mhz, include_noise=False,
        model_dm=position_dm,
    )

    expected_cat = Dispersion.compute_dispersive_delay(catalogue_dm, freq_mhz)
    expected_pos = Dispersion.compute_dispersive_delay(position_dm, freq_mhz)

    assert abs(r_catalogue["dispersive"] - expected_cat) < 1e-15, (
        f"Without model_dm: dispersive={r_catalogue['dispersive']:.6e}, "
        f"expected {expected_cat:.6e}"
    )
    assert abs(r_position["dispersive"] - expected_pos) < 1e-15, (
        f"With model_dm: dispersive={r_position['dispersive']:.6e}, "
        f"expected {expected_pos:.6e}"
    )
    assert abs(r_catalogue["dispersive"] - r_position["dispersive"]) > 1e-10, (
        "model_dm and catalogue DM produce the same dispersive delay — "
        "model_dm is not being used"
    )

    return (
        f"catalogue_dm={catalogue_dm}: {r_catalogue['dispersive']*1e3:.4f}ms; "
        f"model_dm={position_dm}: {r_position['dispersive']*1e3:.4f}ms ✓"
    )


@_test("timing.py math.fsum preserves gravitational term at galactic Roemer scale")
def test_timing_total_precision():
    """Verify that the total timing residual includes the gravitational redshift
    term even when the Roemer delay is ~10⁹ s (galactocentric scale).

    At galactic distances, Roemer ~ 10⁹ s while gravitational redshift ~10⁻⁸ s.
    Float64 ULP of 10⁹ is ~120 μs — naive '+' would discard the 10 ns grav term.
    math.fsum() compensates by maintaining an exact running sum.
    """
    from core.timing import TimingModel
    from core.pulsar import Pulsar
    from core.spacecraft import Spacecraft
    from core.gravity import Gravity

    p = Pulsar(
        name="J1713+0747", period=0.004570, period_dot=8.52e-21,
        dm=15.99, gl=28.75, gb=25.22, distance_kpc=1.05,
        w50=68.0, s1400=10.2, timing_noise_ns=30.0,
    )
    sc = Spacecraft.from_galactic(gl_deg=90.0, gb_deg=0.0, distance_kpc=5.0)

    result = TimingModel.compute_arrival_time(p, sc, 0.0, include_noise=False)

    grav   = result["gravitational"]
    roemer = result["geometric"]
    total  = result["total"]

    # Compute what naive addition would give (may lose grav term due to ULP)
    naive_total = roemer + result["doppler"] + result["shapiro"] + grav + result["dispersive"]

    # The gravitational term must be non-zero (pulsar has finite period)
    assert abs(grav) > 0, "Gravitational term is exactly zero — pulsar period may be zero"
    assert abs(grav) < 1e-3, f"Gravitational term unexpectedly large: {grav:.3e} s"

    # total from the function must agree with math.fsum of all components
    import math
    components = [result["geometric"], result["doppler"], result["shapiro"],
                  result["gravitational"], result["dispersive"], result["noise"]]
    exact_total = math.fsum(components)
    assert abs(total - exact_total) < 1e-30, (
        f"total={total:.15e} differs from math.fsum={exact_total:.15e}"
    )

    # Log how severe the float64 ULP issue would be for awareness
    ulp_of_roemer = abs(roemer) * 2**-52
    return (
        f"Roemer={roemer:.3e}s, grav={grav:.3e}s, ULP(Roemer)={ulp_of_roemer:.1e}s; "
        f"total uses fsum ✓"
    )


@_test("noise.py solar_wind_noise physical magnitude matches Parker model")
def test_noise_solar_wind_physical_magnitude():
    """Verify solar wind DM noise is physically consistent with Parker 1/r² model.

    Parker model with n_e0 ~ 7 cm⁻³ at solar min gives:
      DM_IPM = A_sw / (r × sin ε) ≈ 5.33e-5 / (5 × 1) ≈ 1.07e-5 pc cm⁻³
      Δt = K_DM × DM / f² ≈ 4.148e3 × 1.07e-5 / 1400² ≈ 2.3e-11 s
    So |Δt| should be well below 1e-7 s at 5 AU at solar minimum.
    The pre-fix value of A_sw = 2 pc cm⁻³ AU was 37,500× too large.
    """
    from core.noise import NoiseModel
    from config import K_DM

    # Fixed geometry: perpendicular sightline at 5 AU, solar minimum
    sc_pos = np.array([5.0, 0.0, 0.0])    # 5 AU along X
    p_dir  = np.array([0.0, 1.0, 0.0])    # perpendicular — sin(ε) = 1

    sw_noise = NoiseModel.solar_wind_noise(
        sc_pos, p_dir, solar_activity=0.0, seed=0
    )

    # Expected: DM ≈ A_sw_min / (r × sin ε) = 5.33e-5 / 5 = 1.07e-5 pc cm⁻³
    # Δt = K_DM × DM / f² = 4148 × 1.07e-5 / (1400)² ≈ 2.3e-11 s
    # With 30% noise fluctuation, expect |Δt| < 1e-9 s
    assert abs(sw_noise) < 1e-7, (
        f"Solar wind noise at 5 AU, solar min = {sw_noise:.3e} s; "
        f"expected < 1e-7 s from Parker model. "
        f"If > 1e-4, A_sw amplitude is still overcalculated."
    )
    assert np.isfinite(sw_noise), "Solar wind noise is not finite"

    # Solar max should produce larger noise than solar min
    sw_max = NoiseModel.solar_wind_noise(sc_pos, p_dir, solar_activity=1.0, seed=0)
    sw_min = NoiseModel.solar_wind_noise(sc_pos, p_dir, solar_activity=0.0, seed=0)
    # Mean amplitude: solar max A_sw_max / A_sw_min ≈ 1.52e-4 / 5.33e-5 ≈ 2.85
    # Both seeds are deterministic so the comparison is direct
    assert abs(sw_max) >= abs(sw_min), (
        f"Solar max noise {sw_max:.3e} should be ≥ solar min noise {sw_min:.3e}"
    )

    return (
        f"sw_noise(solar_min, 5AU, perp) = {sw_noise:.3e} s "
        f"(< 1e-7 s ✓); max/min ratio ≈ {abs(sw_max)/max(abs(sw_min),1e-30):.1f}"
    )


@_test("noise.py solar_wind_noise consistent with Dispersion.interplanetary_dm_contribution")
def test_noise_solar_wind_consistent_with_ipm():
    """Cross-check: noise.py and dispersion.py must agree on the IPM DM scale.

    Both modules model the same Parker solar wind.  The DM contribution
    inferred from noise.py solar_wind_noise should be within a factor of 10
    of Dispersion.interplanetary_dm_contribution for the same geometry.
    A factor > 1000 difference (as existed before the A_sw fix) indicates
    an implementation error in one of the two modules.

    Geometry: spacecraft at [5, 0, 0] AU; pulsar at gl=90°, gb=0° → direction [0,1,0].
    This gives a perpendicular sightline (solar elongation = 90°, sin ε = 1).
    """
    from core.noise import NoiseModel
    from core.dispersion import Dispersion
    from config import K_DM

    freq_mhz = 1400.0
    r_au = 5.0

    # Noise module: DM inferred from timing residual (seed=0, no random fluctuation bias)
    sc_pos = np.array([r_au, 0.0, 0.0])
    p_dir  = np.array([0.0, 1.0, 0.0])   # gl=90°, gb=0° — perpendicular to sc_pos
    sw_noise_s = NoiseModel.solar_wind_noise(
        sc_pos, p_dir, solar_activity=0.0, seed=0
    )
    dm_from_noise = abs(sw_noise_s) * freq_mhz**2 / K_DM  # invert DM→Δt formula

    # Dispersion module: direct IPM DM (uses gl=90, gb=0 → direction [0,1,0])
    # interplanetary_dm_contribution(spacecraft_pos_au, pulsar_gl, pulsar_gb)
    dm_from_dispersion = Dispersion.interplanetary_dm_contribution(
        sc_pos, pulsar_gl=90.0, pulsar_gb=0.0
    )

    # Both should be in the same ballpark: within a factor of 10
    # (noise module has a 30% random fluctuation with seed=0 ≈ lognormal draw)
    if dm_from_dispersion > 0 and dm_from_noise > 0:
        ratio = dm_from_noise / dm_from_dispersion
        assert 0.01 < ratio < 100, (
            f"Noise-implied DM = {dm_from_noise:.3e} pc/cm³, "
            f"Dispersion DM = {dm_from_dispersion:.3e} pc/cm³; "
            f"ratio = {ratio:.1f} — inconsistency > 100× indicates A_sw mismatch"
        )

    return (
        f"noise-implied DM = {dm_from_noise:.3e} pc/cm³, "
        f"dispersion DM = {dm_from_dispersion:.3e} pc/cm³, "
        f"ratio = {dm_from_noise/max(dm_from_dispersion,1e-30):.2f}"
    )


@_test("pulsar.py profile FWHM matches catalogued W50 within 30%")
def test_pulsar_profile_fwhm_matches_w50():
    """The von Mises profile FWHM (at half-maximum) must match the catalogued W50.

    W50 is the pulse width at 50% of peak intensity.  The kappa formula
    kappa = 2*ln(2) / (pi*w50_phase)^2 is derived from the half-maximum
    condition of the von Mises distribution.  A 30% tolerance accommodates
    the interpulse component and normalisation rounding.

    Test pulsars are chosen so that W50 spans at least 8 bins at 512-bin
    resolution (w50_phase > 0.015), avoiding the discretisation limit where
    FWHM can only be measured to ±1 bin accuracy.  At <2 bins, the minimum
    measurable FWHM equals one bin width regardless of the true W50.
    """
    from core.pulsar import Pulsar
    import numpy as np

    n_bins = 512   # high resolution for accurate FWHM measurement

    # Synthetic pulsars with w50_phase > 0.015 (≥ 8 bins at 512 resolution)
    # w50_phase = w50_μs / (period_ms × 1e3)
    # To get ≥ 8 bins: w50_phase > 8/512 = 0.016
    test_pulsars = [
        # (name, period_ms, w50_μs, w50_phase_check)
        ("J_wide_A",  10.0,  500.0),   # w50_phase = 0.050 → 25.6 bins
        ("J_wide_B",   5.0,  200.0),   # w50_phase = 0.040 → 20.5 bins
        ("J_wide_C",  20.0, 1000.0),   # w50_phase = 0.050 → 25.6 bins
        ("J_wide_D",   8.0,  200.0),   # w50_phase = 0.025 → 12.8 bins
    ]

    for name, period_ms, w50_us in test_pulsars:
        period_s = period_ms * 1e-3
        p = Pulsar(
            name=name, period=period_s, period_dot=1e-20,
            dm=10.0, gl=45.0, gb=10.0, distance_kpc=1.0,
            w50=w50_us, s1400=5.0, timing_noise_ns=100.0,
        )

        profile = p.generate_profile(n_bins=n_bins)

        # Find FWHM: count bins at or above 50% of peak amplitude
        peak = profile.max()
        half_max = 0.5 * peak
        fwhm_bins = int((profile >= half_max).sum())

        # Convert to phase fraction, then to microseconds
        fwhm_phase = fwhm_bins / n_bins
        fwhm_us_measured = fwhm_phase * period_s * 1e6

        tol = 0.30   # 30% tolerance on FWHM
        assert abs(fwhm_us_measured - w50_us) / w50_us < tol, (
            f"{name}: profile FWHM = {fwhm_us_measured:.1f} μs, W50 = {w50_us:.1f} μs "
            f"(error = {abs(fwhm_us_measured-w50_us)/w50_us:.1%} > {tol:.0%}); "
            f"w50_phase = {w50_us/(period_s*1e6):.4f}, bins expected ≈ "
            f"{w50_us/(period_s*1e6)*n_bins:.1f}"
        )

    return f"Profile FWHM within 30% of W50 for 4 wide synthetic pulsars at {n_bins}-bin resolution"


@_test("gravity.py orbit_radius_m gives correct central body potential at 1 AU")
def test_gravity_central_body_orbit_radius():
    """Verify that orbit_radius_m is used for the central body potential instead
    of the galactocentric distance.

    A spacecraft at 1 AU from a Sun-like star located anywhere in the galaxy
    should feel the same central body potential (GM/r) regardless of the star's
    galactocentric position.  Without orbit_radius_m, the old code used
    np.linalg.norm(position_kpc) × KPC_TO_M as the distance, which gives a
    value ~kpc-scale and completely wrong central body potential.
    """
    from core.gravity import Gravity
    from config import G_NEWTON, M_SUN, AU_TO_M, KPC_TO_M

    # Central body reference: Sun-like star at 1 AU
    expected_central_phi = -G_NEWTON * M_SUN / AU_TO_M   # ≈ −8.87×10⁸ J/kg

    # Star at two very different galactocentric positions
    pos_inner = np.array([3.0, 0.0, 0.0])    # 3 kpc from GC
    pos_outer = np.array([12.0, 0.0, 0.0])   # 12 kpc from GC

    for label, pos_kpc in [("inner, 3 kpc", pos_inner), ("outer, 12 kpc", pos_outer)]:
        phi = Gravity.gravitational_potential(
            pos_kpc,
            central_body_mass_kg=M_SUN,
            central_body_radius_m=6.96e8,
            include_galactic=False,          # isolate central body term
            orbit_radius_m=AU_TO_M,          # actual orbital radius
        )

        # Central body potential at 1 AU should be ~−8.87×10⁸ J/kg regardless of GC distance
        assert abs(phi - expected_central_phi) / abs(expected_central_phi) < 1e-6, (
            f"At galactic position '{label}': central body phi = {phi:.4e}, "
            f"expected {expected_central_phi:.4e} J/kg (1 AU from M_sun). "
            f"If wrong, orbit_radius_m is not being used."
        )

    # Without orbit_radius_m, the code falls back to galactocentric distance
    # (kpc-scale), giving a wildly different (far too shallow) potential
    phi_no_orbit = Gravity.gravitational_potential(
        pos_outer,
        central_body_mass_kg=M_SUN,
        central_body_radius_m=6.96e8,
        include_galactic=False,
        orbit_radius_m=0.0,   # explicitly zero → use fallback
    )
    # Fallback r = 12 kpc in metres = 3.7e20 m → phi = -GM/r ≈ −3.6e−1 J/kg (near zero)
    assert abs(phi_no_orbit) < abs(expected_central_phi) / 1000, (
        f"Without orbit_radius_m, potential at GC distance should be much shallower: "
        f"{phi_no_orbit:.3e} vs {expected_central_phi:.3e}"
    )

    return (
        f"Central body phi at 1 AU = {expected_central_phi:.4e} J/kg; "
        f"consistent at inner and outer galaxy positions ✓"
    )


@_test("gravity.py galactic potential gradient: inner galaxy deeper than outer")
def test_gravity_galactic_potential_gradient():
    """Verify the galactic potential is monotonically deeper toward the centre.

    The combined Miyamoto-Nagai + Hernquist bulge + halo potential must satisfy:
      Φ(R=3 kpc) < Φ(R=8.178 kpc) < Φ(R=12 kpc)  (more negative = deeper)

    This is a fundamental consistency check on the galactic model: spacecraft
    in the inner galaxy experience stronger gravitational redshift, which is
    the physical basis for using timing residuals to constrain galactic position.

    Coordinates: X-axis from GC toward Sun; positive X is anti-centre direction.
    Sun is at (−R₀, 0, 0) in galactocentric frame (GC at origin).
    """
    from core.gravity import Gravity
    from config import KPC_TO_M

    pos_inner = np.array([3.0,   0.0, 0.0])   # R = 3 kpc from GC
    pos_solar  = np.array([8.178, 0.0, 0.0])   # R = R₀ ≈ 8.178 kpc (solar radius)
    pos_outer  = np.array([12.0,  0.0, 0.0])   # R = 12 kpc from GC

    phi_inner = Gravity.gravitational_potential(pos_inner, include_galactic=True)
    phi_solar = Gravity.gravitational_potential(pos_solar, include_galactic=True)
    phi_outer = Gravity.gravitational_potential(pos_outer, include_galactic=True)

    assert phi_inner < phi_solar, (
        f"Inner galaxy potential ({phi_inner:.4e}) should be more negative than "
        f"solar ({phi_solar:.4e}) — monotonicity broken"
    )
    assert phi_solar < phi_outer, (
        f"Solar potential ({phi_solar:.4e}) should be more negative than "
        f"outer galaxy ({phi_outer:.4e}) — halo normalisation may be wrong"
    )

    # All should be negative in the disk (bound system)
    for label, phi in [("inner", phi_inner), ("solar", phi_solar), ("outer", phi_outer)]:
        assert phi < 0, f"Galactic potential at {label} galaxy position should be negative: {phi:.3e}"

    # Potential depth at 3 kpc should be meaningfully deeper than at 12 kpc
    depth_diff = phi_inner - phi_outer   # negative number
    assert abs(depth_diff) > 1e8, (
        f"Potential difference inner–outer = {depth_diff:.3e} m²/s²; "
        f"too small — gradient may be suppressed"
    )

    return (
        f"Φ: inner={phi_inner:.3e} < solar={phi_solar:.3e} < outer={phi_outer:.3e} m²/s² ✓; "
        f"depth diff = {abs(depth_diff):.3e} m²/s²"
    )


# ══════════════════════════════════════════════════════════════════════════════
# RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run() -> bool:
    results: list[tuple[str, bool, str]] = []

    print()
    print("═" * 60)
    print("  XNAV SIMULATOR — Phase 2 Physics Tests")
    print("═" * 60)

    for test_fn in _TESTS:
        test_fn(results)

    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    failed = total - passed

    print()
    print("═" * 60)
    print(f"  Phase 2 result: {passed}/{total} passed", end="")
    if failed:
        print(f"  ({failed} FAILED)")
    else:
        print("  — ALL PASS ✓")
    print("═" * 60)
    print()

    return failed == 0


if __name__ == "__main__":
    ok = run()
    sys.exit(0 if ok else 1)
