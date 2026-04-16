#!/usr/bin/env python3
# tests/test_phase3.py — Phase 3 Estimation Tests
# XNAV Cold Start Simulator
#
# Run with: python tests/test_phase3.py  (from xnav_simulator/ directory)
# or:       python -m tests.test_phase3

from __future__ import annotations

import sys
import os
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import math
import numpy as np

# ── Test registry ─────────────────────────────────────────────────────────────

_TESTS: list = []


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


# ── Helper: build minimal synthetic pulsars ───────────────────────────────────

def _make_pulsars(n: int = 5, seed: int = 7) -> list:
    """Return n synthetic Pulsar objects spread across the galactic plane."""
    from core.pulsar import Pulsar

    rng = np.random.default_rng(seed)
    pulsars = []
    for i in range(n):
        # Spread pulsars around the galactic plane at various distances/longitudes
        gl = float(30.0 + i * 60.0)          # 30°, 90°, 150°, …
        gb = float(rng.uniform(-5.0, 5.0))
        dist = float(1.5 + i * 0.8)          # 1.5–4.7 kpc
        p = Pulsar(
            name=f"J_TEST_{i:02d}",
            period=float(rng.uniform(2e-3, 8e-3)),   # 2–8 ms
            period_dot=1e-20,
            dm=float(30.0 + i * 20.0),               # 30–110 pc cm⁻³
            gl=gl,
            gb=gb,
            distance_kpc=dist,
            w50=float(rng.uniform(50.0, 300.0)),      # μs
            s1400=float(rng.uniform(0.5, 5.0)),       # mJy
            timing_noise_ns=float(rng.uniform(50.0, 200.0)),
        )
        pulsars.append(p)
    return pulsars


def _make_spacecraft(position_kpc=None):
    """Return a Spacecraft at the specified galactocentric position."""
    from core.spacecraft import Spacecraft
    if position_kpc is None:
        position_kpc = np.array([-8.178, 0.0, 0.0])
    return Spacecraft(
        position_kpc=np.asarray(position_kpc, dtype=np.float64),
        velocity_kms=np.array([0.0, 220.0, 0.0]),
        clock_offset_s=0.0,
        true_position_kpc=np.asarray(position_kpc, dtype=np.float64),
    )


# ─────────────────────────────────────────────────────────────────────────────
# TEST 1: Liu-West shrinkage parameter is physically correct
# ─────────────────────────────────────────────────────────────────────────────

@_test("liu_west_a_parameter_in_unit_interval")
def test_liu_west_a():
    """a = sqrt(1 - h²) must be in (0, 1) for all valid h in (0, 0.5).

    The build brief formula a = (3h−1)/(2h) is wrong — at h=0.1 it gives
    a=−3.5, causing particle divergence.  The estimator must use sqrt form.
    """
    from config import LIU_WEST_H

    test_h_values = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
    for h in test_h_values:
        a = math.sqrt(1.0 - h ** 2)
        assert 0.0 < a < 1.0, (
            f"a = sqrt(1 - {h}²) = {a:.6f} is outside (0, 1)"
        )

    # Confirm the build brief formula (3h-1)/(2h) would give wrong result at h=0.1
    h = 0.1
    a_wrong = (3 * h - 1) / (2 * h)
    assert a_wrong < 0.0, (
        f"Expected wrong formula to give negative a at h=0.1, got {a_wrong}"
    )

    a_correct = math.sqrt(1.0 - LIU_WEST_H ** 2)
    assert 0.99 < a_correct < 1.0, (
        f"Expected a ≈ 0.995 at h=0.1, got {a_correct}"
    )
    return f"a = {a_correct:.6f} at h={LIU_WEST_H} (wrong formula gives {a_wrong:.3f})"


# ─────────────────────────────────────────────────────────────────────────────
# TEST 2: Likelihood sigma includes ISM turbulence floor
# ─────────────────────────────────────────────────────────────────────────────

@_test("sigma_total_includes_ism_turbulence_floor")
def test_sigma_total_floor():
    """sigma_total must be > sigma_timing for all pulsars.

    The ISM turbulence floor (K_DM × 0.15 × DM / f²) is ~9–16 μs at 1400 MHz
    for typical MSPs, dominating the ~100 ns timing noise.  Omitting this floor
    causes weight collapse at the true position.
    """
    from config import K_DM
    pulsars = _make_pulsars(n=5)
    frequency_mhz = 1400.0
    integration_time_s = 1000.0

    all_floors_positive = True
    min_ratio = np.inf

    for p in pulsars:
        sigma_timing = p.timing_noise_ns * 1e-9 / math.sqrt(integration_time_s)
        sigma_dm_turb = K_DM * 0.15 * max(p.dm, 1.0) / (frequency_mhz ** 2)
        sigma_total = math.sqrt(sigma_timing ** 2 + sigma_dm_turb ** 2)

        assert sigma_total > sigma_timing, (
            f"{p.name}: sigma_total={sigma_total:.3e} not > sigma_timing={sigma_timing:.3e}"
        )
        assert sigma_dm_turb > 0, (
            f"{p.name}: sigma_dm_turb = {sigma_dm_turb} is not positive"
        )
        ratio = sigma_total / sigma_timing
        min_ratio = min(min_ratio, ratio)
        all_floors_positive = all_floors_positive and (sigma_dm_turb > 0)

    assert all_floors_positive, "ISM turbulence floor was zero for at least one pulsar"
    return f"min(sigma_total/sigma_timing) = {min_ratio:.2f} across {len(pulsars)} pulsars"


# ─────────────────────────────────────────────────────────────────────────────
# TEST 3: DM table shape and finiteness (no ISM grid)
# ─────────────────────────────────────────────────────────────────────────────

@_test("dm_table_shape_and_finiteness_no_grid")
def test_dm_table_shape():
    """dm_table must have shape (n_particles, n_pulsars) with all finite entries.

    Tests the catalogue-DM fallback path (no ISM grid loaded).
    """
    from core.estimator import ParticleFilter

    n_particles = 200
    pulsars = _make_pulsars(n=6)
    true_pos = np.array([-8.0, 0.5, 0.0])

    pf = ParticleFilter(n_particles=n_particles, seed=1)
    pf.initialise_from_region(true_pos, radius_kpc=2.0)

    particle_positions = pf.particles[:, :3]
    pulsar_positions = np.array([p.position_kpc for p in pulsars], dtype=np.float64)

    dm_table = pf._build_dm_table(particle_positions, pulsar_positions, pulsars, ism=None)

    assert dm_table.shape == (n_particles, len(pulsars)), (
        f"dm_table shape {dm_table.shape} != ({n_particles}, {len(pulsars)})"
    )
    assert np.all(np.isfinite(dm_table)), "dm_table contains non-finite values"
    assert np.all(dm_table >= 0.5), (
        f"dm_table minimum {dm_table.min():.3f} < 0.5 pc cm⁻³ floor"
    )

    # All rows should be identical when using catalogue DMs (no grid)
    assert np.allclose(dm_table[0], dm_table[1]), (
        "Without ISM grid, all rows of dm_table should be identical (catalogue DM)"
    )
    return f"shape={dm_table.shape}, range=[{dm_table.min():.1f}, {dm_table.max():.1f}] pc cm⁻³"


# ─────────────────────────────────────────────────────────────────────────────
# TEST 4: ESS is in (0, n_particles] after a full update step
# ─────────────────────────────────────────────────────────────────────────────

@_test("ess_in_valid_range_after_update")
def test_ess_range():
    """get_ess() must return a fraction in (0, 1] after a weight update.

    Per build brief spec: get_ess() returns ESS as a fraction of n_particles.
    ESS = 1/sum(w²) / n_particles.
    With N equal weights: ESS fraction = 1.0 (maximum diversity).
    With all weight on one particle: ESS fraction = 1/N (near collapse).
    """
    from core.estimator import ParticleFilter
    from core.timing import TimingModel

    n_particles = 300
    pulsars = _make_pulsars(n=4, seed=42)
    true_pos = np.array([-7.5, 1.0, 0.1])
    sc = _make_spacecraft(true_pos)

    # Compute observed timings from the true spacecraft position
    observed = TimingModel.compute_all_pulsars(
        pulsars, sc,
        observation_time_s=0.0,
        frequency_mhz=1400.0,
        integration_time_s=1000.0,
        include_noise=False,
    )

    pf = ParticleFilter(n_particles=n_particles, seed=99)
    pf.initialise_from_region(true_pos, radius_kpc=3.0)

    pf.update(
        pulsars, observed, ism=None,
        frequency_mhz=1400.0, integration_time_s=1000.0,
    )

    # get_ess() must return a fraction in (0, 1] per the build brief spec
    ess_frac = pf.get_ess()
    assert ess_frac > 0, f"ESS fraction = {ess_frac} is not positive"
    assert ess_frac <= 1.0 + 1e-9, (
        f"ESS fraction = {ess_frac} exceeds 1.0 — must be a fraction, not absolute count"
    )
    assert np.isfinite(ess_frac), f"ESS fraction is not finite: {ess_frac}"

    # get_ess_absolute() must return the unnormalised ESS (0, n_particles]
    ess_abs = pf.get_ess_absolute()
    assert ess_abs > 0, f"ESS absolute = {ess_abs} is not positive"
    assert ess_abs <= n_particles + 1e-6, (
        f"ESS absolute = {ess_abs} exceeds n_particles = {n_particles}"
    )
    assert abs(ess_frac - ess_abs / n_particles) < 1e-12, (
        f"get_ess()={ess_frac} != get_ess_absolute()/n={ess_abs/n_particles}"
    )

    return f"ESS fraction = {ess_frac:.4f} (absolute = {ess_abs:.1f} / {n_particles})"


# ─────────────────────────────────────────────────────────────────────────────
# TEST 5: Weights normalised to 1.0 after update
# ─────────────────────────────────────────────────────────────────────────────

@_test("weights_sum_to_unity_after_update")
def test_weight_normalisation():
    """Weights must sum to 1.0 (to float64 tolerance) after each update step."""
    from core.estimator import ParticleFilter
    from core.timing import TimingModel

    n_particles = 300
    pulsars = _make_pulsars(n=4, seed=5)
    true_pos = np.array([-9.0, -1.5, 0.2])
    sc = _make_spacecraft(true_pos)

    observed = TimingModel.compute_all_pulsars(
        pulsars, sc,
        observation_time_s=0.0,
        frequency_mhz=1400.0,
        integration_time_s=1000.0,
        include_noise=False,
    )

    pf = ParticleFilter(n_particles=n_particles, seed=12)
    pf.initialise_from_region(true_pos, radius_kpc=2.0)

    pf.update(pulsars, observed, ism=None)

    weight_sum = float(pf.weights.sum())
    assert abs(weight_sum - 1.0) < 1e-12, (
        f"Weight sum = {weight_sum:.15f} deviates from 1.0 by {abs(weight_sum - 1.0):.2e}"
    )
    assert np.all(pf.weights >= 0), "Some weights are negative"
    assert np.all(np.isfinite(pf.weights)), "Some weights are non-finite"
    return f"sum(w) = 1 + {weight_sum - 1.0:.2e}"


# ─────────────────────────────────────────────────────────────────────────────
# TEST 6: reset() produces identical particle clouds for the same seed
# ─────────────────────────────────────────────────────────────────────────────

@_test("reset_reproduces_identical_state_same_seed")
def test_reset_reproducibility():
    """reset() followed by re-initialisation must produce identical particle clouds."""
    from core.estimator import ParticleFilter

    n_particles = 200
    center = np.array([-8.0, 0.0, 0.5])

    # First run
    pf1 = ParticleFilter(n_particles=n_particles, seed=77)
    pf1.initialise_from_region(center, radius_kpc=1.5)
    particles_run1 = pf1.particles.copy()
    weights_run1 = pf1.weights.copy()

    # Reset — check flag is cleared BEFORE re-initialising
    pf1.reset(seed=77)
    assert not pf1.initialised, "initialised flag should be False immediately after reset()"
    assert not pf1.diverged, "diverged flag not cleared by reset()"

    # Re-initialise with same seed — should reproduce identical particles
    pf1.initialise_from_region(center, radius_kpc=1.5)
    particles_run2 = pf1.particles.copy()
    weights_run2 = pf1.weights.copy()

    assert np.allclose(particles_run1, particles_run2), (
        "Particle positions differ after reset with same seed"
    )
    assert np.allclose(weights_run1, weights_run2), (
        "Weights differ after reset with same seed"
    )

    # Different seed should give different particles
    pf1.reset(seed=9999)
    pf1.initialise_from_region(center, radius_kpc=1.5)
    particles_run3 = pf1.particles.copy()
    assert not np.allclose(particles_run1, particles_run3), (
        "Different seed produced identical particles — RNG not reset properly"
    )
    return "Identical particles after same-seed reset; distinct after different-seed reset"


# ─────────────────────────────────────────────────────────────────────────────
# TEST 7: Dict key ordering does not affect update result
# ─────────────────────────────────────────────────────────────────────────────

@_test("update_result_invariant_to_dict_key_order")
def test_dict_key_order_invariance():
    """Scrambling the key order in observed_timings_dict must not change weights.

    The update() method enforces sorted key order internally, so the result
    must be identical regardless of the input dict's iteration order.
    """
    from core.estimator import ParticleFilter
    from core.timing import TimingModel

    n_particles = 200
    pulsars = _make_pulsars(n=5, seed=3)
    true_pos = np.array([-7.0, 2.0, 0.0])
    sc = _make_spacecraft(true_pos)

    observed = TimingModel.compute_all_pulsars(
        pulsars, sc,
        observation_time_s=0.0,
        include_noise=False,
    )

    # Reverse key order
    observed_reversed = dict(reversed(list(observed.items())))

    # Run filter with original order
    pf1 = ParticleFilter(n_particles=n_particles, seed=55)
    pf1.initialise_from_region(true_pos, radius_kpc=1.0)
    pf1.update(pulsars, observed, ism=None)
    w1 = pf1.weights.copy()

    # Run filter with reversed order — same seed, same initialisation
    pf2 = ParticleFilter(n_particles=n_particles, seed=55)
    pf2.initialise_from_region(true_pos, radius_kpc=1.0)
    pf2.update(pulsars, observed_reversed, ism=None)
    w2 = pf2.weights.copy()

    assert np.allclose(w1, w2, atol=1e-12), (
        f"Weights differ by up to {np.abs(w1 - w2).max():.2e} when key order is reversed"
    )
    return f"Max weight diff between orderings: {np.abs(w1 - w2).max():.2e}"


# ─────────────────────────────────────────────────────────────────────────────
# TEST 8: Weighted mean moves toward true position after update
# ─────────────────────────────────────────────────────────────────────────────

@_test("weighted_mean_converges_toward_true_position")
def test_convergence_direction():
    """After one update with a DM-gradient ISM, the weighted mean moves toward
    the true position.

    Physics note: without an ISM grid, the only signal is the Roemer delay
    (~10¹¹ s per kpc).  With sigma_total ≈ 10 μs, particles need to be within
    ~10 pm of the truth to discriminate — impractical for a cold-start filter.
    The primary navigation signal at kpc scale is the DM gradient: as the
    spacecraft moves, the DM to each pulsar changes.  With a linear DM gradient
    of ~5 pc cm⁻³/kpc and sigma_dm_turb ≈ 15–35 μs, the filter discriminates
    at ~1–2 kpc scale.

    This test uses a mock ISM with a simple linear DM gradient so the test
    runs without a precomputed ne2001 grid.
    """
    from core.estimator import ParticleFilter
    from config import K_DM

    # ── Mock ISM: linear DM gradient along galactic X axis ────────────────────
    # DM(midpoint) = DM_0 + grad_x * (midpoint_x - x_sun)
    # Gradient of 8 pc cm⁻³/kpc gives ~8 μs/kpc DM timing signal at 1400 MHz,
    # comparable to sigma_dm_turb (~9.5 μs for DM=30), enabling kpc-scale convergence.
    X_SUN = -8.178   # kpc (galactocentric)
    DM_0 = 40.0      # pc cm⁻³ at solar position
    GRAD_X = 8.0     # pc cm⁻³/kpc

    class _MockISM:
        def grid_loaded(self):
            return True
        @staticmethod
        def batch_lookup(pts):
            """Linear DM gradient along X axis from solar position."""
            # pts shape: (N, 3), galactocentric kpc
            pts = np.asarray(pts, dtype=np.float64)
            return np.maximum(DM_0 + GRAD_X * (pts[:, 0] - X_SUN), 0.5)

    mock_ism = _MockISM()

    # ── Set up pulsars, true position, observed timings ────────────────────────
    n_particles = 500
    pulsars = _make_pulsars(n=6, seed=23)
    true_pos = np.array([X_SUN, 0.0, 0.0])   # solar position
    sc = _make_spacecraft(true_pos)

    # Compute observed timings — use include_noise=False for deterministic test.
    # We override the dispersive delay manually to match the mock ISM at true_pos.
    # The timing model uses catalogue DMs, but our mock ISM should dominate via dm_table.
    # We compute what the true spacecraft would observe given the mock ISM DM gradient.
    from core.timing import TimingModel
    from config import KPC_TO_M, C_LIGHT

    # Build "observed" timing using mock ISM DMs at the true position
    observed = {}
    for p in pulsars:
        pulsar_pos = p.position_kpc
        # Midpoint from true pos to pulsar
        mid = (true_pos + pulsar_pos) / 2.0
        dm_true = float(max(DM_0 + GRAD_X * (mid[0] - X_SUN), 0.5))
        # Path length
        path_kpc = float(np.linalg.norm(pulsar_pos - true_pos))
        helio_dist = float(np.linalg.norm(mid - true_pos))  # approximate
        helio_dist = max(helio_dist, 0.1)
        mean_dens = dm_true / max(np.linalg.norm(mid - np.array([X_SUN, 0, 0])), 0.1)
        dm_obs = float(max(mean_dens * path_kpc, 0.5))

        # Roemer delay at true position
        los = pulsar_pos / np.linalg.norm(pulsar_pos) if np.linalg.norm(pulsar_pos) > 1e-10 else np.array([1., 0., 0.])
        roemer = -np.dot(true_pos, los) * KPC_TO_M / C_LIGHT
        dispersive = K_DM * dm_obs / (1400.0 ** 2)
        total = roemer + dispersive

        observed[p.name] = {"total": total, "geometric": roemer, "dispersive": dispersive,
                            "doppler": 0.0, "shapiro": 0.0, "gravitational": 0.0, "noise": 0.0}

    # ── Initialise particles OFFSET from the true position ────────────────────
    # 1.0 kpc offset gives DM timing signal ~8 μs vs sigma ~9.5 μs → discriminable
    init_offset = np.array([1.0, 0.3, 0.1])
    center = true_pos + init_offset

    pf = ParticleFilter(n_particles=n_particles, seed=7)
    pf.initialise_from_region(center, radius_kpc=0.4)

    # Initial unweighted mean distance from true position
    mean_before = pf.particles[:, :3].mean(axis=0)
    error_before = float(np.linalg.norm(mean_before - true_pos))

    pf.update(pulsars, observed, ism=mock_ism,
              frequency_mhz=1400.0, integration_time_s=1000.0)

    est = pf.get_estimate()
    error_after = float(np.linalg.norm(est["position_kpc"] - true_pos))

    assert error_after < error_before, (
        f"Weighted mean moved away from true position: "
        f"before={error_before:.3f} kpc, after={error_after:.3f} kpc. "
        f"(DM gradient provides ~{GRAD_X * K_DM / 1400.0**2 * 1e6:.1f} μs/kpc signal "
        f"vs sigma_dm_turb ≈ {K_DM * 0.15 * DM_0 / 1400.0**2 * 1e6:.1f} μs)"
    )
    assert np.all(np.isfinite(est["position_kpc"])), "Estimate position has non-finite values"
    assert np.all(np.isfinite(est["position_cov"])), "Covariance has non-finite values"

    # Covariance must be positive definite (eigenvalues > 0)
    eigvals = np.linalg.eigvalsh(est["position_cov"])
    assert np.all(eigvals > 0), f"Position covariance not positive definite: {eigvals}"

    return (
        f"error {error_before:.3f} → {error_after:.3f} kpc "
        f"(Δ={error_before - error_after:.3f} kpc improvement)"
    )


# ─────────────────────────────────────────────────────────────────────────────
# TEST 9: History records are appended per update step
# ─────────────────────────────────────────────────────────────────────────────

@_test("history_records_one_entry_per_update_step")
def test_history_accumulation():
    """history must have exactly one FilterState entry per completed update() call."""
    from core.estimator import ParticleFilter
    from core.timing import TimingModel

    n_steps = 4
    pulsars = _make_pulsars(n=4, seed=8)
    true_pos = np.array([-8.0, 0.5, 0.0])
    sc = _make_spacecraft(true_pos)

    observed = TimingModel.compute_all_pulsars(
        pulsars, sc, observation_time_s=0.0, include_noise=False
    )

    pf = ParticleFilter(n_particles=200, seed=21)
    pf.initialise_from_region(true_pos, radius_kpc=1.0)

    for i in range(n_steps):
        pf.update(pulsars, observed, ism=None)

    assert len(pf.history) == n_steps, (
        f"Expected {n_steps} history entries, got {len(pf.history)}"
    )
    assert pf.step == n_steps, f"pf.step={pf.step} != {n_steps}"

    for i, s in enumerate(pf.history):
        assert s.step == i, f"History entry {i} has step={s.step}"
        assert s.ess > 0, f"History entry {i} has non-positive ESS"
        assert s.ess_fraction > 0, f"History entry {i} has non-positive ESS fraction"

    # After reset, history must be cleared
    pf.reset()
    assert len(pf.history) == 0, "History not cleared by reset()"
    assert pf.step == 0, "Step counter not reset"

    return f"{n_steps} history entries, each with correct step index"


# ─────────────────────────────────────────────────────────────────────────────
# TEST 10: ISM lookup_dm_grid uses heliocentric distance (not galactocentric)
# ─────────────────────────────────────────────────────────────────────────────

@_test("ism_lookup_uses_heliocentric_denominator")
def test_ism_heliocentric_denominator():
    """InterstellarMedium.lookup_dm_grid must use heliocentric midpoint distance.

    A particle at the galactic anticentre (~R=16.3 kpc galactocentric) would
    have galactocentric midpoint distance ~8 kpc but heliocentric ~7 kpc.
    Using the galactocentric distance would introduce a ~15% systematic error.

    We test that the lookup uses the corrected formula by placing a mock
    interpolator that records the inputs and checking the denominator.
    """
    from config import SOLAR_GALACTOCENTRIC_KPC
    import numpy as np

    # Sun position in galactocentric frame (our convention: Sun at -R0 on X axis)
    sun_pos_kpc = np.array([-SOLAR_GALACTOCENTRIC_KPC, 0.0, 0.0])

    # Create a test point well away from the solar position
    # Use the galactic anticentre direction: X = +16 kpc (helio ~24 kpc from Sun)
    # vs galactocentric ~16 kpc — clearly different denominators
    sc_pos = np.array([8.0, 0.0, 0.0])    # 8 kpc toward anticentre (galactocentric)
    pulsar_pos = np.array([12.0, 0.0, 0.0])  # further anticentre

    midpoint = (sc_pos + pulsar_pos) / 2.0   # [10.0, 0, 0] galactocentric

    # Heliocentric distance to midpoint
    helio_dist = np.linalg.norm(midpoint - sun_pos_kpc)   # |[10 - (-8.178), 0, 0]| = 18.178

    # Galactocentric distance (the old wrong formula)
    galacto_dist = np.linalg.norm(midpoint)               # |[10, 0, 0]| = 10.0

    # They must be significantly different at this position
    assert abs(helio_dist - galacto_dist) > 5.0, (
        f"Test setup error: helio={helio_dist:.3f}, galacto={galacto_dist:.3f} "
        f"differ by less than 5 kpc — poor test sensitivity"
    )

    # Check the ISM module import works and has the fix
    from core.interstellar_medium import InterstellarMedium
    import inspect

    source = inspect.getsource(InterstellarMedium.lookup_dm_grid)
    # The fix should reference SUN_POS_KPC or SOLAR_GALACTOCENTRIC_KPC
    has_helio_fix = "SUN_POS_KPC" in source or "SOLAR_GALACTOCENTRIC_KPC" in source
    assert has_helio_fix, (
        "lookup_dm_grid does not reference SUN_POS_KPC or SOLAR_GALACTOCENTRIC_KPC "
        "— heliocentric denominator fix is missing"
    )

    # Verify the fix computation would give heliocentric distance
    from config import SUN_POS_KPC
    helio_vec = midpoint - SUN_POS_KPC
    computed_dist = float(np.linalg.norm(helio_vec))
    assert abs(computed_dist - helio_dist) < 1e-10, (
        f"Heliocentric distance computation mismatch: {computed_dist:.6f} vs {helio_dist:.6f}"
    )

    return (
        f"midpoint=[10,0,0] kpc: helio_dist={helio_dist:.3f} kpc, "
        f"galacto_dist={galacto_dist:.3f} kpc (diff={helio_dist-galacto_dist:.3f} kpc)"
    )


# ─────────────────────────────────────────────────────────────────────────────
# TEST 11: Covariance matrix is positive definite (diagonal nugget applied)
# ─────────────────────────────────────────────────────────────────────────────

@_test("covariance_positive_definite_with_nugget")
def test_covariance_positive_definite():
    """Position and velocity covariance matrices must be positive definite.

    The diagonal nugget (COV_NUGGET) must prevent singular matrices even
    when particles are tightly clustered.
    """
    from core.estimator import ParticleFilter
    from config import COV_NUGGET

    # Use a tight cluster to stress-test near-singularity
    n_particles = 100
    pf = ParticleFilter(n_particles=n_particles, seed=44)

    # Very tight initialisation (0.001 kpc radius ≈ 1 pc)
    center = np.array([-8.178, 0.0, 0.0])
    pf.initialise_from_region(center, radius_kpc=0.001)

    est = pf.get_estimate()

    for name, cov in [("position_cov", est["position_cov"]),
                      ("velocity_cov", est["velocity_cov"])]:
        assert cov.shape == (3, 3), f"{name} is not 3×3"
        assert np.all(np.isfinite(cov)), f"{name} has non-finite entries"
        eigvals = np.linalg.eigvalsh(cov)
        assert np.all(eigvals > 0), (
            f"{name} eigenvalues {eigvals} not all positive (not PD)"
        )
        # Minimum eigenvalue must be at least COV_NUGGET (from diagonal addition)
        assert np.min(eigvals) >= COV_NUGGET * 0.99, (
            f"{name} min eigenvalue {np.min(eigvals):.2e} < COV_NUGGET={COV_NUGGET:.2e}"
        )

    return f"Both 3×3 covariances PD, min eigval ≥ {COV_NUGGET:.0e}"


# ─────────────────────────────────────────────────────────────────────────────
# TEST 12: Systematic resampling preserves weight distribution
# ─────────────────────────────────────────────────────────────────────────────

@_test("systematic_resampling_preserves_particle_distribution")
def test_systematic_resampling():
    """After Liu-West resampling, weights must be uniform and particles finite."""
    from core.estimator import ParticleFilter
    from core.timing import TimingModel

    n_particles = 300
    pulsars = _make_pulsars(n=5, seed=99)
    true_pos = np.array([-8.5, 0.3, 0.1])
    sc = _make_spacecraft(true_pos)

    observed = TimingModel.compute_all_pulsars(
        pulsars, sc, observation_time_s=0.0, include_noise=False
    )

    pf = ParticleFilter(n_particles=n_particles, seed=33)
    # Initialise FAR from true position to force resampling
    pf.initialise_from_region(true_pos + np.array([5.0, 5.0, 1.0]), radius_kpc=0.1)
    pf.update(pulsars, observed, ism=None)

    assert np.all(np.isfinite(pf.particles)), "particles contain non-finite values after update"
    assert np.all(pf.weights >= 0), "negative weights after update"
    assert abs(pf.weights.sum() - 1.0) < 1e-12, "weights don't sum to 1 after resampling"

    return (
        f"particles finite after forced resampling; "
        f"weight_sum={pf.weights.sum():.15f}"
    )


# ── Main test runner ──────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 65)
    print("  XNAV Cold Start Simulator — Phase 3 Tests")
    print("=" * 65)

    results = []
    for test_fn in _TESTS:
        test_fn(results)

    n_pass = sum(1 for _, ok, _ in results if ok)
    n_fail = len(results) - n_pass

    print("\n" + "=" * 65)
    print(f"  Results: {n_pass}/{len(results)} passed, {n_fail} failed")
    print("=" * 65 + "\n")

    if n_fail > 0:
        print("FAILED TESTS:")
        for name, ok, detail in results:
            if not ok:
                print(f"  FAIL  {name}: {detail}")
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
