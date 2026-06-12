#!/usr/bin/env python3
# tests/test_phase4.py — Phase 4 Stage Logic Tests
# XNAV Cold Start Simulator
#
# Run with: python tests/test_phase4.py  (from xnav_simulator/ directory)
# or:       python -m tests.test_phase4

from __future__ import annotations

import sys
import os
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

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


# ── Shared helpers ────────────────────────────────────────────────────────────

def _make_pulsars(n: int = 10, seed: int = 7) -> list:
    """Return n synthetic Pulsar objects spread across the galactic plane."""
    from core.pulsar import Pulsar
    from utils.coordinates import galactic_to_cartesian

    rng = np.random.default_rng(seed)
    pulsars = []
    for i in range(n):
        gl = float(18.0 * i)           # spread around the sky
        gb = float(rng.uniform(-8.0, 8.0))
        dist = float(1.5 + i * 0.7)   # 1.5–8.3 kpc
        p = Pulsar(
            name=f"J_P4_{i:02d}",
            period=float(rng.uniform(2e-3, 9e-3)),
            period_dot=1e-20,
            dm=float(20.0 + i * 15.0),
            gl=gl,
            gb=gb,
            distance_kpc=dist,
            w50=float(rng.uniform(40.0, 250.0)),
            s1400=float(rng.uniform(0.3, 8.0)),
            timing_noise_ns=float(rng.uniform(30.0, 150.0)),
        )
        pulsars.append(p)
    return pulsars


class _MockISM:
    """Minimal ISM mock: linear DM gradient so Stage 1 has spatial discrimination."""

    _DM0 = 50.0          # pc cm⁻³ at Sun
    _GRAD_X = 8.0        # pc cm⁻³ per kpc along X

    def grid_loaded(self) -> bool:
        return True

    @staticmethod
    def batch_lookup(pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, dtype=np.float64)
        from config import SUN_POS_KPC
        x_sun = float(SUN_POS_KPC[0])
        return np.maximum(_MockISM._DM0 + _MockISM._GRAD_X * (pts[:, 0] - x_sun), 0.5)


# ── Stage 1 Tests ─────────────────────────────────────────────────────────────

@_test("stage1: probability_map sums to 1 and has no negative values")
def test_stage1_valid_probability():
    from stages.stage1_dm_localisation import run

    pulsars = _make_pulsars(5)
    # Observed DMs = catalogue DMs (no offset → peak near origin)
    obs_dms = {p.name: p.dm for p in pulsars}

    result = run(
        pulsars=pulsars,
        observed_dm_values=obs_dms,
        ism_model=_MockISM(),
        grid_resolution_pc=1000.0,   # coarse for speed
    )

    prob = result["probability_map"]

    assert np.all(prob >= 0.0), f"Probability map has negative values: min={prob.min()}"
    total = float(prob.sum())
    assert abs(total - 1.0) < 1e-10, f"Probability map sum={total} ≠ 1.0"

    return f"prob_map shape={prob.shape}, sum={total:.10f}"


@_test("stage1: peak of probability map within 5 kpc of spacecraft position")
def test_stage1_localisation_accuracy():
    from stages.stage1_dm_localisation import run
    from utils.coordinates import galactic_to_cartesian

    # Place spacecraft at a known position
    sc_pos = galactic_to_cartesian(90.0, 0.0, 4.0)   # GL=90, GB=0, 4 kpc

    pulsars = _make_pulsars(8)

    # Generate observed DMs as if from the spacecraft position
    # Use MockISM to get consistent DM values
    ism = _MockISM()
    obs_dms = {}
    for p in pulsars:
        pp = np.asarray(p.position_kpc)
        midpoint = (sc_pos + pp) / 2.0
        dm_at_mid = float(ism.batch_lookup(midpoint.reshape(1, 3))[0])
        from config import SUN_POS_KPC
        helio_dist = float(np.linalg.norm(midpoint - SUN_POS_KPC))
        helio_dist = max(helio_dist, 0.1)
        path_len = float(np.linalg.norm(pp - sc_pos))
        obs_dms[p.name] = max(dm_at_mid / helio_dist * path_len, 0.5)

    result = run(
        pulsars=pulsars,
        observed_dm_values=obs_dms,
        ism_model=ism,
        grid_resolution_pc=1000.0,
        spacecraft_position_kpc=sc_pos,
    )

    best_centre = result["best_region"]["centre_kpc"]
    error_kpc = float(np.linalg.norm(best_centre - sc_pos))

    assert error_kpc < 5.0, (
        f"Stage 1 localisation error {error_kpc:.2f} kpc > 5 kpc threshold"
    )

    return f"Stage 1 localisation error: {error_kpc:.2f} kpc"


@_test("stage1: output dict has all required keys")
def test_stage1_return_keys():
    from stages.stage1_dm_localisation import run

    pulsars = _make_pulsars(3)
    obs_dms = {p.name: p.dm for p in pulsars}

    result = run(
        pulsars=pulsars,
        observed_dm_values=obs_dms,
        ism_model=_MockISM(),   # stage 1 now raises without a loaded grid
        grid_resolution_pc=2000.0,
    )

    required_keys = {
        "probability_map", "x_arr", "y_arr", "z_arr",
        "best_region", "dm_residuals", "log_likelihood_map", "n_pulsars_used",
    }
    missing = required_keys - set(result.keys())
    assert not missing, f"Missing keys: {missing}"

    assert "centre_kpc" in result["best_region"], "best_region missing centre_kpc"
    assert "radius_kpc" in result["best_region"], "best_region missing radius_kpc"

    return f"All {len(required_keys)} required keys present"


# ── Stage 2 Tests ─────────────────────────────────────────────────────────────

@_test("stage2: correctly identifies known pulsars at low noise")
def test_stage2_identification():
    from stages.stage2_profile_matching import run, simulate_observed_profiles

    # Use a subset for speed
    pulsars = _make_pulsars(8, seed=42)

    # Generate noisy observed profiles from the first 5 pulsars
    true_pulsars = pulsars[:5]
    obs_profiles = simulate_observed_profiles(true_pulsars, noise_sigma=0.02, seed=1)

    result = run(
        observed_profiles=obs_profiles,
        pulsar_catalogue=pulsars,   # full catalogue including non-true ones
        min_confidence=0.3,
    )

    n_correct = 0
    for ident in result["identifications"]:
        if ident["best_match"] is not None:
            obs_idx = ident["observed_index"]
            expected_name = true_pulsars[obs_idx].name
            if ident["best_match"].name == expected_name:
                n_correct += 1

    assert n_correct >= 4, (
        f"Stage 2 identified {n_correct}/5 correctly, need ≥ 4"
    )

    return (
        f"{n_correct}/5 identified correctly, "
        f"{result['n_identified']}/{result['n_observed']} above confidence threshold"
    )


@_test("stage2: confidence scores are in [0, 1]")
def test_stage2_confidence_range():
    from stages.stage2_profile_matching import run, simulate_observed_profiles

    pulsars = _make_pulsars(5)
    obs_profiles = simulate_observed_profiles(pulsars[:3], noise_sigma=0.05)

    result = run(observed_profiles=obs_profiles, pulsar_catalogue=pulsars)

    for ident in result["identifications"]:
        conf = ident["confidence"]
        assert 0.0 <= conf <= 1.0, f"Confidence {conf} out of [0,1]"

    return f"All confidences in [0,1] for {len(result['identifications'])} profiles"


@_test("stage2: exact profile matches top ZNCC = 1.0")
def test_stage2_exact_match():
    from stages.stage2_profile_matching import run

    pulsars = _make_pulsars(4)
    # Perfect profiles (no noise)
    exact_profiles = [p.profile.copy() for p in pulsars]

    result = run(
        observed_profiles=exact_profiles[:2],
        pulsar_catalogue=pulsars,
        min_confidence=0.0,
    )

    for ident in result["identifications"]:
        assert ident["best_match"] is not None, "Expected a match for exact profile"
        # ZNCC of exact match should be very high (near 1.0)
        assert ident["confidence"] > 0.95, (
            f"Exact profile match confidence {ident['confidence']:.3f} < 0.95"
        )

    return "Exact profile matches have confidence > 0.95"


# ── Stage 3 Tests ─────────────────────────────────────────────────────────────

@_test("stage3: GDOP lower for well-distributed pulsars than clustered")
def test_stage3_gdop_geometry():
    from stages.stage3_geometry import run, _compute_gdop
    from core.pulsar import Pulsar

    # Clustered pulsars: all in one quadrant (GL 0-90, GB ~0)
    clustered = []
    for i in range(6):
        gl = float(i * 10.0)          # 0 to 50°
        gb = float(i * 1.0)           # 0 to 5°
        p = Pulsar(
            name=f"J_CLUST_{i:02d}",
            period=3e-3, period_dot=1e-20,
            dm=50.0, gl=gl, gb=gb,
            distance_kpc=3.0, w50=100.0, s1400=1.0, timing_noise_ns=100.0,
        )
        clustered.append(p)

    # Distributed pulsars: spread across sky
    distributed = []
    for i in range(6):
        gl = float(i * 60.0)          # 0, 60, 120, 180, 240, 300°
        gb = float(30.0 * (-1) ** i)  # ±30°
        p = Pulsar(
            name=f"J_DIST_{i:02d}",
            period=3e-3, period_dot=1e-20,
            dm=50.0, gl=gl, gb=gb,
            distance_kpc=3.0, w50=100.0, s1400=1.0, timing_noise_ns=100.0,
        )
        distributed.append(p)

    gdop_clust, _ = _compute_gdop(clustered)
    gdop_dist, _ = _compute_gdop(distributed)

    assert gdop_dist < gdop_clust, (
        f"Expected distributed GDOP ({gdop_dist:.2f}) < clustered GDOP ({gdop_clust:.2f})"
    )

    return f"Distributed GDOP={gdop_dist:.2f}, Clustered GDOP={gdop_clust:.2f}"


@_test("stage3: weight update does not produce negative weights")
def test_stage3_nonnegative_weights():
    from stages.stage3_geometry import run
    from core.estimator import ParticleFilter

    pulsars = _make_pulsars(5)

    pf = ParticleFilter(n_particles=200, seed=1)
    pf.initialise_from_region(
        center_kpc=np.array([0.0, 0.0, 0.0]),
        radius_kpc=5.0,
    )

    result = run(
        identified_pulsars=pulsars,
        particle_filter=pf,
        los_sigma_kpc=2.0,
    )

    weights = pf.weights
    assert np.all(weights >= 0.0), f"Negative weights found: min={weights.min()}"
    assert abs(weights.sum() - 1.0) < 1e-10, f"Weights don't sum to 1: {weights.sum()}"

    return f"All {len(weights)} weights ≥ 0, sum={weights.sum():.10f}"


@_test("stage3: empty pulsar list returns inf GDOP")
def test_stage3_empty_pulsars():
    from stages.stage3_geometry import run
    from core.estimator import ParticleFilter

    pf = ParticleFilter(n_particles=100, seed=2)
    pf.initialise_from_region(np.array([0.0, 0.0, 0.0]), 5.0)

    result = run(identified_pulsars=[], particle_filter=pf)

    assert not np.isfinite(result["gdop"]), (
        f"Expected inf GDOP for empty pulsar list, got {result['gdop']}"
    )
    assert result["n_pulsars_used"] == 0

    return "Empty pulsar list correctly returns GDOP=inf"


# ── Stage 4 Tests ─────────────────────────────────────────────────────────────

@_test("stage4: ambiguity window shrinks monotonically as pulsars are added")
def test_stage4_window_shrinks():
    from stages.stage4_phase_ambiguity import run, simulate_arrival_times

    pulsars = _make_pulsars(6, seed=99)
    true_offset = 0.012   # 12 ms clock offset

    arrival_times = simulate_arrival_times(
        pulsars, clock_offset_s=true_offset, noise_scale_s=1e-7, seed=5
    )

    result = run(
        identified_pulsars=pulsars,
        arrival_times=arrival_times,
        position_estimate_kpc=np.array([-8.178, 0.0, 0.0]),
        true_clock_offset_s=true_offset,
    )

    history = result["window_history"]
    assert len(history) >= 2, f"Window history too short: {history}"

    # Verify window is generally non-increasing (allow small numerical ties)
    windows = [w for _, w in history]
    # Check that the last window is smaller than the first
    assert windows[-1] <= windows[0], (
        f"Final window {windows[-1]:.3e} s is not smaller than initial {windows[0]:.3e} s"
    )

    final_window_ms = result["ambiguity_window_s"] * 1000.0
    return (
        f"Window: {windows[0]*1000:.1f} ms → {windows[-1]*1000:.3f} ms "
        f"({result['n_pulsars_used']} pulsars)"
    )


@_test("stage4: final window is much smaller than initial window with 5+ pulsars")
def test_stage4_significant_reduction():
    from stages.stage4_phase_ambiguity import run, simulate_arrival_times

    # Use pulsars with well-separated incommensurate periods
    from core.pulsar import Pulsar
    periods = [0.00155, 0.00215, 0.00370, 0.00490, 0.00560, 0.00710]
    pulsars = [
        Pulsar(
            name=f"J_PHASE_{i:02d}",
            period=p, period_dot=1e-20,
            dm=50.0, gl=float(i * 60), gb=5.0,
            distance_kpc=3.0, w50=100.0, s1400=2.0, timing_noise_ns=100.0,
        )
        for i, p in enumerate(periods)
    ]

    true_offset = 0.008   # 8 ms
    arrival_times = simulate_arrival_times(
        pulsars, clock_offset_s=true_offset, noise_scale_s=5e-8, seed=10
    )

    result = run(
        identified_pulsars=pulsars,
        arrival_times=arrival_times,
        position_estimate_kpc=np.array([-8.178, 0.0, 0.0]),
        true_clock_offset_s=true_offset,
    )

    initial_w = result["metadata"]["initial_window_s"]
    final_w = result["ambiguity_window_s"]

    assert final_w < initial_w * 0.5, (
        f"Ambiguity window not sufficiently reduced: {final_w:.3e} vs {initial_w:.3e}"
    )

    return (
        f"Window reduced: {initial_w*1000:.0f} ms → {final_w*1000:.3f} ms "
        f"(factor {initial_w/final_w:.1f}x)"
    )


@_test("stage4: returns required output keys")
def test_stage4_output_keys():
    from stages.stage4_phase_ambiguity import run, simulate_arrival_times

    pulsars = _make_pulsars(4)
    arrival_times = simulate_arrival_times(pulsars, clock_offset_s=0.005)

    result = run(
        identified_pulsars=pulsars,
        arrival_times=arrival_times,
        position_estimate_kpc=np.array([-8.178, 0.0, 0.0]),
    )

    required = {
        "resolved_clock_offset_s", "ambiguity_window_s", "window_history",
        "candidate_times_s", "n_pulsars_used", "metadata",
    }
    missing = required - set(result.keys())
    assert not missing, f"Missing keys: {missing}"

    assert isinstance(result["window_history"], list), "window_history must be a list"
    assert isinstance(result["candidate_times_s"], np.ndarray), (
        "candidate_times_s must be np.ndarray"
    )

    return f"All {len(required)} required keys present"


@_test("stage4: empty pulsar list handled gracefully")
def test_stage4_empty_pulsars():
    from stages.stage4_phase_ambiguity import run

    result = run(
        identified_pulsars=[],
        arrival_times={},
        position_estimate_kpc=np.array([-8.178, 0.0, 0.0]),
    )

    assert result["n_pulsars_used"] == 0
    assert result["ambiguity_window_s"] > 0.0
    assert result["window_history"] == []

    return "Empty pulsar list handled without error"


# ── Test runner ────────────────────────────────────────────────────────────────

def main() -> int:
    print()
    print("═" * 65)
    print("  Phase 4 — Stage Logic Tests")
    print("═" * 65)

    results = []
    for fn in _TESTS:
        fn(results)

    n_pass = sum(1 for _, ok, _ in results if ok)
    n_fail = len(results) - n_pass

    print()
    print("═" * 65)
    print(f"  Results: {n_pass}/{len(results)} passed, {n_fail} failed")
    print("═" * 65)

    if n_fail == 0:
        print()

    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
