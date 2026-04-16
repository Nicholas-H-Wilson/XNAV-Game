#!/usr/bin/env python3
# tests/test_phase6.py — Phase 6 UI module tests
# XNAV Cold Start Simulator
#
# Runs without a live Streamlit server.  Tests call figure-builder functions
# directly (the testable, server-free half of each ui/ module).
#
# Run from xnav_simulator/ directory: python tests/test_phase6.py

from __future__ import annotations

import pathlib
import sys
import traceback
from typing import Callable

import numpy as np

_ROOT = pathlib.Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── Test runner (same pattern as all previous phases) ────────────────────────

_results: list[dict] = []


def _test(name: str):
    def decorator(fn: Callable):
        def wrapper(results: list):
            try:
                msg = fn()
                results.append({"name": name, "passed": True, "msg": msg or ""})
                print(f"  PASS  {name}  — {msg or ''}")
            except Exception as exc:
                tb = traceback.format_exc()
                results.append({"name": name, "passed": False, "msg": str(exc), "tb": tb})
                print(f"  FAIL  {name}  — {exc}")
        return wrapper
    return decorator


# ── Shared test data ──────────────────────────────────────────────────────────

def _make_data(**overrides) -> dict:
    base = {
        "pulsars": [],
        "sc_pos_kpc": np.zeros(3),
        "true_pos_kpc": None,
        "sun_pos_kpc": np.array([-8.178, 0.0, 0.0]),
        "uncertainty_kpc": 5.0,
        "blind_mode": False,
        "particles_kpc": np.random.default_rng(0).standard_normal((50, 3)) * 2.0,
        "weights": np.ones(50) / 50.0,
        "history": [],
        "estimate_kpc": np.array([1.0, 2.0, 0.5]),
        "ess_pre": 0.25,
        "ess_post": 1.0,
        "stage_status": {},
        "window_history": [],
        "clock_candidates": [],
        "clock_estimate_s": 1e-5,
        "stage4_complete": False,
        "phi_m2s2": -2.2e11,
        "phi_uncertainty": 6e10,
        "true_phi_m2s2": -2.2e11,
        "timing_noise_ns": 100.0,
        "dm_value": 50.0,
        "n_pulsars": 10,
        "collecting_area_m2": 1.0,
        "integration_time_s": 1000.0,
        "frequency_mhz": 1400.0,
        "dm_residuals": {},
        "iteration": 5,
        "selected_pulsar": None,
    }
    base.update(overrides)
    return base


# ── Test 1: all figure builders return Plotly Figure objects ─────────────────

@_test("all figure builders return plotly Figure objects")
def test_figure_builders():
    import plotly.graph_objects as go
    from ui import galaxy_map, convergence_panel, phase_panel, gravity_panel
    from ui import timing_panel

    data = _make_data()
    figs = [
        ("galaxy_map.build_topdown_figure", galaxy_map.build_topdown_figure(data)),
        ("galaxy_map.build_skymap_figure", galaxy_map.build_skymap_figure(data)),
        ("convergence_panel.build_particle_figure",
         convergence_panel.build_particle_figure(data)),
        ("convergence_panel.build_uncertainty_timeline_figure",
         convergence_panel.build_uncertainty_timeline_figure(data)),
        ("timing_panel.build_dispersive_sweep_figure",
         timing_panel.build_dispersive_sweep_figure(data)),
        ("timing_panel.build_dm_residual_figure",
         timing_panel.build_dm_residual_figure(data)),
        ("timing_panel.build_residual_breakdown_figure",
         timing_panel.build_residual_breakdown_figure(data)),
        ("phase_panel.build_phase_dials_figure",
         phase_panel.build_phase_dials_figure(data)),
        ("phase_panel.build_ambiguity_timeline_figure",
         phase_panel.build_ambiguity_timeline_figure(data)),
        ("gravity_panel.build_potential_figure",
         gravity_panel.build_potential_figure(data)),
        ("gravity_panel.build_reality_check_figure",
         gravity_panel.build_reality_check_figure(data)),
    ]
    non_figures = [name for name, fig in figs if not isinstance(fig, go.Figure)]
    assert not non_figures, f"Not a Figure: {non_figures}"
    return f"all {len(figs)} figure builders return go.Figure"


# ── Test 2: gravity physics assertions (Appendix D.1) ────────────────────────

@_test("gravity panel physics: gravity << DM turbulence (Appendix D.1)")
def test_gravity_physics():
    from ui.gravity_panel import _compute_noise_budget

    data = _make_data(phi_m2s2=-2.2e11, integration_time_s=1000.0,
                      dm_value=50.0, frequency_mhz=1400.0, timing_noise_ns=100.0)
    budget = _compute_noise_budget(data)

    gravity_ns = budget["gravity_ns"]
    dm_turb_ns = budget["dm_turb_ns"]
    timing_ns = budget["timing_noise_ns"]

    # CRITICAL: gravity < DM turbulence (primary confusion source)
    assert gravity_ns < dm_turb_ns, (
        f"gravity ({gravity_ns:.2e} ns) ≥ DM turbulence ({dm_turb_ns:.2e} ns) — "
        "should be buried under DM turbulence"
    )
    # Sanity: gravity >> timing noise at 1000 s (as per Appendix D.1 ordering)
    assert gravity_ns > timing_ns, (
        f"gravity ({gravity_ns:.2e} ns) ≤ timing noise ({timing_ns:.2e} ns) — "
        "gravity should be well above timing floor at 1000 s integration"
    )
    ratio = dm_turb_ns / gravity_ns
    return (f"gravity={gravity_ns:.2e} ns, DM_turb={dm_turb_ns:.2e} ns, "
            f"DM is {ratio:.1f}× larger — gravity undetectable")


# ── Test 3: gravity bar chart annotation trigger is DM turbulence ─────────────

@_test("gravity bar chart annotation triggers on dm_turb not timing_noise")
def test_gravity_annotation_trigger():
    from ui.gravity_panel import build_reality_check_figure

    data = _make_data(phi_m2s2=-2.2e11, integration_time_s=1000.0,
                      dm_value=50.0, frequency_mhz=1400.0)
    fig = build_reality_check_figure(data)

    # Find annotations in the figure
    annotations = fig.layout.annotations
    annotation_texts = [a.text for a in annotations if a.text]
    full_text = " ".join(annotation_texts)

    assert "DM turbulence" in full_text or "ISM" in full_text, (
        "Annotation must mention DM turbulence as the detection barrier. "
        f"Got: {full_text!r}"
    )
    assert "Current technology" not in full_text, (
        "Wrong annotation text: must not say 'Current technology' — "
        "the limit is ISM model fidelity, not detector technology"
    )
    return "annotation correctly names DM turbulence as detection barrier"


# ── Test 4: gravity roadmap text frames ISM as binding constraint ─────────────

@_test("gravity roadmap: binding constraint is ISM not collecting area")
def test_gravity_roadmap_framing():
    from ui.gravity_panel import build_roadmap_text

    data = _make_data()
    text = build_roadmap_text(data)

    assert "ISM" in text, "Roadmap must mention ISM as the binding constraint"
    assert "not the limiting factor" in text.lower() or \
           "not the limiting" in text.lower(), (
        "Roadmap must explicitly state collecting area/timing noise are NOT limiting"
    )
    # Must not tell user to build a bigger detector as the solution
    wrong_framings = ["buy a larger", "get a bigger", "increase collecting area to solve"]
    for phrase in wrong_framings:
        assert phrase.lower() not in text.lower(), (
            f"Roadmap must not frame solution as detector improvement: '{phrase}'"
        )
    return "roadmap correctly frames limit as ISM model fidelity"


# ── Test 5: gravity signal scales with integration time ───────────────────────

@_test("gravity signal scales linearly with integration time T")
def test_gravity_signal_scales_with_T():
    from ui.gravity_panel import _compute_noise_budget

    phi = -2.2e11
    base = _make_data(phi_m2s2=phi, integration_time_s=1000.0)
    double = _make_data(phi_m2s2=phi, integration_time_s=2000.0)

    b1 = _compute_noise_budget(base)
    b2 = _compute_noise_budget(double)

    ratio = b2["gravity_ns"] / b1["gravity_ns"]
    assert abs(ratio - 2.0) < 0.01, (
        f"Gravity signal should double when T doubles: got ratio={ratio:.4f}"
    )
    return f"gravity scales linearly: T×2 → signal×{ratio:.2f}"


# ── Test 6: convergence panel uses correct ESS label (pre vs post) ───────────

@_test("convergence panel exposes ess_pre field (not post-resample)")
def test_convergence_ess_pre():
    from ui.convergence_panel import build_uncertainty_timeline_figure

    # Build history with pre-resample ESS values
    history = [
        {"step": i, "uncertainty_kpc": 10.0 / (i + 1),
         "ess_pre": 0.05,   # low pre-resample (realistic in Roemer regime)
         "ess_post": 1.0}   # post-resample always 1.0
        for i in range(5)
    ]
    data = _make_data(history=history, ess_pre=0.05)
    fig = build_uncertainty_timeline_figure(data)

    # ESS trace should exist — verify the figure has 2 traces (uncertainty + ESS)
    assert len(fig.data) >= 2, "Expected at least 2 traces (uncertainty + ESS)"

    # Find the ESS trace by its name
    ess_traces = [t for t in fig.data if "ESS" in (t.name or "")]
    assert ess_traces, "No ESS trace found in timeline figure"

    # The ESS trace name must indicate pre-resample
    ess_name = ess_traces[0].name
    assert "pre" in ess_name.lower(), (
        f"ESS trace name must indicate 'pre-resample', got: '{ess_name}'"
    )
    return f"ESS trace name correctly identifies pre-resample: '{ess_name}'"


# ── Test 7: galaxy map renders pulsars with DM colouring ─────────────────────

@_test("galaxy map renders pulsar scatter coloured by DM")
def test_galaxy_map_pulsars():
    from ui.galaxy_map import build_topdown_figure

    pulsars = [
        {"name": f"J{i:04d}+0000", "gl": float(i * 30), "gb": 0.0,
         "distance_kpc": 2.0, "dm": float(10 + i * 5),
         "timing_noise_ns": 100.0, "identified": i % 2 == 0, "confidence": 0.8}
        for i in range(8)
    ]
    data = _make_data(pulsars=pulsars)
    fig = build_topdown_figure(data)

    # Find the pulsar scatter trace (has colorscale)
    pulsar_traces = [t for t in fig.data
                     if hasattr(t, "marker") and
                     hasattr(t.marker, "colorscale") and
                     t.marker.colorscale is not None and
                     len(t.marker.colorscale) > 0]
    assert pulsar_traces, "No DM-coloured pulsar scatter trace found"
    assert len(pulsar_traces[0].x) == 8, "Expected 8 pulsars in trace"
    return "8 pulsars rendered with DM colour scale"


# ── Test 8: phase panel ambiguity timeline shows exponential collapse ─────────

@_test("phase panel ambiguity timeline renders window_history correctly")
def test_phase_timeline():
    from ui.phase_panel import build_ambiguity_timeline_figure

    window_history = [
        {"n_pulsars": i + 1, "window_s": 0.2 / (10 ** i)}
        for i in range(5)
    ]
    data = _make_data(window_history=window_history)
    fig = build_ambiguity_timeline_figure(data)

    assert len(fig.data) >= 1, "Expected at least one trace in timeline"
    trace = fig.data[0]
    assert list(trace.x) == [1, 2, 3, 4, 5], "x-axis should be pulsar counts 1–5"
    assert trace.y[0] > trace.y[-1], "Window should decrease over the timeline"
    return f"Timeline shows collapse from {trace.y[0]:.2e} s to {trace.y[-1]:.2e} s"


# ── Test 9: utils/logger.py writes JSON-lines events ─────────────────────────

@_test("SimLogger writes JSON-lines events to disk")
def test_sim_logger():
    import json
    import tempfile
    import pathlib
    from utils.logger import SimLogger

    with tempfile.TemporaryDirectory() as tmpdir:
        # Monkeypatch the log dir so we don't write to the real logs/ directory
        logger = SimLogger(run_id="test_run")
        logger._log_path = pathlib.Path(tmpdir) / "run_test.jsonl"

        logger.log_event("test_event", {"foo": 42})
        logger.log_iteration(step=3, error_kpc=1.5, ess=0.3)

        lines = logger.log_path.read_text().strip().split("\n")
        assert len(lines) == 2, f"Expected 2 log lines, got {len(lines)}"

        rec0 = json.loads(lines[0])
        assert rec0["event"] == "test_event"
        assert rec0["foo"] == 42

        rec1 = json.loads(lines[1])
        assert rec1["event"] == "iteration"
        assert rec1["step"] == 3

    return "SimLogger writes 2 JSON-lines events correctly"


# ── Test 10: timing panel DM turbulence bar is present and separate ───────────

@_test("timing panel includes separate DM turbulence residual bar")
def test_timing_dm_turbulence_bar():
    from ui.timing_panel import build_residual_breakdown_figure
    from core.pulsar import Pulsar

    p = Pulsar(name="J1713+0747", period=0.00457, period_dot=8.5e-21,
               dm=15.9, gl=28.8, gb=25.2, distance_kpc=1.05,
               w50=3.0, s1400=10.0, timing_noise_ns=30.0)

    pulsars = [{
        "name": p.name, "dm": p.dm, "period": p.period,
        "distance_kpc": p.distance_kpc, "gl": p.gl, "gb": p.gb,
        "timing_noise_ns": p.timing_noise_ns, "w50": p.w50,
        "identified": True, "confidence": 0.95,
        "roemer_s": 1e-3, "dispersive_s": 1.5e-5,
        "timing_noise_s": 3e-8, "photon_noise_s": 5e-7,
        "dm_turbulence_s": 4.148e3 * 0.15 * p.dm / (1400.0 ** 2),
    }]
    data = _make_data(pulsars=pulsars, selected_pulsar=p.name)
    fig = build_residual_breakdown_figure(data)

    bar_labels = list(fig.data[0].y)
    full_labels = " ".join(bar_labels)

    assert "DM turbulence" in full_labels or "turbulence" in full_labels.lower(), (
        "Timing panel must include a separate 'DM turbulence residual' bar. "
        f"Got labels: {bar_labels}"
    )
    assert any("Corrected" in lbl or "corrected" in lbl or "dispersive" in lbl.lower()
               for lbl in bar_labels), (
        "Timing panel must have a 'Corrected dispersive delay' bar (not just 'Dispersive delay')"
    )
    return (f"DM turbulence bar present; {len(bar_labels)} bars total: "
            f"{[l[:20] for l in bar_labels]}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print()
    print("══════════════════════════════════════════════════════════════════════")
    print("  Phase 6 — UI Module Tests")
    print("══════════════════════════════════════════════════════════════════════")
    print()

    results: list[dict] = []
    tests = [
        test_figure_builders,
        test_gravity_physics,
        test_gravity_annotation_trigger,
        test_gravity_roadmap_framing,
        test_gravity_signal_scales_with_T,
        test_convergence_ess_pre,
        test_galaxy_map_pulsars,
        test_phase_timeline,
        test_sim_logger,
        test_timing_dm_turbulence_bar,
    ]
    for t in tests:
        t(results)

    n_pass = sum(1 for r in results if r["passed"])
    n_fail = len(results) - n_pass

    print()
    print("══════════════════════════════════════════════════════════════════════")
    print(f"  Results: {n_pass}/{len(results)} passed, {n_fail} failed")
    print("══════════════════════════════════════════════════════════════════════")
    print()

    if n_fail:
        for r in results:
            if not r["passed"]:
                print(f"FAIL: {r['name']}")
                if "tb" in r:
                    print(r["tb"])
        sys.exit(1)
    sys.exit(0)
