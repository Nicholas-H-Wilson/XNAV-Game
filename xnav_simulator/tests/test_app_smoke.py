#!/usr/bin/env python3
# tests/test_app_smoke.py — App-level smoke tests (Streamlit AppTest)
# XNAV Cold Start Simulator
#
# These tests exercise app.py itself — the one code path the phase tests never
# touched (the t_int NameError shipped in a "90/90 passing" build because no
# test imported app.py).  They boot the real Streamlit script headlessly via
# streamlit.testing.v1.AppTest and click the actual RUN button.
#
# Run from xnav_simulator/ directory: python tests/test_app_smoke.py

from __future__ import annotations

import pathlib
import sys
import traceback
from typing import Callable

import numpy as np

_ROOT = pathlib.Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_APP_PATH = str(_ROOT / "app.py")

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


def _boot_app(timeout: float = 120.0):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(_APP_PATH, default_timeout=timeout)
    at.run()
    return at


def _assert_no_exception(at) -> None:
    if at.exception:
        raise AssertionError(
            f"app raised {len(at.exception)} exception(s); first: "
            f"{at.exception[0].value}"
        )


# ── Tests ─────────────────────────────────────────────────────────────────────

@_test("app boots without exception")
def test_app_boots():
    at = _boot_app()
    _assert_no_exception(at)
    return "app.py renders cleanly"


@_test("RUN SIMULATION completes without exception")
def test_run_simulation():
    at = _boot_app(timeout=280.0)
    _assert_no_exception(at)

    run_buttons = [b for b in at.sidebar.button if "RUN" in (b.label or "")]
    assert run_buttons, "RUN SIMULATION button not found in sidebar"
    run_buttons[0].click()
    at.run()
    _assert_no_exception(at)

    # The simulation loop must actually have advanced the filter.
    iteration = at.session_state["iteration"]
    assert iteration > 0, f"simulation did not advance (iteration={iteration})"
    history = at.session_state["history"]
    assert len(history) == iteration, (
        f"history length {len(history)} != iteration {iteration}"
    )
    err = history[-1]["error_kpc"]
    assert np.isfinite(err), f"final error is not finite: {err}"
    return f"ran {iteration} iterations, final error {err:.2f} kpc"


@_test("observations builder returns finite, filter-consistent values")
def test_observations_builder():
    from core.catalogue import Catalogue
    from core.interstellar_medium import InterstellarMedium
    from core.observations import build_observations

    pulsars = Catalogue().get_top_n(10)
    ism = InterstellarMedium()
    rng = np.random.default_rng(0)
    sc_pos = np.array([-6.0, 1.0, 0.1])

    obs, dms = build_observations(
        pulsars, sc_pos, ism, rng=rng, integration_time_s=500.0,
    )
    assert set(obs.keys()) == {p.name for p in pulsars}
    for name, vals in obs.items():
        for key in ("total", "roemer_s", "dispersive_s", "timing_noise_s"):
            assert np.isfinite(vals[key]), f"{name}.{key} not finite"
        assert dms[name] > 0, f"{name} observed DM not positive"
    return f"{len(obs)} pulsars, all finite"


@_test("RESET clears simulation state")
def test_reset():
    at = _boot_app(timeout=280.0)
    run_buttons = [b for b in at.sidebar.button if "RUN" in (b.label or "")]
    run_buttons[0].click()
    at.run()
    _assert_no_exception(at)
    assert at.session_state["iteration"] > 0

    reset_buttons = [b for b in at.sidebar.button if "Reset" in (b.label or "")]
    assert reset_buttons, "Reset button not found in sidebar"
    reset_buttons[0].click()
    at.run()
    _assert_no_exception(at)
    assert at.session_state["iteration"] == 0, "iteration not reset"
    assert at.session_state["filter"] is None, "filter not cleared"
    assert at.session_state["history"] == [], "history not cleared"
    return "state cleared"


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print()
    print("══════════════════════════════════════════════════════════════════════")
    print("  Phase 8 — App smoke tests (Streamlit AppTest)")
    print("══════════════════════════════════════════════════════════════════════")
    print()

    results: list[dict] = []
    tests = [
        test_app_boots,
        test_run_simulation,
        test_observations_builder,
        test_reset,
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
