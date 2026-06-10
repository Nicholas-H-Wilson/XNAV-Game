#!/usr/bin/env python3
# tests/run_all_tests.py — Master test runner for XNAV Cold Start Simulator
# XNAV Cold Start Simulator
#
# Imports and runs all phase test modules in order (Phases 1–6).
# Catches all exceptions and marks as FAIL rather than crashing.
# Prints a summary table at the end.
# Exit code 0 if all pass, 1 if any fail.
#
# Run from xnav_simulator/ directory: python tests/run_all_tests.py

from __future__ import annotations

import importlib.util
import io
import pathlib
import sys
import time
import traceback

# Ensure xnav_simulator/ is on the path
_ROOT = pathlib.Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_TESTS_DIR = pathlib.Path(__file__).parent

# ── Phase test module definitions ─────────────────────────────────────────────

_PHASES = [
    ("Phase 1 — Foundation",    "test_phase1", "16/16"),
    ("Phase 2 — Physics",       "test_phase2", "34/34"),
    ("Phase 3 — Estimation",    "test_phase3", "12/12"),
    ("Phase 4 — Stage Logic",   "test_phase4", "13/13"),
    ("Phase 5 — Integration",   "test_phase5", " 5/5"),
    ("Phase 6 — UI",            "test_phase6", "10/10"),
    ("Phase 8 — App smoke",     "test_app_smoke", "4/4"),
]


def _run_phase_module(module_name: str) -> tuple[int, int, str]:
    """Import and run a phase test module.

    Returns (n_pass, n_total, error_message).
    Captures stdout from the module's own _results list.
    """
    module_path = _TESTS_DIR / f"{module_name}.py"
    if not module_path.exists():
        return 0, 0, f"Module not found: {module_path}"

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None:
        return 0, 0, f"Cannot create spec for {module_path}"

    module = importlib.util.module_from_spec(spec)
    # Inject the module into sys.modules so it can be imported as a package
    sys.modules[module_name] = module

    # Capture stdout while running
    captured = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = captured

    n_pass = n_total = 0
    error_msg = ""
    try:
        spec.loader.exec_module(module)

        # The test modules execute via their `if __name__ == "__main__"` block
        # when run directly.  Here we call the main execution directly.
        # Each module defines a top-level `_results` list populated by running
        # the test functions.  We replicate what __main__ does.
        if hasattr(module, "_results"):
            results = module._results
        else:
            # Run the tests by calling each test function manually
            results = []
            tests_attr = getattr(module, "tests", None)
            if tests_attr is None:
                # Try to find all test functions decorated with _test
                tests_attr = [
                    v for k, v in vars(module).items()
                    if callable(v) and not k.startswith("_")
                ]
            if tests_attr:
                for t in tests_attr:
                    t(results)

        n_pass = sum(1 for r in results if r.get("passed", False))
        n_total = len(results)

    except SystemExit as exc:
        # Modules call sys.exit(0) or sys.exit(1) at the end
        if exc.code not in (0, 1):
            error_msg = f"sys.exit({exc.code})"
        # Parse results from captured output
        output = captured.getvalue()
        pass_count = output.count("  PASS  ")
        fail_count = output.count("  FAIL  ")
        n_pass = pass_count
        n_total = pass_count + fail_count

    except Exception as exc:
        error_msg = f"{type(exc).__name__}: {exc}"

    finally:
        sys.stdout = old_stdout

    return n_pass, n_total, error_msg


def _run_phase_subprocess(module_name: str) -> tuple[int, int, str]:
    """Run a phase test module in a subprocess and parse the results line."""
    import subprocess

    module_path = str(_TESTS_DIR / f"{module_name}.py")
    try:
        result = subprocess.run(
            [sys.executable, module_path],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(_ROOT),
        )
        output = result.stdout + result.stderr
        # Parse "Results: N/M passed" from output
        import re
        match = re.search(r"Results:\s+(\d+)/(\d+)\s+passed", output)
        if match:
            n_pass = int(match.group(1))
            n_total = int(match.group(2))
            if n_pass < n_total:
                # Extract failure details
                fail_lines = [l for l in output.split("\n") if "  FAIL  " in l]
                error_msg = "; ".join(fail_lines[:3])
            else:
                error_msg = ""
            return n_pass, n_total, error_msg
        elif result.returncode == 0:
            # No parse — assume passed
            pass_count = output.count("  PASS  ")
            fail_count = output.count("  FAIL  ")
            return pass_count, pass_count + fail_count, ""
        else:
            return 0, 0, f"Process exited {result.returncode}: {output[-200:]}"
    except subprocess.TimeoutExpired:
        return 0, 0, "TIMEOUT (>300 s)"
    except Exception as exc:
        return 0, 0, f"{type(exc).__name__}: {exc}"


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    print()
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║         XNAV Cold Start Simulator — Full Test Suite                 ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print()

    rows = []
    total_pass = total_tests = 0
    all_passed = True

    for phase_label, module_name, expected in _PHASES:
        sys.stdout.write(f"  Running {phase_label}...")
        sys.stdout.flush()
        t0 = time.monotonic()
        n_pass, n_total, error_msg = _run_phase_subprocess(module_name)
        elapsed = time.monotonic() - t0

        passed = (n_pass == n_total) and (n_total > 0) and not error_msg
        total_pass += n_pass
        total_tests += n_total
        if not passed:
            all_passed = False

        status = "PASS" if passed else "FAIL"
        rows.append((phase_label, status, n_pass, n_total, elapsed, error_msg))

        result_str = f"{n_pass}/{n_total}" if n_total > 0 else "?"
        check = "✓" if passed else "✗"
        print(f"\r  {check} {phase_label:<30}  {result_str:>7}  {elapsed:5.1f}s")

        if error_msg and not passed:
            # Print first failure detail indented
            print(f"     → {error_msg[:100]}")

    # ── Summary table ─────────────────────────────────────────────────────────
    print()
    print("──────────────────────────────────────────────────────────────────────")
    print(f"  {'Phase':<30}  {'Status':>6}  {'Pass/Total':>10}  {'Time':>6}")
    print("──────────────────────────────────────────────────────────────────────")
    for phase_label, status, n_pass, n_total, elapsed, _ in rows:
        result_str = f"{n_pass}/{n_total}" if n_total > 0 else "n/a"
        mark = "✓" if status == "PASS" else "✗"
        print(f"  {mark} {phase_label:<30}  {status:>6}  {result_str:>10}  {elapsed:5.1f}s")
    print("──────────────────────────────────────────────────────────────────────")
    overall = "ALL PASS" if all_passed else "FAILURES PRESENT"
    print(f"  Total: {total_pass}/{total_tests} passed  ·  {overall}")
    print()

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
