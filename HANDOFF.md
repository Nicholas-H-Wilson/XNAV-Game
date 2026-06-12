# XNAV Cold Start Simulator — Agent Handoff

**Date:** 2026-06-12
**Branch:** `claude/project-status-review-al6pao`
**App entry:** `xnav_simulator/app.py` (Streamlit)

---

## 1. Project overview

A Streamlit app simulating X-ray pulsar navigation (XNAV) from a galactic cold start:

```
Stage 1 (DM localisation) → Particle filter → Stage 2 (profile match)
  → Stage 3 (geometry) → Stage 4 (phase ambiguity)
```

Liu-West particle filter (shrinkage `a = sqrt(1 - h²)`) with **adaptive likelihood
tempering** (see §3), ESS-based resampling, Numba JIT likelihood kernel.
YMW16 ISM DM grid (200 pc, `data/ne2001_grid.npz`), heliocentric denominator
convention, `SUN_POS_KPC = [-8.178, 0, 0]`, `K_DM = 4.148e3`.

---

## 2. Current state (all green)

- `python tests/run_all_tests.py` → **94/94 PASS** (incl. 4 app-level AppTest
  smoke tests that boot app.py and click RUN/RESET for real).
- `python tests/run_convergence_study.py --n 25` → **100/100 converged**
  (< 2 kpc), median error 0.58 kpc, p90 1.6 kpc, Quick Look tier.
- `pip install -r requirements.txt` works on a fresh machine (pygedm/psrqpy
  are optional, lazily imported, only needed to regenerate bundled data).
- The previous blockers (t_int NameError crash on every RUN; filter freezing
  on a single particle at iteration 0) are fixed — see git log for details:
  - `04dca76` Phase A: crash fix, smoke tests, installable requirements
  - `95a28db` Phase B: adaptive tempering — genuine iterative convergence
  - `4022ea6` Phase C+D: real YMW16 grid, robust Stage-1 init, stage fixes

---

## 3. Key design decisions (don't undo without understanding)

- **Adaptive likelihood tempering** (`estimator.py`, `TEMPER_TARGET_ESS`):
  raw timing likelihoods are ~10²⁶× sharper than a kpc-scale cloud; applied
  untempered they argmax-collapse the filter in one step (the original
  "instant CONVERGED" bug). Each update scales log-likelihoods by beta found
  via **log-space** bisection (beta can be ~1e-30) targeting ESS ≈ 50%.
  Resample whenever beta < 1; 1 pc roughening floor on the Liu-West kernel.
- **Stage 1 init seeds 25% of particles uniformly over the disk**
  (`initialise_from_stage1(uniform_fraction=0.25)`): the DM-only map is
  multi-modal and can be confidently wrong; without the uniform floor the
  filter converges tightly on a wrong basin (observed 12 kpc error with
  0.003 kpc claimed uncertainty).
- **Roemer convention:** LOS is `origin → pulsar` in BOTH the observation
  builder and the filter kernel. The shared forward model lives in
  `core/observations.py` (used by app.py AND run_convergence_study.py) —
  do not fork it. Switching either side alone creates a ~1e11 s mismatch.
- **Stage 4 receives clock-only residuals** (`simulate_arrival_times`):
  phase ambiguity needs position to ~c×P (~hundreds of km); a kpc-scale fix
  leaves Roemer residuals that randomise every phase. The docstring's
  `position_estimate_kpc` "Roemer correction" was never implemented and
  cannot rescue raw totals. Stage 2 remains labelled illustrative likewise.
- **Convergence is judged blind-mode safe**: filter uncertainty
  (< `CONVERGE_UNCERTAINTY_KPC` = 2.0) + plateau (<10% improvement over 3
  iters), never the true error. Badge: CONVERGED / NOT CONVERGED / DIVERGED.
- **pygedm expects distances in PARSECS** and returns astropy Quantities —
  `interstellar_medium.py` handles both; passing kpc silently zeroed the DM
  signal in the original build.
- **ESS health is judged on PRE-resample ESS.** Post-resample ESS ≈ 1.0 is
  normal. `history[i]["beta"]` records the tempering exponent per iteration.
- **UI boundary:** `ui/*.py` receive plain dicts only; no core/stages imports.
- **Stage 2 `best_match` holds Pulsar objects, not names** — compare via
  `.name` (three call sites in app.py do this correctly now).

## 4. Regenerating bundled data

- DM grid: needs pygedm. On Debian/Ubuntu: `apt install f2c`, then build
  pygedm in a venv with `setuptools<70` and `scipy==1.11.4` (its build and
  import are both fragile on newer toolchains). Then:
  `ism.precompute_grid(resolution_pc=200)` (~6 min) → writes
  `data/ne2001_grid.npz` (1.4 MB, z symmetric ±2 kpc).
- Catalogue: `Catalogue().refresh_from_atnf()` needs psrqpy; writes
  atomically via temp file + rename.

## 5. Known remaining limitations (documented, not bugs)

- The DM grid midpoint × (path/heliocentric distance) formula is a
  single-sample LOS approximation (~30% error on steep gradients). It is
  self-consistent between observations and filter, so it does not bias
  navigation within the simulation.
- Stage 2 profile matching uses synthetic von Mises templates (illustrative;
  real system would use the EPN database). Stage 4 is post-fix illustrative
  (see §3).
- Tier `grid_resolution_pc` finer than the bundled 200 pc grid does not
  recompute it (needs pygedm); the app shows an info notice instead.
- `expected_runtime_seconds` per tier are estimates from the original brief;
  actual iteration cost is far lower since the DM table is vectorised.

## 6. User preferences (from auto-memory)

- Prefers autonomous execution (don't ask for confirmation on routine edits)
- Terse responses, no trailing summaries; rely on the diff
- Single bundled PRs preferred over churn-y splits for refactors
