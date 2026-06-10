# XNAV Cold Start Simulator — Agent Handoff

**Date:** 2026-06-10
**Branch:** `main` (PR #1 merged)
**Working dir:** `/workspaces/XNAV-Game`
**App entry:** `xnav_simulator/app.py` (Streamlit)

---

## 1. Project overview

A Streamlit app simulating X-ray pulsar navigation (XNAV) from a galactic cold start. The pipeline is:

```
Stage 1 (DM localisation) → Particle filter → Stage 2 (profile match)
  → Stage 3 (geometry) → Stage 4 (phase ambiguity)
```

Implements a Liu-West particle filter with shrinkage `a = sqrt(1 - h²)`, ESS-based resampling/reinjection, systematic resampling via Numba JIT. Uses NE2001/YMW16-style ISM DM grid (heliocentric, `SUN_POS_KPC = [-8.178, 0, 0]`).

Constants: `K_DM = 4.148e3` (MHz² pc⁻¹ cm³ s).

---

## 2. **CURRENT BLOCKER** — runtime crash

When the user runs the simulation in the browser, they get:

```
NameError: name 't_int' is not defined
  File "/workspaces/XNAV-Game/xnav_simulator/app.py", line 612, in _run_one_phase5_pipeline
    obs_timings, dm_vals = _build_observations(...)
  File "/workspaces/XNAV-Game/xnav_simulator/app.py", line 359, in _build_observations
    / np.sqrt(max(t_int, 1.0)))
```

**Root cause:** `_build_observations()` (app.py:295) was previously patched to scale timing noise by `1/sqrt(t_int)` (radiometer equation), but `t_int` was never added to the function signature. Line 359 references an undefined `t_int`.

**Fix needed:**
1. Add `integration_time_s: float = 1000.0` parameter to `_build_observations()` signature
2. Use `integration_time_s` (not `t_int`) on line 359
3. Update the call site at `app.py:612` (and any other callers) to pass `integration_time_s=...` from settings

Search for all call sites of `_build_observations` before editing — there may be multiple in `_run_one_phase5_pipeline()` (initial obs + iteration loop).

---

## 3. Recently completed work (in main)

Last 3 commits on main:
- `11689ce` XNAV Simulator: complete build, quality sweep & UX overhaul
- `352f84f` UX overhaul: implement all 26 findings from UI/UX audit
- `33894ee` Add README, install script, and UX improvements

What's in those commits:
- Liu-West particle filter end-to-end
- All 4 pipeline stages implemented
- 5 UI panels: Convergence, Galaxy Map, Timing, Phase, Gravity
- Onboarding, progress bar, footer metrics, CSV download
- 26 UI/UX audit findings (sidebar collapsible advanced settings, themed empty states, persisted pulsar selection, etc.)
- README + `install.sh`
- All 90/90 tests passing as of commit 8fbfe2d

---

## 4. Convergence study script (incomplete)

File: `xnav_simulator/tests/run_convergence_study.py` (untracked, not yet committed)

Runs the full Stage1→PF→Stage2→Stage3→Stage4 pipeline N times across 4 spacecraft-position presets (random, near_sun, gc, void) and reports convergence rate vs the 95% target.

**Known bugs in this script that block it from running:**
1. Calls `ism.build_grid(pulsars)` — method does not exist. Use `ism.precompute_grid(resolution_pc=...)` and check `ism.grid_loaded()` first. Note: data/ne2001_grid.npz already exists, so just construct `InterstellarMedium()` and it should load automatically.
2. Calls `ism.interpolate_dm(sc_pos, p.position_kpc)` — method does not exist. Use `ism.lookup_dm_grid(sc_pos, p.position_kpc)` (returns float) or `ism.batch_lookup(points)` (vectorised).
3. Its own inline `_build_observations()` already takes `t_int` correctly — no fix needed there.

Fix these, then run `python tests/run_convergence_study.py --n 25` from `xnav_simulator/` to verify ≥95% convergence.

---

## 5. Known physics/runtime bugs (NOT YET FIXED)

Found by two parallel quality agents. Listed in priority order.

### Critical (likely cause of <95% convergence or crashes)
1. **app.py:612 `_build_observations(t_int)` undefined** — see §2.
2. **app.py:734–758 `id_pulsars` scope** — variable defined inside `if step == 3:` block but used inside `if step >= 6:` block. Works if step 3 ran first, but fragile. Initialise `id_pulsars = pulsars[:6]` before the loop.
3. **app.py:724** — Stage 3 fallback uses first 6 pulsars when none identified. Should *skip* Stage 3 instead (wrong pulsars contaminate geometry).
4. **noise.py:152** — `frequency_mhz` not guarded against zero division.

### Major (significant convergence impact)
5. **estimator.py:385–387** — LOS direction computed from galactocentric origin, not per-particle. For nearby pulsars the error can be 57°+. Use mean particle position for the LOS bearing in the kernel (documented approximation needed).
6. **estimator.py:520** — DM path length may double-count for non-Sun-aligned cases.
7. **stage4_phase_ambiguity.py:127–134** — Tolerance widening on no-candidates can break CRT. Change to "skip this pulsar" logic.
8. **dispersion.py:156–162** — IPM Parker model off by `r₀_au` factor. Correct form: `n_e0 * r0_au * π / (2 * sin_elng) * AU_TO_PC`.
9. **stage1_dm_localisation.py:156** — Silent catastrophic fallback to uniform likelihood map when grid unavailable. Raise instead.
10. **stage3_geometry.py:135** — Weights applied even when GDOP undefined (coplanar pulsars).
11. **interstellar_medium.py:212–214** — Escaped particles silently get fill_value DM. Log/clip.

### Minor
12. **estimator.py:457** — Missing ESS guard after reinjection.
13. **estimator.py:770** — Searchsorted index needs clipping.
14. **gravity.py:127–135** — Log cancellation; use ratio form.
15. **stage1_dm_localisation.py:164** — DM sigma floor should be frequency-scaled.
16. **stage4_phase_ambiguity.py:104** — Grid step can exceed window.
17. **app.py:284–285** — Inline reset doesn't call `_do_reset()`; logic duplicated.
18. **stage2_profile_matching.py:140–145** — Zero-norm profiles should warn.

---

## 6. Key invariants and conventions (don't break these)

- **Roemer convention:** LOS is `origin → pulsar` (`-sc_pos · los_dir / c`). Particle filter kernel uses the same convention so the true particle has zero Roemer residual. **Do not switch to spacecraft→pulsar** — that creates a ~10¹¹ s mismatch and no convergence.
- **ISM grid is heliocentric.** When dividing cumulative DM by distance, use the **heliocentric** distance (`norm(midpoint - SUN_POS_KPC)`), NOT galactocentric.
- **ESS health is judged on PRE-resample ESS.** Post-resample ESS = 1.0 is normal in the Roemer-dominated regime; do not colour-code on it.
- **Phase 4 ambiguity window:** removed the spurious `/2.0` factor previously — don't reintroduce.
- **Gravity panel "buried under DM turbulence" trigger:** `gravity_ns < dm_turb_ns` (NOT vs timing noise). See `ui/gravity_panel.py:219`.
- **UI boundary:** `ui/*.py` files receive plain dicts only. **No `core/` or `stages/` imports permitted.**

---

## 7. User preferences (from auto-memory)

See `~/.claude/projects/-workspaces-XNAV-Game/memory/`:
- Prefers autonomous execution (don't ask for confirmation on routine local edits)
- Terse responses, no trailing summaries; rely on the diff
- Single bundled PRs preferred over churn-y splits for refactors

---

## 8. Suggested next steps in order

1. **Fix `t_int` NameError in `_build_observations()`** (§2) — unblocks the app.
2. Initialise `id_pulsars = pulsars[:6]` at top of `_run_one_phase5_pipeline()` to fix scope bug (§5.2).
3. Fix the convergence study script (§4): replace `build_grid` → `precompute_grid` / load-on-init, replace `interpolate_dm` → `lookup_dm_grid`.
4. Run convergence study with `--n 10` first as a smoke test, then `--n 25` for the real number.
5. If convergence <95%, work through major bugs §5.5–5.11 in order.
6. Commit and push.

---

## 9. File map (key files only)

```
xnav_simulator/
  app.py                          # Streamlit entry, _build_observations, _run_one_phase5_pipeline
  config.py                       # K_DM, SUN_POS_KPC, ACCURACY_TIERS, colours
  core/
    estimator.py                  # ParticleFilter (Liu-West)
    interstellar_medium.py        # ISM grid (precompute_grid, lookup_dm_grid, batch_lookup)
    catalogue.py                  # Pulsar catalogue loader
    spacecraft.py                 # Spacecraft.random_deep_space etc.
    timing.py                     # TimingModel (roemer_delay, dispersive_delay)
    noise.py                      # TimingNoise
    dispersion.py                 # IPM model (has Parker bug)
    gravity.py                    # Gravity potential estimate
  stages/
    stage1_dm_localisation.py
    stage2_profile_matching.py
    stage3_geometry.py
    stage4_phase_ambiguity.py
  ui/
    convergence_panel.py          # 3D particle cloud + ESS timeline + stage status
    galaxy_map.py                 # 2D galaxy view + sky map
    timing_panel.py               # Dispersive sweep + residual breakdown
    phase_panel.py                # Phase dials + ambiguity timeline
    gravity_panel.py              # Potential readout + noise budget
    sidebar.py                    # Settings panel
  tests/
    run_convergence_study.py      # NEW, broken — needs §4 fixes
  data/
    ne2001_grid.npz               # Precomputed DM grid (exists)
    atnf_cache.json               # Pulsar catalogue cache
README.md
install.sh
```
