# XNAV Cold Start Simulator

A physics-accurate interactive simulator for **X-ray pulsar navigation (XNAV)** — the technique
that lets a spacecraft determine its position anywhere in the Milky Way using only the millisecond-
precision timing signals of X-ray pulsars.

The "cold start" problem is the hardest case: no GPS, no ground contact, no prior position fix —
just a detector, a pulsar catalogue, and the laws of physics.

---

## What it demonstrates

| Stage | What happens | Why it's hard |
|-------|-------------|---------------|
| **1 — DM Localisation** | Dispersion measure of each pulsar constrains which 3D region the spacecraft is in | DM is noisy; ISM turbulence broadens the distribution |
| **2 — Profile Matching** | Cross-correlate observed X-ray pulse profiles against catalogue templates to identify pulsars | Photon noise, interstellar scattering, and DM uncertainty distort profiles |
| **3 — Geometry** | Line-of-sight directions triangulate position to ~kpc accuracy | Need 4+ non-coplanar pulsars; GDOP quantifies geometry quality |
| **4 — Phase Ambiguity** | Successive pulsar timing resolves the millisecond clock offset | Integer pulse-count ambiguity; ~millisecond window must shrink to ~μs |

A **particle filter (Liu-West)** runs throughout, maintaining a 3D probability cloud over the
spacecraft's galactic position, converging from ~15 kpc uncertainty to sub-kpc accuracy.

---

## Features

- **Interactive galactic map** — top-down Milky Way disk view with a procedurally
  rendered photographic-style backdrop (patchy spiral arms, bar, bulge, dense star
  field), point-of-light markers, particle cloud overlay, and uncertainty circle;
  updates live as the filter converges. Pinch/scroll to zoom, drag to pan (the view
  persists across filter iterations), and tap any point of light for its data card —
  pulsars show spin period, characteristic age, surface magnetic field, spin-down
  luminosity, distance, DM, and flux; the Sun shows its stellar class, age, and
  effective temperature
- **30,000 catalogued stars** — every proper-named star plus the brightest of the
  HYG database (Hipparcos/Yale/Gliese) is on the map as a tappable point of light
  with spectral type, effective temperature, distance, luminosity, and magnitudes;
  zoom into the Sun to resolve the solar neighbourhood. (Real parallax-measured
  stars all lie within ~1 kpc of the Sun, so they form a dense knot at Sol that
  resolves on zoom — that clustering is physically accurate.) Bundled locally
  (`data/hyg_stars.json`, regenerable via `tools/curate_hyg_stars.py`)
- **Galaxy-wide objects** — 184 distributed, tappable objects give coverage across
  the whole disk and halo: 157 globular clusters (Harris 1996, 2010 ed.), plus
  curated nebulae, molecular clouds, supernova remnants, open clusters and black
  holes (Sagittarius A* and stellar-mass X-ray binaries like Cygnus X-1). Each
  card names the object, its class, distance and galactic coordinates. Bundled in
  `data/galactic_objects.json` (`tools/curate_galactic_objects.py`)
- **Level-of-detail rendering** — the full 30k-star field draws when the map is
  idle; during an active simulation only the distributed objects and navigation
  markers (spacecraft estimate, pulsars, true position) render, so the per-iteration
  filter loop stays fast. Main markers are always visible
- **Pulsar sky map** — all active pulsars in galactic coordinates (GL/GB), colour-coded by timing
  noise; identified pulsars highlighted
- **3D particle cloud** — full Liu-West posterior in 3D galactocentric space with playback scrubber
  to replay convergence history
- **Timing & DM panel** — dispersive sweep heatmap, per-pulsar timing residual breakdown,
  multi-pulsar DM residual plot
- **Phase resolution panel** — ambiguity window timeline showing how each additional pulsar
  shrinks the clock-offset window; candidate scatter plot
- **Gravity well panel** — gravitational potential at current position (galactic background +
  optional central body)
- **Spacecraft presets** — Random deep space / Near a star / Galactic centre / Void between arms /
  Manual (GL, GB, distance)
- **Accuracy tiers** — Quick Look (20 pulsars, ~20s) / Balanced (40 pulsars, ~60s) /
  High Fidelity (80 pulsars, ~180s)
- **Noise controls** — timing noise, photon noise, ISM turbulence, solar wind activity all tunable
- **Blind mode** — hides true position for a realistic navigation challenge
- **Results export** — download convergence history as CSV after each run

---

## Run it on your phone (Streamlit Community Cloud)

The app is mobile-optimised (tested at Pixel-class viewports) and deploys to
Streamlit Community Cloud for free, straight from this GitHub repo. The whole
flow works from an Android browser:

1. Open [share.streamlit.io](https://share.streamlit.io) and sign in with the
   GitHub account that owns this repository.
2. Tap **Create app** → **Deploy a public app from GitHub**.
3. Fill in:
   - **Repository:** `Nicholas-H-Wilson/XNAV-Game`
   - **Branch:** `main` (or the branch you want to serve)
   - **Main file path:** `xnav_simulator/app.py`
4. (Optional) Under **Advanced settings**, select **Python 3.12**.
5. Tap **Deploy**. First build takes a few minutes; after that the app lives at
   a permanent `https://<your-app-name>.streamlit.app` URL.

On the phone: the big **▶ RUN SIMULATION** button is in the main view (the
sidebar with tier/preset/noise settings opens via the **»** chevron, top-left).
A Quick Look run completes in seconds. Add the URL to your home screen
(browser menu → *Add to Home screen*) for an app-like experience.

The repo is pre-configured for this: root `requirements.txt`,
`.streamlit/config.toml` (dark theme), and `.python-version` are all in place,
and all data (pulsar catalogue + DM grid) is bundled — no network access or
compiled dependencies needed at runtime.

---

## Quick Start (existing Python environment)

```bash
git clone https://github.com/Nicholas-H-Wilson/XNAV-Game.git
cd XNAV-Game/xnav_simulator
pip install -r requirements.txt
streamlit run app.py
```

The app opens automatically at `http://localhost:8501`.

---

## Full Installation (fresh machine)

### Prerequisites

- **Python 3.11 or 3.12** — [python.org/downloads](https://www.python.org/downloads/)
- **Git** — [git-scm.com](https://git-scm.com/)

### One-line setup (macOS / Linux)

```bash
curl -fsSL https://raw.githubusercontent.com/Nicholas-H-Wilson/XNAV-Game/main/install.sh | bash
```

Or clone first and run locally:

```bash
git clone https://github.com/Nicholas-H-Wilson/XNAV-Game.git
cd XNAV-Game
bash install.sh
```

The script:
1. Checks your Python version (3.11+ required)
2. Creates an isolated virtual environment in `.venv/`
3. Installs all dependencies
4. Launches the app

### Manual installation (Windows or step-by-step)

```bash
# 1. Clone
git clone https://github.com/Nicholas-H-Wilson/XNAV-Game.git
cd XNAV-Game

# 2. Create virtual environment
python -m venv .venv

# 3. Activate
# macOS / Linux:
source .venv/bin/activate
# Windows (PowerShell):
.venv\Scripts\Activate.ps1

# 4. Install dependencies
pip install -r xnav_simulator/requirements.txt

# 5. Launch
streamlit run xnav_simulator/app.py
```

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| streamlit | ≥1.32 | Web UI framework |
| numpy | ≥1.26 | Numerical arrays |
| scipy | ≥1.12 | Statistical routines |
| plotly | ≥5.20 | Interactive charts |
| numba | ≥0.59 | JIT-compiled particle filter |
| astropy | ≥6.0 | Coordinate transforms |
| pandas | ≥2.2 | Data handling |

The app is fully offline: the repo bundles an ATNF pulsar catalogue snapshot
(`data/atnf_cache.json`, 103 millisecond pulsars) and a precomputed YMW16 DM grid
(`data/ne2001_grid.npz`, 200 pc resolution). Two **optional** packages are only
needed to regenerate that data: `psrqpy` (live ATNF refresh) and `pygedm`
(YMW16 electron-density model; requires a C/Fortran toolchain including `f2c`
to compile, which is why it is not a hard dependency).

---

## Usage Guide

### 1. Choose your scenario (sidebar)

| Setting | What it does |
|---------|-------------|
| **Accuracy Tier** | Trade-off between speed and accuracy. Start with *Quick Look* to learn the interface. |
| **Spacecraft Position** | Where in the galaxy the spacecraft is placed. *Random deep space* puts it anywhere in the disk. |
| **Manual mode** | Set exact galactic longitude (GL), latitude (GB), and heliocentric distance. |
| **Integration time** | How long the X-ray detector observes each pulsar per iteration (seconds). Longer = less noise. |
| **Blind mode** | Hides the true position — you see only what the navigation algorithm knows. |

### 2. Run the simulation

Click **▶ RUN SIMULATION**. The filter runs 1 iteration per Streamlit rerun cycle; watch the
particle cloud contract on the **Convergence** and **Galaxy Map** tabs.

The status indicator (top right) shows:
- ◯ **NOT STARTED** — ready to run
- ● **RUNNING** — iteration in progress
- ✓ **CONVERGED** — filter uncertainty < 2 kpc (judged on the filter's own
  uncertainty, never the true position — so it works in blind mode too)
- △ **NOT CONVERGED** — run ended with uncertainty above the threshold
- ⚠ **DIVERGED** — filter collapsed (try resetting and increasing integration time)

### 3. Read the results

**Galaxy Map tab** — The cyan star (◈) is the current estimate; the red circle is the true
position (non-blind mode); the dashed circle is the 1σ uncertainty radius; faint dots are
the particle cloud projected onto the galactic plane.

**Convergence tab** — Use the playback scrubber to step through the filter history. The
timeline below shows uncertainty (kpc) and ESS health (green/amber/red) per iteration.

**Timing & DM tab** — Select a pulsar from the dropdown to inspect its dispersive sweep,
timing residual breakdown, and DM offset from the catalogue model.

**Phase Resolution tab** — The ambiguity window timeline shows the clock offset being
narrowed by successive pulsars. A well-converged run ends with a window < 1 ms.

**Footer** — Always shows current best estimate in GL/GB/distance and Cartesian XYZ,
plus 1σ uncertainty and error vs truth (non-blind mode).

### 4. Export results

After a run completes, click **⬇ Download results (CSV)** in the footer to export the
full convergence history (iteration, position estimate, error, ESS).

### 5. Reset

Click **↺ RESET** to start a new scenario. Changing the accuracy tier while a run is in
progress prompts for confirmation before resetting.

---

## Physics notes

- Dispersion constant: K_DM = 4.148 × 10³ MHz² pc⁻¹ cm³ s (IAU 2016)
- Solar galactocentric distance: R₀ = 8.178 kpc (GRAVITY Collaboration 2019)
- Timing noise: σ_TOA ∝ 1/√T_int (radiometer equation)
- Liu-West kernel bandwidth h = 0.1; ESS threshold for resampling = 50%
- Adaptive likelihood tempering: each update is raised to a power β chosen by
  log-space bisection so the post-update ESS stays near 50% — pulsar timing
  likelihoods are ~10²⁶× sharper than a kpc-scale cloud, and applying them
  untempered collapses the filter onto one particle in a single step
- ISM DM model: YMW16 via pygedm, precomputed to a 200 pc galactocentric grid
  with a 15% log-normal turbulence field baked in
- Roemer delay uses filter-consistent LOS convention (origin → pulsar)
- Phase ambiguity (Stage 4) operates on clock-only residuals: resolving the
  integer pulse count requires position knowledge of order c×P (~hundreds of
  km), so it illustrates the final refinement after the position fix

---

## Project structure

```
XNAV-Game/
├── install.sh                    # One-line setup script
├── xnav_simulator/
│   ├── app.py                    # Streamlit entry point
│   ├── config.py                 # Physical constants and tier definitions
│   ├── requirements.txt
│   ├── core/
│   │   ├── catalogue.py          # ATNF pulsar catalogue wrapper
│   │   ├── estimator.py          # Liu-West particle filter (adaptive tempering)
│   │   ├── observations.py       # Synthetic observation forward model (shared)
│   │   ├── spacecraft.py         # Spacecraft state and factory presets
│   │   ├── timing.py             # Timing model (Roemer, dispersive, noise)
│   │   ├── dispersion.py         # Solar wind + ISM DM models
│   │   ├── interstellar_medium.py # YMW16 DM grid (precompute + lookup)
│   │   ├── galaxy.py             # Disk geometry, uniform/map sampling
│   │   ├── noise.py              # Timing/DM noise models
│   │   ├── pulsar.py             # Pulsar data model + profile generation
│   │   └── gravity.py            # Galactic + central-body potential
│   ├── stages/
│   │   ├── stage1_dm_localisation.py
│   │   ├── stage2_profile_matching.py
│   │   ├── stage3_geometry.py
│   │   └── stage4_phase_ambiguity.py
│   ├── ui/
│   │   ├── sidebar.py
│   │   ├── galaxy_map.py
│   │   ├── convergence_panel.py
│   │   ├── timing_panel.py
│   │   ├── phase_panel.py
│   │   └── gravity_panel.py
│   └── tests/
│       ├── run_all_tests.py      # Master test runner: python tests/run_all_tests.py
│       ├── run_convergence_study.py # Batch convergence benchmark (95% target)
│       ├── test_phase1.py        # Foundation (16 tests)
│       ├── test_phase2.py        # Physics (34 tests)
│       ├── test_phase3.py        # Particle filter (12 tests)
│       ├── test_phase4.py        # Stage logic (13 tests)
│       ├── test_phase5.py        # Integration (5 tests)
│       ├── test_phase6.py        # UI (10 tests)
│       └── test_app_smoke.py     # App-level smoke tests via Streamlit AppTest (4 tests)
```

---

## Running the tests

```bash
cd xnav_simulator
python tests/run_all_tests.py
```

Expected output: `Total: 94/94 passed · ALL PASS`

The suite includes app-level smoke tests (`test_app_smoke.py`) that boot the real
Streamlit script headlessly and click RUN/RESET, so regressions in `app.py` itself
are caught — not just in the physics modules.

For a statistical convergence benchmark across all spacecraft presets:

```bash
python tests/run_convergence_study.py --n 25
```

Expected: ≥95% of runs converge to < 2 kpc final error (typically 100%, median
error ~0.6 kpc at Quick Look tier).

Note: tests use a custom `_test("name")` decorator pattern, not pytest.

---

## Licence

MIT
