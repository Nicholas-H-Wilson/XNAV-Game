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

- **Interactive galactic map** — top-down Milky Way disk view with spiral arms, pulsar catalogue,
  particle cloud overlay, and uncertainty circle; updates live as the filter converges
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
| psrqpy | ≥1.2 | ATNF pulsar catalogue access |
| pygedm | ≥1.1 | NE2001/YMW16 ISM DM model |
| pandas | ≥2.2 | Data handling |

First run downloads the ATNF pulsar catalogue (~5 MB) and caches it locally — internet
connection required for the very first launch only.

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
- ✓ **CONVERGED** — error < 2 kpc
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
- ISM DM model: NE2001 / YMW16 via pygedm, smoothed to grid
- Roemer delay uses filter-consistent LOS convention (origin → pulsar)

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
│   │   ├── estimator.py          # Liu-West particle filter
│   │   ├── spacecraft.py         # Spacecraft state and factory presets
│   │   ├── timing.py             # Timing model (Roemer, dispersive, noise)
│   │   ├── dispersion.py         # Solar wind + ISM DM models
│   │   ├── interstellar_medium.py # NE2001/YMW16 DM grid
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
│       ├── test_phase1.py        # Foundation (16 tests)
│       ├── test_phase2.py        # Physics (34 tests)
│       ├── test_phase3.py        # Particle filter (12 tests)
│       ├── test_phase4.py        # Stage logic (13 tests)
│       ├── test_phase5.py        # Integration (5 tests)
│       └── test_phase6.py        # UI (10 tests)
```

---

## Running the tests

```bash
cd xnav_simulator
python tests/run_all_tests.py
```

Expected output: `Total: 90/90 passed · ALL PASS`

Note: tests use a custom `_test("name")` decorator pattern, not pytest.

---

## Licence

MIT
