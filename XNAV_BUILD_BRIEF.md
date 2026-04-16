# XNAV Cold Start Simulator — Full Build Brief
## For Claude Code

---

## Overview

Build a polished Streamlit web application simulating a spacecraft performing a true
galactic cold start using real ATNF pulsar data. The user places a spacecraft anywhere
in the galaxy, tunes noise and accuracy/speed tradeoff parameters, and watches an
iterative solver converge on position and gravity well depth through successive stages.

This is a physics simulation grounded in real data. Every approximation must be
documented inline. The user cannot read code — all testing must be automated and
all test results must be printed clearly to the terminal.

---

## Stack

- Python 3.11+
- Streamlit (UI)
- numpy, scipy, astropy, plotly, pandas, numba, pygedm, psrqpy, h5py
- See full requirements.txt at end of this document

---

## Repository Structure

```
xnav_simulator/
├── app.py                          # Streamlit entry point
├── requirements.txt
├── config.py                       # Global constants, defaults, performance tiers
├── data/
│   ├── atnf_cache.json             # Committed static ATNF snapshot (top 100 MSPs)
│   └── ne2001_grid.npz             # Precomputed electron density grid
├── core/
│   ├── pulsar.py                   # Pulsar data model
│   ├── catalogue.py                # ATNF loader and filter
│   ├── spacecraft.py               # Spacecraft state model
│   ├── galaxy.py                   # Galactic coordinate system and geometry
│   ├── interstellar_medium.py      # Electron density via pygedm
│   ├── dispersion.py               # DM computation and chromatic correction
│   ├── timing.py                   # Pulse arrival time simulation
│   ├── noise.py                    # All noise source models
│   ├── gravity.py                  # Gravitational potential and well depth
│   └── estimator.py                # Particle filter (Liu-West, ESS protection)
├── stages/
│   ├── stage1_dm_localisation.py   # Coarse galactic localisation from DM pattern
│   ├── stage2_profile_matching.py  # Pulsar ID from folded profiles
│   ├── stage3_geometry.py          # Geometric triangulation
│   └── stage4_phase_ambiguity.py   # Phase resolution via CRT approach
├── ui/
│   ├── galaxy_map.py               # Interactive galactic placement map
│   ├── timing_panel.py             # Pulse timing and DM correction visualisation
│   ├── convergence_panel.py        # Position uncertainty shrinking over iterations
│   ├── phase_panel.py              # Phase ambiguity resolution visualisation
│   ├── gravity_panel.py            # Gravity well depth estimation and reality check
│   └── sidebar.py                  # All sliders and controls including accuracy tier
├── tests/
│   ├── run_all_tests.py            # Master test runner — prints pass/fail for all phases
│   ├── test_phase1.py              # Phase 1 self-tests
│   ├── test_phase2.py              # Phase 2 self-tests
│   ├── test_phase3.py              # Phase 3 self-tests
│   ├── test_phase4.py              # Phase 4 self-tests
│   └── test_phase5.py              # Phase 5 self-tests (stages and estimator)
└── utils/
    ├── coordinates.py              # Galactic / ICRF / cartesian conversions
    ├── plotting.py                 # Shared Plotly figure builders and colour scheme
    └── logger.py                   # Simulation run logging to JSON
```

---

## Accuracy / Speed Tradeoff — Selectable Tiers

This is a first-class feature. The user selects a tier from the sidebar which controls
the number of pulsars used, the particle count, and the grid resolution simultaneously.
Each tier has a clearly labelled expected runtime displayed in the UI.

Define these tiers in config.py:

```python
ACCURACY_TIERS = {
    "Quick Look (20 pulsars)": {
        "n_pulsars": 20,
        "n_particles": 2_000,
        "grid_resolution_pc": 500,       # parsecs per grid cell
        "expected_runtime_seconds": 20,
        "description": "Fast rough estimate. Good for exploration."
    },
    "Balanced (40 pulsars)": {
        "n_pulsars": 40,
        "n_particles": 5_000,
        "grid_resolution_pc": 200,
        "expected_runtime_seconds": 60,
        "description": "Good accuracy. Recommended starting point."
    },
    "Detailed (60 pulsars)": {
        "n_pulsars": 60,
        "n_particles": 15_000,
        "grid_resolution_pc": 100,
        "expected_runtime_seconds": 120,
        "description": "High accuracy. Takes around 2 minutes."
    },
    "High Fidelity (80 pulsars)": {
        "n_pulsars": 80,
        "n_particles": 30_000,
        "grid_resolution_pc": 75,
        "expected_runtime_seconds": 300,
        "description": "Near-maximum accuracy. Takes around 5 minutes."
    },
    "Maximum (100 pulsars)": {
        "n_pulsars": 100,
        "n_particles": 50_000,
        "grid_resolution_pc": 50,
        "expected_runtime_seconds": 600,
        "description": "Full simulation. Best results, longest runtime."
    }
}
```

The tier selector is a radio button group in the sidebar, not a slider.
Display the expected runtime next to each option so the user can make an
informed choice before hitting Run.

Pulsars are always selected by ranking on timing quality (lowest timing noise first)
so that adding more pulsars always improves results monotonically.

---

## Critical Implementation Rules

### Particle Filter — Divergence Protection

This is the highest-risk component. Implement all of the following:

**1. Never initialise with a uniform galactic prior.**
Stage 1 DM localisation runs BEFORE the particle filter initialises.
Stage 1 produces a coarse likelihood map over galactic regions.
The particle filter samples its initial particles FROM that map.
This is mandatory — a uniform prior over galactic volume will cause
immediate weight collapse.

**2. Use Liu-West regularised resampling, not multinomial resampling.**
Liu-West adds small kernel-smoothed perturbations during resampling
to prevent particle degeneracy. Use parameter h=0.1 (standard value).

```python
def liu_west_resample(particles, weights, h=0.1):
    # a = (3*h - 1) / (2*h), shrinkage toward mean
    # perturb resampled particles with N(0, h^2 * variance)
    # standard Liu-West algorithm
```

**3. Check ESS at every iteration and enforce minimum diversity.**

```python
def check_ess(weights):
    ess = 1.0 / np.sum(weights**2)
    return ess

# After every weight update:
ess = check_ess(weights)
if ess < n_particles / 2:
    particles, weights = liu_west_resample(particles, weights)
if ess < n_particles / 10:
    # Reinject 20% fresh particles sampled from current best-estimate region
    n_reinject = n_particles // 5
    # replace lowest-weight particles with new samples near current mean
    reinject_diversity(particles, weights, n_reinject)
```

**4. Log ESS at every iteration** so the UI can display a health indicator
showing whether the filter is healthy or struggling.

**5. Add a convergence guard.** If ESS stays below n_particles / 10 for
three consecutive iterations, halt and display a warning:
"Filter diverged — try increasing particle count or reducing noise levels."

### Streamlit Session State

Streamlit reruns the entire script on every UI interaction.
Use this exact pattern to protect simulation state:

```python
# At top of app.py, before any UI elements
if 'filter' not in st.session_state:
    st.session_state.filter = None
if 'history' not in st.session_state:
    st.session_state.history = []
if 'iteration' not in st.session_state:
    st.session_state.iteration = 0
if 'running' not in st.session_state:
    st.session_state.running = False
```

The "Run Simulation" button sets st.session_state.running = True and
initialises the filter. Display controls (zoom, colour scheme, which
pulsar to inspect) must NEVER trigger filter reinitialisation.

Separate clearly in the UI:
- SETUP controls (tier, spacecraft position, noise settings) — changing
  these resets the simulation with a confirmation prompt
- DISPLAY controls (visualisation options) — changing these only affects
  rendering, never the filter

### ATNF Data

Bundle data/atnf_cache.json as a committed static file in the repository.
This file contains the top 100 millisecond pulsars ranked by timing quality.
The app must work fully offline using this file.

psrqpy is used only for an optional "Refresh catalogue from ATNF" button
in the sidebar. If psrqpy fails (network unavailable), catch the exception
and log a message — never crash the app.

The static JSON must include for each pulsar:
- name, period (P0), period derivative (P1), DM, GL, GB, distance (kpc),
  w50 (pulse width at 50% peak), s1400 (flux density), timing noise level

### Electron Density Model

Use pygedm for all DM computations. Never implement YMW16 manually.

```python
import pygedm
dm, tau = pygedm.dist_to_dm(gl, gb, distance_kpc, method='ymw16')
```

At startup, precompute a galactic DM lookup grid and save to
data/ne2001_grid.npz. Use this grid for fast interpolation during
particle filter iterations. Never call pygedm inside the filter loop.

Show a Streamlit progress bar during grid precomputation at startup.
Precomputation should complete in under 60 seconds.

Add stochastic ISM turbulence to the grid — multiply each cell by a
log-normal random factor with sigma=0.15 to simulate patchy ISM.
This represents the irreducible model uncertainty and is fixed at
startup (same seed per session).

### Performance Requirements

- Single iteration of the particle filter at Balanced tier (5,000 particles,
  40 pulsars) must complete in under 10 seconds on a modern laptop.
- Quick Look tier must complete full convergence in under 30 seconds.
- Balanced tier full convergence must complete in under 90 seconds.
- The likelihood evaluation loop in the particle filter MUST be Numba
  JIT compiled. This is the only code path requiring Numba.
- All DM lookups during filter iterations use the precomputed grid —
  never call pygedm inside the filter inner loop.
- Display a live progress bar and elapsed time during simulation.

### Profile Matching (Stage 2)

Generate synthetic folded profiles using von Mises distributions
parameterised by real w50 pulse width values from the ATNF catalogue.
Each pulsar gets a deterministic profile fingerprint (seeded by pulsar name).

Cross-correlation matching is internally consistent and physically
illustrative. Label this stage clearly in the UI:
"Profile matching (illustrative — real profiles would use EPN database)"

### Gravity Panel — Reality Check

The gravity well depth panel must include a bar chart comparing:
- The estimated gravity signal magnitude (timing residual in nanoseconds)
- Pulsar timing noise floor (nanoseconds)
- DM correction residual uncertainty (nanoseconds)
- Photon counting noise contribution (nanoseconds)

This chart makes viscerally clear why gravity well estimation from
pulsar timing is currently beyond reach and what improvement factors
would be needed. Label actual current technology vs simulation capability.

---

## Physics Approximation Documentation Standard

Every place where the simulation approximates real physics, add this
comment pattern immediately above the relevant code:

```python
# APPROXIMATION: [what the real system does]
# [why this approximation is acceptable for simulation purposes]
# [what the error magnitude is, if quantifiable]
```

Example:
```python
# APPROXIMATION: Real Shapiro delay integrates over the full curved
# spacetime path through the solar system using numerical GR.
# Here we use the weak-field linearised approximation valid for
# gravitational potentials << c^2. Error < 1 microsecond for
# solar system geometries, negligible vs pulsar timing noise floor.
delta_t_shapiro = -2 * G * M_sun / c**3 * np.log(r / r_ref)
```

---

## Module Specifications

### config.py

Global constants and tier definitions.

```python
# Physical constants
C_LIGHT = 2.998e8          # m/s
G_NEWTON = 6.674e-11       # m^3 kg^-1 s^-2
M_SUN = 1.989e30           # kg
PC_TO_M = 3.086e16         # metres per parsec
KPC_TO_M = 3.086e19        # metres per kiloparsec

# Dispersion constant
K_DM = 4.148e3             # MHz^2 pc^-1 cm^3 s (standard value)

# Galactic geometry (approximate)
GALAXY_RADIUS_KPC = 15.0
GALAXY_THICKNESS_KPC = 1.0
SOLAR_GALACTOCENTRIC_KPC = 8.5

# Particle filter defaults
DEFAULT_TIER = "Balanced (40 pulsars)"
LIU_WEST_H = 0.1
ESS_RESAMPLE_THRESHOLD = 0.5   # fraction of n_particles
ESS_REINJECT_THRESHOLD = 0.1   # fraction of n_particles
REINJECT_FRACTION = 0.2        # fraction of particles to reinject

# Accuracy tiers (as specified above)
ACCURACY_TIERS = { ... }
```

### core/pulsar.py

Data model for a single pulsar. Dataclass or simple class.

Fields: name, period, period_dot, dm, gl, gb, distance_kpc, w50, s1400,
timing_noise_ns, profile (numpy array generated at construction).

Method generate_profile(n_bins=128): returns a von Mises profile array
seeded deterministically from the pulsar name hash. Uses w50 to set
the concentration parameter kappa.

Method timing_quality_score(): returns a float used for ranking.
Lower timing noise and higher flux density = higher score.

### core/catalogue.py

Loads ATNF data from data/atnf_cache.json as primary source.
Returns a list of Pulsar objects sorted by timing_quality_score descending.

Method get_top_n(n): returns the n highest-quality pulsars.
This is what each accuracy tier uses.

Optional method refresh_from_atnf(): attempts psrqpy fetch,
updates the cache file if successful, silently fails if not.

### core/spacecraft.py

Spacecraft state dataclass.

Fields:
- position_kpc: np.ndarray shape (3,) — galactic cartesian XYZ
- velocity_kms: np.ndarray shape (3,)
- clock_offset_s: float — how much the local clock differs from barycentric
- true_position_kpc: np.ndarray shape (3,) — ground truth, hidden in blind mode
- central_body_mass_kg: float — for gravity well computation
- central_body_radius_m: float

Method gravitational_potential(): returns Φ at current position in m^2/s^2
from central body plus simplified galactic background potential.

### core/interstellar_medium.py

Wraps pygedm for DM computation.

Method compute_dm(gl, gb, distance_kpc): calls pygedm.dist_to_dm,
returns DM value with added stochastic turbulence component.

Method precompute_grid(resolution_pc, output_path): builds a 3D grid
of DM values over the galactic volume at the specified resolution.
Saves to output_path as .npz. Shows progress via tqdm.

Method lookup_dm_grid(position_kpc, pulsar_direction): fast trilinear
interpolation into the precomputed grid. This is what runs inside
the particle filter loop.

### core/dispersion.py

Method compute_dispersive_delay(dm, frequency_mhz): returns delay in seconds.
Uses K_DM constant from config.

Method simulate_multifreq_arrival(dm, n_channels, freq_low_mhz, freq_high_mhz):
returns array of arrival times across frequency channels showing dispersive sweep.

Method correct_dm_chromatic(arrival_times, frequencies_mhz): fits for DM
from multi-frequency arrival times. Returns corrected arrival time and
residual DM uncertainty estimate.

Method interplanetary_dm_contribution(spacecraft_pos_au, pulsar_gl, pulsar_gb):
estimates solar wind electron column contribution. Scales with 1/r^2
from Sun, anisotropic by ecliptic latitude.
# APPROXIMATION: Uses simple radial solar wind model. Real IPM is
# highly structured and time-variable with solar activity.

### core/timing.py

Central module. Computes what a spacecraft at a given position would observe
for each pulsar.

Method compute_arrival_time(pulsar, spacecraft, observation_time):
Returns the observed pulse arrival time including:
1. Geometric Roemer delay (light travel time correction for position
   within galaxy relative to barycentric reference)
2. Doppler shift from spacecraft velocity
3. Shapiro delay (weak field approximation)
4. Gravitational redshift (local clock slowing from potential)
5. Dispersive delay (from dispersion.py)
6. Timing noise (from noise.py)

Each contribution is computed and stored separately so the UI can
show a breakdown of what each effect contributes.

### core/noise.py

All noise sources. Every method takes a seed parameter for reproducibility.

Method timing_noise(pulsar, integration_time_s): returns timing residual
in seconds. Scales as pulsar.timing_noise_ns / sqrt(integration_time_s).
Add red noise component with power law index -2 for realistic MSPs.

Method photon_noise(pulsar, distance_kpc, collecting_area_m2,
integration_time_s): returns phase uncertainty from Poisson photon statistics.
Flux scales as 1/distance^2. Fewer photons = worse profile SNR = worse timing.

Method dm_turbulence(dm, baseline_kpc): returns DM uncertainty from
unmodelled ISM structure. Scales with DM and path length.

Method solar_wind_noise(spacecraft_pos_au, pulsar_direction, solar_activity):
returns variable IPM contribution. solar_activity in [0,1].

### core/gravity.py

Method gravitational_potential(position_kpc, central_body_mass_kg,
central_body_radius_m): returns Φ in m^2/s^2.
Includes central body term plus simplified galactic potential (Miyamoto-Nagai disk).

Method clock_slowing_factor(potential): returns Δf/f = Φ/c^2.

Method timing_residual_from_potential(potential): returns the timing
offset in seconds that appears identically in all pulsar observations
regardless of direction. This is the monopolar gravity signal.

Method extract_monopolar_residual(timing_residuals_per_pulsar):
takes a dict of timing residuals and extracts the common component
across all pulsars. Returns estimate and uncertainty.
The uncertainty is dominated by pulsar timing noise divided by sqrt(n_pulsars).

Method gravity_well_depth(potential, central_body_mass_kg,
central_body_radius_m): returns depth from surface to current altitude in joules/kg.

### core/estimator.py

The particle filter. Most complex module. Numba JIT on inner loop.

Class ParticleFilter:

__init__(n_particles, tier_config):
    Allocates particle arrays. Does NOT initialise positions yet.
    Waits for Stage 1 to provide the initial distribution.

initialise_from_stage1(coarse_likelihood_map):
    Samples particle positions from the Stage 1 likelihood map.
    Sets uniform weights. This replaces the dangerous uniform prior.

update(pulsar_observations, noise_params):
    For each particle, compute likelihood of observing the given
    timing residuals given that particle's position.
    Uses precomputed DM grid for fast lookup.
    Updates weights. Checks ESS. Resamples if needed.

    Inner loop MUST be decorated with @numba.jit(nopython=True).

get_estimate():
    Returns weighted mean position, velocity, clock offset.
    Returns covariance matrix (for uncertainty ellipsoid display).

get_ess():
    Returns current effective sample size as fraction of n_particles.

history property:
    Returns list of (iteration, estimate, covariance, ess) tuples.
    Used by convergence panel to animate the solution collapsing.

### stages/stage1_dm_localisation.py

Runs before particle filter initialisation.

Function run(pulsars, observed_dm_values, ism_model):
    For each point on a coarse galactic grid:
        For each pulsar:
            Compute expected DM from that grid point using ism_model
            Compare to observed DM
            Accumulate log-likelihood
    Return a probability map over the galactic grid.
    This map is what the particle filter samples from.

The galactic grid resolution is set by the accuracy tier (grid_resolution_pc).

Return value includes:
- probability_map: 3D numpy array over galactic volume
- best_region: centre and radius of highest-probability region
- dm_residuals: observed minus expected DM per pulsar (for display)

### stages/stage2_profile_matching.py

Function run(observed_profiles, pulsar_catalogue):
    For each observed profile:
        Cross-correlate with every catalogue profile
        Score matches by correlation coefficient
        Return ranked candidate list with confidence
    Return identification dict: observed index -> (pulsar, confidence)

Label this stage as illustrative in all returned metadata.

### stages/stage3_geometry.py

Function run(identified_pulsars, particle_filter):
    Given confirmed pulsar identifications and their known galactic positions,
    compute geometric line-of-sight constraints.
    Update particle filter weights based on geometric consistency.

    Compute GDOP (geometric dilution of precision) from pulsar sky distribution.
    Return GDOP value for display — shows user why sky coverage matters.

### stages/stage4_phase_ambiguity.py

Function run(identified_pulsars, arrival_times, position_estimate_kpc):
    For each pulsar pair with incommensurate periods:
        Find times consistent with both observed phases (CRT approach)
    Intersect candidate times across all pulsars
    Return resolved clock offset and ambiguity window size

    Track ambiguity window size as pulsars are added one by one.
    Return this history for the phase panel visualisation.

---

## UI Specifications

### app.py — Page Structure

```
st.set_page_config(layout="wide", page_title="XNAV Cold Start Simulator")

Header row: title left, status badge right (Not started / Running / Converged)

Main layout: sidebar left (300px), main area right

Main area tabs:
  Tab 1: Galaxy Map
  Tab 2: Timing & DM
  Tab 3: Convergence
  Tab 4: Phase Resolution
  Tab 5: Gravity Well

Footer: current best estimate readout bar (position ± uncertainty, iteration count)
```

Use a dark theme colour palette throughout — space aesthetic.
Primary accent colour: #00D4FF (cyan).
Background: #0A0A1A (near-black).
Grid lines and axes: #1A1A3A.

### ui/sidebar.py

```
Section: SIMULATION SETUP
  Accuracy Tier: radio buttons (5 options from config.ACCURACY_TIERS)
    Show expected runtime next to each option
    Show pulsar count and particle count for each option

  Spacecraft Position
    Preset scenarios: dropdown
      - Random deep space (default)
      - Near a Sun-like star
      - Galactic centre region
      - Void between spiral arms
      - Manual (shows GL/GB/Distance sliders)
    If Manual: three sliders for GL (0-360°), GB (-90 to 90°), Distance (0-30 kpc)

Section: DETECTOR SETTINGS
  Collecting area: slider 0.1 → 50 m² (default 1.0 m²)
    Label: "Larger = better photon statistics, faster profile matching"
  Frequency band low: 0.5 → 3.0 keV (default 1.0)
  Frequency band high: 3.0 → 12.0 keV (default 10.0)
  Integration time per pulsar: 100 → 10,000 s (default 1000)
    Label: "Longer = less photon noise, slower per-iteration"

Section: NOISE CONTROLS
  Timing noise: toggle on/off, amplitude multiplier 0.1x → 3x
  Photon noise: toggle on/off
  ISM turbulence: toggle on/off, amplitude multiplier 0.1x → 3x
  Solar wind: radio (Quiet / Moderate / Active)

Section: ENVIRONMENT
  Central body: dropdown (None / Earth-mass / Jupiter-mass / Solar-mass / 10x Solar)
  Include galactic background potential: checkbox (default on)

Section: DISPLAY
  Blind mode: toggle (hides true position from all panels)
    When on, show a lock icon and "True position hidden"
  Animation speed: slow / normal / fast

[RUN SIMULATION] button — prominent, full sidebar width, cyan
[RESET] button — smaller, below run
```

### ui/galaxy_map.py

Two-panel layout side by side:

Left panel — Top-down galaxy view (Plotly scatter):
  - Stylised galactic disk background using a 2D Gaussian density field
    representing spiral arm structure (log-spiral arms, 4 arms)
  - All available pulsars plotted as small dots, coloured by DM value
    (colorscale: viridis, low DM = purple, high DM = yellow)
  - Spacecraft position as a bright cyan star marker
  - In non-blind mode: true position as dim white marker
  - Uncertainty ellipse (current 1σ from particle filter covariance)
    drawn as a Plotly shape, updated each iteration
  - Lines of sight from spacecraft to each active pulsar
    (thin, semi-transparent, coloured by identification confidence)
  - Sun position marked for reference

Right panel — Pulsar sky map (Mollweide projection, Plotly):
  - All pulsars plotted in galactic coordinates as seen from spacecraft
  - Coloured by timing quality score
  - Identified pulsars highlighted with larger markers
  - GDOP indicator: colour the sky regions showing where additional
    pulsars would most improve the geometric solution

Hover tooltip on each pulsar: name, DM, period, distance, timing noise.

### ui/timing_panel.py

Pulsar selector: dropdown showing the active pulsars by name.

For the selected pulsar, show three sub-panels:

Top: Dispersive sweep plot
  - X axis: time offset (microseconds)
  - Y axis: frequency channel (keV)
  - Heatmap showing pulse intensity across frequency vs time
  - Shows the diagonal sweep before DM correction
  - Toggle: Before correction / After correction / Overlay
  - After correction: sweep collapses to vertical stripe

Middle: Timing residual breakdown bar chart
  - Horizontal bars for each contribution:
    Geometric Roemer delay / Doppler / Shapiro / Gravitational redshift /
    Dispersive delay / Timing noise / Photon noise
  - Shows relative magnitude of each effect
  - Makes clear which effects dominate

Bottom: Multi-pulsar DM residual plot
  - All active pulsars on X axis (sorted by GL)
  - Y axis: observed minus expected DM
  - Colour: confidence of DM model at that sky direction
  - Should show scatter around zero with ISM turbulence uncertainty envelope

### ui/convergence_panel.py

This is the centrepiece. Three sub-panels:

Top: 3D particle cloud (Plotly 3D scatter)
  - Particles coloured by weight (low weight = faint blue, high weight = bright cyan)
  - True position shown as red sphere (hidden in blind mode)
  - Current estimate shown as bright cyan sphere
  - Uncertainty ellipsoid as a wireframe surface
  - Playback controls: scrub slider through iteration history, play/pause

Middle: Uncertainty timeline (Plotly line chart)
  - X axis: iteration number
  - Y axis: position uncertainty radius (kpc, log scale)
  - Shows uncertainty collapsing across iterations
  - Vertical dashed lines marking when each stage completed
  - Second line: ESS as fraction of n_particles (right Y axis)
    Colour the ESS line green/amber/red based on health thresholds

Bottom: Stage completion status row
  - Four boxes: Stage 1 / Stage 2 / Stage 3 / Stage 4
  - Each shows: Not started (grey) / Running (amber pulse) / Complete (green tick)
  - On hover: show what that stage did and what it contributed to accuracy

### ui/phase_panel.py

Two sub-panels:

Left: Phase dial array
  - One circular dial per active pulsar (up to 12 shown, scrollable)
  - Each dial shows: observed phase as a line, ambiguity candidates as arcs
  - As CRT resolution proceeds, candidates eliminate each other
  - Colour: unresolved = amber, resolved = green

Right: Ambiguity window timeline
  - X axis: number of pulsars included in CRT solution
  - Y axis: ambiguity window size in seconds (log scale)
  - Shows exponential collapse as each pulsar is added
  - Annotate the line where ambiguity drops below 1 second, 1 millisecond, etc.
  - Include a horizontal reference line: "Required for 1 AU position accuracy"

### ui/gravity_panel.py

Three sub-panels:

Top: Local potential estimate
  - Large numerical readout: estimated Φ in m²/s² with ± uncertainty
  - Conversion display: "Equivalent to X km above a Y solar-mass body"
  - Confidence interval shown as a horizontal error bar on a number line
  - In non-blind mode: true value shown as a tick mark on the same number line

Middle: Reality check bar chart (MANDATORY)
  - Title: "Why this is hard: signal vs noise"
  - Horizontal log-scale bar chart
  - Bars (in timing residual nanoseconds):
      Gravity signal (estimated) — cyan
      Pulsar timing noise floor — red
      DM correction residual — orange
      Photon counting noise — yellow
      Solar wind contribution — purple
  - If gravity signal bar is shorter than any noise bar: annotate
    "Current technology cannot detect this signal"
  - Show what collecting area or pulsar count would be needed to detect it

Bottom: Improvement roadmap
  - Simple text panel: "To detect the gravity signal at this location,
    you would need: [X] m² collecting area OR [Y] pulsars with [Z] timing noise"
  - Computed dynamically from the current noise model

---

## Testing Requirements

### Automated Test Philosophy

The user cannot read or review code. Therefore:
- Every module must have a corresponding self-test
- Every self-test must print PASS or FAIL clearly with a reason
- Tests must be runnable with: python tests/run_all_tests.py
- run_all_tests.py prints a summary table at the end
- Exit code 0 if all pass, 1 if any fail

### tests/run_all_tests.py

Master runner. Imports and runs all phase test modules in order.
Catches all exceptions and marks as FAIL rather than crashing.
Prints final summary:

```
═══════════════════════════════════════
XNAV SIMULATOR — TEST RESULTS
═══════════════════════════════════════
Phase 1 — Foundation
  config.py constants          PASS
  coordinates.py transforms    PASS
  plotting.py figure creation  PASS
  pulsar.py data model         PASS
  catalogue.py ATNF load       PASS  (loaded 100 pulsars)

Phase 2 — Physics
  interstellar_medium.py       PASS
  dispersion.py correction     PASS
  timing.py arrival times      PASS
  noise.py all sources         PASS
  gravity.py potential         PASS
  spacecraft.py state          PASS

Phase 3 — Estimation
  estimator.py init            PASS
  estimator.py ESS check       PASS
  estimator.py liu-west        PASS
  estimator.py convergence     PASS  (converged in 8 iterations)

Phase 4 — Stage Logic
  stage1 dm localisation       PASS  (error: 2.3 kpc from true position)
  stage2 profile matching      PASS  (identified 18/20 pulsars correctly)
  stage3 geometry              PASS  (GDOP: 2.1)
  stage4 phase ambiguity       PASS  (resolved to 0.3 ms window)

Phase 5 — Integration
  full cold start pipeline     PASS  (final error: 0.8 kpc, 12 iterations)
  session state isolation      PASS
  tier switching               PASS

═══════════════════════════════════════
TOTAL: 24/24 passed
═══════════════════════════════════════
```

### tests/test_phase1.py — Foundation Tests

Test: config constants are physically reasonable
  - C_LIGHT within 0.1% of 2.998e8 m/s → PASS/FAIL

Test: coordinate round-trip
  - Convert galactic (l=45, b=30, d=5kpc) to cartesian and back
  - Assert round-trip error < 0.001 kpc → PASS/FAIL

Test: plotting module creates figure without error
  - Call plotting.make_empty_galaxy_figure()
  - Assert returns a plotly Figure object → PASS/FAIL

Test: pulsar profile generation is deterministic
  - Generate profile for same pulsar twice
  - Assert arrays are identical → PASS/FAIL
  - Assert profile sums to 1.0 (normalised) → PASS/FAIL

Test: catalogue loads successfully
  - Load from data/atnf_cache.json
  - Assert at least 50 pulsars loaded → PASS/FAIL
  - Assert all pulsars have valid period > 0 → PASS/FAIL
  - Assert all pulsars have valid DM > 0 → PASS/FAIL
  - Print: "Loaded N pulsars. Best timing quality: [name] at [noise] ns"

### tests/test_phase2.py — Physics Tests

Test: dispersion delay scaling
  - Compute delay at 1 GHz and 2 GHz for DM=100
  - Assert delay at 1 GHz is exactly 4x delay at 2 GHz (1/f^2 law)
  - → PASS/FAIL

Test: chromatic DM correction recovery
  - Set true DM = 150 pc/cm^3
  - Simulate 16-channel arrival times with that DM
  - Run chromatic correction
  - Assert recovered DM within 1% of true value → PASS/FAIL

Test: timing arrival time includes all components
  - Create a spacecraft at known position
  - Create a pulsar at known position
  - Compute arrival time
  - Assert return value is a dict with keys:
    geometric, doppler, shapiro, gravitational, dispersive, noise, total
  - → PASS/FAIL

Test: gravitational redshift sign and magnitude
  - Compute timing residual from potential at 1 AU from solar mass body
  - Assert residual is positive (clock runs slow deeper in well)
  - Assert magnitude within 10% of analytical value GM/rc^2
  - → PASS/FAIL

Test: noise sources produce finite values
  - Call each noise method with typical inputs
  - Assert no NaN or Inf returned → PASS/FAIL
  - Assert values are within physically plausible ranges → PASS/FAIL

Test: ISM grid precomputed file exists and loads
  - Assert data/ne2001_grid.npz exists → PASS/FAIL
  - Load the grid and assert shape is 3D → PASS/FAIL
  - Sample a DM value from the grid and assert > 0 → PASS/FAIL

### tests/test_phase3.py — Estimator Tests

Test: particle filter initialises from Stage 1 map
  - Create a mock Stage 1 likelihood map (Gaussian centred at known point)
  - Initialise particle filter from it
  - Assert particle centroid within 2 kpc of the Gaussian centre → PASS/FAIL
  - Assert ESS = n_particles at initialisation (uniform weights) → PASS/FAIL

Test: ESS computation is correct
  - Create weight array with all weight on one particle
  - Assert ESS ≈ 1 → PASS/FAIL
  - Create uniform weight array
  - Assert ESS = n_particles → PASS/FAIL

Test: Liu-West resampling preserves mean
  - Create particles with known weighted mean
  - Run Liu-West resampling
  - Assert post-resample mean within 5% of pre-resample mean → PASS/FAIL
  - Assert post-resample ESS > pre-resample ESS → PASS/FAIL

Test: filter converges on known position (key integration test)
  - Place spacecraft at known galactic position (e.g. 5 kpc from Sun, GL=90)
  - Simulate observations from top 20 pulsars with low noise
  - Run 10 filter iterations
  - Assert final position estimate within 1 kpc of true position → PASS/FAIL
  - Print iteration-by-iteration uncertainty for inspection

Test: divergence protection triggers correctly
  - Artificially set all particle weights to near zero
  - Call update()
  - Assert ESS check triggers resampling → PASS/FAIL
  - Assert reinject_diversity is called when ESS < 10% threshold → PASS/FAIL

### tests/test_phase4.py — Stage Logic Tests

Test: Stage 1 produces coarser estimate than true position
  - Run Stage 1 with 20 pulsars from known spacecraft position
  - Assert returned probability map has its peak within 5 kpc of true position
  - Print "Stage 1 localisation error: X kpc" → PASS/FAIL

Test: Stage 1 output is a valid probability distribution
  - Assert probability map sums to 1.0 (normalised) → PASS/FAIL
  - Assert no negative values → PASS/FAIL

Test: Stage 2 correctly identifies known pulsars
  - Generate synthetic profiles for 5 known pulsars
  - Add moderate noise
  - Run profile matching
  - Assert at least 4/5 identified correctly → PASS/FAIL
  - Print identification confidence scores

Test: Stage 3 GDOP is lower for well-separated pulsars
  - Compute GDOP for 5 pulsars clustered in one sky quadrant
  - Compute GDOP for 5 pulsars evenly distributed across sky
  - Assert GDOP is lower (better) for distributed case → PASS/FAIL

Test: Stage 4 phase ambiguity window shrinks monotonically
  - Run CRT phase resolution adding pulsars one by one
  - Assert ambiguity window decreases at each step → PASS/FAIL
  - Print final ambiguity window in milliseconds

### tests/test_phase5.py — Integration Tests

Test: Full cold start pipeline (most important test)
  - Place spacecraft at random known galactic position
  - Run full pipeline: Stage1 → init filter → Stage2 → Stage3 → Stage4 → iterate
  - Use Quick Look tier (20 pulsars, 2000 particles)
  - Assert final position error < 2 kpc → PASS/FAIL
  - Assert convergence in < 15 iterations → PASS/FAIL
  - Assert total runtime < 60 seconds → PASS/FAIL
  - Print full convergence history

Test: Tier switching resets filter correctly
  - Initialise filter with Quick Look tier
  - Run 3 iterations
  - Switch to Balanced tier
  - Assert filter is fully reset (no history carryover) → PASS/FAIL
  - Assert new particle count matches Balanced tier config → PASS/FAIL

Test: Blind mode hides true position from all getters
  - Set blind_mode=True on spacecraft
  - Assert spacecraft.get_display_position() returns None → PASS/FAIL
  - Assert spacecraft.true_position_kpc is not None (still stored) → PASS/FAIL

Test: Gravity panel reality check values are ordered correctly
  - For a typical scenario, get gravity signal and noise values
  - Assert gravity signal < timing noise floor (it should be — this is the point)
  - Print the bar chart values so the user can verify they look reasonable

---

## Build Order

Build phases in this exact order. Do not start a phase until the
previous phase's tests all pass.

### Phase 1 — Foundation
1. config.py
2. utils/coordinates.py
3. utils/plotting.py
4. core/pulsar.py
5. core/catalogue.py
6. tests/test_phase1.py
7. RUN: python tests/test_phase1.py — ALL must pass before continuing

### Phase 2 — Physics
8. core/interstellar_medium.py
9. core/noise.py
10. core/dispersion.py
11. core/timing.py
12. core/gravity.py
13. core/spacecraft.py
14. core/galaxy.py
15. tests/test_phase2.py
16. RUN: python tests/test_phase2.py — ALL must pass before continuing

Also at end of Phase 2: run the grid precomputation script and commit
the output to data/ne2001_grid.npz. This only runs once.

### Phase 3 — Estimation
17. core/estimator.py (with Numba JIT on likelihood loop)
18. tests/test_phase3.py
19. RUN: python tests/test_phase3.py — ALL must pass before continuing

The convergence test in Phase 3 is the most important validation gate.
If the filter does not converge on a known position in test conditions,
do not proceed to Phase 4.

### Phase 4 — Stage Logic
20. stages/stage1_dm_localisation.py
21. stages/stage2_profile_matching.py
22. stages/stage3_geometry.py
23. stages/stage4_phase_ambiguity.py
24. tests/test_phase4.py
25. RUN: python tests/test_phase4.py — ALL must pass before continuing

### Phase 5 — Integration Tests
26. tests/test_phase5.py
27. RUN: python tests/test_phase5.py — ALL must pass before continuing

### Phase 6 — UI
28. ui/sidebar.py
29. ui/galaxy_map.py
30. ui/timing_panel.py
31. ui/convergence_panel.py
32. ui/phase_panel.py
33. ui/gravity_panel.py
34. app.py
35. utils/logger.py

### Phase 7 — Final
36. tests/run_all_tests.py (master runner)
37. RUN: python tests/run_all_tests.py — print full results
38. Launch: streamlit run app.py
39. Verify app loads without errors and all 5 tabs render

---

## requirements.txt

```
streamlit>=1.32
numpy>=1.26
scipy>=1.12
astropy>=6.0
psrqpy>=1.2
pygedm>=1.1
plotly>=5.20
pandas>=2.2
numba>=0.59
h5py>=3.10
tqdm>=4.66
```

---

## How to Start the Session

Say to Claude Code:

"Read this entire document before writing any code.
Then build Phase 1 exactly as specified.
After completing Phase 1, run the Phase 1 tests and show me the output.
Do not proceed to Phase 2 until all Phase 1 tests pass.
After all phases are complete, run python tests/run_all_tests.py and
show me the full output."

---

## Notes for Claude Code

- This is a physics simulation, not a web app that happens to use physics.
  Get the physics right first. The UI is secondary.
- Every approximation must be documented with the APPROXIMATION comment pattern.
- The particle filter divergence protection is non-negotiable. Do not simplify it.
- The ATNF static cache file must be bundled — the app must work offline.
- The gravity reality check bar chart is mandatory — it is the most
  scientifically important element of the UI.
- pygedm wraps YMW16 — do not implement electron density models manually.
- The Numba JIT decorator on the likelihood loop is mandatory for performance.
- All test output must be human-readable. The user cannot review code.
  Tests are the only feedback mechanism available to them.

---

## Appendix A — Phase 2 Amendments (post-astrophysics review)

These corrections were identified during a full physics panel review after Phase 2
was built. All are implemented and tested. Carry these forward into every phase.

### A.1 Physical Constants
```python
C_LIGHT = 2.99792458e8      # IAU 2012 exact (was 2.998e8 — 0.003% error)
SOLAR_GALACTOCENTRIC_KPC = 8.178  # GRAVITY Collaboration 2019 (was 8.5 kpc)
```

### A.2 Von Mises Profile Kappa (core/pulsar.py)
The correct half-maximum condition for von Mises gives:
```python
kappa = 2.0 * np.log(2) / (np.pi * w50_phase) ** 2   # CORRECT
# NOT: kappa = 1.0 / (np.pi * w50_phase) ** 2        # 39% too small → profiles 16% too broad
```

### A.3 Solar Wind Amplitude (core/noise.py)
Parker 1/r² model with n_e0 ≈ 7 cm⁻³ at 1 AU gives:
```python
A_SW_MIN = 5.33e-5   # pc cm⁻³ AU  (solar minimum)
A_SW_MAX = 1.52e-4   # pc cm⁻³ AU  (solar maximum)
# Previous value A_sw = 2.0 to 20.0 was 37,500× too large
```

### A.4 Dispersive Delay in Particle Filter — CRITICAL
The catalogue DM (pulsar.dm) is measured Sun→pulsar and is invariant with spacecraft
position. Using it for the dispersive delay provides **zero navigation signal** in
Stage 1 localisation. The filter MUST use position-dependent DM from the ISM grid:

```python
# core/timing.py — compute_arrival_time now accepts:
model_dm: Optional[float] = None
# If provided, uses this instead of pulsar.dm for the dispersive delay.
# Pass spacecraft-to-pulsar DM from ism.lookup_dm_grid() in the filter loop.

# core/timing.py — compute_all_pulsars now accepts:
model_dms: Optional[dict] = None   # {pulsar.name: dm_value}
```

### A.5 Central Body Potential (core/gravity.py, core/spacecraft.py)
The central body potential must use the orbital radius, not the galactocentric distance:
```python
# Spacecraft dataclass now includes:
orbit_radius_m: float = 0.0   # actual orbit radius in metres
# Pass to Gravity.gravitational_potential(orbit_radius_m=...) 
# Without this, a spacecraft at 1 AU from a star at R=12 kpc from GC
# would compute the potential at 12 kpc distance instead of 1 AU distance.
```

### A.6 Precision Summation (core/timing.py)
Galactocentric Roemer delays (~10¹¹ s) exceed the float64 ULP of nanosecond-scale
gravitational terms by ~10¹⁶. Use math.fsum() for the total:
```python
total_s = math.fsum([roemer, doppler, shapiro, grav_redshift, dispersive, noise])
```

### A.7 Roemer Delay Coordinate Convention
The filter uses galactocentric spacecraft positions for the Roemer delay.
This is valid: the Sun's galactocentric position creates a constant offset
(−R₀·n̂/c) that cancels exactly in the filter's differential likelihood computation.
The SSB-relative and galactocentric formulations give identical particle differentials.

---

## Appendix B — Phase 3 Amendments (pre-build guidance)

### B.1 Numba JIT Boundary
The Numba JIT region must use only numeric arrays. Python objects (Pulsar,
Spacecraft) cannot enter nopython mode. Structure the update loop as:

**Outside JIT (Python):**
1. Extract pulsar positions, periods, noise sigmas as numpy arrays
2. Build `dm_table[n_particles, n_pulsars]` via ISM grid lookups
3. Extract particle positions/velocities as float arrays

**Inside JIT (nopython=True):**
- Roemer delay: `-dot(sc_pos, los_dir) * kpc_to_m / c_light`
- Doppler delay: `v_radial_ms / c_light * observation_time_s`
- Dispersive delay: `k_dm * dm_table[i,j] / freq_mhz²`
- Gaussian log-likelihood: `-0.5 * ((t_obs - t_pred) / sigma)²`

Shapiro and gravitational redshift are below timing noise floor; omitting them
from the JIT is acceptable — the sigma per pulsar absorbs the residual.

### B.2 Particle Initialisation
Particles are NOT Spacecraft objects. Store positions, velocities, and clock
offsets as separate numpy arrays of shape (n_particles, 3), (n_particles, 3),
and (n_particles,). Create a lightweight Spacecraft proxy only when needed by
the full timing model outside the JIT loop.

### B.3 Stage 1 Map Interface
`initialise_from_stage1(probability_map, x_arr, y_arr, z_arr)` uses
`Galaxy.sample_from_map()` which already exists in core/galaxy.py.

### B.4 Divergence Guard
If ESS remains below `n_particles × ESS_REINJECT_THRESHOLD` for three consecutive
iterations, set `filter.diverged = True` and stop iterating.
The UI reads this flag to display the divergence warning.


---

## Appendix C — Phase 3 Amendments (Estimation)

### C.1 Corrections to Phase 3 Design

**Liu-West shrinkage formula (CORRECTED)**
The brief's formula `a = (3h−1)/(2h)` is mathematically wrong — at h=0.1 it
gives a=−3.5, which causes negative covariance scaling and particle divergence.
Correct formula per Liu & West (2001) eq. 3:
```python
a = np.sqrt(1.0 - h**2)   # ≈ 0.995 at h=0.1
```
This is implemented in `core/estimator.py` with an explicit comment.

**ISM denominator (CORRECTED)**
`interstellar_medium.py lookup_dm_grid()` was using galactocentric distance
`norm(midpoint)` as the heliocentric path-length denominator.  The ISM grid
stores cumulative DM from the Sun.  Fixed to:
```python
helio_dist = norm(midpoint - SUN_POS_KPC)  # SUN_POS_KPC = [-8.178, 0, 0]
```
Error before fix: factor ~2 for off-Sun particles.

### C.2 Likelihood Sigma — Required Formula

The likelihood sigma MUST include the ISM turbulence floor:
```python
sigma_dm_turb = K_DM * 0.15 * dm_catalogue / frequency_mhz**2   # 15% grid turbulence
sigma_total = np.sqrt(sigma_timing**2 + sigma_dm_turb**2)
```
For typical MSPs (DM=30–100 pc cm⁻³ at 1400 MHz):
  sigma_dm_turb ≈ 9–35 μs >> sigma_timing ≈ 100 ns
Omitting the ISM floor causes weight collapse even at the true position.

### C.3 DM Table Construction — Required Vectorisation

The double Python loop (200,000 interpolator calls at Balanced tier) must be
replaced with a vectorised broadcast:
```python
midpoints = (particles[:, np.newaxis, :] + pulsar_positions[np.newaxis, :, :]) / 2.0
flat = midpoints.reshape(-1, 3)
dm_flat = ism._interpolator(flat)   # single RegularGridInterpolator call
dm_table = dm_flat.reshape(n_particles, n_pulsars)
```
This is implemented in `ParticleFilter._build_dm_table()`.

### C.4 Stage 1 Return Value Contract

`stages/stage1_dm_localisation.py run()` must return a dict with these keys
(consumed by `ParticleFilter.initialise_from_stage1()`):
```python
{
    "probability_map": np.ndarray,   # 3D float64, shape (nx, ny, nz), non-negative
    "x_arr": np.ndarray,             # 1D float64, galactocentric kpc
    "y_arr": np.ndarray,
    "z_arr": np.ndarray,
    "best_region": {"centre_kpc": np.ndarray, "radius_kpc": float},
    "dm_residuals": dict,            # {pulsar.name: observed_dm - expected_dm}
    "velocity_scale_kms": 200.0,     # optional; consumed by initialise_from_stage1()
}
```

### C.5 Public API Clarifications

- `get_ess()` returns ESS as a **fraction** of n_particles (0.0–1.0)
- `get_ess_absolute()` returns ESS as absolute count (0–n_particles)
- `get_estimate()` returns separate 3×3 position and velocity covariances
  (keys: `position_cov`, `velocity_cov`) NOT a single 6×6 matrix

### C.6 Navigation Signal Note

Without the ISM grid loaded, all particles receive the same catalogue DM,
so the ONLY signal is the Roemer delay (~1×10¹¹ s/kpc).  With sigma_total
~10 μs, particle discriminability is ~10 pm — far below kpc initial uncertainty.
The ISM grid's DM gradient (~5–10 pc cm⁻³/kpc) is the primary cold-start signal.
Timing noise becomes dominant only at the final sub-pc convergence phase.

---

## Appendix D — Phase 5 Amendments (Integration)

### D.1 Gravity Panel Test Spec Correction (CRITICAL)

The build brief at line 952 specifies:
  "Assert gravity signal < timing noise floor (it should be — this is the point)"

This assertion is **physically wrong** at T = 1000 s integration and must not be
used in the Phase 6 gravity panel implementation or any future test.

Correct physics at T = 1000 s:

  gravity signal  = |Φ/c²| × T ≈ 340 μs
  timing noise    = σ_ns / sqrt(T) ≈ 3 ns       (100 ns MSP, 1000 s integration)
  DM turbulence   = K_DM × 0.15 × DM / f²  ≈ 16 ms  (DM=50 pc cm⁻³, 1400 MHz)

  Ordering: timing_noise << gravity << photon_noise << DM_turbulence

The gravity signal is NOT undetectable because it is below timing noise — it is
five orders of magnitude ABOVE timing noise at 1000 s integration.

The gravity signal is undetectable because ISM DM turbulence (~16 ms) is ~46×
larger than the gravity signal (~340 μs).  DM turbulence produces a
quasi-monopolar timing offset (correlated across pulsars through shared
line-of-sight ISM structure) that cannot be distinguished from the gravitational
redshift signature with current ISM models.

The Phase 6 gravity panel reality check bar chart MUST:
1. Show DM turbulence as the dominant noise bar (not timing noise)
2. Label the annotation: "Gravity signal undetectable — buried under ISM DM turbulence"
3. NOT assert or imply that gravity is below the timing noise floor

The correct test assertion (implemented in test_phase5.py Test 4) is:
  assert gravity_signal < dm_noise_floor    # ~340 μs << 16 ms  PASS
  assert gravity_signal > timing_noise_floor  # ~340 μs >> 3 ns  PASS

### D.2 LOS Convention — Filter-Consistent Observation Generation

The timing model (timing.py) computes Roemer delay using:
  los_dir = (pulsar_pos - sc_pos) / |pulsar_pos - sc_pos|   (spacecraft → pulsar)

The particle filter kernel computes:
  los_dir = pulsar_pos / |pulsar_pos|                        (origin → pulsar)

These differ by up to ~30° for spacecraft at ~8 kpc galactocentric radius.
The mismatch creates a systematic Roemer residual at the true particle position
of ~10¹¹ s — orders of magnitude above sigma_total (~10 μs) — making all
particles equally bad and preventing convergence.

For integration tests (and for the UI simulation loop), synthetic observations
MUST be generated using the same LOS convention as the filter.  Use:
  los_dir = pulsar_pos / |pulsar_pos|   (origin → pulsar)
for both observation generation and the particle filter kernel.

This is implemented in test_phase5.py via `_build_observed_timings()`.
The app.py simulation loop must follow the same convention.

### D.3 ESS = 1.0 After Resampling is Correct

In the Roemer-dominated regime (sigma_total ~ 10 μs, Roemer discrimination
~10¹¹ s/kpc), weight collapse to a single particle after every update is
physically correct.  Liu-West resampling immediately fires, restoring uniform
weights — post-resampling ESS = 1.0.

Do not treat ESS = 1.0 as a filter health warning in this regime.
The ESS health thresholds (green/amber/red) in the convergence panel apply to
the PRE-resampling ESS, not the post-resampling ESS.
The UI should display: "ESS before resample: X / N_particles" as the health metric.

The primary kpc-scale convergence signal is DM (from the ISM grid and Stage 1
probability map), not the Roemer-driven weight updates.  Stage 1 initialises
particles in the correct galactic region (~few kpc radius); the DM gradient
then narrows the estimate per iteration.
