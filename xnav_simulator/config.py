# config.py — Global constants, defaults, and accuracy tier definitions
# XNAV Cold Start Simulator

import pathlib

import numpy as np

# ── Physical constants ────────────────────────────────────────────────────────
C_LIGHT = 2.99792458e8      # m/s  — speed of light (IAU 2012, exact)
G_NEWTON = 6.674e-11        # m^3 kg^-1 s^-2  (CODATA 2018)
M_SUN = 1.989e30            # kg
PC_TO_M = 3.086e16          # metres per parsec
KPC_TO_M = 3.086e19         # metres per kiloparsec
AU_TO_M = 1.496e11          # metres per astronomical unit

# Dispersion constant
# K_DM = 4.148e3 MHz^2 pc^-1 cm^3 s  (standard IAU 2016 value)
K_DM = 4.148e3              # MHz^2 pc^-1 cm^3 s

# ── Galactic geometry (approximate) ──────────────────────────────────────────
# APPROXIMATION: Milky Way modelled as a flat disk with constant thickness.
# Real galaxy has flares, warps, and bulge. Error: ~10-30% in outer disk.
GALAXY_RADIUS_KPC = 15.0
GALAXY_THICKNESS_KPC = 1.0
SOLAR_GALACTOCENTRIC_KPC = 8.178  # kpc — GRAVITY Collaboration 2019 (A&A 625 L10)

# ── Sun's galactocentric position ─────────────────────────────────────────────
# Sun lies on the negative-X axis in galactocentric Cartesian coordinates.
# GRAVITY Collaboration 2019 (A&A 625 L10): R₀ = 8.178 ± 0.026 kpc.
# This must be used wherever heliocentric distances are computed from
# galactocentric positions (e.g. ISM DM grid lookups).
SUN_POS_KPC: np.ndarray = np.array([-8.178, 0.0, 0.0], dtype=np.float64)

# ── Particle filter defaults ──────────────────────────────────────────────────
DEFAULT_TIER = "Balanced (40 pulsars)"
LIU_WEST_H = 0.1            # Liu-West kernel bandwidth (standard value)
ESS_RESAMPLE_THRESHOLD = 0.5   # fraction of n_particles → trigger resampling
ESS_REINJECT_THRESHOLD = 0.1   # fraction of n_particles → trigger reinjection
REINJECT_FRACTION = 0.2        # fraction of particles to reinject

# Covariance regularisation nugget — prevents singular covariance matrices
# when particles are near-degenerate (well below physical uncertainty scales).
COV_NUGGET: float = 1e-6

# ── Accuracy / speed tiers ────────────────────────────────────────────────────
ACCURACY_TIERS = {
    "Quick Look (20 pulsars)": {
        "n_pulsars": 20,
        "n_particles": 2_000,
        "grid_resolution_pc": 500,
        "expected_runtime_seconds": 20,
        "description": "Fast rough estimate. Good for exploration.",
    },
    "Balanced (40 pulsars)": {
        "n_pulsars": 40,
        "n_particles": 5_000,
        "grid_resolution_pc": 200,
        "expected_runtime_seconds": 60,
        "description": "Good accuracy. Recommended starting point.",
    },
    "Detailed (60 pulsars)": {
        "n_pulsars": 60,
        "n_particles": 15_000,
        "grid_resolution_pc": 100,
        "expected_runtime_seconds": 120,
        "description": "High accuracy. Takes around 2 minutes.",
    },
    "High Fidelity (80 pulsars)": {
        "n_pulsars": 80,
        "n_particles": 30_000,
        "grid_resolution_pc": 75,
        "expected_runtime_seconds": 300,
        "description": "Near-maximum accuracy. Takes around 5 minutes.",
    },
    "Maximum (100 pulsars)": {
        "n_pulsars": 100,
        "n_particles": 50_000,
        "grid_resolution_pc": 50,
        "expected_runtime_seconds": 600,
        "description": "Full simulation. Best results, longest runtime.",
    },
}

# ── UI colour scheme ──────────────────────────────────────────────────────────
COLOUR_ACCENT = "#00D4FF"       # cyan
COLOUR_BG = "#0A0A1A"           # near-black
COLOUR_GRID = "#1A1A3A"         # dark blue-grey for grid lines

# ── Data paths ────────────────────────────────────────────────────────────────
ROOT_DIR = pathlib.Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
ATNF_CACHE_PATH = DATA_DIR / "atnf_cache.json"
DM_GRID_PATH = DATA_DIR / "ne2001_grid.npz"
