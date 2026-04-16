# core/interstellar_medium.py — Electron density and DM grid via pygedm
# XNAV Cold Start Simulator

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.interpolate import RegularGridInterpolator

logger = logging.getLogger(__name__)


class InterstellarMedium:
    """Wraps pygedm for DM computation and manages the precomputed DM grid.

    The DM grid is the key performance component: pygedm calls are expensive
    (~1 ms each), so we precompute a 3D grid at startup and use fast trilinear
    interpolation inside the particle filter loop.

    Stochastic ISM turbulence is baked into the grid at construction time using
    a log-normal multiplicative factor (sigma=0.15 per cell).  The seed is
    fixed per session so the turbulence field is reproducible.
    """

    # APPROXIMATION: The ISM is modelled using YMW16 (Yao, Manchester & Wang 2017)
    # via pygedm.  YMW16 is a smooth, time-independent free-electron model.
    # Real ISM has time-variable turbulence, discrete H II regions, supernova
    # remnants, and structures below the YMW16 resolution (~pc scale).
    # Error: DM predictions accurate to ~20% for individual sightlines.

    def __init__(
        self,
        grid_path: Optional[Path] = None,
        turbulence_seed: int = 42,
    ) -> None:
        if grid_path is None:
            from config import DM_GRID_PATH
            grid_path = DM_GRID_PATH

        self._grid_path = Path(grid_path)
        self._turbulence_seed = turbulence_seed
        self._interpolator: Optional[RegularGridInterpolator] = None
        self._grid_coords: Optional[tuple] = None   # (x_arr, y_arr, z_arr)

        if self._grid_path.exists():
            self._load_grid()

    # ── pygedm wrapper ────────────────────────────────────────────────────────

    def compute_dm(self, gl: float, gb: float, distance_kpc: float) -> float:
        """Return DM (pc cm⁻³) from pygedm for a direction and distance.

        Adds a log-normal stochastic turbulence component (sigma=0.15).
        The turbulence seed is fixed so results are reproducible within
        a session.

        APPROXIMATION: Turbulence multiplier is spatially uncorrelated
        between calls.  Real ISM turbulence is correlated on scales of
        tens to hundreds of parsecs.  This model overstates independence
        between nearby sightlines.  Error: ~15% per sightline.
        """
        try:
            import pygedm
            dm, _ = pygedm.dist_to_dm(gl, gb, distance_kpc, method="ymw16")
            dm_val = float(dm)
        except Exception as exc:
            logger.warning("pygedm.dist_to_dm failed for (gl=%.1f, gb=%.1f, d=%.2f): %s",
                           gl, gb, distance_kpc, exc)
            # Fallback: simple exponential disk model (very rough)
            # APPROXIMATION: DM ∝ distance × exp(-|sin(gb)| / 0.2) × 30 pc cm⁻³/kpc
            gb_rad = np.radians(gb)
            dm_val = max(1.0, distance_kpc * 30.0 * np.exp(-abs(np.sin(gb_rad)) / 0.2))

        # Stochastic turbulence: log-normal multiplier, sigma=0.15
        # APPROXIMATION: Same sigma for all distances and latitudes.
        # Real turbulence amplitude is higher at low latitudes and increases
        # with path length (structure function scaling).  Error: factor ~2
        # in variance for high-DM sightlines.
        rng = np.random.default_rng(
            self._turbulence_seed ^ int(abs(gl * 100)) ^ int(abs(gb * 100))
        )
        turb_factor = rng.lognormal(mean=0.0, sigma=0.15)
        return dm_val * turb_factor

    # ── Grid precomputation ───────────────────────────────────────────────────

    def precompute_grid(
        self,
        resolution_pc: float,
        output_path: Optional[Path] = None,
        progress_callback=None,
    ) -> None:
        """Build a 3D DM grid over the galactic volume and save to .npz.

        Grid covers the galactic disk volume in galactocentric Cartesian
        coordinates (X, Y, Z in kpc).  Resolution is in parsecs.

        progress_callback(fraction: float) is called with values in [0, 1]
        so callers can drive a Streamlit progress bar.

        Grid turbulence is fixed with self._turbulence_seed so the same
        noise field appears every session (reproducible uncertainty estimates).
        """
        if output_path is None:
            output_path = self._grid_path

        from config import GALAXY_RADIUS_KPC, GALAXY_THICKNESS_KPC

        step_kpc = resolution_pc / 1000.0   # pc → kpc

        # Grid axes in galactocentric kpc
        x_arr = np.arange(-GALAXY_RADIUS_KPC, GALAXY_RADIUS_KPC + step_kpc, step_kpc)
        y_arr = np.arange(-GALAXY_RADIUS_KPC, GALAXY_RADIUS_KPC + step_kpc, step_kpc)
        z_arr = np.arange(
            -GALAXY_THICKNESS_KPC / 2.0,
            GALAXY_THICKNESS_KPC / 2.0 + step_kpc,
            step_kpc,
        )

        nx, ny, nz = len(x_arr), len(y_arr), len(z_arr)
        dm_grid = np.zeros((nx, ny, nz), dtype=np.float32)

        from utils.coordinates import cartesian_to_galactic

        logger.info(
            "Precomputing DM grid: %dx%dx%d cells at %g pc resolution …",
            nx, ny, nz, resolution_pc,
        )

        total_cells = nx * ny * nz
        completed = 0

        # Fix log-normal turbulence field for reproducibility
        rng = np.random.default_rng(self._turbulence_seed)

        # APPROXIMATION: Z cells outside ±GALAXY_THICKNESS_KPC/2 are set to
        # a floor DM of 1.0 pc cm⁻³ rather than querying pygedm (they are
        # far from the galactic plane and have negligible electron density).
        # Real ISM extends above the midplane with an exponential scale height
        # of ~1 kpc for the warm ionised medium. Floor error < 5 pc cm⁻³ for
        # sources at |z| > 0.5 kpc.

        try:
            import pygedm
        except ImportError:
            pygedm = None
            logger.warning("pygedm not available; DM grid will use fallback model.")

        for ix, x in enumerate(x_arr):
            for iy, y in enumerate(y_arr):
                for iz, z in enumerate(z_arr):
                    # Convert galactocentric (X,Y,Z) → heliocentric (gl, gb, d)
                    gl, gb, d_kpc = cartesian_to_galactic(np.array([x, y, z]))

                    if d_kpc < 0.01:
                        # At or very near the Sun — set a small floor DM
                        dm_val = 1.0
                    elif pygedm is not None:
                        try:
                            dm_val, _ = pygedm.dist_to_dm(gl, gb, d_kpc, method="ymw16")
                            dm_val = float(max(dm_val, 0.0))
                        except Exception:
                            dm_val = self._fallback_dm(gb, d_kpc)
                    else:
                        dm_val = self._fallback_dm(gb, d_kpc)

                    # Log-normal turbulence: baked into grid at precompute time
                    turb = rng.lognormal(mean=0.0, sigma=0.15)
                    dm_grid[ix, iy, iz] = np.float32(max(dm_val * turb, 0.5))

                    completed += 1

                if progress_callback is not None and ny > 0:
                    progress_callback(completed / total_cells)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_path,
            dm_grid=dm_grid,
            x_arr=x_arr,
            y_arr=y_arr,
            z_arr=z_arr,
            resolution_pc=np.array([resolution_pc]),
            turbulence_seed=np.array([self._turbulence_seed]),
        )

        logger.info("DM grid saved to %s (%d MB)",
                    output_path,
                    dm_grid.nbytes // (1024 * 1024))

        self._load_grid()   # immediately load the freshly computed grid

    # ── Grid loading and fast lookup ──────────────────────────────────────────

    def _load_grid(self) -> None:
        """Load a precomputed DM grid from .npz and build the interpolator."""
        data = np.load(self._grid_path)
        dm_grid = data["dm_grid"].astype(np.float64)
        x_arr = data["x_arr"]
        y_arr = data["y_arr"]
        z_arr = data["z_arr"]

        self._grid_coords = (x_arr, y_arr, z_arr)
        self._interpolator = RegularGridInterpolator(
            (x_arr, y_arr, z_arr),
            dm_grid,
            method="linear",
            bounds_error=False,
            fill_value=1.0,   # outside grid → minimal DM (far from plane)
        )
        logger.info(
            "DM grid loaded: %dx%dx%d, resolution %.0f pc",
            len(x_arr), len(y_arr), len(z_arr),
            float(data.get("resolution_pc", [np.nan])[0]),
        )

    def lookup_dm_grid(
        self,
        spacecraft_pos_kpc: np.ndarray,
        pulsar_pos_kpc: np.ndarray,
    ) -> float:
        """Fast trilinear DM lookup — call this inside the particle filter loop.

        Interpolates the precomputed grid at the midpoint between spacecraft
        and pulsar, scaled by distance.  Returns DM in pc cm⁻³.

        APPROXIMATION: Uses the grid value at the line-of-sight midpoint
        rather than integrating along the path.  This is adequate when the
        spacecraft and pulsar are both within the galactic disk.  Error is
        largest for high-latitude sightlines crossing the disk edge: ~30%.
        """
        if self._interpolator is None:
            raise RuntimeError(
                "DM grid not loaded. Call precompute_grid() or ensure "
                "data/ne2001_grid.npz exists before running the filter."
            )

        # Sample at the LOS midpoint — a better representative point than
        # either endpoint alone.
        #
        # APPROXIMATION: The grid stores cumulative DM from the Sun to each
        # cell (pc cm⁻³).  Dividing by the cell's distance from the Sun gives
        # a mean electron density along that line of sight (pc cm⁻³ / kpc),
        # which we then scale by the spacecraft→pulsar path length.
        # This is a single-sample approximation to the LOS integral.
        # Error: ~30% for paths crossing steep density gradients near the
        # galactic centre or high-latitude disk edges.
        midpoint = (spacecraft_pos_kpc + pulsar_pos_kpc) / 2.0
        point = midpoint.reshape(1, 3)
        dm_at_midpoint = float(self._interpolator(point)[0])

        # Convert cumulative DM (Sun→midpoint) to mean electron density.
        # IMPORTANT: The DM grid stores cumulative DM integrated from the Sun
        # outward in heliocentric coordinates.  The denominator must therefore
        # be the HELIOCENTRIC distance to the midpoint, not the galactocentric
        # distance.  Using galactocentric distance (norm of galactocentric XYZ)
        # would give a wrong denominator by up to a factor of ~2 for off-Sun
        # particles, producing a systematic error in the DM prediction.
        from config import SUN_POS_KPC
        helio_vec = midpoint - SUN_POS_KPC
        mid_dist_from_sun = float(np.linalg.norm(helio_vec))
        if mid_dist_from_sun > 0.1:
            mean_density_dm_per_kpc = dm_at_midpoint / mid_dist_from_sun
        else:
            # Near galactic centre: use a representative floor density
            mean_density_dm_per_kpc = 30.0

        distance_kpc = float(np.linalg.norm(pulsar_pos_kpc - spacecraft_pos_kpc))
        return max(mean_density_dm_per_kpc * distance_kpc, 0.5)

    def batch_lookup(self, points_kpc: np.ndarray) -> np.ndarray:
        """Vectorised DM grid lookup for an array of galactocentric positions.

        Parameters
        ----------
        points_kpc: shape (N, 3) float64 — galactocentric Cartesian kpc

        Returns
        -------
        dm_values: shape (N,) float64 — cumulative DM (pc cm⁻³) from Sun to
            each point, as stored in the precomputed grid.

        This is the public batch interface used by the particle filter's
        vectorised DM table construction.  Callers should convert the
        cumulative DM to a path DM using the heliocentric path length
        (see ParticleFilter._build_dm_table).
        """
        if self._interpolator is None:
            raise RuntimeError(
                "DM grid not loaded. Call precompute_grid() or ensure "
                "data/ne2001_grid.npz exists before running the filter."
            )
        pts = np.asarray(points_kpc, dtype=np.float64)
        if pts.ndim == 1:
            pts = pts.reshape(1, 3)
        return self._interpolator(pts)

    def grid_loaded(self) -> bool:
        """True if the precomputed grid is ready for fast lookups."""
        return self._interpolator is not None

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _fallback_dm(gb_deg: float, distance_kpc: float) -> float:
        """Simple exponential disk DM model used when pygedm is unavailable.

        APPROXIMATION: DM = distance × 30 × exp(−|sin(gb)| / 0.2) pc cm⁻³/kpc
        Based on the warm ionised medium scale height of ~200 pc.
        Accuracy: factor ~3 for individual sightlines at low latitude.
        """
        gb_rad = np.radians(gb_deg)
        return float(distance_kpc * 30.0 * np.exp(-abs(np.sin(gb_rad)) / 0.2))
