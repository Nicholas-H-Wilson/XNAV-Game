# core/galaxy.py — Galactic coordinate system and geometry helpers
# XNAV Cold Start Simulator

from __future__ import annotations

import numpy as np

from config import (
    GALAXY_RADIUS_KPC,
    GALAXY_THICKNESS_KPC,
    SOLAR_GALACTOCENTRIC_KPC,
)


class Galaxy:
    """Galactic geometry helpers used by Stage 1 and the particle filter.

    Provides grid generation, boundary checks, and sampling utilities
    over the galactic volume.
    """

    # ── Volume grid ───────────────────────────────────────────────────────────

    @staticmethod
    def make_volume_grid(resolution_pc: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (x_arr, y_arr, z_arr) axis arrays for a galactic volume grid.

        Covers ±GALAXY_RADIUS_KPC in X/Y and ±GALAXY_THICKNESS_KPC/2 in Z.
        Resolution is in parsecs.

        Used by Stage 1 to build the coarse DM likelihood map and by the
        ISM module to precompute the DM grid.
        """
        step_kpc = resolution_pc / 1000.0
        x_arr = np.arange(-GALAXY_RADIUS_KPC, GALAXY_RADIUS_KPC + step_kpc, step_kpc)
        y_arr = np.arange(-GALAXY_RADIUS_KPC, GALAXY_RADIUS_KPC + step_kpc, step_kpc)
        z_arr = np.arange(
            -GALAXY_THICKNESS_KPC / 2.0,
            GALAXY_THICKNESS_KPC / 2.0 + step_kpc,
            step_kpc,
        )
        return x_arr, y_arr, z_arr

    @staticmethod
    def in_galaxy(position_kpc: np.ndarray) -> bool:
        """Return True if the position is within the modelled galactic volume.

        Disk criterion: R_gc < GALAXY_RADIUS_KPC and |Z| < GALAXY_THICKNESS_KPC/2.

        APPROXIMATION: Flat, cylindrical disk with sharp edges.  Real galaxy
        has a flared outer disk and no sharp boundary.  Used only to reject
        clearly unphysical particle positions during filter resampling.
        """
        x, y, z = position_kpc
        r_gc = np.sqrt(x**2 + y**2)
        return (
            r_gc < GALAXY_RADIUS_KPC
            and abs(z) < GALAXY_THICKNESS_KPC / 2.0
        )

    @staticmethod
    def clip_to_galaxy(position_kpc: np.ndarray) -> np.ndarray:
        """Clip a position to stay within the modelled galactic volume."""
        pos = position_kpc.copy()
        x, y, z = pos
        r = np.sqrt(x**2 + y**2)
        if r > GALAXY_RADIUS_KPC and r > 0:
            scale = GALAXY_RADIUS_KPC / r
            pos[0] *= scale
            pos[1] *= scale
        pos[2] = np.clip(pos[2], -GALAXY_THICKNESS_KPC / 2.0, GALAXY_THICKNESS_KPC / 2.0)
        return pos

    # ── Sampling ──────────────────────────────────────────────────────────────

    @staticmethod
    def sample_uniform(
        n: int,
        rng: np.random.Generator,
        r_min_kpc: float = 0.0,
    ) -> np.ndarray:
        """Sample n points uniformly within the galactic disk volume.

        Returns array of shape (n, 3) in galactocentric Cartesian kpc.

        APPROXIMATION: Uniform spatial prior.  The actual galactic stellar
        density is exponentially concentrated toward the centre.  A uniform
        prior over the disk volume is intentionally conservative — it places
        no assumption on where the spacecraft is.  The particle filter's
        Stage 1 DM initialisation will immediately break this symmetry.
        """
        positions = np.zeros((n, 3))
        placed = 0
        batch = max(n * 2, 1000)

        while placed < n:
            # Rejection sample within the disk cylinder
            xy = rng.uniform(-GALAXY_RADIUS_KPC, GALAXY_RADIUS_KPC, size=(batch, 2))
            z = rng.uniform(
                -GALAXY_THICKNESS_KPC / 2.0,
                GALAXY_THICKNESS_KPC / 2.0,
                size=batch,
            )
            r = np.sqrt(xy[:, 0]**2 + xy[:, 1]**2)
            mask = r < GALAXY_RADIUS_KPC
            if r_min_kpc > 0:
                mask &= r > r_min_kpc

            good = np.column_stack([xy[mask], z[mask]])
            take = min(len(good), n - placed)
            positions[placed:placed + take] = good[:take]
            placed += take

        return positions

    @staticmethod
    def sample_from_map(
        probability_map: np.ndarray,
        x_arr: np.ndarray,
        y_arr: np.ndarray,
        z_arr: np.ndarray,
        n: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Sample n 3D positions from a probability map by inverse CDF.

        probability_map has shape (nx, ny, nz) and need not be normalised —
        the function normalises internally.

        Returns array of shape (n, 3) in galactocentric Cartesian kpc.
        Positions are jittered within their grid cell to avoid particle
        clustering on the grid nodes.

        This is called by ParticleFilter.initialise_from_stage1().
        """
        # Flatten and normalise
        flat = probability_map.ravel().astype(np.float64)
        flat = np.maximum(flat, 0.0)
        total = flat.sum()
        if total <= 0:
            raise ValueError("probability_map sums to zero — cannot sample from it")
        flat /= total

        # Draw n cell indices
        indices = rng.choice(len(flat), size=n, replace=True, p=flat)

        nx, ny, nz = len(x_arr), len(y_arr), len(z_arr)
        ix = indices // (ny * nz)
        iy = (indices % (ny * nz)) // nz
        iz = indices % nz

        dx = (x_arr[1] - x_arr[0]) if len(x_arr) > 1 else 1.0
        dy = (y_arr[1] - y_arr[0]) if len(y_arr) > 1 else 1.0
        dz = (z_arr[1] - z_arr[0]) if len(z_arr) > 1 else 1.0

        # Jitter within grid cell
        jitter = rng.uniform(-0.5, 0.5, size=(n, 3))
        positions = np.column_stack([
            x_arr[ix] + jitter[:, 0] * dx,
            y_arr[iy] + jitter[:, 1] * dy,
            z_arr[iz] + jitter[:, 2] * dz,
        ])

        return positions

    # ── Geometric helpers ─────────────────────────────────────────────────────

    @staticmethod
    def gdop(pulsar_positions_kpc: np.ndarray, spacecraft_pos_kpc: np.ndarray) -> float:
        """Compute Geometric Dilution of Precision (GDOP) for a pulsar set.

        GDOP measures how well the pulsars' sky directions span 3D space.
        Lower GDOP = better geometry = smaller position error for a given
        timing precision.

        Uses the standard navigation definition:
            GDOP = sqrt(trace(H^T H)^{-1})
        where H is the unit-vector matrix (each row = LOS direction to one pulsar).

        APPROXIMATION: Assumes all pulsars have equal timing weight.  A proper
        weighted GDOP would use the inverse timing noise variance as weights.
        This version correctly captures the geometry; the weighting affects
        the absolute position uncertainty but not the ranking of geometries.

        Returns np.inf if fewer than 4 pulsars (GDOP undefined — underdetermined).
        """
        if len(pulsar_positions_kpc) < 4:
            return np.inf

        sc = spacecraft_pos_kpc
        H_rows = []
        for p_pos in pulsar_positions_kpc:
            los = p_pos - sc
            dist = np.linalg.norm(los)
            if dist < 1e-10:
                continue
            H_rows.append(los / dist)

        if len(H_rows) < 4:
            return np.inf

        H = np.array(H_rows)   # shape (n_pulsars, 3)
        try:
            HtH_inv = np.linalg.inv(H.T @ H)
            gdop = float(np.sqrt(np.trace(HtH_inv)))
        except np.linalg.LinAlgError:
            gdop = np.inf

        return gdop

    @staticmethod
    def spiral_arm_density(position_kpc: np.ndarray) -> float:
        """Return a dimensionless density weight [0, 1] for spiral arm structure.

        Used for visualisation and as a prior modifier in Stage 1 when the
        spacecraft is known to be near a stellar population.

        APPROXIMATION: 4 log-spiral arms with pitch angle 12°, offset by 90°.
        Real Milky Way arm geometry is poorly constrained beyond 2–3 kpc from Sun.
        This is for illustrative purposes only — the particle filter does not
        use this as a prior.
        """
        x, y = position_kpc[0], position_kpc[1]
        r = np.sqrt(x**2 + y**2)
        if r < 0.1:
            return 1.0

        theta = np.arctan2(y, x)
        pitch = np.radians(12.0)

        max_density = 0.0
        for i in range(4):
            phi0 = i * np.pi / 2.0
            arm_theta = np.log(r / 2.0) / np.tan(pitch) + phi0
            delta = (theta - arm_theta) % (2 * np.pi)
            if delta > np.pi:
                delta -= 2 * np.pi
            density = np.exp(-delta**2 / (2 * 0.3**2)) * np.exp(-r / 6.0)
            max_density = max(max_density, density)

        # Background disk density
        disk_density = np.exp(-r / 4.0) * 0.4
        return float(min(max_density + disk_density, 1.0))
