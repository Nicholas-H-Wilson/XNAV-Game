# core/spacecraft.py — Spacecraft state model
# XNAV Cold Start Simulator

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from config import G_NEWTON, M_SUN, KPC_TO_M

logger = logging.getLogger(__name__)


@dataclass
class Spacecraft:
    """Full spacecraft state for the cold-start navigation problem.

    Coordinates are in galactocentric Cartesian kpc throughout.
    Velocities in km/s.  Clock offset in seconds.

    In blind mode (blind_mode=True), get_display_position() returns None so
    the UI cannot reveal the true position to the user.  The true position is
    still stored internally for computing synthetic observations.
    """

    # ── State ─────────────────────────────────────────────────────────────────
    position_kpc: np.ndarray           # galactocentric Cartesian XYZ (kpc)
    velocity_kms: np.ndarray           # velocity (km/s)
    clock_offset_s: float              # local clock error vs barycentric (s)
    true_position_kpc: np.ndarray      # ground truth — never exposed in blind mode
    central_body_mass_kg: float = 0.0  # mass of orbited body (0 = free space)
    central_body_radius_m: float = 0.0 # radius of orbited body (m)
    orbit_radius_m: float = 0.0        # orbital radius around central body (m); 0 = free space
    blind_mode: bool = False           # hide true position from display methods

    def __post_init__(self) -> None:
        # Ensure arrays are float64
        self.position_kpc = np.asarray(self.position_kpc, dtype=np.float64)
        self.velocity_kms = np.asarray(self.velocity_kms, dtype=np.float64)
        self.true_position_kpc = np.asarray(self.true_position_kpc, dtype=np.float64)

    # ── Display interface ─────────────────────────────────────────────────────

    def get_display_position(self) -> Optional[np.ndarray]:
        """Return true position for display, or None if blind mode is active."""
        if self.blind_mode:
            return None
        return self.true_position_kpc.copy()

    # ── Gravitational potential at current position ───────────────────────────

    def gravitational_potential(self, include_galactic: bool = True) -> float:
        """Return Φ at the spacecraft's current position (m²/s²).

        Delegates to gravity.Gravity for the full multi-component model.
        Uses orbit_radius_m (if set) for the central body term so the potential
        is computed at the actual orbital distance, not the galactocentric distance.
        """
        from core.gravity import Gravity
        return Gravity.gravitational_potential(
            self.position_kpc,
            central_body_mass_kg=self.central_body_mass_kg,
            central_body_radius_m=self.central_body_radius_m,
            include_galactic=include_galactic,
            orbit_radius_m=self.orbit_radius_m,
        )

    # ── Galactic coordinates ──────────────────────────────────────────────────

    def galactic_coords(self) -> tuple[float, float, float]:
        """Return heliocentric (gl_deg, gb_deg, distance_kpc) of spacecraft."""
        from utils.coordinates import cartesian_to_galactic
        return cartesian_to_galactic(self.position_kpc)

    def true_galactic_coords(self) -> tuple[float, float, float]:
        """Return heliocentric galactic coords of the *true* spacecraft position."""
        from utils.coordinates import cartesian_to_galactic
        return cartesian_to_galactic(self.true_position_kpc)

    # ── Convenience factory methods ───────────────────────────────────────────

    @classmethod
    def random_deep_space(cls, rng: Optional[np.random.Generator] = None) -> "Spacecraft":
        """Create a spacecraft at a random deep-space galactic position.

        Samples directly in galactocentric cylindrical coordinates so the
        position is always inside the modelled disk (R < GALAXY_RADIUS_KPC,
        |Z| < GALAXY_THICKNESS_KPC/2).  The previous heliocentric (l, b, d)
        sampling could place the craft up to ~5 kpc above the disk plane and
        beyond the disk edge — outside both the galaxy model and the DM grid,
        where observations carry no usable navigation signal.
        """
        if rng is None:
            rng = np.random.default_rng()

        from config import GALAXY_RADIUS_KPC, GALAXY_THICKNESS_KPC

        # Uniform over the disk area (sqrt for areal uniformity), inside edges
        r_gc = (GALAXY_RADIUS_KPC - 1.0) * np.sqrt(rng.uniform(0.0, 1.0))
        r_gc = max(r_gc, 1.0)
        phi = rng.uniform(0.0, 2.0 * np.pi)
        z = rng.uniform(-0.9, 0.9) * GALAXY_THICKNESS_KPC / 2.0

        pos = np.array([r_gc * np.cos(phi), r_gc * np.sin(phi), z])
        vel = rng.normal(0.0, 30.0, size=3)   # typical galactic dispersion ~30 km/s

        return cls(
            position_kpc=pos,
            velocity_kms=vel,
            clock_offset_s=rng.normal(0.0, 1e-4),
            true_position_kpc=pos.copy(),
        )

    @classmethod
    def interarm_void(cls, rng: Optional[np.random.Generator] = None) -> "Spacecraft":
        """Create a spacecraft in the void between spiral arms.

        Per build brief Appendix E.6: galactocentric radius 10–14 kpc, within
        the disk plane.  Low electron density and low stellar density — the
        hardest realistic navigation environment inside the disk.
        """
        if rng is None:
            rng = np.random.default_rng()

        from config import GALAXY_THICKNESS_KPC

        r_gc = rng.uniform(10.0, 14.0)
        phi = rng.uniform(0.0, 2.0 * np.pi)
        z = rng.uniform(-0.8, 0.8) * GALAXY_THICKNESS_KPC / 2.0

        pos = np.array([r_gc * np.cos(phi), r_gc * np.sin(phi), z])

        return cls(
            position_kpc=pos,
            velocity_kms=rng.normal(0.0, 20.0, size=3),   # low dispersion in void
            clock_offset_s=rng.normal(0.0, 1e-4),
            true_position_kpc=pos.copy(),
        )

    @classmethod
    def near_sun_like_star(cls, rng: Optional[np.random.Generator] = None) -> "Spacecraft":
        """Create a spacecraft in a circular orbit around a Sun-like star."""
        if rng is None:
            rng = np.random.default_rng()

        from config import SOLAR_GALACTOCENTRIC_KPC
        from utils.coordinates import galactic_to_cartesian

        gl = rng.uniform(0.0, 360.0)
        gb = rng.uniform(-5.0, 5.0)
        dist = rng.uniform(0.5, 3.0)

        pos = galactic_to_cartesian(gl, gb, dist)

        orbit_au = rng.uniform(0.5, 5.0)   # random orbit between 0.5 and 5 AU
        from config import AU_TO_M
        return cls(
            position_kpc=pos,
            velocity_kms=rng.normal(0.0, 30.0, size=3),
            clock_offset_s=rng.normal(0.0, 1e-6),
            true_position_kpc=pos.copy(),
            central_body_mass_kg=M_SUN,
            central_body_radius_m=6.96e8,   # solar radius
            orbit_radius_m=orbit_au * AU_TO_M,
        )

    @classmethod
    def at_galactic_centre(cls, rng: Optional[np.random.Generator] = None) -> "Spacecraft":
        """Create a spacecraft near the galactic centre region.

        Samples galactocentrically (0.5–3 kpc from the GC, in the disk).  The
        previous version sampled 0.5–3 kpc from the *Sun* toward GL≈0°, which
        left the craft 5–8 kpc from the actual centre.
        """
        if rng is None:
            rng = np.random.default_rng()

        from config import GALAXY_THICKNESS_KPC

        r_gc = rng.uniform(0.5, 3.0)
        phi = rng.uniform(0.0, 2.0 * np.pi)
        z = rng.uniform(-0.8, 0.8) * GALAXY_THICKNESS_KPC / 2.0

        pos = np.array([r_gc * np.cos(phi), r_gc * np.sin(phi), z])

        return cls(
            position_kpc=pos,
            velocity_kms=rng.normal(0.0, 100.0, size=3),  # high velocity dispersion near GC
            clock_offset_s=rng.normal(0.0, 1e-5),
            true_position_kpc=pos.copy(),
        )

    @classmethod
    def from_galactic(
        cls,
        gl_deg: float,
        gb_deg: float,
        distance_kpc: float,
        velocity_kms: Optional[np.ndarray] = None,
        clock_offset_s: float = 0.0,
        blind_mode: bool = False,
        **kwargs,
    ) -> "Spacecraft":
        """Create a spacecraft from explicit galactic coordinates.

        Positions outside the modelled galactic disk are allowed (the position
        is the user's to choose) but logged: the galaxy model and DM grid only
        cover the disk, so navigation signal degrades sharply out there.
        """
        from core.galaxy import Galaxy
        from utils.coordinates import galactic_to_cartesian

        pos = galactic_to_cartesian(gl_deg, gb_deg, distance_kpc)
        if not Galaxy.in_galaxy(pos):
            logger.warning(
                "Position %s kpc is outside the modelled galactic disk; "
                "DM navigation signal will be degraded there.", np.round(pos, 2),
            )
        if velocity_kms is None:
            velocity_kms = np.zeros(3)

        return cls(
            position_kpc=pos,
            velocity_kms=np.asarray(velocity_kms, dtype=np.float64),
            clock_offset_s=clock_offset_s,
            true_position_kpc=pos.copy(),
            blind_mode=blind_mode,
            **kwargs,
        )

    def __repr__(self) -> str:
        gl, gb, d = self.galactic_coords()
        return (
            f"Spacecraft(gl={gl:.1f}°, gb={gb:.1f}°, d={d:.2f} kpc, "
            f"clock_offset={self.clock_offset_s:.3e} s)"
        )
