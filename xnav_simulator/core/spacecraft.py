# core/spacecraft.py — Spacecraft state model
# XNAV Cold Start Simulator

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from config import G_NEWTON, M_SUN, KPC_TO_M


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
        """Create a spacecraft at a random deep-space galactic position."""
        if rng is None:
            rng = np.random.default_rng()

        from config import GALAXY_RADIUS_KPC, GALAXY_THICKNESS_KPC
        from utils.coordinates import galactic_to_cartesian

        # Uniform random in galactic (l, b, d) — within the disk
        gl = rng.uniform(0.0, 360.0)
        gb = rng.uniform(-20.0, 20.0)      # stay near galactic plane
        dist = rng.uniform(1.0, GALAXY_RADIUS_KPC - 1.0)

        pos = galactic_to_cartesian(gl, gb, dist)
        vel = rng.normal(0.0, 30.0, size=3)   # typical galactic dispersion ~30 km/s

        return cls(
            position_kpc=pos,
            velocity_kms=vel,
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
        """Create a spacecraft near the galactic centre region."""
        if rng is None:
            rng = np.random.default_rng()

        from utils.coordinates import galactic_to_cartesian

        gl = rng.uniform(355.0, 365.0) % 360.0
        gb = rng.uniform(-5.0, 5.0)
        dist = rng.uniform(0.5, 3.0)   # kpc from Sun, near GC direction

        pos = galactic_to_cartesian(gl, gb, dist)

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
        """Create a spacecraft from explicit galactic coordinates."""
        from utils.coordinates import galactic_to_cartesian

        pos = galactic_to_cartesian(gl_deg, gb_deg, distance_kpc)
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
