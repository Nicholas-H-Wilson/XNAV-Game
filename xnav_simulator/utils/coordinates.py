# utils/coordinates.py — Galactic / ICRF / Cartesian coordinate conversions
# XNAV Cold Start Simulator

import numpy as np

# ── IAU galactic–ICRS rotation matrix (J2000) ────────────────────────────────
# Source: Hipparcos Catalog (ESA 1997), Perryman et al. 1997, Vol. 1, Sec. 1.5.3
# Transforms ICRS Cartesian unit vectors → Galactic Cartesian unit vectors.
# Columns are the ICRS representations of the galactic x, y, z axes.
# Constructed from: NGP at (α, δ) = (192.859508°, 27.128336°) J2000,
# and galactic longitude of ascending node l_Ω = 32.932°.
# To go galactic → ICRS multiply by the TRANSPOSE (matrix is orthogonal).
_R_EQ_TO_GAL = np.array([
    [-0.0548755604162154, -0.8734370902348850, -0.4838350155487132],
    [ 0.4941094278755837, -0.4448296299600112,  0.7469822444972189],
    [-0.8676661490190047, -0.1980763734312015,  0.4559837761750669],
], dtype=np.float64)


def galactic_to_cartesian(gl_deg: float, gb_deg: float, distance_kpc: float) -> np.ndarray:
    """Convert galactic longitude, latitude, distance → Galactic-centred Cartesian XYZ (kpc).

    Convention:
      X: towards Galactic centre (l=0, b=0)
      Y: in direction of Galactic rotation (l=90, b=0)
      Z: towards Galactic north pole (b=90)

    The Sun sits at (−SOLAR_GALACTOCENTRIC_KPC, 0, 0) in this frame.
    """
    gl = np.radians(gl_deg)
    gb = np.radians(gb_deg)

    # Position relative to the Sun (heliocentric, galactic-aligned)
    x_hel = distance_kpc * np.cos(gb) * np.cos(gl)
    y_hel = distance_kpc * np.cos(gb) * np.sin(gl)
    z_hel = distance_kpc * np.sin(gb)

    # Shift to Galactocentric frame
    # APPROXIMATION: Sun placed exactly on the X-axis at SOLAR_GALACTOCENTRIC_KPC
    # (8.178 kpc, GRAVITY Collaboration 2019).  Real Sun has a small vertical
    # offset (~17 pc above midplane) that is neglected here.
    # Error < 0.2% in position magnitude for sources > 1 kpc away.
    from config import SOLAR_GALACTOCENTRIC_KPC
    x_gc = x_hel - SOLAR_GALACTOCENTRIC_KPC
    y_gc = y_hel
    z_gc = z_hel

    return np.array([x_gc, y_gc, z_gc])


def cartesian_to_galactic(xyz_kpc: np.ndarray) -> tuple[float, float, float]:
    """Convert Galactocentric Cartesian XYZ (kpc) → (gl_deg, gb_deg, distance_kpc).

    Returns:
        gl_deg: galactic longitude in [0, 360)
        gb_deg: galactic latitude in [-90, 90]
        distance_kpc: heliocentric distance
    """
    from config import SOLAR_GALACTOCENTRIC_KPC

    # Shift back to heliocentric frame
    x_hel = xyz_kpc[0] + SOLAR_GALACTOCENTRIC_KPC
    y_hel = xyz_kpc[1]
    z_hel = xyz_kpc[2]

    distance_kpc = np.sqrt(x_hel**2 + y_hel**2 + z_hel**2)

    if distance_kpc < 1e-10:
        return 0.0, 0.0, 0.0

    gb_rad = np.arcsin(np.clip(z_hel / distance_kpc, -1.0, 1.0))
    gl_rad = np.arctan2(y_hel, x_hel)

    gl_deg = np.degrees(gl_rad) % 360.0
    gb_deg = np.degrees(gb_rad)

    return gl_deg, gb_deg, distance_kpc


def galactic_to_icrf(gl_deg: float, gb_deg: float) -> tuple[float, float]:
    """Convert galactic (l, b) coordinates to ICRS J2000 (RA, Dec) in degrees.

    Uses the IAU rotation matrix _R_EQ_TO_GAL.  Galactic → ICRS is the
    transpose of that matrix (orthogonal matrix, inverse = transpose).

    This replaces a formula-based implementation that contained a sign error
    (cos vs sin in the declination equation), which produced errors up to 90°.
    The matrix approach is the same method used by astropy, ERFA, and SLALIB.
    """
    gl = np.radians(gl_deg)
    gb = np.radians(gb_deg)

    v_gal = np.array([
        np.cos(gb) * np.cos(gl),
        np.cos(gb) * np.sin(gl),
        np.sin(gb),
    ])

    # Galactic → ICRS: multiply by transpose of eq→gal matrix
    v_eq = _R_EQ_TO_GAL.T @ v_gal
    x, y, z = v_eq

    dec = np.degrees(np.arcsin(np.clip(z, -1.0, 1.0)))
    ra = np.degrees(np.arctan2(y, x)) % 360.0

    return ra, dec


def icrf_to_galactic(ra_deg: float, dec_deg: float) -> tuple[float, float]:
    """Convert ICRS J2000 (RA, Dec) in degrees → galactic (l, b) in degrees.

    Uses the IAU rotation matrix _R_EQ_TO_GAL directly (ICRS → galactic).
    """
    ra = np.radians(ra_deg)
    dec = np.radians(dec_deg)

    v_eq = np.array([
        np.cos(dec) * np.cos(ra),
        np.cos(dec) * np.sin(ra),
        np.sin(dec),
    ])

    v_gal = _R_EQ_TO_GAL @ v_eq
    x_g, y_g, z_g = v_gal

    gb = np.degrees(np.arcsin(np.clip(z_g, -1.0, 1.0)))
    gl = np.degrees(np.arctan2(y_g, x_g)) % 360.0

    return gl, gb


def angular_separation_deg(gl1: float, gb1: float, gl2: float, gb2: float) -> float:
    """Great-circle angular separation between two galactic positions (degrees)."""
    gl1, gb1, gl2, gb2 = map(np.radians, [gl1, gb1, gl2, gb2])
    cos_sep = (np.sin(gb1) * np.sin(gb2)
               + np.cos(gb1) * np.cos(gb2) * np.cos(gl1 - gl2))
    return np.degrees(np.arccos(np.clip(cos_sep, -1.0, 1.0)))


def direction_vector(gl_deg: float, gb_deg: float) -> np.ndarray:
    """Unit vector pointing from origin toward galactic (gl, gb). Shape (3,)."""
    gl = np.radians(gl_deg)
    gb = np.radians(gb_deg)
    return np.array([
        np.cos(gb) * np.cos(gl),
        np.cos(gb) * np.sin(gl),
        np.sin(gb),
    ])
