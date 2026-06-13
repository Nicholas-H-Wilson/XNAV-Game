#!/usr/bin/env python3
"""Curate a 200k Gaia DR3 star sample for galaxy-wide map coverage.

Source: Gaia DR3 via the ngwnos/gaia-dr3-chunked GitHub mirror (binary chunks
of 100k stars each, sorted by apparent magnitude). Columns per star:
  ra, dec, parallax, pmra, pmdec, rv, gmag, bp_rp, teff_k, dist_pc, gl, gb

Real parallax-measured stars thin out with distance, so a naive "brightest N"
selection clusters at the Sun. This tool pools many chunks and samples them
*distance-stratified* — deliberately over-weighting the far shells — so the
output spreads real stars across the near disk and beyond, out to ~12 kpc
(Gaia is extinction-limited toward the far side, which stays sparse — that is
physically real, not a shortcut).

Output: xnav_simulator/data/gaia_stars.npz with float32 columns
  x_kpc, y_kpc, z_kpc  galactocentric (Sun at [-8.178, 0, 0], app convention)
  mag                  Gaia G apparent magnitude
  teff_k               effective temperature (K; 0 if unknown)
  dist_pc              distance (parsecs)
  bp_rp                BP-RP colour index

Usage:  python tools/curate_gaia_stars.py /tmp/gaia/chunk_*.bin [--n 200000]
"""

from __future__ import annotations

import argparse
import glob
import math
import pathlib
import sys

import numpy as np

_REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "xnav_simulator"))

from config import SUN_POS_KPC  # noqa: E402

# Distance shells (kpc) and the share of the final sample drawn from each.
# Over-weights the far shells relative to their natural abundance to flatten
# the radial profile and fill the disk.
_SHELLS = [
    ((0.0, 1.0), 0.20),
    ((1.0, 3.0), 0.34),
    ((3.0, 5.0), 0.22),
    ((5.0, 8.0), 0.15),
    ((8.0, 12.0), 0.07),
    ((12.0, 20.0), 0.02),
]


def _galactic_to_xyz(gl_deg, gb_deg, d_kpc):
    """Vectorised galactic (l, b, d) → galactocentric Cartesian kpc (app frame)."""
    l = np.radians(gl_deg)
    b = np.radians(gb_deg)
    # Heliocentric Cartesian (X toward GC, Y toward rotation, Z toward NGP)
    xh = d_kpc * np.cos(b) * np.cos(l)
    yh = d_kpc * np.cos(b) * np.sin(l)
    zh = d_kpc * np.sin(b)
    # Sun at galactocentric [-8.178, 0, 0]; +X heliocentric points to GC (+x_gc dir)
    x = SUN_POS_KPC[0] + xh
    y = SUN_POS_KPC[1] + yh
    z = SUN_POS_KPC[2] + zh
    return x, y, z


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("chunks", nargs="+", help="chunk_*.bin files")
    ap.add_argument("--n", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=20260613)
    args = ap.parse_args()

    files = sorted({f for pat in args.chunks for f in glob.glob(pat)})
    if not files:
        print("No chunk files matched.")
        sys.exit(1)

    cols = []
    for f in files:
        a = np.fromfile(f, dtype="<f4").reshape(-1, 12)
        cols.append(a)
    pool = np.concatenate(cols, axis=0)
    # ra dec plx pmra pmdec rv gmag bprp teff dist gl gb
    gmag = pool[:, 6]
    bp_rp = pool[:, 7]
    teff = pool[:, 8]
    dist_pc = pool[:, 9]
    gl = pool[:, 10]
    gb = pool[:, 11]

    valid = np.isfinite(dist_pc) & (dist_pc > 0) & (dist_pc < 20000) & np.isfinite(gmag)
    idx_all = np.nonzero(valid)[0]
    d_kpc_all = dist_pc[idx_all] / 1000.0

    rng = np.random.default_rng(args.seed)
    chosen = []
    for (lo, hi), share in _SHELLS:
        want = int(round(args.n * share))
        in_shell = idx_all[(d_kpc_all >= lo) & (d_kpc_all < hi)]
        if len(in_shell) == 0 or want == 0:
            continue
        take = min(want, len(in_shell))
        chosen.append(rng.choice(in_shell, size=take, replace=False))
    sel = np.concatenate(chosen)
    rng.shuffle(sel)

    d_kpc = dist_pc[sel] / 1000.0
    x, y, z = _galactic_to_xyz(gl[sel], gb[sel], d_kpc)

    out_path = _REPO / "xnav_simulator" / "data" / "gaia_stars.npz"
    np.savez_compressed(
        out_path,
        x_kpc=x.astype(np.float32),
        y_kpc=y.astype(np.float32),
        z_kpc=z.astype(np.float32),
        mag=gmag[sel].astype(np.float32),
        teff_k=np.nan_to_num(teff[sel]).astype(np.float32),
        dist_pc=dist_pc[sel].astype(np.float32),
        bp_rp=np.nan_to_num(bp_rp[sel]).astype(np.float32),
    )
    size_mb = out_path.stat().st_size / 1e6
    # Report radial spread
    r_gc = np.sqrt(x ** 2 + y ** 2)
    print(f"Wrote {len(sel):,} Gaia stars to {out_path} ({size_mb:.1f} MB)")
    print(f"  heliocentric distance: median {np.median(d_kpc):.2f} kpc, "
          f"max {d_kpc.max():.1f} kpc")
    print(f"  galactocentric R: {r_gc.min():.1f}–{r_gc.max():.1f} kpc")
    for lo, hi in [(0, 1), (1, 3), (3, 5), (5, 8), (8, 20)]:
        n = int(((d_kpc >= lo) & (d_kpc < hi)).sum())
        print(f"    {lo}-{hi} kpc: {n:,}")


if __name__ == "__main__":
    main()
