#!/usr/bin/env python3
"""Curate the galactic-objects catalogue for the XNAV map.

Real individually-catalogued stars (HYG) all lie within ~1 kpc of the Sun —
that is the parallax limit, so galaxy-wide *clickable* coverage must come from
genuinely distributed objects.  This tool emits
xnav_simulator/data/galactic_objects.json containing:

  - Globular clusters (Harris 1996, 2010 edition; ~157 objects, halo-wide),
    parsed from tools/mwgc.dat with accurate L/B/distance.
  - A curated set of famous nebulae, molecular clouds, HII regions, supernova
    remnants, open clusters and black holes (Sgr A* + stellar-mass X-ray
    binaries), with positions and distances from standard references.

Each object carries a galactocentric position in the app frame
(Sun at [-8.178, 0, 0] kpc), a kind, and a short description for the data card.

Usage:  python tools/curate_galactic_objects.py [tools/mwgc.dat]
"""

from __future__ import annotations

import json
import math
import pathlib
import sys

import numpy as np

_REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "xnav_simulator"))

from utils.coordinates import galactic_to_cartesian  # noqa: E402


def _xyz(gl: float, gb: float, d_kpc: float) -> dict:
    p = galactic_to_cartesian(gl, gb, d_kpc)
    return {"x_kpc": round(float(p[0]), 4),
            "y_kpc": round(float(p[1]), 4),
            "z_kpc": round(float(p[2]), 4)}


# ── Harris globular clusters ──────────────────────────────────────────────────

def parse_globulars(path: pathlib.Path) -> list[dict]:
    lines = path.read_text().splitlines()
    start = None
    for i, l in enumerate(lines):
        if l.lstrip().startswith("ID") and "R_Sun" in l:
            start = i + 1
            break
    if start is None:
        raise RuntimeError("Could not find Part I header in Harris catalogue")

    out = []
    for l in lines[start:]:
        if not l.strip():
            continue
        if "Part" in l or "___" in l or "Key to" in l:
            break
        # Fixed-width: id col 0-11, name 11-23, RA 23-37, DEC 37-51,
        # L 51-59, B 59-67, R_Sun 67-74 ...
        try:
            cid = l[0:11].strip()
            name = l[11:23].strip()
            gl = float(l[51:59])
            gb = float(l[59:67])
            r_sun = float(l[67:74])
        except (ValueError, IndexError):
            continue
        if not cid:
            continue
        label = f"{cid}" + (f" ({name})" if name else "")
        out.append({
            "name": label,
            "kind": "globular",
            "desc": "Globular cluster — ancient gravitationally bound swarm of "
                    "10⁴–10⁶ stars in the galactic halo",
            "d_kpc": round(r_sun, 2),
            "gl": round(gl, 3), "gb": round(gb, 3),
            **_xyz(gl, gb, r_sun),
        })
    return out


# ── Curated exotic / nebular / cluster catalogue ──────────────────────────────
# (gl_deg, gb_deg, distance_kpc, name, kind, description)
# Positions/distances from SIMBAD / standard references; distances are best
# modern estimates and rounded.

_CURATED = [
    # ── Black holes ──────────────────────────────────────────────────────────
    (0.0, 0.0, 8.178, "Sagittarius A*", "blackhole",
     "Supermassive black hole at the Galactic Centre — 4.15 million M☉"),
    (71.33, 3.07, 2.22, "Cygnus X-1", "blackhole",
     "Stellar-mass black hole (~21 M☉) accreting from a blue supergiant; "
     "first widely accepted black-hole candidate"),
    (322.12, -0.96, 3.0, "GRO J1655-40", "blackhole",
     "Microquasar — stellar-mass black hole (~6 M☉) with relativistic jets"),
    (49.0, -3.86, 1.86, "GRS 1915+105", "blackhole",
     "Microquasar — rapidly spinning stellar-mass black hole (~12 M☉)"),
    (188.0, 2.0, 2.96, "A0620-00", "blackhole",
     "Nearest known black hole X-ray binary (~6.6 M☉)"),
    (302.0, -1.0, 9.8, "V404 Cygni", "blackhole",
     "Black-hole X-ray binary (~9 M☉) known for dramatic outbursts"),
    # ── Nebulae & star-forming regions ──────────────────────────────────────
    (209.01, -19.38, 0.39, "Orion Nebula (M42)", "nebula",
     "Nearest large star-forming region — emission nebula in Orion's sword"),
    (6.02, -1.22, 1.6, "Lagoon Nebula (M8)", "nebula",
     "Giant emission nebula and active star-forming region in Sagittarius"),
    (8.0, -0.3, 1.7, "Trifid Nebula (M20)", "nebula",
     "Combined emission, reflection and dark nebula in Sagittarius"),
    (15.0, -0.7, 1.74, "Eagle Nebula (M16)", "nebula",
     "Star-forming region hosting the 'Pillars of Creation'"),
    (80.22, 0.79, 1.5, "North America Nebula", "nebula",
     "Large emission nebula in Cygnus shaped like the continent"),
    (130.07, 3.08, 0.95, "Heart Nebula (IC 1805)", "nebula",
     "Emission nebula and open cluster in Cassiopeia"),
    (84.0, -1.0, 0.4, "Veil / Cygnus Loop", "snr",
     "Supernova remnant — expanding shell from a star that exploded ~10–20 kyr ago"),
    (184.56, -5.78, 2.0, "Crab Nebula (M1)", "snr",
     "Supernova remnant from SN 1054, powered by the Crab pulsar"),
    (327.6, 14.6, 2.2, "SN 1006 remnant", "snr",
     "Remnant of the brightest stellar event in recorded history (SN 1006)"),
    (111.7, -2.1, 3.4, "Cassiopeia A", "snr",
     "Youngest known galactic supernova remnant (~350 yr); strong radio source"),
    (10.0, -0.4, 1.5, "Omega Nebula (M17)", "nebula",
     "Bright star-forming emission nebula in Sagittarius"),
    (267.95, -1.06, 1.32, "Carina Nebula", "nebula",
     "Vast star-forming region hosting Eta Carinae and Trumpler clusters"),
    (291.0, -0.5, 4.0, "Eta Carinae", "exotic",
     "Luminous blue variable (~5 million L☉) — candidate future supernova"),
    # ── Molecular clouds ─────────────────────────────────────────────────────
    (158.0, -20.0, 0.14, "Taurus Molecular Cloud", "cloud",
     "Nearby cold molecular cloud — low-mass star-forming region"),
    (353.0, 17.0, 0.13, "Rho Ophiuchi cloud", "cloud",
     "Nearest active star-forming molecular cloud complex"),
    (265.0, 1.5, 0.45, "Vela Molecular Ridge", "cloud",
     "Massive molecular cloud complex along the Vela arm"),
    # ── Notable open clusters ────────────────────────────────────────────────
    (166.57, -23.52, 0.136, "Pleiades (M45)", "cluster",
     "Young open cluster of hot blue stars (~100 Myr) in Taurus"),
    (180.0, -20.0, 0.047, "Hyades", "cluster",
     "Nearest open cluster to the Sun (~625 Myr)"),
    (160.5, 12.4, 2.5, "Double Cluster (h+χ Per)", "cluster",
     "Pair of young open clusters in Perseus"),
    (266.4, -5.4, 0.16, "IC 2602 (Southern Pleiades)", "cluster",
     "Bright young open cluster in Carina"),
    # ── Other galaxies / satellites (very distant; on the map edge) ──────────
    (280.47, -32.89, 50.0, "Large Magellanic Cloud", "galaxy",
     "Satellite dwarf galaxy of the Milky Way (~50 kpc); off the disk"),
]


def curated_objects() -> list[dict]:
    out = []
    for gl, gb, d, name, kind, desc in _CURATED:
        out.append({
            "name": name, "kind": kind, "desc": desc,
            "d_kpc": round(d, 3), "gl": round(gl, 3), "gb": round(gb, 3),
            **_xyz(gl, gb, d),
        })
    return out


def main() -> None:
    harris_path = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 \
        else _REPO / "tools" / "mwgc.dat"
    objs = curated_objects()
    if harris_path.exists():
        gc = parse_globulars(harris_path)
        objs += gc
        print(f"Parsed {len(gc)} globular clusters from {harris_path.name}")
    else:
        print(f"WARNING: {harris_path} not found — globulars omitted")

    out_path = _REPO / "xnav_simulator" / "data" / "galactic_objects.json"
    payload = {
        "metadata": {
            "source": "Harris (1996, 2010 ed.) globulars; curated nebulae/"
                      "black holes/clusters from SIMBAD and standard references",
            "frame": "galactocentric kpc, Sun at [-8.178, 0, 0] (app convention)",
            "count": len(objs),
        },
        "objects": objs,
    }
    out_path.write_text(json.dumps(payload, separators=(",", ":")))
    by_kind: dict[str, int] = {}
    for o in objs:
        by_kind[o["kind"]] = by_kind.get(o["kind"], 0) + 1
    print(f"Wrote {len(objs)} objects to {out_path} "
          f"({out_path.stat().st_size // 1024} KB)")
    print("  by kind:", dict(sorted(by_kind.items())))


if __name__ == "__main__":
    main()
