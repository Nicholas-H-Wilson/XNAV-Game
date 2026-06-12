#!/usr/bin/env python3
"""Curate a local star catalogue for the XNAV galaxy map from the HYG database.

Source: HYG v4.1 (https://github.com/astronexus/HYG-Database, CC BY-SA 4.0,
compiled from Hipparcos, Yale BSC and Gliese). This script selects every star
with a proper name plus the brightest stars by apparent magnitude up to a cap,
and writes xnav_simulator/data/hyg_stars.json with:

  name      display name (proper > Bayer+constellation > Gliese > HIP/HD id)
  spect     spectral type string from the catalogue ("" if unknown)
  d_pc      heliocentric distance in parsecs
  mag       apparent magnitude
  absmag    absolute magnitude
  lum       luminosity (L_sun)
  teff_k    effective temperature from B-V via Ballesteros (2012), or null
  x/y/z_kpc galactocentric Cartesian position in the app convention
            (Sun at [-8.178, 0, 0] kpc; same frame as Pulsar.position_kpc)

Usage:  python tools/curate_hyg_stars.py path/to/hygdata_v41.csv [--n 3000]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import sys

import numpy as np

_REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "xnav_simulator"))

from config import SUN_POS_KPC                      # noqa: E402
from utils.coordinates import galactic_to_cartesian  # noqa: E402


def _teff_ballesteros(ci: float) -> float | None:
    """Effective temperature (K) from B-V colour index, Ballesteros (2012)."""
    try:
        return round(4600.0 * (1.0 / (0.92 * ci + 1.7)
                               + 1.0 / (0.92 * ci + 0.62)))
    except (TypeError, ZeroDivisionError):
        return None


def _radec_to_galactic(ra_deg: float, dec_deg: float) -> tuple[float, float]:
    """Equatorial (J2000, degrees) → galactic (l, b) in degrees."""
    # IAU galactic pole / centre (J2000)
    ra_p = math.radians(192.85948)    # NGP right ascension
    dec_p = math.radians(27.12825)    # NGP declination
    l_ncp = math.radians(122.93192)   # galactic longitude of NCP

    ra = math.radians(ra_deg)
    dec = math.radians(dec_deg)

    sin_b = (math.sin(dec_p) * math.sin(dec)
             + math.cos(dec_p) * math.cos(dec) * math.cos(ra - ra_p))
    b = math.asin(max(-1.0, min(1.0, sin_b)))
    y = math.cos(dec) * math.sin(ra - ra_p)
    x = (math.cos(dec_p) * math.sin(dec)
         - math.sin(dec_p) * math.cos(dec) * math.cos(ra - ra_p))
    l = (l_ncp - math.atan2(y, x)) % (2 * math.pi)
    return math.degrees(l), math.degrees(b)


def _display_name(row: dict) -> str:
    if row["proper"]:
        return row["proper"]
    if row["bf"]:
        return row["bf"]
    if row["gl"]:
        return f"Gliese {row['gl'].strip('Gl ').strip()}"
    if row["hip"]:
        return f"HIP {row['hip']}"
    if row["hd"]:
        return f"HD {row['hd']}"
    return f"HYG {row['id']}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("--n", type=int, default=3000,
                    help="total stars to keep (named stars always included)")
    args = ap.parse_args()

    rows = []
    with open(args.csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            if row["id"] == "0":        # Sol — drawn separately on the map
                continue
            try:
                dist = float(row["dist"])
                mag = float(row["mag"])
            except ValueError:
                continue
            if dist <= 0 or dist >= 100000:   # 100000 = unknown distance in HYG
                continue
            rows.append(row)

    rows.sort(key=lambda r: float(r["mag"]))
    named = [r for r in rows if r["proper"]]
    bright = [r for r in rows if not r["proper"]][: max(args.n - len(named), 0)]
    keep = sorted(named + bright, key=lambda r: float(r["mag"]))

    out = []
    for r in keep:
        gl, gb = _radec_to_galactic(float(r["ra"]) * 15.0, float(r["dec"]))
        d_kpc = float(r["dist"]) / 1000.0
        pos = galactic_to_cartesian(gl, gb, d_kpc)
        ci = None
        try:
            ci = float(r["ci"])
        except ValueError:
            pass
        out.append({
            "name": _display_name(r),
            "spect": r["spect"] or "",
            "d_pc": round(float(r["dist"]), 2),
            "mag": round(float(r["mag"]), 2),
            "absmag": round(float(r["absmag"]), 2),
            "lum": round(float(r["lum"]), 4) if r["lum"] else None,
            "teff_k": _teff_ballesteros(ci) if ci is not None else None,
            "x_kpc": round(float(pos[0]), 5),
            "y_kpc": round(float(pos[1]), 5),
            "z_kpc": round(float(pos[2]), 5),
        })

    out_path = _REPO / "xnav_simulator" / "data" / "hyg_stars.json"
    payload = {
        "metadata": {
            "source": "HYG v4.1 (github.com/astronexus/HYG-Database), CC BY-SA 4.0",
            "selection": f"all proper-named stars + brightest others, {len(out)} total",
            "frame": "galactocentric kpc, Sun at [-8.178, 0, 0] (app convention)",
            "teff": "Ballesteros (2012) from B-V colour index",
        },
        "stars": out,
    }
    out_path.write_text(json.dumps(payload, separators=(",", ":")))
    named_count = sum(1 for s in out if not s["name"].startswith(("HIP", "HD", "Gliese", "HYG")))
    print(f"Wrote {len(out)} stars ({named_count} proper-named) to {out_path} "
          f"({out_path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
