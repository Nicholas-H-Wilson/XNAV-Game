# core/catalogue.py — ATNF pulsar catalogue loader and filter
# XNAV Cold Start Simulator

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from core.pulsar import Pulsar

logger = logging.getLogger(__name__)


class Catalogue:
    """Loads and manages the ATNF millisecond pulsar catalogue.

    Primary source: data/atnf_cache.json (committed static file, offline-safe).
    Optional: refresh_from_atnf() fetches live data via psrqpy and updates cache.
    """

    def __init__(self, cache_path: Optional[Path] = None) -> None:
        if cache_path is None:
            from config import ATNF_CACHE_PATH
            cache_path = ATNF_CACHE_PATH

        self._cache_path = Path(cache_path)
        self._pulsars: list[Pulsar] = []
        self._load()

    # ── Loading ───────────────────────────────────────────────────────────────

    def _load(self) -> None:
        """Load pulsars from the static JSON cache, sorted by timing quality."""
        if not self._cache_path.exists():
            raise FileNotFoundError(
                f"ATNF cache not found at {self._cache_path}. "
                "Ensure data/atnf_cache.json is present in the repository."
            )

        with open(self._cache_path, "r") as fh:
            data = json.load(fh)

        raw_list = data.get("pulsars", [])
        pulsars = []
        for rec in raw_list:
            try:
                # Support both key variants: "gl_deg"/"gb_deg" (psrqpy refresh)
                # and "gl"/"gb" (static cache compact keys)
                gl_val = rec.get("gl_deg", rec.get("gl"))
                gb_val = rec.get("gb_deg", rec.get("gb"))
                if gl_val is None or gb_val is None:
                    raise KeyError("gl_deg/gl")
                p = Pulsar(
                    name=rec["name"],
                    period=float(rec["period_s"]),
                    period_dot=float(rec["period_dot"]),
                    dm=float(rec["dm"]),
                    gl=float(gl_val),
                    gb=float(gb_val),
                    distance_kpc=float(rec["distance_kpc"]),
                    w50=float(rec["w50_us"]),
                    s1400=float(rec["s1400_mJy"]),
                    timing_noise_ns=float(rec["timing_noise_ns"]),
                )
                pulsars.append(p)
            except (KeyError, ValueError, TypeError) as exc:
                logger.warning("Skipping malformed pulsar record %r: %s", rec.get("name"), exc)

        # Sort descending by timing quality (best pulsars first)
        self._pulsars = sorted(pulsars, key=lambda p: p.timing_quality_score(), reverse=True)
        logger.info("Loaded %d pulsars from %s", len(self._pulsars), self._cache_path)

    # ── Access ────────────────────────────────────────────────────────────────

    @property
    def all_pulsars(self) -> list[Pulsar]:
        """All pulsars sorted by timing quality (best first)."""
        return list(self._pulsars)

    def get_top_n(self, n: int) -> list[Pulsar]:
        """Return the n highest-quality pulsars for a given accuracy tier.

        Always returns the top-N by timing_quality_score so that increasing
        N always adds pulsars and never removes them — results improve
        monotonically with tier.
        """
        n = min(n, len(self._pulsars))
        return list(self._pulsars[:n])

    def __len__(self) -> int:
        return len(self._pulsars)

    def __repr__(self) -> str:
        if self._pulsars:
            best = self._pulsars[0]
            return (
                f"Catalogue({len(self._pulsars)} pulsars, "
                f"best={best.name} [{best.timing_noise_ns:.0f} ns])"
            )
        return "Catalogue(empty)"

    # ── Optional live refresh ─────────────────────────────────────────────────

    def refresh_from_atnf(self) -> bool:
        """Attempt to fetch current ATNF data via psrqpy and update cache.

        Returns True on success, False on any failure (network, missing package).
        Never raises — the app must remain functional offline.
        """
        try:
            import psrqpy

            logger.info("Fetching ATNF catalogue via psrqpy …")
            query = psrqpy.QueryATNF(
                params=[
                    "JNAME", "P0", "P1", "DM", "GL", "GB",
                    "DIST", "W50", "S1400", "EPHEM",
                ],
                condition="P0 < 0.03 AND DM > 0",  # millisecond pulsars only
            )
            df = query.pandas

            records = []
            for _, row in df.iterrows():
                try:
                    # Estimate timing noise from characteristic age and P0
                    # APPROXIMATION: timing noise ∝ P0^(1/2) / flux^(1/2)
                    # Real timing noise depends on pulsar spin noise, DM
                    # variations, and receiver noise. Order-of-magnitude only.
                    p0 = float(row["P0"])
                    s14 = float(row["S1400"]) if row["S1400"] > 0 else 0.5
                    noise_estimate = max(50.0, 200.0 * (p0 / 0.005) ** 0.5 / (s14 ** 0.3))

                    records.append({
                        "name": str(row["JNAME"]),
                        "period_s": p0,
                        "period_dot": float(row["P1"]),
                        "dm": float(row["DM"]),
                        "gl_deg": float(row["GL"]),
                        "gb_deg": float(row["GB"]),
                        "distance_kpc": float(row["DIST"]),
                        "w50_us": float(row["W50"]) if row["W50"] > 0 else 500.0,
                        "s1400_mJy": s14,
                        "timing_noise_ns": noise_estimate,
                    })
                except (ValueError, TypeError):
                    continue

            if len(records) < 10:
                logger.warning("psrqpy returned too few valid records (%d); keeping cache.", len(records))
                return False

            # Sort by timing quality score and keep top 100.
            # Compute the score directly on raw fields to avoid key-name
            # translation between the psrqpy record schema and Pulsar.__init__.
            # Score mirrors Pulsar.timing_quality_score(): log10(flux / noise).
            import math
            def _raw_score(rec: dict) -> float:
                flux = max(rec["s1400_mJy"], 1e-6)
                noise = max(rec["timing_noise_ns"], 1e-3)
                return math.log10(flux / noise)

            records.sort(key=_raw_score, reverse=True)
            top100 = records[:100]

            cache_data = {
                "metadata": {
                    "description": "Live fetch from ATNF via psrqpy",
                    "fields": "name, period_s, period_dot, dm_pc_cm3, gl_deg, gb_deg, distance_kpc, w50_us, s1400_mJy, timing_noise_ns",
                    "source": "psrqpy live query",
                },
                "pulsars": top100,
            }

            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._cache_path, "w") as fh:
                json.dump(cache_data, fh, indent=2)

            self._load()  # reload from newly written cache
            logger.info("Cache refreshed: %d pulsars written.", len(top100))
            return True

        except ImportError:
            logger.warning("psrqpy not installed; cannot refresh ATNF catalogue.")
        except Exception as exc:
            logger.warning("ATNF refresh failed (network?): %s", exc)

        return False
