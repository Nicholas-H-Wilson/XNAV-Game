#!/usr/bin/env python3
"""
Convergence study for XNAV Cold Start Simulator.

Runs the full Stage1→filter→Stage2→Stage3→Stage4 pipeline across N random
spacecraft positions and reports:
  - Convergence rate (error < threshold at end)
  - Median / mean / p90 final error
  - Filter divergence rate
  - Per-stage success rate
  - Timing (iterations to converge)

Usage (from xnav_simulator/):
    python tests/run_convergence_study.py [--n 50] [--tier "Quick Look (20 pulsars)"]
"""

from __future__ import annotations

import argparse
import sys
import time
import pathlib
import warnings

import numpy as np

warnings.filterwarnings("ignore")

_ROOT = pathlib.Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.logger import configure_root_logger
configure_root_logger(level="WARNING")   # suppress info spam during batch run

from config import ACCURACY_TIERS, K_DM, SUN_POS_KPC
from core.catalogue import Catalogue
from core.interstellar_medium import InterstellarMedium
from core.estimator import ParticleFilter
from core.spacecraft import Spacecraft
from stages import (
    stage1_dm_localisation,
    stage2_profile_matching,
    stage3_geometry,
    stage4_phase_ambiguity,
)


# ── Synthetic observation helper ───────────────────────────────────────────────
# Exactly the same forward model app.py uses — shared via core/observations.py
# so the study measures what the app actually does.

from core.observations import build_observations


def _build_observations(pulsars, sc_pos, ism, freq_mhz=1400.0, rng=None,
                         timing_scale=1.0, ism_scale=1.0, t_int=1000.0):
    return build_observations(
        pulsars, sc_pos, ism,
        frequency_mhz=freq_mhz,
        rng=rng,
        timing_noise_scale=timing_scale,
        ism_turb_scale=ism_scale,
        integration_time_s=t_int,
    )


# ── Single run ─────────────────────────────────────────────────────────────────

def run_one(seed: int, pulsars, ism, tier_cfg: dict,
            preset: str = "random") -> dict:
    """Run the full pipeline from scratch. Returns a result dict."""
    rng = np.random.default_rng(seed)
    freq_mhz = 1400.0
    t_int = 1000.0

    # Place spacecraft
    if preset == "random":
        sc = Spacecraft.random_deep_space(rng=rng)
    elif preset == "near_sun":
        sc = Spacecraft.near_sun_like_star(rng=rng)
    elif preset == "gc":
        sc = Spacecraft.at_galactic_centre(rng=rng)
    elif preset == "void":
        sc = Spacecraft.interarm_void(rng=rng)
    else:
        sc = Spacecraft.random_deep_space(rng=rng)

    true_pos = sc.true_position_kpc.copy()
    result = {
        "seed": seed, "preset": preset,
        "true_pos": true_pos,
        "converged": False, "diverged": False,
        "final_error_kpc": float("inf"),
        "iterations": 0,
        "stage1_ok": False, "stage2_ok": False,
        "stage3_ok": False, "stage4_ok": False,
        "n_identified": 0,
        "final_ess": 0.0,
    }

    # Initialise particle filter
    pf = ParticleFilter(
        n_particles=tier_cfg["n_particles"],
        tier_config=tier_cfg,
        seed=seed + 1000,
    )

    converge_threshold_kpc = 2.0
    max_iterations = 20
    id_pulsars: list = []

    for step in range(max_iterations):
        iter_rng = np.random.default_rng(seed + step * 100)
        obs_timings, dm_vals = _build_observations(
            pulsars, sc.true_position_kpc, ism,
            freq_mhz=freq_mhz, rng=iter_rng, t_int=t_int,
        )

        # Stage 1 (step 0 only)
        if step == 0:
            obs_dms = {n: v["dispersive_s"] * (freq_mhz ** 2) / K_DM
                       for n, v in obs_timings.items()}
            try:
                s1 = stage1_dm_localisation.run(
                    pulsars, obs_dms, ism,
                    grid_resolution_pc=float(tier_cfg["grid_resolution_pc"]),
                    spacecraft_position_kpc=sc.true_position_kpc,
                    frequency_mhz=freq_mhz,
                )
                pf.initialise_from_stage1(s1)
                result["stage1_ok"] = True
            except Exception as e:
                pf.initialise_from_region(sc.true_position_kpc, 10.0)

        # Particle filter update
        try:
            pf.update(pulsars, obs_timings, ism=ism,
                      frequency_mhz=freq_mhz, integration_time_s=t_int)
        except RuntimeError:
            result["diverged"] = True
            result["iterations"] = step
            return result

        if pf.diverged:
            result["diverged"] = True
            result["iterations"] = step
            return result

        # Stage 2 (step 3) — mirror app.py: skip Stages 3/4 when nothing identified
        if step == 3:
            obs_profiles = [p.generate_profile() for p in pulsars]
            try:
                s2 = stage2_profile_matching.run(obs_profiles, pulsars)
                result["stage2_ok"] = True
                result["n_identified"] = s2.get("n_identified", 0)
                # best_match holds Pulsar objects (or None) — see stage2 docstring
                id_names = {r["best_match"].name
                            for r in s2.get("identifications", [])
                            if r.get("best_match") is not None}
                id_pulsars = [p for p in pulsars if p.name in id_names]
            except Exception:
                id_pulsars = []

            if id_pulsars:
                try:
                    stage3_geometry.run(id_pulsars, pf)
                    result["stage3_ok"] = True
                except Exception:
                    pass

        # Stage 4 (step 6)
        if step == 6 and id_pulsars:
            try:
                arrival_times = {p.name: obs_timings[p.name]["total"]
                                 for p in id_pulsars if p.name in obs_timings}
                stage4_phase_ambiguity.run(
                    id_pulsars, arrival_times, pf.get_estimate()["position_kpc"],
                    true_clock_offset_s=sc.clock_offset_s,
                )
                result["stage4_ok"] = True
            except Exception:
                pass

        # Check convergence
        est = pf.get_estimate()
        error = float(np.linalg.norm(est["position_kpc"] - true_pos))
        if error < converge_threshold_kpc:
            result["converged"] = True
            result["final_error_kpc"] = error
            result["iterations"] = step + 1
            result["final_ess"] = pf.get_ess()
            return result

    # End of loop — not converged but not diverged
    est = pf.get_estimate()
    result["final_error_kpc"] = float(np.linalg.norm(est["position_kpc"] - true_pos))
    result["iterations"] = max_iterations
    result["final_ess"] = pf.get_ess()
    return result


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=30, help="Number of runs per preset")
    parser.add_argument("--tier", default="Quick Look (20 pulsars)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    tier_cfg = ACCURACY_TIERS.get(args.tier)
    if tier_cfg is None:
        print(f"Unknown tier '{args.tier}'. Available: {list(ACCURACY_TIERS.keys())}")
        sys.exit(1)

    print(f"\n  XNAV Convergence Study")
    print(f"  Tier: {args.tier}  |  N={args.n} per preset  |  seed={args.seed}")
    print(f"  Loading catalogue and ISM...", end="", flush=True)
    t0 = time.monotonic()

    cat = Catalogue()
    pulsars = cat.get_top_n(tier_cfg["n_pulsars"])
    ism = InterstellarMedium()   # loads data/ne2001_grid.npz automatically
    if not ism.grid_loaded():
        print("\n  WARNING: DM grid not loaded — catalogue DMs carry no position"
              " signal; convergence will be poor.")
    print(f" done ({time.monotonic()-t0:.1f}s, {len(pulsars)} pulsars)")

    presets = ["random", "near_sun", "gc", "void"]
    all_results = []

    for preset in presets:
        print(f"\n  Preset: {preset}")
        preset_results = []
        for i in range(args.n):
            # Stable per-preset offset (hash() is salted per process — not reproducible)
            seed = args.seed + i * 13 + sum(ord(c) for c in preset) % 1000
            t_run = time.monotonic()
            r = run_one(seed, pulsars, ism, tier_cfg, preset=preset)
            elapsed = time.monotonic() - t_run
            status = "CONV" if r["converged"] else ("DIV" if r["diverged"] else "FAIL")
            print(f"    [{i+1:3d}/{args.n}] seed={seed:6d} {status:4s}  "
                  f"err={r['final_error_kpc']:6.2f} kpc  "
                  f"iter={r['iterations']:3d}  "
                  f"s2={r['n_identified']:2d}id  {elapsed:.1f}s")
            preset_results.append(r)
        all_results.extend(preset_results)

        # Per-preset summary
        conv = [r for r in preset_results if r["converged"]]
        div  = [r for r in preset_results if r["diverged"]]
        fail = [r for r in preset_results if not r["converged"] and not r["diverged"]]
        conv_rate = len(conv) / len(preset_results) * 100
        errors_conv = [r["final_error_kpc"] for r in conv]
        print(f"    → {preset}: conv={conv_rate:.0f}%  "
              f"div={len(div)}  fail={len(fail)}  "
              f"median_err={np.median(errors_conv):.3f} kpc" if errors_conv
              else f"    → {preset}: conv={conv_rate:.0f}%  div={len(div)}  fail={len(fail)}")

    # Overall summary
    total = len(all_results)
    conv_all = [r for r in all_results if r["converged"]]
    div_all  = [r for r in all_results if r["diverged"]]
    fail_all = [r for r in all_results if not r["converged"] and not r["diverged"]]
    errors_all = [r["final_error_kpc"] for r in conv_all]

    print(f"\n{'─'*65}")
    print(f"  OVERALL  ({total} runs, tier={args.tier})")
    print(f"{'─'*65}")
    print(f"  Converged:  {len(conv_all):3d}/{total}  ({len(conv_all)/total*100:.1f}%)")
    print(f"  Diverged:   {len(div_all):3d}/{total}  ({len(div_all)/total*100:.1f}%)")
    print(f"  Timed out:  {len(fail_all):3d}/{total}  ({len(fail_all)/total*100:.1f}%)")
    if errors_all:
        print(f"  Error (converged runs):")
        print(f"    Median:  {np.median(errors_all):.3f} kpc")
        print(f"    Mean:    {np.mean(errors_all):.3f} kpc")
        print(f"    p90:     {np.percentile(errors_all, 90):.3f} kpc")
        print(f"    Max:     {np.max(errors_all):.3f} kpc")
    stage_rates = {
        "stage1": sum(r["stage1_ok"] for r in all_results) / total * 100,
        "stage2": sum(r["stage2_ok"] for r in all_results) / total * 100,
        "stage3": sum(r["stage3_ok"] for r in all_results) / total * 100,
        "stage4": sum(r["stage4_ok"] for r in all_results) / total * 100,
    }
    print(f"  Stage success rates: " +
          "  ".join(f"{k}={v:.0f}%" for k, v in stage_rates.items()))
    print(f"{'─'*65}\n")

    target = 95.0
    actual = len(conv_all) / total * 100
    if actual >= target:
        print(f"  ✓ Convergence rate {actual:.1f}% meets 95% target")
    else:
        print(f"  ✗ Convergence rate {actual:.1f}% BELOW 95% target")
    print()
    return 0 if actual >= target else 1


if __name__ == "__main__":
    sys.exit(main())
