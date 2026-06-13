# app.py — Streamlit entry point for XNAV Cold Start Simulator
# XNAV Cold Start Simulator
"""
Main Streamlit application.

Layout
------
  Sidebar (300px)   — Simulation controls (ui/sidebar.py)
  Main area (5 tabs) — Galaxy Map / Timing & DM / Convergence / Phase Resolution / Gravity Well
  Footer            — Current best estimate readout bar

Simulation loop
---------------
  Each click of RUN SIMULATION runs the full cold-start pipeline:
    Stage 1 → init → 3×update → Stage 2 → Stage 3 → N×update → Stage 4

  The "running" flag in session_state gates one iteration per rerun.
  Streamlit rerun() is called at the end of each iteration to advance the loop.

CRITICAL — LOS convention (Appendix D.2):
  Synthetic observations MUST use los_dir = pulsar_pos / |pulsar_pos| (origin→pulsar).
  Using TimingModel.compute_arrival_time() creates a ~10^11 s Roemer residual at
  the true particle position, preventing all convergence.  _build_observations()
  below implements the filter-consistent convention.

Session state keys
------------------
  filter        ParticleFilter instance (or None before init)
  spacecraft    Spacecraft instance
  catalogue     Catalogue instance
  ism           InterstellarMedium instance
  pulsars       list[Pulsar] — active pulsars for this run
  history       list of sim_step dicts
  iteration     int — current iteration counter
  running       bool — simulation is actively stepping
  stage_status  dict {stage1..4: str}
  stage_results dict — results from each stage
  settings      dict — last settings used to build the simulation objects
  pending_reset bool — awaiting user confirmation before resetting
  staged_tier   str  — tier the user requested (pending confirmation)
  current_tier  str  — tier currently in use
"""

from __future__ import annotations

import sys
import time
import pathlib

# Ensure xnav_simulator/ is on the path when run as `streamlit run app.py`
_ROOT = pathlib.Path(__file__).parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import streamlit as st

import config
from config import (
    C_LIGHT, K_DM, KPC_TO_M, SUN_POS_KPC,
    ACCURACY_TIERS, DEFAULT_TIER, COLOUR_ACCENT, COLOUR_BG,
)
from utils.logger import configure_root_logger, SimLogger, get_logger

logger = get_logger(__name__)

# ── Page configuration ────────────────────────────────────────────────────────

st.set_page_config(
    layout="wide",
    page_title="XNAV Cold Start Simulator",
    page_icon="🛸",
)

# ── Configure logging (once per session) ─────────────────────────────────────

configure_root_logger()

# ── Dark theme CSS injection ──────────────────────────────────────────────────

st.markdown(
    f"""
    <style>
      .stApp {{ background-color: {COLOUR_BG}; color: #CCCCDD; }}
      .stTabs [data-baseweb="tab-list"] {{
        border-bottom: 1px solid #1A1A3A;
        gap: 4px;
      }}
      .stTabs [data-baseweb="tab"] {{
        background-color: #111128;
        color: #888;
        border-radius: 4px 4px 0 0;
        padding: 6px 16px;
        font-size: 0.85em;
        letter-spacing: 0.02em;
      }}
      .stTabs [aria-selected="true"] {{
        background-color: #1A1A3A;
        color: {COLOUR_ACCENT};
        border-bottom: 2px solid {COLOUR_ACCENT};
      }}
      .stMetric label {{ color: #AAAACC !important; }}
      .stAlert {{ border-radius: 6px; }}
      footer {{ visibility: hidden; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Session state initialisation ──────────────────────────────────────────────
# All expensive objects are guarded here.  Display-only widget changes must
# NEVER reach this block — only the simulation init path should recreate these.

if "filter" not in st.session_state:
    st.session_state.filter = None
if "spacecraft" not in st.session_state:
    st.session_state.spacecraft = None
if "catalogue" not in st.session_state:
    st.session_state.catalogue = None
if "ism" not in st.session_state:
    st.session_state.ism = None
if "pulsars" not in st.session_state:
    st.session_state.pulsars = []
if "history" not in st.session_state:
    st.session_state.history = []
if "iteration" not in st.session_state:
    st.session_state.iteration = 0
if "running" not in st.session_state:
    st.session_state.running = False
if "stage_status" not in st.session_state:
    st.session_state.stage_status = {
        "stage1": "pending", "stage2": "pending",
        "stage3": "pending", "stage4": "pending",
    }
if "stage_results" not in st.session_state:
    st.session_state.stage_results = {}
if "settings" not in st.session_state:
    st.session_state.settings = {}
if "pending_reset" not in st.session_state:
    st.session_state.pending_reset = False
if "staged_tier" not in st.session_state:
    st.session_state.staged_tier = DEFAULT_TIER
if "current_tier" not in st.session_state:
    st.session_state.current_tier = DEFAULT_TIER
if "sim_logger" not in st.session_state:
    st.session_state.sim_logger = SimLogger()
if "dm_residuals" not in st.session_state:
    st.session_state.dm_residuals = {}
if "observed_timings" not in st.session_state:
    st.session_state.observed_timings = {}

# ── Header ────────────────────────────────────────────────────────────────────

col_title, col_status = st.columns([4, 1])
with col_title:
    st.markdown(
        f'<h1 style="color:{COLOUR_ACCENT}; margin:0; font-size:2em;">'
        "🛸 XNAV Cold Start Simulator</h1>"
        '<p style="color:#888; font-size:0.85em; margin:0;">'
        "X-ray pulsar navigation — determine your position anywhere in the galaxy, from scratch</p>",
        unsafe_allow_html=True,
    )
with col_status:
    if st.session_state.running:
        st.markdown(
            f'<div style="background:#FF8800; color:#000; padding:4px 12px; '
            f'border-radius:20px; text-align:center; font-weight:bold; '
            f'font-size:0.8em; margin-top:10px;">● RUNNING</div>',
            unsafe_allow_html=True,
        )
    elif st.session_state.filter is not None and st.session_state.iteration > 0:
        pf = st.session_state.filter
        if pf.diverged:
            status_text, status_col = "⚠ DIVERGED", "#FF4444"
        else:
            # Blind-mode safe: judge convergence on the filter's own
            # uncertainty, not the true error.
            _est = pf.get_estimate()
            _unc = float(np.mean(_est["position_std_kpc"]))
            if _unc < config.CONVERGE_UNCERTAINTY_KPC:
                status_text, status_col = "✓ CONVERGED", "#00CC66"
            else:
                status_text, status_col = "△ NOT CONVERGED", "#FFAA00"
        st.markdown(
            f'<div style="background:{status_col}33; color:{status_col}; '
            f'padding:4px 12px; border-radius:20px; text-align:center; '
            f'font-weight:bold; font-size:0.8em; margin-top:10px;">'
            f'{status_text}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div style="background:#33334A; color:#888; padding:4px 12px; '
            f'border-radius:20px; text-align:center; font-size:0.8em; '
            f'margin-top:10px;">◯ NOT STARTED</div>',
            unsafe_allow_html=True,
        )

st.divider()

# ── Helper: reset simulation state ───────────────────────────────────────────

def _do_reset() -> None:
    """Reset all mutable simulation state in session_state."""
    if st.session_state.filter is not None:
        st.session_state.filter.reset()
    st.session_state.filter = None
    st.session_state.spacecraft = None
    st.session_state.pulsars = []
    st.session_state.history = []
    st.session_state.iteration = 0
    st.session_state.running = False
    st.session_state.stage_status = {
        "stage1": "pending", "stage2": "pending",
        "stage3": "pending", "stage4": "pending",
    }
    st.session_state.stage_results = {}
    st.session_state.dm_residuals = {}
    st.session_state.observed_timings = {}
    st.session_state.sim_logger = SimLogger()
    logger.info("Simulation reset.")


# ── Onboarding expander ───────────────────────────────────────────────────────
if st.session_state.filter is None:
    with st.expander("How XNAV navigation works — quick start guide", expanded=True):
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.markdown(
                f'<div style="background:#111128; border:1px solid #1A1A3A; '
                f'padding:12px; border-radius:6px; min-height:140px;">'
                f'<p style="color:{COLOUR_ACCENT}; font-weight:bold; margin:0 0 6px 0;">'
                f'🛸 The Cold Start Problem</p>'
                f'<p style="color:#AAAACC; font-size:0.85em; margin:0;">'
                f'Your spacecraft is somewhere in the Milky Way. No GPS. No ground contact. '
                f'No prior position fix. The only navigation tool is an X-ray detector '
                f'pointed at millisecond pulsars — the most precise natural clocks in the universe.</p>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with col_b:
            st.markdown(
                f'<div style="background:#111128; border:1px solid #1A1A3A; '
                f'padding:12px; border-radius:6px; min-height:140px;">'
                f'<p style="color:{COLOUR_ACCENT}; font-weight:bold; margin:0 0 6px 0;">'
                f'📡 How it works</p>'
                f'<p style="color:#AAAACC; font-size:0.85em; margin:0;">'
                f'<b style="color:#CCC;">Stage 1</b> — Dispersion measure constrains your region.<br>'
                f'<b style="color:#CCC;">Stage 2</b> — Profile matching identifies which pulsars you see.<br>'
                f'<b style="color:#CCC;">Stage 3</b> — Line-of-sight geometry triangulates position.<br>'
                f'<b style="color:#CCC;">Stage 4</b> — Phase timing resolves the clock offset.<br>'
                f'A <b style="color:#CCC;">particle filter</b> fuses all of this into a converging '
                f'position estimate.</p>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with col_c:
            st.markdown(
                f'<div style="background:#111128; border:1px solid #1A1A3A; '
                f'padding:12px; border-radius:6px; min-height:140px;">'
                f'<p style="color:{COLOUR_ACCENT}; font-weight:bold; margin:0 0 6px 0;">'
                f'🚀 Quick start</p>'
                f'<p style="color:#AAAACC; font-size:0.85em; margin:0;">'
                f'1. Choose a <b style="color:#CCC;">spacecraft preset</b> in the sidebar '
                f'(or leave as Random deep space).<br>'
                f'2. Click <b style="color:{COLOUR_ACCENT};">▶ RUN SIMULATION</b>.<br>'
                f'3. Watch the particle cloud on the <b style="color:#CCC;">Galaxy Map</b> '
                f'and <b style="color:#CCC;">Convergence</b> tabs contract toward the true position.<br>'
                f'4. Explore the <b style="color:#CCC;">Timing</b> and '
                f'<b style="color:#CCC;">Phase Resolution</b> tabs to see the physics.'
                f'</p>'
                f'</div>',
                unsafe_allow_html=True,
            )


# ── Sidebar ───────────────────────────────────────────────────────────────────

from ui.sidebar import render as render_sidebar
settings = render_sidebar()


# ── Confirmation prompt for setup changes ─────────────────────────────────────
# See Appendix D architecture guidance: pending_reset flag survives Streamlit
# reruns; staged_tier holds the new config before confirmation.

if settings["tier"] != st.session_state.current_tier and st.session_state.filter is not None:
    st.session_state.staged_tier = settings["tier"]
    st.session_state.pending_reset = True

if st.session_state.pending_reset:
    st.warning(
        f"Changing accuracy tier from **{st.session_state.current_tier}** "
        f"to **{st.session_state.staged_tier}** will reset the simulation."
    )
    col_confirm, col_cancel = st.columns(2)
    if col_confirm.button("✓ Confirm reset", type="primary", key="confirm_reset"):
        st.session_state.pending_reset = False
        st.session_state.current_tier = st.session_state.staged_tier
        _do_reset()
        st.rerun()
    if col_cancel.button("✗ Cancel", key="cancel_reset"):
        st.session_state.pending_reset = False
        st.rerun()
    st.stop()   # don't render the rest of the page while confirmation is pending


# ── Helper: build observed timings (filter-consistent LOS convention) ─────────
# Shared with tests/run_convergence_study.py — see core/observations.py for the
# forward model and the Appendix D.2 LOS convention notes.

from core.observations import build_observations as _build_observations


# ── Helper: initialise catalogue and ISM (once per session) ──────────────────

def _ensure_catalogue_and_ism(settings: dict) -> bool:
    """Load catalogue and ISM grid if not already loaded.  Returns success."""
    tier_config = settings["tier_config"]
    n_pulsars = tier_config["n_pulsars"]
    grid_res_pc = tier_config["grid_resolution_pc"]

    if st.session_state.catalogue is None:
        from core.catalogue import Catalogue
        cat = Catalogue()
        st.session_state.catalogue = cat
        logger.info("Catalogue loaded: %d pulsars", len(cat))

    if st.session_state.ism is None:
        from core.interstellar_medium import InterstellarMedium
        ism = InterstellarMedium()

        if not ism.grid_loaded():
            progress_bar = st.progress(0, text="Precomputing DM grid…")
            try:
                ism.precompute_grid(
                    resolution_pc=grid_res_pc,
                    progress_callback=lambda frac: progress_bar.progress(
                        int(frac * 100), text=f"DM grid: {int(frac*100)}%"
                    ),
                )
                progress_bar.empty()
                logger.info("DM grid computed at %d pc resolution.", grid_res_pc)
            except Exception as exc:
                progress_bar.empty()
                st.warning(
                    f"DM grid unavailable ({exc}); falling back to catalogue DMs. "
                    "Navigation accuracy will be reduced."
                )
                logger.warning("DM grid precomputation failed: %s", exc)

        st.session_state.ism = ism

    # Surface a mismatch between the tier's requested grid resolution and
    # what the loaded (bundled) grid actually provides — regenerating the
    # grid requires pygedm, so we inform rather than silently recompute.
    ism = st.session_state.ism
    if ism is not None and ism.grid_loaded():
        actual_res = ism.resolution_pc
        if np.isfinite(actual_res) and actual_res > grid_res_pc * 1.5:
            st.info(
                f"Bundled DM grid resolution is {actual_res:.0f} pc; the "
                f"{settings['tier']} tier requests {grid_res_pc} pc. "
                "Position accuracy is limited by the bundled grid — install "
                "pygedm and delete data/ne2001_grid.npz to regenerate it "
                "at full resolution."
            )

    return True


# ── Helper: build sim_data dict for UI panels ─────────────────────────────────

@st.cache_data(show_spinner=False)
def _load_star_catalogue() -> list:
    """Local HYG star subset for the galaxy map (see tools/curate_hyg_stars.py).

    ~30,000 stars: every proper-named star plus the brightest others, with
    galactocentric positions, spectral types and derived T_eff. Cached
    process-wide — the file is static.
    """
    import json
    path = config.DATA_DIR / "hyg_stars.json"
    try:
        with open(path) as fh:
            return json.load(fh)["stars"]
    except Exception as exc:
        logger.warning("Star catalogue unavailable (%s); map shows pulsars only.", exc)
        return []


@st.cache_data(show_spinner=False)
def _load_galactic_objects() -> list:
    """Distributed galaxy-wide objects for the map (globulars, nebulae, black
    holes, clusters). See tools/curate_galactic_objects.py. Cached process-wide.
    """
    import json
    path = config.DATA_DIR / "galactic_objects.json"
    try:
        with open(path) as fh:
            return json.load(fh)["objects"]
    except Exception as exc:
        logger.warning("Galactic-objects catalogue unavailable (%s).", exc)
        return []


def _build_sim_data(settings: dict) -> dict:
    """Extract all display data from session_state into a plain dict for UI panels.

    UI modules receive this dict only — they never import from core/ or stages/.
    """
    pf = st.session_state.filter
    sc = st.session_state.spacecraft
    pulsars = st.session_state.pulsars
    history = st.session_state.history
    ism = st.session_state.ism
    blind_mode = settings.get("blind_mode", False)

    # Current estimate
    if pf is not None and pf.initialised:
        est = pf.get_estimate()
        pos_est = est["position_kpc"]
        uncertainty = float(np.mean(est["position_std_kpc"]))
        particles_kpc = pf.particles[:, :3].copy()
        weights = pf.weights.copy()
        ess_pre = pf.get_ess()
    else:
        pos_est = np.zeros(3)
        uncertainty = float(config.GALAXY_RADIUS_KPC)
        particles_kpc = np.zeros((1, 3))
        weights = np.ones(1)
        ess_pre = 1.0

    true_pos = None
    if sc is not None and not blind_mode:
        true_pos = sc.true_position_kpc.copy()

    # Gravitational potential
    phi_m2s2 = -2.0e11   # default galactic background estimate
    true_phi = None
    if sc is not None:
        try:
            phi_m2s2 = sc.gravitational_potential(
                include_galactic=settings.get("include_galactic_potential", True)
            )
            if not blind_mode:
                true_phi = phi_m2s2
        except Exception:
            pass

    # Build pulsar dicts for UI panels
    pulsar_dicts = []
    stage2_res = st.session_state.stage_results.get("stage2", {})
    identifications = stage2_res.get("identifications", [])
    # best_match holds Pulsar objects (or None) — see stage2 docstring
    id_names = {r["best_match"].name for r in identifications
                if r.get("best_match") is not None}
    for p in pulsars:
        ident = p.name in id_names
        conf = next((r.get("confidence", 0.0) for r in identifications
                     if r.get("best_match") is not None
                     and r["best_match"].name == p.name), 0.0)
        # Derived astrophysics for the map popups (standard dipole formulas):
        # surface field B ≈ 3.2×10¹⁹ √(P·Ṗ) G; spin-down Ė = 4π²I·Ṗ/P³ (I=10⁴⁵ g cm²)
        pdot = max(p.period_dot, 0.0)
        b_surf_g = 3.2e19 * float(np.sqrt(p.period * pdot)) if pdot > 0 else 0.0
        edot_erg_s = 3.95e46 * pdot / (p.period ** 3) if pdot > 0 else 0.0
        ppos = p.position_kpc
        pulsar_dicts.append({
            "name": p.name, "dm": p.dm, "period": p.period,
            "distance_kpc": p.distance_kpc, "gl": p.gl, "gb": p.gb,
            "timing_noise_ns": p.timing_noise_ns, "w50": p.w50,
            "identified": ident, "confidence": conf,
            # Map sprite data
            "x_kpc": float(ppos[0]), "y_kpc": float(ppos[1]), "z_kpc": float(ppos[2]),
            "period_dot": p.period_dot,
            "age_yr": float(p.characteristic_age_yr),
            "b_surf_g": b_surf_g,
            "edot_erg_s": edot_erg_s,
            "type": ("Millisecond pulsar — recycled neutron star"
                     if p.period < 0.03 else "Rotation-powered pulsar (neutron star)"),
            "s1400_mjy": p.s1400,
            # Timing contributions (populated from observed_timings if available)
            **_pulsar_timing_contributions(p, st.session_state.observed_timings,
                                           settings),
        })

    # Stage 4 results
    s4_res = st.session_state.stage_results.get("stage4", {})
    # Convert list[tuple[n_pulsars, window_s]] → list[dict] for phase_panel
    window_history_raw = s4_res.get("window_history", [])
    window_history = [
        {"n_pulsars": n, "window_s": w} for n, w in window_history_raw
    ]
    clock_candidates = list(s4_res.get("candidate_times_s", []))
    clock_estimate = float(s4_res.get("resolved_clock_offset_s", 0.0))

    return {
        # Galaxy map
        "pulsars": pulsar_dicts,
        "sc_pos_kpc": pos_est,
        "true_pos_kpc": true_pos,
        "sun_pos_kpc": SUN_POS_KPC,
        "uncertainty_kpc": uncertainty,
        "blind_mode": blind_mode,
        "stars": _load_star_catalogue(),
        "galactic_objects": _load_galactic_objects(),
        # Level-of-detail: render the full faint-star field only when idle, so
        # the per-iteration run loop stays fast (bright stars + distributed
        # objects + main markers are always shown).
        "full_detail": not st.session_state.running,
        "particle_pos": particles_kpc if pf is not None and pf.initialised else None,
        "particle_weights": weights if pf is not None and pf.initialised else None,
        # Convergence panel
        "particles_kpc": particles_kpc,
        "weights": weights,
        "estimate_kpc": pos_est,
        "history": history,
        "ess_pre": ess_pre,
        "ess_post": 1.0,  # post-resample ESS is always ~1.0 (Appendix D.3)
        "stage_status": dict(st.session_state.stage_status),
        "iteration": st.session_state.iteration,
        # Timing panel
        "frequency_mhz": 1400.0,
        "integration_time_s": settings.get("integration_time_s", 1000.0),
        "dm_residuals": dict(st.session_state.dm_residuals),
        # Phase panel
        "window_history": window_history,
        "clock_candidates": clock_candidates,
        "clock_estimate_s": clock_estimate,
        "stage4_complete": st.session_state.stage_status.get("stage4") == "complete",
        # Gravity panel
        "phi_m2s2": phi_m2s2,
        "phi_uncertainty": abs(phi_m2s2) * 0.3,
        "true_phi_m2s2": true_phi,
        "timing_noise_ns": (min(p.timing_noise_ns for p in pulsars)
                            if pulsars else 100.0),
        "dm_value": (np.mean([p.dm for p in pulsars]) if pulsars else 30.0),
        "n_pulsars": len(pulsars),
        "collecting_area_m2": settings.get("collecting_area_m2", 1.0),
    }


def _pulsar_timing_contributions(p, observed_timings: dict, settings: dict) -> dict:
    """Extract per-pulsar timing contributions for the timing panel."""
    if p.name not in observed_timings:
        return {
            "roemer_s": 0.0, "dispersive_s": 0.0,
            "timing_noise_s": 0.0, "photon_noise_s": 1e-6,
            "dm_turbulence_s": K_DM * 0.15 * p.dm / (1400.0 ** 2),
            "observed_timing": 0.0,
        }
    obs = observed_timings[p.name]
    freq_mhz = 1400.0
    dm_turb_s = K_DM * 0.15 * p.dm / (freq_mhz ** 2)
    return {
        "roemer_s": obs.get("roemer_s", 0.0),
        "dispersive_s": obs.get("dispersive_s", 0.0),
        "timing_noise_s": obs.get("timing_noise_s", 0.0),
        "photon_noise_s": 1e-6,   # fiducial; not separately tracked
        "dm_turbulence_s": dm_turb_s,
        "observed_timing": obs.get("total", 0.0),
    }


# ── Simulation run logic ──────────────────────────────────────────────────────

def _run_one_phase5_pipeline(settings: dict) -> None:
    """Run the full Stage1→init→updates→Stage2→Stage3→updates→Stage4 pipeline.

    Each call to this function advances the simulation one iteration.
    The function returns quickly; st.rerun() is called by the caller to
    loop back for the next iteration.

    CRITICAL: All observation generation uses the filter-consistent LOS
    convention (origin→pulsar).  See _build_observations() above.
    """
    from core.estimator import ParticleFilter
    from stages import (
        stage1_dm_localisation,
        stage2_profile_matching,
        stage3_geometry,
        stage4_phase_ambiguity,
    )

    pf = st.session_state.filter
    sc = st.session_state.spacecraft
    pulsars = st.session_state.pulsars
    ism = st.session_state.ism
    tier_cfg = settings["tier_config"]
    freq_mhz = 1400.0
    t_int = settings.get("integration_time_s", 1000.0)
    timing_scale = settings.get("timing_noise_scale", 1.0)
    ism_scale = settings.get("ism_turb_scale", 1.0)
    rng = np.random.default_rng(42 + st.session_state.iteration)

    # Generate observations using filter-consistent LOS convention
    obs_timings, dm_vals = _build_observations(
        pulsars, sc.true_position_kpc, ism,
        frequency_mhz=freq_mhz,
        rng=rng,
        timing_noise_scale=timing_scale,
        ism_turb_scale=ism_scale,
        integration_time_s=t_int,
    )
    st.session_state.observed_timings = obs_timings

    step = st.session_state.iteration

    # ── Stage 1: DM localisation (once, at step 0) ────────────────────────────
    if step == 0:
        st.session_state.stage_status["stage1"] = "running"
        obs_dms = {name: vals["dispersive_s"] * (freq_mhz ** 2) / K_DM
                   for name, vals in obs_timings.items()}
        try:
            s1_result = stage1_dm_localisation.run(
                pulsars, obs_dms, ism,
                grid_resolution_pc=float(tier_cfg["grid_resolution_pc"]),
                spacecraft_position_kpc=sc.true_position_kpc,
                frequency_mhz=freq_mhz,
            )
            st.session_state.stage_results["stage1"] = s1_result
        except Exception as exc:
            logger.warning("Stage 1 failed: %s — falling back to region init", exc)
            s1_result = None

        # Initialise filter
        if s1_result is not None:
            try:
                pf.initialise_from_stage1(s1_result)
            except Exception as exc:
                logger.warning("initialise_from_stage1 failed: %s", exc)
                pf.initialise_from_region(sc.true_position_kpc, 10.0)
        else:
            pf.initialise_from_region(sc.true_position_kpc, 10.0)

        st.session_state.stage_status["stage1"] = (
            "complete" if s1_result is not None else "failed"
        )

    # ── Particle filter update ────────────────────────────────────────────────
    try:
        state = pf.update(
            pulsars, obs_timings,
            ism=ism,
            frequency_mhz=freq_mhz,
            integration_time_s=t_int,
        )
    except RuntimeError as exc:
        st.session_state.running = False
        st.error(
            f"**Filter diverged at iteration {step}** — the particle cloud collapsed "
            f"and could not be recovered.\n\n"
            f"**To recover:** ↺ Reset, then try:\n"
            f"- Increase *Integration time* (more signal → less noise)\n"
            f"- Reduce *Timing noise multiplier*\n"
            f"- Switch to *Quick Look* tier (fewer pulsars, faster convergence)\n"
            f"- Try a different spacecraft preset"
        )
        logger.error("Filter diverged: %s", exc)
        return

    # Compute error if true position known
    est = pf.get_estimate()
    error_kpc = float(np.linalg.norm(
        est["position_kpc"] - sc.true_position_kpc
    ))
    ess_pre = float(1.0 / np.sum(pf.weights ** 2)) / pf.n_particles

    # Append to history (includes particle snapshot for playback scrubber — Appendix E.2)
    # Memory cost: Balanced tier 5000 × 3 floats × 20 iters ≈ 1.2 MB — acceptable.
    st.session_state.history.append({
        "step": step,
        "pos_kpc": est["position_kpc"].copy(),
        "uncertainty_kpc": float(np.mean(est["position_std_kpc"])),
        "error_kpc": error_kpc,
        "ess_pre": ess_pre,
        "ess_post": 1.0,   # post-resample ESS always ~1.0 (Appendix D.3)
        "beta": getattr(pf, "last_beta", 1.0),   # tempering exponent this step
        "particles_kpc": pf.particles[:, :3].copy(),   # for convergence panel playback
        "weights": pf.weights.copy(),
    })

    # Update DM residuals
    dm_residuals = {}
    for p in pulsars:
        if p.name in dm_vals:
            dm_residuals[p.name] = dm_vals[p.name] - p.dm
    st.session_state.dm_residuals = dm_residuals

    # Log the iteration
    st.session_state.sim_logger.log_iteration(step, error_kpc, ess_pre)

    # ── Stage 2: Profile matching (at step 3) ─────────────────────────────────
    if step == 3 and "stage2" not in st.session_state.stage_results:
        st.session_state.stage_status["stage2"] = "running"
        obs_profiles = [p.generate_profile() for p in pulsars]
        try:
            s2_result = stage2_profile_matching.run(obs_profiles, pulsars)
            st.session_state.stage_results["stage2"] = s2_result
            st.session_state.stage_status["stage2"] = "complete"
        except Exception as exc:
            logger.warning("Stage 2 failed: %s", exc)
            st.session_state.stage_results["stage2"] = {"identifications": [], "n_identified": 0}
            st.session_state.stage_status["stage2"] = "failed"

    # ── Stage 3: Geometry (at step 3, after Stage 2) ──────────────────────────
    if step == 3 and "stage3" not in st.session_state.stage_results:
        st.session_state.stage_status["stage3"] = "running"
        s2_res = st.session_state.stage_results.get("stage2", {})
        # best_match holds Pulsar objects (or None) — see stage2 docstring
        id_names = {r["best_match"].name
                    for r in s2_res.get("identifications", [])
                    if r.get("best_match") is not None}
        identified_pulsars = [p for p in pulsars if p.name in id_names]
        if not identified_pulsars:
            # No identified pulsars → skip Stage 3 rather than triangulating
            # with arbitrary (possibly misidentified) pulsars.
            logger.warning("Stage 3 skipped: no pulsars identified by Stage 2.")
            st.session_state.stage_status["stage3"] = "skipped"
        else:
            try:
                s3_result = stage3_geometry.run(identified_pulsars, pf)
                st.session_state.stage_results["stage3"] = s3_result
                st.session_state.stage_status["stage3"] = "complete"
            except Exception as exc:
                logger.warning("Stage 3 failed: %s", exc)
                st.session_state.stage_status["stage3"] = "failed"

    # ── Stage 4: Phase ambiguity (at step 6, or convergence, once only) ───────
    if (step >= 6 and "stage4" not in st.session_state.stage_results):
        st.session_state.stage_status["stage4"] = "running"
        s2_res = st.session_state.stage_results.get("stage2", {})
        id_names = {r["best_match"].name
                    for r in s2_res.get("identifications", [])
                    if r.get("best_match") is not None}
        identified_pulsars = [p for p in pulsars if p.name in id_names]
        if not identified_pulsars:
            logger.warning("Stage 4 skipped: no pulsars identified by Stage 2.")
            st.session_state.stage_status["stage4"] = "skipped"
        else:
            # Stage 4 operates on post-position-fix phase residuals: raw
            # arrival totals contain ~1e11 s of Roemer delay, and phase
            # ambiguity is only resolvable once position is known to ~c×P
            # (a few hundred km) — far beyond the filter's kpc-scale fix.
            # We therefore feed it clock-only residuals (the documented
            # illustrative regime, same as the Phase 4 tests).
            arrival_times = stage4_phase_ambiguity.simulate_arrival_times(
                identified_pulsars,
                clock_offset_s=sc.clock_offset_s,
                seed=42 + step,
            )
            try:
                s4_result = stage4_phase_ambiguity.run(
                    identified_pulsars,
                    arrival_times,
                    est["position_kpc"],
                    true_clock_offset_s=sc.clock_offset_s,
                )
                st.session_state.stage_results["stage4"] = s4_result
                st.session_state.stage_status["stage4"] = "complete"
            except Exception as exc:
                logger.warning("Stage 4 failed: %s", exc)
                st.session_state.stage_status["stage4"] = "failed"

    # ── Convergence check ─────────────────────────────────────────────────────
    # Blind-mode safe: judged on the filter's own uncertainty (never the true
    # error).  Stop once the uncertainty is small AND has plateaued (< 10%
    # improvement over the last 3 iterations) — stopping on the threshold
    # alone would cut the convergence story short on easy scenarios.
    max_iterations = 20
    uncertainty_kpc = float(np.mean(est["position_std_kpc"]))
    hist = st.session_state.history
    plateaued = (
        len(hist) >= 4
        and uncertainty_kpc > 0.9 * float(hist[-4]["uncertainty_kpc"])
    )
    if (uncertainty_kpc < config.CONVERGE_UNCERTAINTY_KPC and plateaued) \
            or step >= max_iterations:
        st.session_state.running = False
        logger.info(
            "Simulation complete at step %d: uncertainty=%.3f kpc, "
            "error=%.3f kpc, ess=%.3f, beta=%.3g",
            step, uncertainty_kpc, error_kpc, ess_pre,
            getattr(pf, "last_beta", float("nan")),
        )

    st.session_state.iteration += 1


# ── Button handlers ───────────────────────────────────────────────────────────

def _handle_run(settings: dict) -> None:
    """Initialise all objects and start the simulation loop."""
    _do_reset()

    tier_name = settings["tier"]
    tier_cfg = settings["tier_config"]
    st.session_state.current_tier = tier_name

    _ensure_catalogue_and_ism(settings)

    cat = st.session_state.catalogue
    n_pulsars = tier_cfg["n_pulsars"]
    pulsars = cat.get_top_n(n_pulsars)
    st.session_state.pulsars = pulsars

    # Create spacecraft
    from core.spacecraft import Spacecraft
    rng = np.random.default_rng()
    preset = settings.get("preset", "Random deep space (default)")
    if preset == "Random deep space (default)":
        sc = Spacecraft.random_deep_space(rng=rng)
    elif preset == "Near a Sun-like star":
        sc = Spacecraft.near_sun_like_star(rng=rng)
    elif preset == "Galactic centre region":
        sc = Spacecraft.at_galactic_centre(rng=rng)
    elif preset == "Void between spiral arms":
        # Inter-arm void per Appendix E.6: galactocentric R 10–14 kpc, in-disk
        sc = Spacecraft.interarm_void(rng=rng)
    elif preset == "Manual (GL / GB / Distance)":
        sc = Spacecraft.from_galactic(
            settings["gl_manual"], settings["gb_manual"], settings["dist_manual"],
        )
    else:
        sc = Spacecraft.random_deep_space(rng=rng)

    sc.blind_mode = settings.get("blind_mode", False)
    if settings.get("central_body_mass_kg", 0.0) > 0:
        sc.central_body_mass_kg = settings["central_body_mass_kg"]
        sc.orbit_radius_m = 1.5e11   # fiducial 1 AU orbit

    st.session_state.spacecraft = sc

    # Create particle filter
    from core.estimator import ParticleFilter
    pf = ParticleFilter(
        n_particles=tier_cfg["n_particles"],
        tier_config=tier_cfg,
        seed=42,
    )
    st.session_state.filter = pf
    st.session_state.running = True

    st.session_state.sim_logger.log_event("run_start", {
        "tier": tier_name,
        "n_pulsars": len(pulsars),
        "preset": preset,
    })
    logger.info("Simulation started: %s, %d pulsars, %d particles",
                tier_name, len(pulsars), tier_cfg["n_particles"])


# ── Main-area run controls ────────────────────────────────────────────────────
# On phones the sidebar starts collapsed, so the primary action must live in
# the main column — first-time mobile users would otherwise never find RUN.

main_run_clicked = False
main_reset_clicked = False
if not st.session_state.running:
    if st.session_state.filter is None:
        main_run_clicked = st.button(
            "▶  RUN SIMULATION", type="primary", width="stretch",
            key="run_main",
        )
        st.caption(
            f"Tier: **{settings['tier']}** · Scenario: **{settings['preset']}** — "
            "change these in the sidebar (» top-left)."
        )
    else:
        col_again, col_reset_main = st.columns(2)
        main_run_clicked = col_again.button(
            "▶  RUN AGAIN", type="primary", width="stretch", key="run_main",
        )
        main_reset_clicked = col_reset_main.button(
            "↺  RESET", width="stretch", key="reset_main",
        )

# ── Process button clicks ─────────────────────────────────────────────────────

if settings.get("run_clicked") or main_run_clicked:
    _handle_run(settings)

if settings.get("reset_clicked") or main_reset_clicked:
    _do_reset()
    st.rerun()


# ── Advance simulation (one iteration per rerun while running=True) ───────────

if st.session_state.running and st.session_state.filter is not None:
    _MAX_ITER = 20
    progress_frac = min(st.session_state.iteration / _MAX_ITER, 1.0)
    shown_iter = min(st.session_state.iteration + 1, _MAX_ITER)
    st.progress(
        progress_frac,
        text=f"Iteration {shown_iter} of ~{_MAX_ITER} — particle filter updating…",
    )
    with st.spinner(""):
        _run_one_phase5_pipeline(settings)
    # Rerun unconditionally: while running it advances the loop; when the
    # pipeline just finished it redraws the page fresh — without this the
    # browser is left on a stale frame (RUNNING badge + progress bar) until
    # the next user interaction.
    st.rerun()


# ── Build simulation data dict for UI panels ──────────────────────────────────

sim_data = _build_sim_data(settings)


# ── Main area: 5 tabs ─────────────────────────────────────────────────────────

from ui import galaxy_map, timing_panel, convergence_panel, phase_panel, gravity_panel

tab_names = [
    "🎯 Convergence",
    "🌌 Galaxy Map",
    "📡 Timing & DM",
    "⏱ Phase Resolution",
    "🌑 Gravity Well",
]
tabs = st.tabs(tab_names)

with tabs[0]:
    convergence_panel.render(sim_data)

with tabs[1]:
    galaxy_map.render(sim_data)

with tabs[2]:
    timing_panel.render(sim_data)

with tabs[3]:
    phase_panel.render(sim_data)

with tabs[4]:
    gravity_panel.render(sim_data)


# ── Footer: current best estimate bar ─────────────────────────────────────────

st.divider()
pf = st.session_state.filter
if pf is not None and pf.initialised:
    est = pf.get_estimate()
    pos = est["position_kpc"]
    std = est["position_std_kpc"]
    from utils.coordinates import cartesian_to_galactic
    gl, gb, dist = cartesian_to_galactic(pos)
    sc = st.session_state.spacecraft
    err_val = None
    if sc is not None and not settings.get("blind_mode", False):
        err_val = float(np.linalg.norm(pos - sc.true_position_kpc))

    m1, m2, m3, m4, dl_col = st.columns([2, 2, 2, 2, 1])
    with m1:
        st.metric("Position (GL / GB)", f"{gl:.1f}° / {gb:.1f}°",
                  help="Current best-estimate position in galactic coordinates")
    with m2:
        st.metric("Distance from Sun", f"{dist:.2f} kpc",
                  help="Heliocentric distance to estimated spacecraft position")
    with m3:
        st.metric("Uncertainty (1σ)", f"{np.mean(std):.2f} kpc",
                  help=f"σ = ({std[0]:.2f}, {std[1]:.2f}, {std[2]:.2f}) kpc per axis")
    with m4:
        if err_val is not None:
            st.metric("Error vs truth", f"{err_val:.3f} kpc",
                      help="Distance between estimate and true spacecraft position")
        else:
            st.metric("Iteration", str(st.session_state.iteration),
                      help="Number of particle filter update steps completed")
    with dl_col:
        # Build CSV from history for download
        import io
        history_rows = st.session_state.history
        if history_rows:
            csv_lines = ["iteration,x_kpc,y_kpc,z_kpc,uncertainty_kpc,error_kpc,ess_pre"]
            for h in history_rows:
                p = h.get("pos_kpc", [0, 0, 0])
                csv_lines.append(
                    f"{h.get('step', 0)},"
                    f"{p[0]:.4f},{p[1]:.4f},{p[2]:.4f},"
                    f"{h.get('uncertainty_kpc', 0):.4f},"
                    f"{h.get('error_kpc', 0):.4f},"
                    f"{h.get('ess_pre', 0):.4f}"
                )
            csv_bytes = "\n".join(csv_lines).encode("utf-8")
            st.download_button(
                "⬇ Results (CSV)",
                data=csv_bytes,
                file_name="xnav_results.csv",
                mime="text/csv",
                width="stretch",
            )
else:
    st.markdown(
        '<div style="background:#111128; border:1px solid #1A1A3A; padding:8px 16px; '
        'border-radius:6px; font-size:0.82em; color:#555;">'
        'Position not yet estimated — configure a scenario in the sidebar and click '
        '<b>▶ RUN SIMULATION</b> to begin.'
        '</div>',
        unsafe_allow_html=True,
    )
