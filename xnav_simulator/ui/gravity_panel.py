# ui/gravity_panel.py — Gravity well panel for XNAV Cold Start Simulator
# UI boundary: receives plain data only. No core/ or stages/ imports permitted.

"""
Three sub-panels:
  Top    — Gravitational potential estimate (large numerical readout)
  Middle — Reality check bar chart: signal vs noise (log scale)
  Bottom — Improvement roadmap text

PHYSICS CORRECTIONS (per Appendix D, Phase 5 testing):
  - Detection bar chart MUST show DM turbulence as dominant noise, not timing noise.
  - Annotation trigger: gravity_signal < dm_turbulence (NOT gravity < timing_noise).
  - Annotation text: "buried under ISM DM turbulence" (NOT "cannot detect").
  - Improvement roadmap: binding constraint is ISM model fidelity, NOT collecting area.
  - gravity_signal_ns = |Φ/c²| × T_integration × 1e9  (scales with integration time T).

Data dict keys expected:
    phi_m2s2:            float — gravitational potential in m²/s² (negative)
    phi_uncertainty:     float — ±σ in m²/s²
    true_phi_m2s2:       float or None — ground truth (hidden in blind mode)
    integration_time_s:  float — per-pulsar integration time
    timing_noise_ns:     float — RMS timing noise of best pulsar (ns)
    dm_value:            float — current DM estimate (pc cm⁻³)
    frequency_mhz:       float — observing frequency (MHz)
    collecting_area_m2:  float — detector collecting area
    n_pulsars:           int   — number of active pulsars
    blind_mode:          bool
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

import config


_ACCENT = config.COLOUR_ACCENT
_BG = config.COLOUR_BG
_GRID = config.COLOUR_GRID
_PLOT_BG = "#0D0D22"

_C2 = config.C_LIGHT ** 2    # m²/s²  — used in gravity signal formula


# ── Gravity signal and noise model ───────────────────────────────────────────

def _compute_noise_budget(data: dict) -> dict:
    """Compute all noise components in nanoseconds.

    Returns a dict with keys: gravity_ns, dm_turb_ns, timing_noise_ns,
    photon_noise_ns, solar_wind_ns.

    PHYSICS (per Appendix D.1):
      Ordering: timing_noise << gravity << photon_noise << DM_turbulence
      At T=1000s: gravity~340μs, DM_turb~16ms, photon~5μs, timing~3ns.

    gravity_signal_ns = |Φ/c²| × T × 1e9   (scales with integration time T)
    dm_turb_ns        = K_DM × 0.15 × DM / f²  × 1e9
    photon_noise_ns   ≈ σ_TOA from photon statistics (simplified model)
    timing_noise_ns   = pulsar RMS timing residual (hardware floor)
    solar_wind_ns     = A_sw × DM_solar / f²  × 1e9  (APPROXIMATION: Parker model)
    """
    phi = abs(data.get("phi_m2s2", -2.0e11))
    T = data.get("integration_time_s", 1000.0)
    timing_ns = data.get("timing_noise_ns", 100.0)
    dm = max(data.get("dm_value", 30.0), 0.5)
    freq_mhz = data.get("frequency_mhz", 1400.0)
    area_m2 = max(data.get("collecting_area_m2", 1.0), 0.01)
    n_pulsars = max(data.get("n_pulsars", 1), 1)

    # Gravity timing signal (Appendix D.1 formula)
    gravity_ns = (phi / _C2) * T * 1e9

    # DM turbulence: 15% of catalogue DM, dominant noise source
    dm_turb_s = config.K_DM * 0.15 * dm / (freq_mhz ** 2)
    dm_turb_ns = dm_turb_s * 1e9

    # APPROXIMATION: Photon noise model uses simplified TOA uncertainty.
    # σ_TOA ≈ W50 / (S/N) where S/N ∝ sqrt(A × T × S1400).
    # Uses fiducial S1400=1 mJy, W50=100μs. Error < factor 3 across MSP catalogue.
    # Real calculation requires per-pulsar flux which is not always available.
    w50_s = 100e-6   # fiducial 100 μs pulse width
    flux_jy = 1e-3   # fiducial 1 mJy
    # S/N ∝ sqrt(area × T × flux) / w50
    sn = np.sqrt(area_m2 * T * flux_jy) * 1e3 / (w50_s * 1e6)
    photon_noise_s = max(w50_s / max(sn, 1.0), 1e-9)
    photon_noise_ns = photon_noise_s * 1e9

    # APPROXIMATION: Solar wind contribution using Parker model (Appendix A).
    # A_sw ≈ 8.5e-5 pc cm⁻³ AU for quiet solar wind (mid-range of build brief spec).
    # Dispersive delay: δt = K_DM × A_sw / (f² × r_AU) where r_AU ≈ distance_AU.
    # For galactic pulsars, solar wind is negligible vs ISM DM; included for completeness.
    A_sw = 8.5e-5    # pc cm⁻³ AU (quiet solar wind)
    r_au = 1.0       # at 1 AU from Sun reference
    solar_wind_ns = config.K_DM * A_sw / (r_au * freq_mhz ** 2) * 1e9

    return {
        "gravity_ns": gravity_ns,
        "dm_turb_ns": dm_turb_ns,
        "timing_noise_ns": timing_ns,
        "photon_noise_ns": photon_noise_ns,
        "solar_wind_ns": solar_wind_ns,
    }


# ── Top: potential readout ────────────────────────────────────────────────────

def build_potential_figure(data: dict) -> go.Figure:
    """Number line showing estimated Φ with uncertainty band.

    True value shown as a tick mark in non-blind mode.
    """
    phi = data.get("phi_m2s2", -2.0e11)
    phi_unc = data.get("phi_uncertainty", abs(phi) * 0.3)
    true_phi = data.get("true_phi_m2s2")
    blind_mode = data.get("blind_mode", False)

    fig = go.Figure()

    # Estimate with error bar
    fig.add_trace(go.Scatter(
        x=[phi], y=[0.0],
        mode="markers",
        marker=dict(size=14, color=_ACCENT, symbol="circle"),
        error_x=dict(type="data", array=[phi_unc], visible=True,
                     color=_ACCENT, thickness=3, width=12),
        name="Estimate",
        hovertemplate=f"Φ = {phi:.3e} ± {phi_unc:.2e} m²/s²<extra></extra>",
    ))

    # True value in non-blind mode
    if not blind_mode and true_phi is not None:
        fig.add_trace(go.Scatter(
            x=[true_phi], y=[0.0],
            mode="markers",
            marker=dict(size=14, color="#FF4444", symbol="line-ns",
                        line=dict(width=3, color="#FF4444")),
            name="True value",
            hovertemplate=f"True Φ = {true_phi:.3e} m²/s²<extra></extra>",
        ))

    fig.update_layout(
        paper_bgcolor=_BG,
        plot_bgcolor=_PLOT_BG,
        font=dict(color="#AAAACC", size=11),
        height=110,
        margin=dict(l=50, r=20, t=30, b=30),
        title=dict(text="Gravitational Potential Φ (m²/s²)",
                   font=dict(color=_ACCENT, size=12), x=0.5),
        xaxis=dict(showgrid=False, zeroline=True, zerolinecolor=_GRID,
                   color="#888", tickformat=".2e"),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   range=[-0.5, 0.5]),
        showlegend=True,
        legend=dict(bgcolor="rgba(0,0,0,0.5)", font=dict(size=10),
                    orientation="h", y=-0.25),
    )
    return fig


# ── Middle: reality check bar chart ──────────────────────────────────────────

def build_reality_check_figure(data: dict) -> go.Figure:
    """Horizontal log-scale bar chart: signal vs noise.

    CRITICAL physics requirements (Appendix D.1):
    1. DM turbulence MUST appear as the longest/dominant bar.
    2. Annotation trigger: gravity < dm_turbulence (NOT gravity < timing_noise).
    3. Annotation text: "Gravity signal undetectable — buried under ISM DM turbulence"
    4. Bar ordering (largest first): DM turbulence, photon noise, gravity,
       solar wind, timing noise floor.
    """
    budget = _compute_noise_budget(data)
    gravity_ns = budget["gravity_ns"]
    dm_turb_ns = budget["dm_turb_ns"]
    timing_ns = budget["timing_noise_ns"]
    photon_ns = budget["photon_noise_ns"]
    solar_ns = budget["solar_wind_ns"]

    # Bars ordered largest → smallest per Appendix D.1
    # (DM turbulence always first — physically it dominates)
    labels = [
        "ISM DM turbulence",
        "Photon noise",
        "Gravity signal",
        "Solar wind",
        "Timing noise floor",
    ]
    values = [dm_turb_ns, photon_ns, gravity_ns, solar_ns, timing_ns]
    colours = ["#FF8800", "#FFCC00", _ACCENT, "#CC88FF", "#FF4444"]

    fig = go.Figure(layout=go.Layout(
        paper_bgcolor=_BG,
        plot_bgcolor=_PLOT_BG,
        font=dict(color="#AAAACC", size=11),
        title=dict(
            text="Why this is hard: signal vs noise",
            font=dict(color=_ACCENT, size=13), x=0.5,
        ),
        height=280,
        margin=dict(l=130, r=100, t=45, b=40),
    ))

    fig.add_trace(go.Bar(
        y=labels,
        x=values,
        orientation="h",
        marker=dict(color=colours, opacity=0.85),
        text=[f"{v:.1e} ns" for v in values],
        textposition="outside",
        textfont=dict(size=9, color="#AAAACC"),
    ))

    # CRITICAL: Annotation when gravity is buried under DM turbulence.
    # Trigger: gravity < DM turbulence (per Appendix D.1).
    # NOT: gravity < timing_noise (that would be physically wrong).
    if gravity_ns < dm_turb_ns:
        ratio = dm_turb_ns / max(gravity_ns, 1e-12)
        fig.add_annotation(
            text=(
                f"Gravity signal undetectable<br>"
                f"buried under ISM DM turbulence<br>"
                f"(DM turbulence is {ratio:.0f}× larger)"
            ),
            xref="paper", yref="paper",
            x=0.98, y=0.5,
            xanchor="right",
            showarrow=False,
            font=dict(size=9, color="#FF8800"),
            bgcolor="rgba(40,20,0,0.7)",
            bordercolor="#FF8800",
            borderwidth=1,
            align="left",
        )

    fig.update_layout(
        xaxis=dict(
            title="Timing residual (ns)",
            type="log",
            showgrid=True, gridcolor=_GRID, zeroline=False, color="#888",
        ),
        yaxis=dict(showgrid=False, color="#AAAACC"),
        showlegend=False,
    )
    return fig


# ── Bottom: improvement roadmap ───────────────────────────────────────────────

def build_roadmap_text(data: dict) -> str:
    """Return the improvement roadmap text.

    CRITICAL (Appendix D.1): The binding constraint is ISM DM model fidelity,
    NOT collecting area or timing noise. The text must NOT suggest "build a
    bigger detector" or "improve timing hardware" as the path to detection.
    """
    budget = _compute_noise_budget(data)
    gravity_ns = budget["gravity_ns"]
    dm_turb_ns = budget["dm_turb_ns"]
    ratio = dm_turb_ns / max(gravity_ns, 1e-12)

    phi = abs(data.get("phi_m2s2", -2.0e11))
    T = data.get("integration_time_s", 1000.0)

    # What ISM improvement factor would be needed?
    # Need: dm_turb_required < gravity_signal
    # dm_turb_required = K_DM × 0.15 × DM / f² × 1e9 × improvement_factor
    # Solve: improvement_factor = gravity_ns / dm_turb_ns
    improvement_factor = gravity_ns / max(dm_turb_ns, 1e-12)

    return (
        f"**Why gravity is undetectable here**\n\n"
        f"Gravity signal: **{gravity_ns:.2e} ns** (at T={T:.0f} s integration)\n\n"
        f"ISM DM turbulence: **{dm_turb_ns:.2e} ns** — **{ratio:.0f}× larger**\n\n"
        f"To detect the gravity signal, ISM DM turbulence modelling must improve "
        f"by a factor of **{1/improvement_factor:.1f}×** — reducing the DM "
        f"residual below **{gravity_ns:.2e} ns**.\n\n"
        f"**Collecting area and timing hardware are not the limiting factors.** "
        f"The gravity signal is already {budget['gravity_ns']/budget['timing_noise_ns']:.0f}× "
        f"above the timing noise floor. The barrier is ISM electron density model "
        f"accuracy — an active research problem with no near-term solution."
    )


# ── Streamlit render ──────────────────────────────────────────────────────────

def render(data: dict) -> None:
    """Render the full gravity panel in Streamlit."""
    if data.get("particle_pos") is None:
        st.markdown(
            '<div style="background:#111128; border:1px solid #1A1A3A; '
            'border-radius:6px; padding:24px; color:#AAAACC; text-align:center;">'
            'Run the simulation to estimate the local gravitational potential.</div>',
            unsafe_allow_html=True,
        )
        return

    phi = data.get("phi_m2s2", -2.0e11)
    phi_unc = data.get("phi_uncertainty", abs(phi) * 0.3)

    # Top: numerical readout + potential figure
    budget = _compute_noise_budget(data)
    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric(
            label="Φ (m²/s²)",
            value=f"{phi:.3e}",
            delta=f"±{phi_unc:.2e}",
            delta_color="off",
        )
        # Equivalent in km above a reference body
        # Δh ≈ |Φ| / g  where g ≈ 9.8 m/s² for Earth reference
        equiv_km = abs(phi) / 9.8 / 1000.0
        st.caption(f"≈ {equiv_km:.2e} km above Earth surface (reference)")

    with col2:
        fig_phi = build_potential_figure(data)
        st.plotly_chart(fig_phi, use_container_width=True)

    # Middle: reality check bar chart
    fig_bar = build_reality_check_figure(data)
    st.plotly_chart(fig_bar, use_container_width=True)

    # Gravity vs DM ratio callout
    ratio = budget["dm_turb_ns"] / max(budget["gravity_ns"], 1e-12)
    st.markdown(
        f'<div style="background:rgba(40,20,0,0.5); border-left:3px solid #FF8800; '
        f'padding:8px 12px; border-radius:4px; font-size:0.82em; color:#FFAA55;">'
        f'Gravity / DM ratio: {budget["gravity_ns"] / budget["dm_turb_ns"]:.3f} '
        f'— gravity is {ratio:.0f}× below DM noise'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Bottom: improvement roadmap
    st.markdown("---")
    st.markdown(build_roadmap_text(data))
