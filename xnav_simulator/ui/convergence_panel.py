# ui/convergence_panel.py — Convergence panel for XNAV Cold Start Simulator
# UI boundary: receives plain data only. No core/ or stages/ imports permitted.

"""
Three sub-panels:
  Top    — 3D particle cloud (Plotly 3D scatter, subsampled to top 1000 by weight)
  Middle — Uncertainty timeline with ESS health indicator (pre-resample ESS)
  Bottom — Stage completion status row

DISPLAY-ONLY APPROXIMATION: The 3D scatter subsamples to the top-1000 particles
by weight for rendering performance at Balanced tier (5000 particles). The
filter state in session_state retains all particles. Accuracy of the visual
representation is sufficient for qualitative convergence assessment.

Data dict keys expected:
    particles_kpc:      (N, 3) all particle positions (subsampled for display)
    weights:            (N,)   particle weights
    history:            list of dicts {step, pos_kpc, uncertainty_kpc, ess_pre, ess_post}
    true_pos_kpc:       (3,) or None
    estimate_kpc:       (3,)
    uncertainty_kpc:    float — current 1σ radius
    ess_pre:            float — PRE-resampling ESS fraction (used for health colouring)
    ess_post:           float — POST-resampling ESS fraction (display only, not for health)
    stage_status:       dict {stage1,stage2,stage3,stage4: "pending"|"running"|"complete"}
    iteration:          int
    blind_mode:         bool
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

import config


_ACCENT = config.COLOUR_ACCENT
_BG = config.COLOUR_BG
_GRID = config.COLOUR_GRID

_STAGE_LABELS = {
    "stage1": "Stage 1\nDM Localisation",
    "stage2": "Stage 2\nProfile Matching",
    "stage3": "Stage 3\nGeometry",
    "stage4": "Stage 4\nPhase Ambiguity",
}
_STAGE_DESCRIPTIONS = {
    "stage1": "Coarse position from DM grid: narrows galaxy to ~few kpc",
    "stage2": "Pulsar ID from profile matching: confirms which pulsars are observed",
    "stage3": "Line-of-sight geometry: tightens position along each pulsar direction",
    "stage4": "Phase ambiguity resolution: pins clock offset to millisecond accuracy",
}


def _ess_colour(ess_pre: float) -> str:
    """Map pre-resample ESS fraction to green/amber/red health indicator colour."""
    if ess_pre >= 0.3:
        return "#00CC66"    # green — healthy
    if ess_pre >= 0.1:
        return "#FFAA00"    # amber — low diversity
    return "#FF4444"        # red — near collapse


# ── 3D particle cloud ─────────────────────────────────────────────────────────

def build_particle_figure(data: dict) -> go.Figure:
    """3D scatter of particle cloud.

    DISPLAY-ONLY APPROXIMATION: Renders top-1000 particles by weight for
    interactive performance. Full particle array is kept in session state.
    At Balanced tier (5000 particles) Plotly 3D serialises ~800 KB per render;
    subsampling to 1000 reduces this to ~160 KB while preserving the visual
    shape of the posterior.
    """
    particles = np.asarray(data.get("particles_kpc", np.zeros((1, 3))))
    weights = np.asarray(data.get("weights", np.ones(len(particles)) / len(particles)))
    estimate = np.asarray(data.get("estimate_kpc", [0, 0, 0]))
    true_pos = data.get("true_pos_kpc")
    blind_mode = data.get("blind_mode", False)
    n_particles = len(particles)

    # Subsample to top 1000 by weight for display performance
    _MAX_DISPLAY = 1000
    if n_particles > _MAX_DISPLAY:
        idx = np.argsort(weights)[-_MAX_DISPLAY:]
        disp_particles = particles[idx]
        disp_weights = weights[idx]
    else:
        disp_particles = particles
        disp_weights = weights

    # Normalise weights for colour mapping
    w_norm = disp_weights / (disp_weights.max() + 1e-300)

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=disp_particles[:, 0],
        y=disp_particles[:, 1],
        z=disp_particles[:, 2],
        mode="markers",
        marker=dict(
            size=3,
            color=w_norm,
            colorscale=[[0, "rgba(0,100,200,0.2)"], [1, f"rgba(0,212,255,0.9)"]],
            opacity=0.7,
        ),
        hoverinfo="skip",
        name="Particles",
        showlegend=True,
    ))

    # Estimated position
    fig.add_trace(go.Scatter3d(
        x=[estimate[0]], y=[estimate[1]], z=[estimate[2]],
        mode="markers",
        marker=dict(size=10, color=_ACCENT, symbol="circle",
                    line=dict(width=2, color="white")),
        name="Estimate",
        hovertemplate=f"Estimate: ({estimate[0]:.2f}, {estimate[1]:.2f}, {estimate[2]:.2f}) kpc<extra></extra>",
    ))

    # True position (non-blind mode only)
    if not blind_mode and true_pos is not None:
        tp = np.asarray(true_pos)
        fig.add_trace(go.Scatter3d(
            x=[tp[0]], y=[tp[1]], z=[tp[2]],
            mode="markers",
            marker=dict(size=10, color="#FF4444", symbol="circle"),
            name="True position",
            hovertemplate=f"True: ({tp[0]:.2f}, {tp[1]:.2f}, {tp[2]:.2f}) kpc<extra></extra>",
        ))

    fig.update_layout(
        paper_bgcolor=_BG,
        plot_bgcolor=_BG,
        scene=dict(
            bgcolor="#0D0D22",
            xaxis=dict(title="X (kpc)", gridcolor=_GRID, color="#888"),
            yaxis=dict(title="Y (kpc)", gridcolor=_GRID, color="#888"),
            zaxis=dict(title="Z (kpc)", gridcolor=_GRID, color="#888"),
        ),
        font=dict(color="#AAAACC", size=11),
        margin=dict(l=0, r=0, t=30, b=0),
        title=dict(
            text=f"Particle Cloud (showing {len(disp_particles):,}/{n_particles:,})",
            font=dict(color=_ACCENT, size=12), x=0.5,
        ),
        height=380,
        legend=dict(bgcolor="rgba(0,0,0,0.5)", font=dict(size=10)),
    )
    return fig


# ── Uncertainty timeline ──────────────────────────────────────────────────────

def build_uncertainty_timeline_figure(data: dict) -> go.Figure:
    """Uncertainty radius + ESS health timeline.

    ESS health colouring uses PRE-resampling ESS (ess_pre field), NOT
    post-resampling ESS. Post-resample ESS = 1.0 is normal in the Roemer-
    dominated regime and must not trigger amber/red colouring (Appendix D.3).
    """
    history = data.get("history", [])
    stage_status = data.get("stage_status", {})

    if not history:
        fig = go.Figure(layout=go.Layout(
            paper_bgcolor=_BG, plot_bgcolor="#0D0D22",
            font=dict(color="#AAAACC", size=11),
            title=dict(text="Uncertainty Timeline", font=dict(color=_ACCENT, size=12), x=0.5),
            height=240,
        ))
        fig.add_annotation(text="Run simulation to see convergence history",
                           showarrow=False, font=dict(color="#888"),
                           xref="paper", yref="paper", x=0.5, y=0.5)
        return fig

    steps = [h["step"] for h in history]
    uncertainties = [h.get("uncertainty_kpc", 0.0) for h in history]
    ess_pre_vals = [h.get("ess_pre", 1.0) for h in history]
    ess_post_vals = [h.get("ess_post", 1.0) for h in history]

    from plotly.subplots import make_subplots
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Uncertainty (left y-axis)
    fig.add_trace(go.Scatter(
        x=steps, y=uncertainties,
        mode="lines+markers",
        line=dict(color=_ACCENT, width=2),
        marker=dict(size=5),
        name="Uncertainty (kpc)",
    ), secondary_y=False)

    # ESS pre-resample (right y-axis, colour-coded health)
    ess_colours = [_ess_colour(e) for e in ess_pre_vals]
    fig.add_trace(go.Scatter(
        x=steps, y=ess_pre_vals,
        mode="lines+markers",
        line=dict(color="#FFAA00", width=1, dash="dot"),
        marker=dict(size=6, color=ess_colours),
        name="ESS pre-resample (fraction)",
        hovertemplate="Step %{x}: ESS=%{y:.3f}<extra></extra>",
    ), secondary_y=True)

    # Stage completion vertical lines
    _STAGE_STEPS = {"stage1": 0, "stage2": 3, "stage3": 3}
    for stage_key, stage_step in _STAGE_STEPS.items():
        status = stage_status.get(stage_key, "pending")
        if status == "complete" and steps:
            # Find the step where this stage completed
            target_step = min(stage_step, max(steps))
            fig.add_vline(
                x=target_step,
                line=dict(color="#448844", dash="dash", width=1),
                annotation_text=stage_key.replace("stage", "S"),
                annotation_font=dict(size=9, color="#448844"),
            )

    fig.update_layout(
        paper_bgcolor=_BG,
        plot_bgcolor="#0D0D22",
        font=dict(color="#AAAACC", size=11),
        title=dict(text="Uncertainty & ESS Timeline", font=dict(color=_ACCENT, size=12), x=0.5),
        legend=dict(bgcolor="rgba(0,0,0,0.5)", font=dict(size=10), x=0.01, y=0.99),
        xaxis=dict(
            title="Iteration", showgrid=True, gridcolor=_GRID,
            zeroline=False, color="#888",
        ),
        height=240,
    )
    fig.update_yaxes(
        title_text="Uncertainty (kpc, log)", type="log",
        showgrid=True, gridcolor=_GRID, color="#888",
        secondary_y=False,
    )
    fig.update_yaxes(
        title_text="ESS fraction (pre-resample)", range=[0, 1.05],
        showgrid=False, color="#FFAA00",
        secondary_y=True,
    )
    return fig


# ── Stage status row ──────────────────────────────────────────────────────────

def render_stage_status(stage_status: dict) -> None:
    """Render four stage status boxes as Streamlit columns."""
    cols = st.columns(4)
    status_colours = {
        "pending": ("#555555", "grey", "◯"),
        "running": ("#FF8800", "#FF8800", "◉"),
        "complete": ("#00CC66", "#00CC66", "✓"),
    }
    for col, (key, label) in zip(cols, _STAGE_LABELS.items()):
        status = stage_status.get(key, "pending")
        bg_col, text_col, icon = status_colours.get(status, status_colours["pending"])
        with col:
            st.markdown(
                f'<div style="background:{bg_col}22; border:1px solid {bg_col}; '
                f'border-radius:6px; padding:8px; text-align:center;">'
                f'<span style="color:{text_col}; font-size:1.4em;">{icon}</span><br>'
                f'<span style="color:{text_col}; font-size:0.75em; font-weight:600;">'
                f'{label.replace(chr(10), "<br>")}</span><br>'
                f'<span style="color:#888; font-size:0.65em;">{status.upper()}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
            if st.button("?", key=f"stage_info_{key}", help=_STAGE_DESCRIPTIONS[key]):
                pass  # tooltip is in the help argument


# ── Streamlit render ──────────────────────────────────────────────────────────

def render(data: dict) -> None:
    """Render the full convergence panel in Streamlit."""
    # 3D particle cloud
    fig3d = build_particle_figure(data)
    st.plotly_chart(fig3d, use_container_width=True)

    # Playback scrubber
    history = data.get("history", [])
    if len(history) > 1:
        scrub_step = st.slider(
            "Playback iteration",
            min_value=0, max_value=len(history) - 1,
            value=len(history) - 1,
            key="convergence_scrubber",
        )
        # Show historical particle cloud at scrub_step if playback enabled
        hist_entry = history[scrub_step]
        if "particles_kpc" in hist_entry:
            data_hist = dict(data, particles_kpc=hist_entry["particles_kpc"],
                             weights=hist_entry.get("weights", data.get("weights")))
            fig_hist = build_particle_figure(data_hist)
            st.plotly_chart(fig_hist, use_container_width=True)

    # Uncertainty / ESS timeline
    fig_timeline = build_uncertainty_timeline_figure(data)
    st.plotly_chart(fig_timeline, use_container_width=True)

    # Stage status row
    st.markdown(
        f'<p style="color:{_ACCENT}; font-size:0.8em; margin-top:0.5em;">STAGE STATUS</p>',
        unsafe_allow_html=True,
    )
    render_stage_status(data.get("stage_status", {}))
