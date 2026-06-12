# ui/galaxy_map.py — Galaxy map panels for XNAV Cold Start Simulator
# UI boundary: receives plain data only. No core/ or stages/ imports permitted.

"""
Two-panel galaxy map:
  Left  — Top-down galactic disk view (Plotly scatter)
  Right — Pulsar sky map in Mollweide-approximated galactic coordinates

Data dict keys expected:
    pulsars:          list of dicts {name, gl, gb, distance_kpc, dm, timing_noise_ns,
                                    identified: bool, confidence: float}
    sc_pos_kpc:       (3,) spacecraft estimated position
    true_pos_kpc:     (3,) or None (None in blind mode)
    particle_pos:     (N,3) particle positions (optional — skipped if absent)
    particle_weights: (N,)  particle weights    (optional)
    uncertainty_kpc:  float — 1σ radius in kpc
    sun_pos_kpc:      (3,) — always [-8.178, 0, 0]
    iteration:        int
    blind_mode:       bool
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

import config


# ── Colour helpers ────────────────────────────────────────────────────────────

_ACCENT = config.COLOUR_ACCENT
_BG = config.COLOUR_BG
_GRID = config.COLOUR_GRID
_AXIS_STYLE = dict(
    showgrid=True,
    gridcolor=_GRID,
    zeroline=False,
    color="#888",
    tickcolor="#888",
)
_PAPER_BG = _BG
_PLOT_BG = "#0D0D22"


def _dark_layout(title: str = "", **kwargs) -> go.Layout:
    return go.Layout(
        title=dict(text=title, font=dict(color=_ACCENT, size=13), x=0.5),
        paper_bgcolor=_PAPER_BG,
        plot_bgcolor=_PLOT_BG,
        font=dict(color="#AAAACC", size=11),
        margin=dict(l=40, r=20, t=40, b=40),
        **kwargs,
    )


# ── Spiral arm background ─────────────────────────────────────────────────────

def _make_spiral_arm_trace() -> go.Scatter:
    """Generate 4 log-spiral arms as faint background traces.

    APPROXIMATION: 4-arm log-spiral with pitch angle 12° — matches Milky Way
    morphology at ~20% accuracy. Purely decorative; does not affect navigation.
    """
    pitch = np.radians(12.0)
    arm_offsets = [0.0, np.pi / 2, np.pi, 3 * np.pi / 2]
    r0 = 3.0  # kpc start radius
    theta = np.linspace(0.0, 3.5 * np.pi, 300)
    xs, ys = [], []
    for offset in arm_offsets:
        r = r0 * np.exp(pitch * theta)
        x = r * np.cos(theta + offset)
        y = r * np.sin(theta + offset)
        xs.extend(list(x) + [None])
        ys.extend(list(y) + [None])
    return go.Scatter(
        x=xs, y=ys,
        mode="lines",
        line=dict(color="rgba(60,40,120,0.35)", width=12),
        hoverinfo="skip",
        showlegend=False,
        name="_arms",
    )


# ── Main figure builders ──────────────────────────────────────────────────────

def build_topdown_figure(data: dict) -> go.Figure:
    """Build the top-down galactic disk view.

    Parameters are extracted from the data dict — no Streamlit calls here.
    """
    pulsars = data.get("pulsars", [])
    sc_pos = np.asarray(data.get("sc_pos_kpc", [0, 0, 0]))
    true_pos = data.get("true_pos_kpc")
    sun_pos = np.asarray(data.get("sun_pos_kpc", config.SUN_POS_KPC))
    blind_mode = data.get("blind_mode", False)
    uncertainty_kpc = data.get("uncertainty_kpc", 1.0)

    fig = go.Figure(layout=_dark_layout("Top-down: Galactic Disk (XY plane)"))

    # Spiral arm background
    fig.add_trace(_make_spiral_arm_trace())

    # Galactic disk boundary circle
    theta_c = np.linspace(0, 2 * np.pi, 200)
    r_disk = config.GALAXY_RADIUS_KPC
    fig.add_trace(go.Scatter(
        x=r_disk * np.cos(theta_c),
        y=r_disk * np.sin(theta_c),
        mode="lines",
        line=dict(color=_GRID, width=1, dash="dot"),
        hoverinfo="skip",
        showlegend=False,
        name="_disk_boundary",
    ))

    # Pulsars — coloured by DM
    if pulsars:
        from utils.coordinates import galactic_to_cartesian
        px, py, pnames, pdms, pconf = [], [], [], [], []
        for p in pulsars:
            pos = galactic_to_cartesian(p["gl"], p["gb"], p["distance_kpc"])
            px.append(pos[0])
            py.append(pos[1])
            pnames.append(p["name"])
            pdms.append(p.get("dm", 0.0))
            pconf.append(p.get("confidence", 0.0))

        fig.add_trace(go.Scatter(
            x=px, y=py,
            mode="markers",
            marker=dict(
                size=5,
                color=pdms,
                colorscale="Viridis",
                colorbar=dict(title="DM (pc cm⁻³)", thickness=10, len=0.6),
                opacity=0.7,
            ),
            text=pnames,
            hovertemplate="<b>%{text}</b><br>DM=%{marker.color:.1f} pc cm⁻³<extra></extra>",
            showlegend=False,
            name="Pulsars",
        ))

    # Particle cloud (subsampled to top-500 by weight for rendering performance)
    particle_pos = data.get("particle_pos")
    particle_weights = data.get("particle_weights")
    if particle_pos is not None and len(particle_pos) > 1:
        part = np.asarray(particle_pos)
        wts = np.asarray(particle_weights) if particle_weights is not None else np.ones(len(part))
        # Subsample: top 500 particles by weight
        n_show = min(500, len(part))
        top_idx = np.argsort(wts)[-n_show:]
        wts_show = wts[top_idx]
        wts_show = wts_show / wts_show.max() if wts_show.max() > 0 else wts_show
        fig.add_trace(go.Scatter(
            x=part[top_idx, 0], y=part[top_idx, 1],
            mode="markers",
            marker=dict(
                size=3,
                color=f"rgba(0,212,255,0.18)",
                opacity=0.5,
            ),
            hoverinfo="skip",
            showlegend=False,
            name="_particles",
        ))

    theta_u = np.linspace(0, 2 * np.pi, 200)
    has_estimate = data.get("particle_pos") is not None

    if not has_estimate:
        # No simulation run yet — show helpful annotation instead of a misleading estimate
        fig.add_annotation(
            text="▶  Run simulation to see position estimate",
            showarrow=False,
            font=dict(color="#AAAACC", size=13),
            xref="paper", yref="paper", x=0.5, y=0.05,
            bgcolor="rgba(17,17,40,0.8)",
            bordercolor="#1A1A3A",
            borderwidth=1,
        )
        fig.update_layout(
            xaxis=dict(**_AXIS_STYLE, title="X (kpc)", range=[-16, 16]),
            yaxis=dict(**_AXIS_STYLE, title="Y (kpc)", range=[-16, 16],
                       scaleanchor="x", scaleratio=1),
            # Horizontal legend below the plot — anchored top-left it overlaps
            # the title on phone-width viewports.
            legend=dict(bgcolor="rgba(0,0,0,0.5)", orientation="h",
                        x=0.5, xanchor="center", y=-0.18, font=dict(size=10)),
            height=420,
        )
        return fig

    # True position — only in non-blind mode; rendered before estimate so estimate is on top
    if not blind_mode and true_pos is not None:
        tp = np.asarray(true_pos)
        fig.add_trace(go.Scatter(
            x=[tp[0]], y=[tp[1]],
            mode="markers",
            marker=dict(size=12, color="#FF4444", symbol="x", opacity=1.0,
                        line=dict(width=2, color="#FF4444")),
            name="True position",
            showlegend=True,
            hovertemplate="True: (%{x:.2f}, %{y:.2f}) kpc<extra></extra>",
        ))

    # Uncertainty circle (labelled for legend)
    fig.add_trace(go.Scatter(
        x=sc_pos[0] + uncertainty_kpc * np.cos(theta_u),
        y=sc_pos[1] + uncertainty_kpc * np.sin(theta_u),
        mode="lines",
        line=dict(color=f"rgba(0,212,255,0.3)", width=1, dash="dash"),
        hoverinfo="skip",
        showlegend=True,
        name=f"Uncertainty (1σ = {uncertainty_kpc:.1f} kpc)",
    ))

    # Spacecraft estimated position — added last so it renders on top
    fig.add_trace(go.Scatter(
        x=[sc_pos[0]], y=[sc_pos[1]],
        mode="markers",
        marker=dict(size=14, color=_ACCENT, symbol="star", line=dict(width=1, color="white")),
        name="Estimate",
        hovertemplate=f"Estimate: ({sc_pos[0]:.2f}, {sc_pos[1]:.2f}) kpc<extra></extra>",
    ))

    # Sun reference
    fig.add_trace(go.Scatter(
        x=[sun_pos[0]], y=[sun_pos[1]],
        mode="markers+text",
        marker=dict(size=8, color="#FFD700", symbol="circle"),
        text=["☀"],
        textposition="top center",
        textfont=dict(size=10, color="#FFD700"),
        name="Sun",
        hovertemplate=f"Sun: ({sun_pos[0]:.2f}, {sun_pos[1]:.2f}) kpc<extra></extra>",
    ))

    fig.update_layout(
        xaxis=dict(**_AXIS_STYLE, title="X (kpc)", range=[-16, 16]),
        yaxis=dict(**_AXIS_STYLE, title="Y (kpc)", range=[-16, 16],
                   scaleanchor="x", scaleratio=1),
        # Horizontal legend below the plot — inside the axes it covers the
        # galaxy on phone-width viewports.
        legend=dict(bgcolor="rgba(0,0,0,0.5)", orientation="h",
                    x=0.5, xanchor="center", y=-0.18, font=dict(size=10)),
        height=420,
    )
    return fig


def build_skymap_figure(data: dict) -> go.Figure:
    """Build the pulsar sky map in galactic coordinates (Mollweide approximation).

    APPROXIMATION: Plotly does not support true Mollweide projection natively.
    We use a flat equirectangular plot with GL on the x-axis and GB on y-axis
    and label it as such. The visual layout matches a standard galactic sky map.
    """
    pulsars = data.get("pulsars", [])
    sc_pos = np.asarray(data.get("sc_pos_kpc", [0, 0, 0]))

    fig = go.Figure(layout=_dark_layout("Pulsar Sky Map (Galactic Coordinates)"))

    if not pulsars:
        fig.add_annotation(
            text="Pulsar catalogue loads when simulation starts",
            showarrow=False,
            font=dict(color="#888", size=13),
            xref="paper", yref="paper", x=0.5, y=0.5,
        )
        fig.update_layout(height=420)
        return fig

    gls, gbs, names, scores, identified = [], [], [], [], []
    for p in pulsars:
        gls.append(p["gl"])
        gbs.append(p["gb"])
        names.append(p["name"])
        scores.append(p.get("timing_noise_ns", 100.0))
        identified.append(p.get("identified", False))

    gls = np.array(gls)
    gbs = np.array(gbs)

    # Unidentified pulsars
    mask_un = ~np.array(identified, dtype=bool)
    if mask_un.any():
        fig.add_trace(go.Scatter(
            x=gls[mask_un], y=gbs[mask_un],
            mode="markers",
            marker=dict(
                size=6,
                color=np.array(scores)[mask_un],
                colorscale="RdYlGn_r",
                cmin=1, cmax=500,
                opacity=0.6,
            ),
            text=np.array(names)[mask_un],
            hovertemplate="<b>%{text}</b><br>GL=%{x:.1f}°, GB=%{y:.1f}°<extra></extra>",
            showlegend=True,
            name="Unidentified",
        ))

    # Identified pulsars (larger, highlighted)
    mask_id = np.array(identified, dtype=bool)
    if mask_id.any():
        fig.add_trace(go.Scatter(
            x=gls[mask_id], y=gbs[mask_id],
            mode="markers",
            marker=dict(
                size=11,
                color=_ACCENT,
                symbol="circle-open",
                line=dict(width=2, color=_ACCENT),
            ),
            text=np.array(names)[mask_id],
            hovertemplate="<b>%{text}</b><br>GL=%{x:.1f}°, GB=%{y:.1f}° (identified)<extra></extra>",
            showlegend=True,
            name="Identified",
        ))

    fig.update_layout(
        xaxis=dict(**_AXIS_STYLE, title="Galactic Longitude GL (°)", range=[0, 360]),
        yaxis=dict(**_AXIS_STYLE, title="Galactic Latitude GB (°)", range=[-90, 90]),
        height=420,
    )
    return fig


# ── Streamlit render ──────────────────────────────────────────────────────────

def render(data: dict) -> None:
    """Render both galaxy map panels side by side in Streamlit."""
    col1, col2 = st.columns(2)
    with col1:
        fig = build_topdown_figure(data)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    with col2:
        fig = build_skymap_figure(data)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
