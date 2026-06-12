# ui/timing_panel.py — Timing and DM panel for XNAV Cold Start Simulator
# UI boundary: receives plain data only. No core/ or stages/ imports permitted.

"""
Three sub-panels for the selected pulsar:
  Top    — Dispersive sweep heatmap (frequency vs time)
  Middle — Timing residual breakdown bar chart
  Bottom — Multi-pulsar DM residual plot

Data dict keys expected:
    pulsars:         list of dicts {name, dm, period, distance_kpc, gl, gb,
                                   timing_noise_ns, observed_timing: float,
                                   roemer_s, dispersive_s, timing_noise_s,
                                   photon_noise_s, dm_turbulence_s}
    selected_pulsar: str — name of the pulsar to detail
    frequency_mhz:   float — observing frequency
    integration_time_s: float
    dm_residuals:    dict {name: float} — observed - expected DM per pulsar
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

import config


_ACCENT = config.COLOUR_ACCENT
_BG = config.COLOUR_BG
_GRID = config.COLOUR_GRID
_PLOT_BG = "#0D0D22"
_AXIS_STYLE = dict(showgrid=True, gridcolor=_GRID, zeroline=False, color="#888")


def _dark_layout(title: str = "", **kwargs) -> go.Layout:
    return go.Layout(
        title=dict(text=title, font=dict(color=_ACCENT, size=13), x=0.5),
        paper_bgcolor=_BG,
        plot_bgcolor=_PLOT_BG,
        font=dict(color="#AAAACC", size=11),
        margin=dict(l=50, r=20, t=40, b=40),
        **kwargs,
    )


# ── Dispersive sweep figure ───────────────────────────────────────────────────

def build_dispersive_sweep_figure(data: dict) -> go.Figure:
    """Heatmap showing the dispersive sweep across frequency channels.

    APPROXIMATION: The simulated sweep uses a quadratic DM delay model
    δt(f) = K_DM × DM / f² over a coarse frequency grid (20 channels).
    Real X-ray detectors have energy-dependent effective area not modelled here.
    Error: < 10% across the NICER energy band (1–10 keV) for the sweep shape.
    """
    pulsars = data.get("pulsars", [])
    selected_name = data.get("selected_pulsar", "")
    frequency_mhz = data.get("frequency_mhz", 1400.0)

    # Find selected pulsar
    pulsar = next((p for p in pulsars if p["name"] == selected_name), None)
    if pulsar is None:
        fig = go.Figure(layout=_dark_layout("Dispersive Sweep"))
        fig.add_annotation(text="Select a pulsar", showarrow=False,
                           font=dict(color="#888"), xref="paper", yref="paper",
                           x=0.5, y=0.5)
        fig.update_layout(height=200)
        return fig

    dm = pulsar.get("dm", 30.0)
    period = pulsar.get("period", 0.005)

    # 20 frequency channels around the observing band
    n_channels = 20
    freqs_mhz = np.linspace(frequency_mhz * 0.85, frequency_mhz * 1.15, n_channels)

    # DM delay per channel (seconds), relative to highest frequency
    k_dm = config.K_DM  # MHz² pc⁻¹ cm³ s
    delays = k_dm * dm / freqs_mhz**2
    delays -= delays[-1]   # reference to highest frequency channel

    # Time axis: ±1.5 periods, 80 bins
    n_time = 80
    t_bins = np.linspace(-1.5 * period, 1.5 * period, n_time)

    # Pulse intensity per channel: Gaussian pulse at delay offset
    pulse_width_s = max(pulsar.get("w50", 100.0) * 1e-6, period * 0.01)
    sigma_t = pulse_width_s * 0.5

    intensity = np.zeros((n_channels, n_time))
    for i, delay in enumerate(delays):
        intensity[i, :] = np.exp(-0.5 * ((t_bins - delay) / sigma_t) ** 2)

    fig = go.Figure(layout=_dark_layout("Dispersive Sweep (before DM correction)"))
    fig.add_trace(go.Heatmap(
        z=intensity,
        x=t_bins * 1e6,        # microseconds
        y=freqs_mhz,
        colorscale="Plasma",
        showscale=False,
        name="Intensity",
    ))
    fig.update_layout(
        xaxis=dict(**_AXIS_STYLE, title="Time offset (μs)"),
        yaxis=dict(**_AXIS_STYLE, title="Frequency (MHz)"),
        height=280,
    )
    fig.update_traces(showscale=True,
                      colorbar=dict(title="Intensity", thickness=8,
                                    tickfont=dict(size=9, color="#888")))
    return fig


# ── Timing residual breakdown figure ─────────────────────────────────────────

def build_residual_breakdown_figure(data: dict) -> go.Figure:
    """Horizontal bar chart showing each timing contribution.

    Bars ordered from largest to smallest (Roemer dominates at kpc scales).
    Includes DM turbulence residual as a separate bar per Physics Inspector
    recommendation (ISSUE 6): "Dispersive delay" is the corrected mean DM
    delay; the turbulence residual is the unmodelled component.
    """
    pulsars = data.get("pulsars", [])
    selected_name = data.get("selected_pulsar", "")
    pulsar = next((p for p in pulsars if p["name"] == selected_name), None)

    if pulsar is None:
        fig = go.Figure(layout=_dark_layout("Timing Residual Breakdown"))
        fig.update_layout(height=260)
        return fig

    roemer_ns = abs(pulsar.get("roemer_s", 1e-3)) * 1e9
    dispersive_ns = abs(pulsar.get("dispersive_s", 1e-5)) * 1e9
    dm_turb_ns = abs(pulsar.get("dm_turbulence_s", 1e-5)) * 1e9
    timing_noise_ns = abs(pulsar.get("timing_noise_ns", 100.0))
    photon_noise_ns = abs(pulsar.get("photon_noise_s", 1e-6)) * 1e9

    # APPROXIMATION: Doppler and Shapiro delays are sub-dominant at kpc distances.
    # Doppler ~ v/c × T_roemer ~ 10⁻⁴ × Roemer; Shapiro ~ GM/(c²r) ~ 10⁻⁶ s.
    # Including them as fixed fractions of Roemer and dispersive delay
    # gives correct order-of-magnitude for display without a full timing model.
    # Error: < 0.1% of the Roemer term for speeds below 500 km/s.
    doppler_ns = roemer_ns * 1e-4
    shapiro_ns = max(dispersive_ns * 1e-3, 1.0)

    labels = [
        "Roemer (geometric)",
        "Corrected dispersive delay",
        "DM turbulence residual",
        "Photon counting noise",
        "Timing noise floor",
        "Doppler (velocity)",
        "Shapiro (gravitational lensing)",
    ]
    values_ns = [
        roemer_ns, dispersive_ns, dm_turb_ns,
        photon_noise_ns, timing_noise_ns,
        doppler_ns, shapiro_ns,
    ]
    colours = [
        _ACCENT, "#4488FF", "#FF8800",
        "#FFCC00", "#FF4444",
        "#88FFAA", "#CC88FF",
    ]

    # Sort largest first
    order = np.argsort(values_ns)[::-1]
    labels = [labels[i] for i in order]
    values_ns = [values_ns[i] for i in order]
    colours = [colours[i] for i in order]

    fig = go.Figure(layout=_dark_layout("Timing Residual Breakdown"))
    fig.add_trace(go.Bar(
        y=labels,
        x=values_ns,
        orientation="h",
        marker=dict(color=colours, opacity=0.85),
        text=[f"{v:.1e} ns" for v in values_ns],
        textposition="outside",
        textfont=dict(size=9, color="#AAAACC"),
    ))
    fig.update_layout(
        xaxis=dict(**_AXIS_STYLE, title="Magnitude (ns)", type="log"),
        yaxis=dict(showgrid=False, color="#AAAACC"),
        height=260,
        showlegend=False,
    )
    return fig


# ── DM residual figure ────────────────────────────────────────────────────────

def build_dm_residual_figure(data: dict) -> go.Figure:
    """Multi-pulsar DM residual: observed minus expected DM, sorted by GL."""
    pulsars = data.get("pulsars", [])
    dm_residuals = data.get("dm_residuals", {})

    if not pulsars or not dm_residuals:
        fig = go.Figure(layout=_dark_layout("Multi-Pulsar DM Residuals"))
        fig.add_annotation(
            text="DM residuals appear after the first simulation iteration",
            showarrow=False, font=dict(color="#888", size=12),
            xref="paper", yref="paper", x=0.5, y=0.5,
        )
        fig.update_layout(height=200)
        return fig

    # Sort by GL
    sorted_p = sorted(pulsars, key=lambda p: p["gl"])
    names = [p["name"] for p in sorted_p]
    residuals = [dm_residuals.get(p["name"], 0.0) for p in sorted_p]

    # ISM turbulence uncertainty envelope: ±15% of catalogue DM
    upper = [abs(p.get("dm", 30.0)) * 0.15 for p in sorted_p]
    lower = [-u for u in upper]

    fig = go.Figure(layout=_dark_layout("Multi-Pulsar DM Residuals (obs − expected)"))

    # Uncertainty envelope
    x_fill = list(range(len(names)))
    fig.add_trace(go.Scatter(
        x=x_fill + x_fill[::-1],
        y=upper + lower[::-1],
        fill="toself",
        fillcolor="rgba(255,136,0,0.15)",
        line=dict(color="rgba(0,0,0,0)"),
        hoverinfo="skip",
        showlegend=True,
        name="±15% ISM uncertainty",
    ))

    # Residuals
    fig.add_trace(go.Scatter(
        x=x_fill, y=residuals,
        mode="markers",
        marker=dict(size=7, color=_ACCENT),
        text=names,
        hovertemplate="<b>%{text}</b><br>ΔDM=%{y:.3f} pc cm⁻³<extra></extra>",
        name="DM residual",
    ))

    # Zero line
    fig.add_hline(y=0, line=dict(color="#888", dash="dash", width=1))

    fig.update_layout(
        xaxis=dict(
            ticktext=names, tickvals=x_fill,
            tickangle=45, tickfont=dict(size=8, color="#888"),
            showgrid=False, color="#888",
            title="Pulsar (sorted by GL)",
        ),
        yaxis=dict(**_AXIS_STYLE, title="ΔDM (pc cm⁻³)"),
        height=200,
    )
    return fig


# ── Streamlit render ──────────────────────────────────────────────────────────

def render(data: dict) -> None:
    """Render the full timing panel in Streamlit."""
    pulsars = data.get("pulsars", [])
    if not pulsars:
        st.markdown(
            '<div style="background:#111128; border:1px solid #1A1A3A; border-radius:6px; '
            'padding:24px; color:#AAAACC; text-align:center;">'
            'Run the simulation to load pulsar timing data.</div>',
            unsafe_allow_html=True,
        )
        return

    pulsar_names = [p["name"] for p in pulsars]
    # Prefer persisted session_state selection over passed-in default
    persisted = st.session_state.get("timing_pulsar_selector")
    default_name = (
        persisted if persisted in pulsar_names
        else (data.get("selected_pulsar") or pulsar_names[0])
    )
    if default_name not in pulsar_names:
        default_name = pulsar_names[0]

    selected = st.selectbox(
        "Select pulsar to inspect",
        pulsar_names,
        index=pulsar_names.index(default_name),
        key="timing_pulsar_selector",
    )
    st.caption("Dispersive sweep and timing breakdown for the selected pulsar")
    data = dict(data, selected_pulsar=selected)

    fig_sweep = build_dispersive_sweep_figure(data)
    st.plotly_chart(fig_sweep, width="stretch", config={"displayModeBar": False})

    fig_breakdown = build_residual_breakdown_figure(data)
    st.plotly_chart(fig_breakdown, width="stretch", config={"displayModeBar": False})

    fig_dm = build_dm_residual_figure(data)
    st.plotly_chart(fig_dm, width="stretch", config={"displayModeBar": False})
