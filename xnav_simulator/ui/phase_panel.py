# ui/phase_panel.py — Phase ambiguity resolution panel for XNAV Cold Start Simulator
# UI boundary: receives plain data only. No core/ or stages/ imports permitted.

"""
Two sub-panels:
  Left  — Phase dial array (one dial per active pulsar, up to 12)
  Right — Ambiguity window timeline (log-scale collapse as pulsars added)

Data dict keys expected:
    pulsars:         list of dicts {name, period, identified: bool}
    window_history:  list of dicts {n_pulsars: int, window_s: float}
                     from stage4_phase_ambiguity output
    clock_candidates: list of floats — surviving clock offset candidates (seconds)
    clock_estimate_s: float — best clock offset estimate
    stage4_complete:  bool
"""

from __future__ import annotations

import math

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

import config


_ACCENT = config.COLOUR_ACCENT
_BG = config.COLOUR_BG
_GRID = config.COLOUR_GRID
_PLOT_BG = "#0D0D22"


# ── Phase dial (one dial = one pulsar) ───────────────────────────────────────

def _single_dial_trace(
    phase_obs: float,
    candidates: list[float],
    period: float,
    resolved: bool,
    cx: float, cy: float,
    r: float = 0.35,
) -> list[go.BaseTraceType]:
    """Return a list of traces representing one phase dial.

    The dial is drawn in normalised figure coordinates centred at (cx, cy).
    phase_obs is in [0,1]; candidates are phases of surviving clock offsets.
    """
    traces = []
    # Dial background circle
    theta = np.linspace(0, 2 * np.pi, 120)
    traces.append(go.Scatter(
        x=cx + r * np.cos(theta),
        y=cy + r * np.sin(theta),
        mode="lines",
        line=dict(color="#334", width=1),
        fill="toself",
        fillcolor="rgba(20,20,40,0.8)",
        hoverinfo="skip",
        showlegend=False,
    ))
    # Candidate arcs (amber if unresolved, green if resolved)
    arc_col = "#00CC66" if resolved else "#FFAA00"
    for cand in candidates[:16]:   # max 16 arcs for legibility
        phi = cand % (2 * np.pi)
        traces.append(go.Scatter(
            x=[cx, cx + r * 0.9 * math.cos(phi)],
            y=[cy, cy + r * 0.9 * math.sin(phi)],
            mode="lines",
            line=dict(color=arc_col, width=1.5),
            hoverinfo="skip",
            showlegend=False,
        ))
    # Observed phase line
    phi_obs = phase_obs * 2 * np.pi
    traces.append(go.Scatter(
        x=[cx, cx + r * math.cos(phi_obs)],
        y=[cy, cy + r * math.sin(phi_obs)],
        mode="lines",
        line=dict(color=_ACCENT, width=2.5),
        hoverinfo="skip",
        showlegend=False,
    ))
    return traces


def build_phase_dials_figure(data: dict) -> go.Figure:
    """Build a grid of phase dials — one per identified pulsar (up to 12)."""
    pulsars = data.get("pulsars", [])
    clock_candidates = data.get("clock_candidates", [0.0])
    clock_estimate_s = data.get("clock_estimate_s", 0.0)
    stage4_complete = data.get("stage4_complete", False)

    identified = [p for p in pulsars if p.get("identified", False)][:12]
    n = len(identified)
    if n == 0:
        fig = go.Figure(layout=go.Layout(
            paper_bgcolor=_BG, plot_bgcolor=_PLOT_BG,
            font=dict(color="#AAAACC", size=11), height=300,
            title=dict(text="Phase Dials", font=dict(color=_ACCENT, size=12), x=0.5),
        ))
        fig.add_annotation(text="Waiting for Stage 4…", showarrow=False,
                           font=dict(color="#888"), xref="paper", yref="paper",
                           x=0.5, y=0.5)
        return fig

    n_cols = min(4, n)
    n_rows = math.ceil(n / n_cols)
    cell_w = 1.0 / n_cols
    cell_h = 1.0 / n_rows

    fig = go.Figure()
    for idx, p in enumerate(identified):
        row = idx // n_cols
        col_idx = idx % n_cols
        cx = (col_idx + 0.5) * cell_w
        cy = 1.0 - (row + 0.5) * cell_h

        period = p.get("period", 0.005)
        # Observed phase from clock estimate
        phase_obs = (clock_estimate_s % period) / period if period > 0 else 0.0
        # Candidate phases
        cand_phases = [(c % period) / period * 2 * np.pi for c in clock_candidates[:16]]

        for trace in _single_dial_trace(
            phase_obs, cand_phases, period,
            resolved=stage4_complete,
            cx=cx, cy=cy, r=0.42 * min(cell_w, cell_h),
        ):
            fig.add_trace(trace)

        fig.add_annotation(
            x=cx, y=cy - 0.44 * min(cell_w, cell_h),
            text=p["name"][:8],
            showarrow=False,
            font=dict(size=9, color="#AAAACC"),
            xref="paper", yref="paper",
        )

    fig.update_layout(
        paper_bgcolor=_BG,
        plot_bgcolor=_PLOT_BG,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, 1]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   range=[0, 1], scaleanchor="x"),
        height=max(220, n_rows * 150),
        margin=dict(l=10, r=10, t=40, b=10),
        title=dict(text="Phase Dials", font=dict(color=_ACCENT, size=12), x=0.5),
        showlegend=False,
    )
    return fig


# ── Ambiguity window timeline ─────────────────────────────────────────────────

def build_ambiguity_timeline_figure(data: dict) -> go.Figure:
    """Log-scale plot of ambiguity window size vs number of pulsars used."""
    window_history = data.get("window_history", [])

    fig = go.Figure(layout=go.Layout(
        paper_bgcolor=_BG, plot_bgcolor=_PLOT_BG,
        font=dict(color="#AAAACC", size=11),
        title=dict(
            text="Clock Ambiguity Window vs Pulsars Added",
            font=dict(color=_ACCENT, size=12), x=0.5,
        ),
        height=300,
    ))

    if not window_history:
        fig.add_annotation(text="Run Stage 4 to see ambiguity collapse",
                           showarrow=False, font=dict(color="#888"),
                           xref="paper", yref="paper", x=0.5, y=0.5)
        return fig

    n_pulsars_list = [h["n_pulsars"] for h in window_history]
    windows_s = [max(h["window_s"], 1e-9) for h in window_history]

    fig.add_trace(go.Scatter(
        x=n_pulsars_list, y=windows_s,
        mode="lines+markers",
        line=dict(color=_ACCENT, width=2),
        marker=dict(size=7, color=_ACCENT),
        name="Ambiguity window",
        hovertemplate="Pulsars: %{x}<br>Window: %{y:.2e} s<extra></extra>",
    ))

    # Reference lines
    for threshold, label in [
        (1.0, "1 s"), (1e-3, "1 ms"), (1e-6, "1 μs"),
        (1e-9, "1 ns"),
    ]:
        if min(windows_s) <= threshold <= max(windows_s) * 2:
            fig.add_hline(
                y=threshold,
                line=dict(color="#446644", dash="dash", width=1),
                annotation_text=label,
                annotation_font=dict(size=9, color="#446644"),
                annotation_position="right",
            )

    # "Required for 1 AU accuracy" reference
    # APPROXIMATION: 1 AU accuracy ↔ Roemer timing error ≤ AU/c ≈ 500 s.
    # Required clock precision: period × (1 AU / 2πr_pulsar) ≈ 0.5 ms for typical MSP.
    fig.add_hline(
        y=5e-4,
        line=dict(color="#AA4400", dash="dot", width=1.5),
        annotation_text="Required for 1 AU accuracy",
        annotation_font=dict(size=9, color="#AA4400"),
    )

    fig.update_layout(
        xaxis=dict(
            title="Number of pulsars included",
            showgrid=True, gridcolor=_GRID, zeroline=False, color="#888",
            tick0=1, dtick=1,
        ),
        yaxis=dict(
            title="Ambiguity window (s)",
            type="log",
            showgrid=True, gridcolor=_GRID, zeroline=False, color="#888",
        ),
        legend=dict(bgcolor="rgba(0,0,0,0.5)", font=dict(size=10)),
    )
    return fig


# ── Streamlit render ──────────────────────────────────────────────────────────

def render(data: dict) -> None:
    """Render the phase panel: dials left, timeline right."""
    iteration = data.get("iteration", 0)
    if iteration < 6 and not data.get("stage4_complete"):
        st.markdown(
            f'<div style="background:#111128; border:1px solid #1A1A3A; border-radius:6px; '
            f'padding:24px; text-align:center; color:#AAAACC;">'
            f'Phase resolution runs at iteration 6. '
            f'Current iteration: <b style="color:#00D4FF;">{iteration}</b> of ~20.'
            f'</div>',
            unsafe_allow_html=True,
        )
        return

    col1, col2 = st.columns([1, 1])
    with col1:
        fig_dials = build_phase_dials_figure(data)
        st.plotly_chart(fig_dials, width="stretch", config={"displayModeBar": False})
    with col2:
        fig_timeline = build_ambiguity_timeline_figure(data)
        st.plotly_chart(fig_timeline, width="stretch", config={"displayModeBar": False})

    # Clock estimate summary
    if data.get("stage4_complete"):
        t_est = data.get("clock_estimate_s", 0.0)
        n_cand = len(data.get("clock_candidates", []))
        wh = data.get("window_history", [])
        final_window_ms = (wh[-1]["window_s"] * 1e3) if wh else float("nan")
        st.markdown(
            f'<div style="color:{_ACCENT}; font-size:0.85em; margin-top:0.5em;">'
            f'Clock offset estimate: <b>{t_est:.6f} s</b> | '
            f'Remaining candidates: <b>{n_cand}</b> | '
            f'Final window: <b>{final_window_ms:.3f} ms</b>'
            f'</div>',
            unsafe_allow_html=True,
        )
