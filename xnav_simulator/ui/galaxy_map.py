# ui/galaxy_map.py — Galaxy map panels for XNAV Cold Start Simulator
# UI boundary: receives plain data only. No core/ or stages/ imports permitted.

"""
Two-panel galaxy map:
  Left  — Top-down galactic disk view (Plotly scatter over a procedurally
          rendered galaxy backdrop; pinch/scroll zoomable, tap a body for
          its astrophysical data card)
  Right — Pulsar sky map in galactic coordinates

Data dict keys expected:
    pulsars:          list of dicts {name, gl, gb, distance_kpc, dm, period,
                                    timing_noise_ns, identified, confidence,
                                    x_kpc, y_kpc, z_kpc, age_yr, b_surf_g,
                                    edot_erg_s, type, s1400_mjy, w50}
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

from functools import lru_cache

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
    gridcolor="rgba(26,26,58,0.55)",
    zeroline=False,
    color="#888",
    tickcolor="#888",
)
_PAPER_BG = _BG
_PLOT_BG = "#06060F"

_HOVERLABEL = dict(
    bgcolor="rgba(13,13,34,0.95)",
    bordercolor=_ACCENT,
    font=dict(color="#DDE6FF", size=12, family="monospace"),
    align="left",
)

_MAP_EXTENT_KPC = 16.0


def _fmt_kpc(value: float) -> str:
    """Format a kpc quantity, switching to parsecs below 0.1 kpc."""
    if value < 0.1:
        return f"{value * 1000:.0f} pc"
    return f"{value:.2f} kpc"


def _dark_layout(title: str = "", **kwargs) -> go.Layout:
    return go.Layout(
        title=dict(text=title, font=dict(color=_ACCENT, size=13), x=0.5),
        paper_bgcolor=_PAPER_BG,
        plot_bgcolor=_PLOT_BG,
        font=dict(color="#AAAACC", size=11),
        margin=dict(l=40, r=20, t=40, b=40),
        hoverlabel=_HOVERLABEL,
        **kwargs,
    )


# ── Procedural galaxy backdrop ────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _galaxy_backdrop():
    """Render a Milky-Way-like backdrop as a PIL image (cached per process).

    Composited from an exponential disk, a central bar + bulge, four
    logarithmic spiral arms (12° pitch, matching the navigation geometry's
    decorative approximation), and a seeded star-speckle field.  The image is
    anchored to data coordinates so it pans and zooms with the axes.

    DISPLAY-ONLY: purely decorative; does not affect navigation.
    """
    from PIL import Image

    n = 640
    ext = _MAP_EXTENT_KPC
    ax = np.linspace(-ext, ext, n)
    X, Y = np.meshgrid(ax, ax)
    R = np.sqrt(X ** 2 + Y ** 2)
    TH = np.arctan2(Y, X)

    # Smooth outer edge of the stellar disk
    edge = 1.0 / (1.0 + np.exp((R - 14.2) / 0.7))
    disk = np.exp(-R / 5.0) * edge

    # Central bulge + bar (bar angle ~25°)
    bulge = np.exp(-((R / 1.3) ** 2))
    phi = np.radians(25.0)
    Xb = X * np.cos(phi) + Y * np.sin(phi)
    Yb = -X * np.sin(phi) + Y * np.cos(phi)
    bar = np.exp(-((Xb / 3.2) ** 2 + (Yb / 1.1) ** 2))

    # Four log-spiral arms: gaussian profile in angular distance to each arm
    pitch = np.tan(np.radians(12.0))
    r_safe = np.maximum(R, 0.3)
    theta_arm = np.log(r_safe / 3.0) / pitch
    arms = np.zeros_like(R)
    for k in range(4):
        d_ang = np.angle(np.exp(1j * (TH - theta_arm - k * np.pi / 2.0)))
        arms += np.exp(-((d_ang / 0.38) ** 2))
    arms *= np.exp(-R / 8.0) * edge * (R > 1.8)

    # Compose intensity channels (gamma-lifted so the faint outer disk and
    # arms survive the dark theme instead of vanishing into the background)
    glow = np.clip(0.85 * disk + 0.80 * arms * (0.4 + 0.6 * disk), 0.0, 1.0) ** 0.65
    core = np.clip(bulge + 0.55 * bar * np.exp(-R / 6.0), 0.0, 1.0)

    # Colour: deep indigo glow, warm white core
    base = np.array([6, 6, 15], dtype=np.float64)
    rgb = np.empty((n, n, 3), dtype=np.float64)
    rgb[..., 0] = base[0] + 80 * glow + 190 * core
    rgb[..., 1] = base[1] + 95 * glow + 170 * core
    rgb[..., 2] = base[2] + 175 * glow + 140 * core

    # Star speckle field, denser inside the disk
    rng = np.random.default_rng(20260612)
    speckle = rng.random((n, n))
    p_star = 0.0012 + 0.004 * disk + 0.004 * arms
    stars = speckle < p_star
    brightness = 90 + 140 * rng.random((n, n))
    for c in range(3):
        rgb[..., c] = np.where(stars, np.minimum(rgb[..., c] + brightness, 255), rgb[..., c])

    img = np.clip(rgb, 0, 255).astype(np.uint8)
    # PIL row 0 is the TOP of the image (y=+ext); meshgrid row 0 is y=−ext.
    return Image.fromarray(np.flipud(img))


def _add_backdrop(fig: go.Figure) -> None:
    ext = _MAP_EXTENT_KPC
    fig.add_layout_image(dict(
        source=_galaxy_backdrop(),
        xref="x", yref="y",
        x=-ext, y=ext, sizex=2 * ext, sizey=2 * ext,
        sizing="stretch", layer="below", opacity=0.95,
    ))


# ── Local coordinate fallback (pure math — no core/ imports) ─────────────────

def _fallback_xy(p: dict) -> tuple[float, float]:
    """Heliocentric (gl, gb, d) → galactocentric XY for dicts without x_kpc."""
    gl_r = np.radians(p["gl"])
    gb_r = np.radians(p["gb"])
    d = p["distance_kpc"]
    sun = config.SUN_POS_KPC
    return (float(sun[0] + d * np.cos(gb_r) * np.cos(gl_r)),
            float(sun[1] + d * np.cos(gb_r) * np.sin(gl_r)))


# ── Popup data cards ──────────────────────────────────────────────────────────

def _fmt_age(age_yr: float) -> str:
    if not np.isfinite(age_yr) or age_yr <= 0:
        return "unknown"
    if age_yr >= 1e9:
        return f"{age_yr / 1e9:.1f} Gyr"
    if age_yr >= 1e6:
        return f"{age_yr / 1e6:.0f} Myr"
    return f"{age_yr:.2e} yr"


def _fmt_pow10(value: float, unit: str) -> str:
    if value <= 0 or not np.isfinite(value):
        return "unknown"
    exp = int(np.floor(np.log10(value)))
    mant = value / 10 ** exp
    return f"{mant:.1f}×10{_sup(exp)} {unit}"


_SUPERSCRIPTS = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


def _sup(n: int) -> str:
    return str(n).translate(_SUPERSCRIPTS)


def _pulsar_customdata(p: dict) -> list:
    """Columns consumed by the pulsar hovertemplate (all pre-formatted)."""
    return [
        p["name"],
        p.get("type", "Millisecond pulsar — recycled neutron star"),
        f"{p.get('period', 0.0) * 1e3:.3f}",
        _fmt_age(p.get("age_yr", float("nan"))),
        _fmt_pow10(p.get("b_surf_g", 0.0), "G"),
        _fmt_pow10(p.get("edot_erg_s", 0.0), "erg s⁻¹"),
        f"{p.get('distance_kpc', 0.0):.2f}",
        f"{p.get('dm', 0.0):.1f}",
        f"{p.get('s1400_mjy', 0.0):.1f}",
        f"{p.get('timing_noise_ns', 0.0):.0f}",
    ]


_PULSAR_HOVER = (
    "<b>PSR %{customdata[0]}</b><br>"
    "%{customdata[1]}<br>"
    "─────────────────<br>"
    "Spin period   %{customdata[2]} ms<br>"
    "Char. age     %{customdata[3]}<br>"
    "Surface B     %{customdata[4]}<br>"
    "Spin-down Ė   %{customdata[5]}<br>"
    "Distance      %{customdata[6]} kpc<br>"
    "DM            %{customdata[7]} pc cm⁻³<br>"
    "Flux 1.4 GHz  %{customdata[8]} mJy<br>"
    "Timing noise  %{customdata[9]} ns"
    "<extra></extra>"
)

_SUN_HOVER = (
    "<b>Sol</b><br>"
    "G2V main-sequence star<br>"
    "─────────────────<br>"
    "Age           4.6 Gyr<br>"
    "T_eff         5772 K<br>"
    "Mass          1.0 M☉<br>"
    "Galactocentric R  8.18 kpc"
    "<extra></extra>"
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
    _add_backdrop(fig)

    # Galactic disk boundary circle
    theta_c = np.linspace(0, 2 * np.pi, 200)
    r_disk = config.GALAXY_RADIUS_KPC
    fig.add_trace(go.Scatter(
        x=r_disk * np.cos(theta_c),
        y=r_disk * np.sin(theta_c),
        mode="lines",
        line=dict(color="rgba(70,80,140,0.35)", width=1, dash="dot"),
        hoverinfo="skip",
        showlegend=False,
        name="_disk_boundary",
    ))

    # Pulsars — glow halo + DM-coloured star core, rich data-card popup
    if pulsars:
        px, py, cdata, ident_x, ident_y = [], [], [], [], []
        pdms = []
        for p in pulsars:
            x = p.get("x_kpc")
            y = p.get("y_kpc")
            if x is None or y is None:
                x, y = _fallback_xy(p)
            px.append(x)
            py.append(y)
            pdms.append(p.get("dm", 0.0))
            cdata.append(_pulsar_customdata(p))
            if p.get("identified", False):
                ident_x.append(x)
                ident_y.append(y)

        # Soft glow halo (non-interactive)
        fig.add_trace(go.Scatter(
            x=px, y=py,
            mode="markers",
            marker=dict(size=13, color="rgba(140,200,255,0.16)", symbol="circle"),
            hoverinfo="skip",
            showlegend=False,
            name="_pulsar_halo",
        ))
        # Core sprite — 4-point star, tinted by DM
        fig.add_trace(go.Scatter(
            x=px, y=py,
            mode="markers",
            marker=dict(
                size=7,
                color=pdms,
                colorscale=[[0.0, "#9fc8ff"], [0.5, "#e8f4ff"], [1.0, "#fff7d6"]],
                showscale=False,
                symbol="star-diamond",
                line=dict(width=0.6, color="rgba(200,230,255,0.85)"),
                opacity=0.95,
            ),
            customdata=cdata,
            hovertemplate=_PULSAR_HOVER,
            showlegend=False,
            name="Pulsars",
        ))
        # Identification rings
        if ident_x:
            fig.add_trace(go.Scatter(
                x=ident_x, y=ident_y,
                mode="markers",
                marker=dict(size=15, color="rgba(0,0,0,0)", symbol="circle-open",
                            line=dict(width=1.4, color=_ACCENT)),
                hoverinfo="skip",
                showlegend=True,
                name="Identified pulsar",
            ))

    # Particle cloud (subsampled to top-500 by weight for rendering performance)
    particle_pos = data.get("particle_pos")
    particle_weights = data.get("particle_weights")
    if particle_pos is not None and len(particle_pos) > 1:
        part = np.asarray(particle_pos)
        wts = np.asarray(particle_weights) if particle_weights is not None else np.ones(len(part))
        n_show = min(500, len(part))
        top_idx = np.argsort(wts)[-n_show:]
        fig.add_trace(go.Scatter(
            x=part[top_idx, 0], y=part[top_idx, 1],
            mode="markers",
            marker=dict(size=3, color="rgba(0,212,255,0.22)"),
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
        _finalise_topdown_layout(fig)
        return fig

    # True position — only in non-blind mode; rendered before estimate so estimate is on top
    if not blind_mode and true_pos is not None:
        tp = np.asarray(true_pos)
        fig.add_trace(go.Scatter(
            x=[tp[0]], y=[tp[1]],
            mode="markers",
            marker=dict(size=17, color="rgba(0,0,0,0)", symbol="circle-open",
                        line=dict(width=1.5, color="#FF5555")),
            hoverinfo="skip",
            showlegend=False,
            name="_true_ring",
        ))
        fig.add_trace(go.Scatter(
            x=[tp[0]], y=[tp[1]],
            mode="markers",
            marker=dict(size=11, color="#FF5555", symbol="cross-thin",
                        line=dict(width=1.6, color="#FF5555")),
            name="True position",
            showlegend=True,
            hovertemplate=("<b>True position</b><br>"
                           f"({tp[0]:.2f}, {tp[1]:.2f}) kpc<extra></extra>"),
        ))

    # Uncertainty circle (labelled for legend)
    fig.add_trace(go.Scatter(
        x=sc_pos[0] + uncertainty_kpc * np.cos(theta_u),
        y=sc_pos[1] + uncertainty_kpc * np.sin(theta_u),
        mode="lines",
        line=dict(color="rgba(0,212,255,0.35)", width=1, dash="dash"),
        hoverinfo="skip",
        showlegend=True,
        name=f"Uncertainty (1σ = {_fmt_kpc(uncertainty_kpc)})",
    ))

    # Spacecraft estimated position — glow + star, added last so it renders on top
    fig.add_trace(go.Scatter(
        x=[sc_pos[0]], y=[sc_pos[1]],
        mode="markers",
        marker=dict(size=26, color="rgba(0,212,255,0.20)", symbol="circle"),
        hoverinfo="skip",
        showlegend=False,
        name="_estimate_glow",
    ))
    fig.add_trace(go.Scatter(
        x=[sc_pos[0]], y=[sc_pos[1]],
        mode="markers",
        marker=dict(size=15, color=_ACCENT, symbol="star",
                    line=dict(width=1, color="white")),
        name="Estimate",
        hovertemplate=("<b>Spacecraft estimate</b><br>"
                       f"({sc_pos[0]:.2f}, {sc_pos[1]:.2f}) kpc<br>"
                       f"1σ uncertainty {_fmt_kpc(uncertainty_kpc)}<extra></extra>"),
    ))

    # Sun reference — warm glow + core with stellar data card
    fig.add_trace(go.Scatter(
        x=[sun_pos[0]], y=[sun_pos[1]],
        mode="markers",
        marker=dict(size=20, color="rgba(255,215,0,0.22)", symbol="circle"),
        hoverinfo="skip",
        showlegend=False,
        name="_sun_glow",
    ))
    fig.add_trace(go.Scatter(
        x=[sun_pos[0]], y=[sun_pos[1]],
        mode="markers",
        marker=dict(size=9, color="#FFD700", symbol="circle",
                    line=dict(width=1, color="#FFF2B0")),
        name="Sun",
        hovertemplate=_SUN_HOVER,
    ))

    _finalise_topdown_layout(fig)
    return fig


def _finalise_topdown_layout(fig: go.Figure) -> None:
    ext = _MAP_EXTENT_KPC
    fig.update_layout(
        xaxis=dict(**_AXIS_STYLE, title="X (kpc)", range=[-ext, ext]),
        yaxis=dict(**_AXIS_STYLE, title="Y (kpc)", range=[-ext, ext],
                   scaleanchor="x", scaleratio=1),
        # Horizontal legend below the plot — inside the axes it covers the
        # galaxy on phone-width viewports.
        legend=dict(bgcolor="rgba(0,0,0,0.5)", orientation="h",
                    x=0.5, xanchor="center", y=-0.18, font=dict(size=10)),
        height=460,
        dragmode="pan",
        # Preserve the user's zoom/pan across Streamlit reruns (e.g. while
        # the filter iterates) — without this every rerun resets the view.
        uirevision="galaxy-topdown",
    )


def build_skymap_figure(data: dict) -> go.Figure:
    """Build the pulsar sky map in galactic coordinates (Mollweide approximation).

    APPROXIMATION: Plotly does not support true Mollweide projection natively.
    We use a flat equirectangular plot with GL on the x-axis and GB on y-axis
    and label it as such. The visual layout matches a standard galactic sky map.
    """
    pulsars = data.get("pulsars", [])

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

    gls = np.array([p["gl"] for p in pulsars])
    gbs = np.array([p["gb"] for p in pulsars])
    scores = np.array([p.get("timing_noise_ns", 100.0) for p in pulsars])
    identified = np.array([p.get("identified", False) for p in pulsars], dtype=bool)
    cdata = [_pulsar_customdata(p) for p in pulsars]

    # Faint galactic-plane band for context
    fig.add_shape(type="rect", x0=0, x1=360, y0=-10, y1=10,
                  fillcolor="rgba(70,90,180,0.10)", line=dict(width=0),
                  layer="below")

    # Unidentified pulsars
    mask_un = ~identified
    if mask_un.any():
        fig.add_trace(go.Scatter(
            x=gls[mask_un], y=gbs[mask_un],
            mode="markers",
            marker=dict(
                size=7,
                color=scores[mask_un],
                colorscale="RdYlGn_r",
                cmin=1, cmax=500,
                symbol="star-diamond",
                opacity=0.85,
                line=dict(width=0.5, color="rgba(255,255,255,0.4)"),
            ),
            customdata=[c for c, m in zip(cdata, mask_un) if m],
            hovertemplate=_PULSAR_HOVER,
            showlegend=True,
            name="Unidentified",
        ))

    # Identified pulsars (larger, highlighted)
    if identified.any():
        fig.add_trace(go.Scatter(
            x=gls[identified], y=gbs[identified],
            mode="markers",
            marker=dict(
                size=11,
                color=_ACCENT,
                symbol="circle-open",
                line=dict(width=2, color=_ACCENT),
            ),
            customdata=[c for c, m in zip(cdata, identified) if m],
            hovertemplate=_PULSAR_HOVER,
            showlegend=True,
            name="Identified",
        ))

    fig.update_layout(
        xaxis=dict(**_AXIS_STYLE, title="Galactic Longitude GL (°)", range=[0, 360]),
        yaxis=dict(**_AXIS_STYLE, title="Galactic Latitude GB (°)", range=[-90, 90]),
        legend=dict(bgcolor="rgba(0,0,0,0.5)", orientation="h",
                    x=0.5, xanchor="center", y=-0.22, font=dict(size=10)),
        height=420,
        uirevision="galaxy-skymap",
    )
    return fig


# ── Streamlit render ──────────────────────────────────────────────────────────

def render(data: dict) -> None:
    """Render both galaxy map panels side by side in Streamlit."""
    col1, col2 = st.columns(2)
    with col1:
        fig = build_topdown_figure(data)
        st.plotly_chart(fig, width="stretch",
                        config={"displayModeBar": False, "scrollZoom": True})
        st.caption("Pinch or scroll to zoom · drag to pan · tap any body for its data card · double-tap to reset")
    with col2:
        fig = build_skymap_figure(data)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
