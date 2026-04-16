# utils/plotting.py — Shared Plotly figure builders and colour scheme
# XNAV Cold Start Simulator

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Colour scheme ─────────────────────────────────────────────────────────────
# The three shared colours are the single source of truth in config.py.
# Additional UI-only colours are defined here.
from config import COLOUR_ACCENT, COLOUR_BG, COLOUR_GRID  # noqa: F401 (re-exported)

COLOUR_TEXT = "#C8D0E0"     # off-white text
COLOUR_TRUE = "#FFFFFF"     # white — true position marker (non-blind mode)
COLOUR_WARN = "#FF6B35"     # amber — warnings / ESS degraded
COLOUR_FAIL = "#FF3355"     # red — filter failure / noise floor
COLOUR_GOOD = "#00FF88"     # green — converged / healthy


_AXIS_DARK = dict(gridcolor=COLOUR_GRID, zerolinecolor=COLOUR_GRID, linecolor=COLOUR_GRID)


def _dark_layout(title: str = "", **kwargs) -> dict:
    """Return a Plotly layout dict with the simulator dark theme.

    Does NOT include xaxis/yaxis so callers can supply their own without
    Plotly raising a 'multiple values for keyword argument' error.
    Merge axis style with _AXIS_DARK at the call site if needed.
    """
    return dict(
        title=dict(text=title, font=dict(color=COLOUR_TEXT, size=14)),
        paper_bgcolor=COLOUR_BG,
        plot_bgcolor=COLOUR_BG,
        font=dict(color=COLOUR_TEXT, family="monospace"),
        margin=dict(l=50, r=20, t=40, b=40),
        **kwargs,
    )


def make_empty_galaxy_figure() -> go.Figure:
    """Return an empty stylised galactic top-down view figure (Plotly Figure).

    Draws a faint 2D Gaussian disk representing the galactic plane, with
    4 log-spiral arms overlaid as density contours.  Axes are in kpc.
    """
    from config import GALAXY_RADIUS_KPC, SOLAR_GALACTOCENTRIC_KPC

    # ── Galactic disk background (2D Gaussian density) ────────────────────
    # APPROXIMATION: Spiral arm structure represented by 4 log-spiral curves
    # offset by 90° each. Real arms have finite width, pitch angle variation,
    # and asymmetry. This is purely decorative — it conveys the geometry
    # without misleading the user about precise arm positions.
    R = GALAXY_RADIUS_KPC
    x_grid = np.linspace(-R, R, 200)
    y_grid = np.linspace(-R, R, 200)
    X, Y = np.meshgrid(x_grid, y_grid)
    r = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(Y, X)

    # Background disk density (exponential radial profile)
    disk = np.exp(-r / 4.0)

    # Spiral arms: 4 arms with pitch angle ~12°
    pitch = np.radians(12.0)
    arm_density = np.zeros_like(disk)
    for i in range(4):
        phi0 = i * np.pi / 2.0
        arm_theta = np.log(np.maximum(r, 0.5) / 2.0) / np.tan(pitch) + phi0
        delta = (theta - arm_theta) % (2 * np.pi)
        delta = np.where(delta > np.pi, delta - 2 * np.pi, delta)
        arm_density += np.exp(-delta**2 / (2 * 0.3**2)) * np.exp(-r / 6.0)

    total_density = 0.4 * disk + 0.6 * arm_density
    total_density = np.clip(total_density, 0, None)

    fig = go.Figure()

    # Background heatmap
    fig.add_trace(go.Heatmap(
        z=total_density,
        x=x_grid,
        y=y_grid,
        colorscale=[
            [0.0, COLOUR_BG],
            [0.3, "#050520"],
            [0.6, "#0A1040"],
            [1.0, "#182060"],
        ],
        showscale=False,
        hoverinfo="skip",
        name="Galactic disk",
    ))

    # Sun marker
    fig.add_trace(go.Scatter(
        x=[-SOLAR_GALACTOCENTRIC_KPC],
        y=[0],
        mode="markers+text",
        marker=dict(symbol="star", size=12, color="#FFD700"),
        text=["Sun"],
        textposition="top center",
        textfont=dict(color="#FFD700", size=10),
        name="Sun",
        hoverinfo="name",
    ))

    fig.update_layout(
        **_dark_layout("Galactic Map"),
        xaxis=dict(
            title="X (kpc, Galactocentric)",
            range=[-R, R],
            gridcolor=COLOUR_GRID,
            zerolinecolor=COLOUR_GRID,
            scaleanchor="y",
        ),
        yaxis=dict(
            title="Y (kpc, Galactocentric)",
            range=[-R, R],
            gridcolor=COLOUR_GRID,
            zerolinecolor=COLOUR_GRID,
        ),
        showlegend=True,
        legend=dict(
            bgcolor="rgba(10,10,26,0.8)",
            bordercolor=COLOUR_GRID,
            borderwidth=1,
        ),
    )

    return fig


def make_empty_skymap_figure() -> go.Figure:
    """Return an empty Mollweide-projection sky map (Plotly Figure)."""
    fig = go.Figure()
    fig.update_layout(
        **_dark_layout("Pulsar Sky Map (Galactic Coordinates)"),
        xaxis=dict(
            title="Galactic Longitude (°)",
            range=[360, 0],  # standard astronomical convention
            gridcolor=COLOUR_GRID,
            zerolinecolor=COLOUR_GRID,
        ),
        yaxis=dict(
            title="Galactic Latitude (°)",
            range=[-90, 90],
            gridcolor=COLOUR_GRID,
            zerolinecolor=COLOUR_GRID,
        ),
    )
    return fig


def make_convergence_timeline_figure() -> go.Figure:
    """Return an empty convergence timeline (uncertainty vs iteration)."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.update_layout(
        **_dark_layout("Position Uncertainty vs Iteration"),
        xaxis=dict(title="Iteration", gridcolor=COLOUR_GRID, zerolinecolor=COLOUR_GRID),
        yaxis=dict(title="Uncertainty radius (kpc)", type="log", gridcolor=COLOUR_GRID),
    )
    fig.update_yaxes(
        title_text="ESS / N_particles",
        secondary_y=True,
        range=[0, 1.05],
        gridcolor="rgba(0,0,0,0)",
    )
    return fig


def make_bar_figure(
    labels: list[str],
    values: list[float],
    colours: list[str],
    title: str = "",
    x_label: str = "",
    log_scale: bool = False,
) -> go.Figure:
    """Return a horizontal bar chart with the dark theme."""
    fig = go.Figure(go.Bar(
        y=labels,
        x=values,
        orientation="h",
        marker=dict(color=colours),
        text=[f"{v:.2g}" for v in values],
        textposition="outside",
        textfont=dict(color=COLOUR_TEXT),
    ))
    fig.update_layout(
        **_dark_layout(title),
        xaxis=dict(
            title=x_label,
            type="log" if log_scale else "linear",
            gridcolor=COLOUR_GRID,
            zerolinecolor=COLOUR_GRID,
        ),
        yaxis=dict(gridcolor=COLOUR_GRID, zerolinecolor=COLOUR_GRID),
        showlegend=False,
    )
    return fig
