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

def _smooth_noise(rng: np.random.Generator, n: int, sigma_px: float) -> np.ndarray:
    """Gaussian-smoothed white noise in [0, 1] via FFT low-pass (numpy only)."""
    noise = rng.random((n, n))
    fy = np.fft.fftfreq(n)[:, None]
    fx = np.fft.rfftfreq(n)[None, :]
    lowpass = np.exp(-2.0 * (np.pi * sigma_px) ** 2 * (fx ** 2 + fy ** 2))
    smooth = np.fft.irfft2(np.fft.rfft2(noise) * lowpass, s=(n, n))
    lo, hi = smooth.min(), smooth.max()
    return (smooth - lo) / max(hi - lo, 1e-12)


@lru_cache(maxsize=1)
def _galaxy_backdrop():
    """Render a photographic-style Milky Way backdrop (cached per process).

    Silvery grey-blue palette: exponential disk, central bar + bulge, four
    logarithmic spiral arms (12° pitch) broken up by two octaves of smoothed
    noise so the arms read as patchy star clouds rather than drawn curves,
    plus a dense magnitude-varying star speckle field and gaussian blooms for
    the brightest field stars.  Anchored to data coordinates so it pans and
    zooms with the axes.

    DISPLAY-ONLY: purely decorative; does not affect navigation.
    """
    from PIL import Image

    n = 768
    ext = _MAP_EXTENT_KPC
    ax = np.linspace(-ext, ext, n)
    X, Y = np.meshgrid(ax, ax)
    R = np.sqrt(X ** 2 + Y ** 2)
    TH = np.arctan2(Y, X)
    rng = np.random.default_rng(20260612)

    # Smooth outer edge of the stellar disk
    edge = 1.0 / (1.0 + np.exp((R - 14.0) / 0.9))
    disk = np.exp(-R / 4.6) * edge

    # Central bulge + bar (bar angle ~25°)
    bulge = np.exp(-((R / 1.5) ** 2))
    phi = np.radians(25.0)
    Xb = X * np.cos(phi) + Y * np.sin(phi)
    Yb = -X * np.sin(phi) + Y * np.cos(phi)
    bar = np.exp(-((Xb / 3.0) ** 2 + (Yb / 1.1) ** 2))

    # Four log-spiral arms: gaussian profile in angular distance to each arm
    pitch = np.tan(np.radians(12.0))
    r_safe = np.maximum(R, 0.3)
    theta_arm = np.log(r_safe / 3.0) / pitch
    arms = np.zeros_like(R)
    for k in range(4):
        d_ang = np.angle(np.exp(1j * (TH - theta_arm - k * np.pi / 2.0)))
        arms += np.exp(-((d_ang / 0.42) ** 2))
    arms *= np.exp(-R / 8.5) * edge * (R > 1.6)

    # Patchiness: two octaves of smoothed noise turn the clean spiral curves
    # into clumpy star clouds (the photographic look)
    clumps = 0.45 + 0.85 * _smooth_noise(rng, n, 7.0)
    grain = 0.70 + 0.45 * _smooth_noise(rng, n, 2.0)
    arms_textured = arms * clumps * grain
    disk_textured = disk * (0.65 + 0.45 * _smooth_noise(rng, n, 11.0))

    # Compose intensity (gamma-lifted so faint structure survives dark theme)
    glow = np.clip(0.62 * disk_textured
                   + 0.85 * arms_textured * (0.35 + 0.65 * disk), 0.0, 1.0) ** 0.62
    core = np.clip(bulge + 0.50 * bar * np.exp(-R / 6.0), 0.0, 1.0)

    # Silvery grey with a faint blue cast; near-white core
    base = np.array([5, 5, 10], dtype=np.float64)
    rgb = np.empty((n, n, 3), dtype=np.float64)
    rgb[..., 0] = base[0] + 132 * glow + 205 * core
    rgb[..., 1] = base[1] + 138 * glow + 205 * core
    rgb[..., 2] = base[2] + 158 * glow + 200 * core

    # Dense star speckle, magnitude-varying, concentrated in disk and arms
    speckle = rng.random((n, n))
    p_star = 0.0025 + 0.010 * disk + 0.012 * arms_textured
    star_mask = speckle < p_star
    brightness = (40 + 215 * rng.random((n, n)) ** 2.2)
    for c in range(3):
        rgb[..., c] = np.where(star_mask,
                               np.minimum(rgb[..., c] + brightness, 255),
                               rgb[..., c])

    # Gaussian blooms for ~220 bright field stars (2-3 px halos)
    weights = (disk + 0.8 * arms).ravel()
    weights /= weights.sum()
    idx = rng.choice(n * n, size=220, replace=False, p=weights)
    iy, ix = np.unravel_index(idx, (n, n))
    stamp = np.exp(-0.5 * (np.arange(-3, 4)[:, None] ** 2
                           + np.arange(-3, 4)[None, :] ** 2) / 1.1 ** 2)
    for y0, x0 in zip(iy, ix):
        if 3 <= y0 < n - 3 and 3 <= x0 < n - 3:
            amp = 120 + 135 * rng.random()
            patch = rgb[y0 - 3:y0 + 4, x0 - 3:x0 + 4, :]
            patch += (amp * stamp)[..., None]
    np.clip(rgb, 0, 255, out=rgb)

    img = rgb.astype(np.uint8)
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

_STAR_HOVER = (
    "<b>%{customdata[0]}</b><br>"
    "%{customdata[1]}<br>"
    "─────────────────<br>"
    "T_eff         %{customdata[2]}<br>"
    "Distance      %{customdata[3]}<br>"
    "Luminosity    %{customdata[4]}<br>"
    "App. mag      %{customdata[5]}<br>"
    "Abs. mag      %{customdata[6]}<br>"
    "Age           not catalogued"
    "<extra></extra>"
)

_SPECTRAL_DESC = {
    "O": "O-type — blue, very hot, massive",
    "B": "B-type — blue-white, hot",
    "A": "A-type — white",
    "F": "F-type — yellow-white",
    "G": "G-type — yellow (Sun-like)",
    "K": "K-type — orange",
    "M": "M-type — red, cool",
    "L": "L-type — brown dwarf",
    "W": "Wolf-Rayet — evolved, very hot",
    "C": "Carbon star — cool giant",
    "S": "S-type — cool giant",
    "D": "White dwarf — stellar remnant",
}


def _star_customdata(s: dict) -> list:
    """Columns for the catalogued-star hovertemplate (pre-formatted)."""
    spect = s.get("spect") or ""
    desc = _SPECTRAL_DESC.get(spect[:1].upper(), "spectral type unknown")
    if spect:
        desc = f"{spect} · {desc.split('—')[-1].strip()}"
    teff = s.get("teff_k")
    lum = s.get("lum")
    d_pc = s.get("d_pc", 0.0)
    return [
        s.get("name", "?"),
        desc,
        f"{teff:,} K" if teff else "unknown",
        f"{d_pc:.1f} pc ({d_pc * 3.262:.0f} ly)",
        f"{lum:,.1f} L☉" if lum else "unknown",
        f"{s.get('mag', 0.0):+.2f}",
        f"{s.get('absmag', 0.0):+.2f}",
    ]


# ── Object (globular / nebula / black hole) styling and popups ────────────────

_OBJECT_STYLE = {
    # kind: (colour, marker size, symbol, legend label)
    "blackhole": ("#FF4D6D", 9, "circle", "Black hole"),
    "globular":  ("#FFE08A", 6, "circle", "Globular cluster"),
    "nebula":    ("#7DF5C0", 8, "circle", "Nebula"),
    "snr":       ("#C792EA", 8, "circle", "Supernova remnant"),
    "cloud":     ("#6FB3FF", 8, "circle", "Molecular cloud"),
    "cluster":   ("#BFE3FF", 6, "circle", "Open cluster"),
    "exotic":    ("#FFB347", 8, "circle", "Exotic star"),
    "galaxy":    ("#E6C2FF", 7, "circle", "Satellite galaxy"),
}

_OBJECT_HOVER = (
    "<b>%{customdata[0]}</b><br>"
    "%{customdata[1]}<br>"
    "─────────────────<br>"
    "%{customdata[2]}<br>"
    "Distance from Sun  %{customdata[3]}<br>"
    "Galactic l, b      %{customdata[4]}"
    "<extra></extra>"
)


def _object_customdata(o: dict) -> list:
    label = _OBJECT_STYLE.get(o.get("kind", ""), ("", 0, "", o.get("kind", "object")))[3]
    d = o.get("d_kpc", 0.0)
    dist = _fmt_kpc(d) if d < 1.0 else f"{d:.1f} kpc ({d * 3.262:.0f} kly)"
    return [
        o.get("name", "?"),
        label,
        o.get("desc", ""),
        dist,
        f"{o.get('gl', 0.0):.1f}°, {o.get('gb', 0.0):.1f}°",
    ]


# Memoised per-process render arrays — the catalogues are static, so the heavy
# per-point formatting (30k stars) runs once and is reused on every rerun.
_STAR_CACHE: dict = {}
_OBJ_CACHE: dict = {}


def _star_render_data(stars: list) -> dict:
    """Split stars into a bright tier (full cards) and faint tier (dense field).

    Returns numpy arrays + customdata lists, keyed/cached on catalogue size.
    """
    key = len(stars)
    if key in _STAR_CACHE:
        return _STAR_CACHE[key]

    mags = np.array([s.get("mag", 6.0) for s in stars], dtype=float)
    xs = np.array([s["x_kpc"] for s in stars], dtype=float)
    ys = np.array([s["y_kpc"] for s in stars], dtype=float)
    sizes = np.clip(4.6 - 0.55 * mags, 1.3, 6.5)

    # Brightest ~4000 carry full hover cards; the rest form the faint field.
    n_bright = min(4000, len(stars))
    order = np.argsort(mags)
    bright_idx = order[:n_bright]
    faint_idx = order[n_bright:]

    bright_cd = [_star_customdata(stars[i]) for i in bright_idx]
    faint_cd = [[stars[i].get("name", "?"),
                 f"{stars[i].get('d_pc', 0.0):.1f} pc"] for i in faint_idx]

    out = {
        "bright": dict(x=xs[bright_idx], y=ys[bright_idx],
                       size=sizes[bright_idx], cd=bright_cd),
        "faint": dict(x=xs[faint_idx], y=ys[faint_idx],
                      size=sizes[faint_idx], cd=faint_cd),
    }
    _STAR_CACHE[key] = out
    return out


def _object_render_data(objects: list) -> dict:
    """Group distributed objects by kind into render arrays (cached)."""
    key = len(objects)
    if key in _OBJ_CACHE:
        return _OBJ_CACHE[key]
    groups: dict[str, dict] = {}
    for o in objects:
        kind = o.get("kind", "globular")
        g = groups.setdefault(kind, {"x": [], "y": [], "cd": []})
        g["x"].append(o.get("x_kpc", 0.0))
        g["y"].append(o.get("y_kpc", 0.0))
        g["cd"].append(_object_customdata(o))
    _OBJ_CACHE[key] = groups
    return groups


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

    full_detail = data.get("full_detail", True)

    # Catalogued stars — points of light, every one tappable for its data card.
    # All real (parallax-measured) stars sit within ~1 kpc of the Sun: a bright
    # knot at galaxy zoom that resolves into the named solar neighbourhood as
    # you zoom in. Rendered in two tiers for performance:
    #   faint field (~26k) — shown only at full detail (idle), light hover
    #   bright tier (~4k)  — always shown, full data cards
    # Both star tiers render only at full detail (idle): the catalogue never
    # changes during a run, so rebuilding 30k points every filter iteration is
    # wasted work (all tabs rebuild each Streamlit rerun). During a run the map
    # keeps the cheap distributed objects + navigation markers; the full star
    # field repopulates the instant the run completes.
    stars = data.get("stars", [])
    if stars and full_detail:
        sr = _star_render_data(stars)
        f = sr["faint"]
        if len(f["x"]):
            fig.add_trace(go.Scattergl(
                x=f["x"], y=f["y"],
                mode="markers",
                marker=dict(size=np.maximum(f["size"], 1.8),
                            color="rgba(228,235,255,0.70)",
                            line=dict(width=0)),
                customdata=f["cd"],
                hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}<extra></extra>",
                showlegend=False,
                name="_faint_stars",
            ))
        b = sr["bright"]
        fig.add_trace(go.Scattergl(
            x=b["x"], y=b["y"],
            mode="markers",
            marker=dict(size=b["size"], color="rgba(240,244,255,0.92)",
                        line=dict(width=0)),
            customdata=b["cd"],
            hovertemplate=_STAR_HOVER,
            showlegend=False,
            name="Catalogued stars",
        ))

    # Distributed galactic objects — globular clusters (halo-wide), nebulae,
    # molecular clouds, supernova remnants, black holes. These give genuine
    # galaxy-wide clickable coverage and are ALWAYS shown (small count).
    objects = data.get("galactic_objects", [])
    if objects:
        groups = _object_render_data(objects)
        for kind, g in groups.items():
            colour, size, symbol, label = _OBJECT_STYLE.get(
                kind, ("#FFFFFF", 6, "circle", kind))
            # Soft halo so objects read against the bright star field
            fig.add_trace(go.Scattergl(
                x=g["x"], y=g["y"], mode="markers",
                marker=dict(size=size + 6, color=colour, opacity=0.18,
                            line=dict(width=0)),
                hoverinfo="skip", showlegend=False, name=f"_{kind}_halo",
            ))
            fig.add_trace(go.Scattergl(
                x=g["x"], y=g["y"], mode="markers",
                marker=dict(size=size, color=colour, opacity=0.95,
                            line=dict(width=0)),
                customdata=g["cd"], hovertemplate=_OBJECT_HOVER,
                showlegend=True, name=label,
            ))

    # Pulsars — points of light: soft halo + bright pinpoint core
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
            marker=dict(size=10, color="rgba(150,205,255,0.22)", symbol="circle",
                        line=dict(width=0)),
            hoverinfo="skip",
            showlegend=False,
            name="_pulsar_halo",
        ))
        # Pinpoint core — DM-tinted point of light (no symbol shapes)
        fig.add_trace(go.Scatter(
            x=px, y=py,
            mode="markers",
            marker=dict(
                size=3.6,
                color=pdms,
                colorscale=[[0.0, "#cfe4ff"], [0.5, "#ffffff"], [1.0, "#fff3cf"]],
                showscale=False,
                symbol="circle",
                line=dict(width=0),
                opacity=1.0,
            ),
            customdata=cdata,
            hovertemplate=_PULSAR_HOVER,
            showlegend=False,
            name="Pulsars",
        ))
        # Identification rings — thin and faint, just enough to read
        if ident_x:
            fig.add_trace(go.Scatter(
                x=ident_x, y=ident_y,
                mode="markers",
                marker=dict(size=12, color="rgba(0,0,0,0)", symbol="circle-open",
                            line=dict(width=1, color="rgba(0,212,255,0.75)")),
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
            marker=dict(size=14, color="rgba(255,90,90,0.18)", symbol="circle",
                        line=dict(width=0)),
            hoverinfo="skip",
            showlegend=False,
            name="_true_glow",
        ))
        fig.add_trace(go.Scatter(
            x=[tp[0]], y=[tp[1]],
            mode="markers",
            marker=dict(size=4.5, color="#FF6666", symbol="circle",
                        line=dict(width=0)),
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

    # Spacecraft estimated position — layered point of light, on top
    for size, alpha in ((22, 0.10), (12, 0.25)):
        fig.add_trace(go.Scatter(
            x=[sc_pos[0]], y=[sc_pos[1]],
            mode="markers",
            marker=dict(size=size, color=f"rgba(0,212,255,{alpha})",
                        symbol="circle", line=dict(width=0)),
            hoverinfo="skip",
            showlegend=False,
            name="_estimate_glow",
        ))
    fig.add_trace(go.Scatter(
        x=[sc_pos[0]], y=[sc_pos[1]],
        mode="markers",
        marker=dict(size=5.5, color="#9FEFFF", symbol="circle",
                    line=dict(width=0)),
        name="Estimate",
        hovertemplate=("<b>Spacecraft estimate</b><br>"
                       f"({sc_pos[0]:.2f}, {sc_pos[1]:.2f}) kpc<br>"
                       f"1σ uncertainty {_fmt_kpc(uncertainty_kpc)}<extra></extra>"),
    ))

    # Sun reference — warm layered point of light with stellar data card
    for size, alpha in ((16, 0.12), (9, 0.30)):
        fig.add_trace(go.Scatter(
            x=[sun_pos[0]], y=[sun_pos[1]],
            mode="markers",
            marker=dict(size=size, color=f"rgba(255,215,120,{alpha})",
                        symbol="circle", line=dict(width=0)),
            hoverinfo="skip",
            showlegend=False,
            name="_sun_glow",
        ))
    fig.add_trace(go.Scatter(
        x=[sun_pos[0]], y=[sun_pos[1]],
        mode="markers",
        marker=dict(size=4.5, color="#FFE9A0", symbol="circle",
                    line=dict(width=0)),
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
        # galaxy on phone-width viewports. Compact font: the object colour key
        # has many entries (tap an entry to toggle that layer).
        legend=dict(bgcolor="rgba(0,0,0,0.5)", orientation="h",
                    x=0.5, xanchor="center", y=-0.16, font=dict(size=8),
                    itemwidth=30),
        height=470,
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
                size=5,
                color=scores[mask_un],
                colorscale="RdYlGn_r",
                cmin=1, cmax=500,
                symbol="circle",
                opacity=0.9,
                line=dict(width=0),
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
        st.caption(
            "Pinch or scroll to zoom · drag to pan · tap any point of light "
            "for its data card · double-tap to reset. Zoom into the Sun to "
            "resolve 3,000 catalogued stars of the solar neighbourhood."
        )
    with col2:
        fig = build_skymap_figure(data)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
