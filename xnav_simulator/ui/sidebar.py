# ui/sidebar.py — Streamlit sidebar for XNAV Cold Start Simulator
# UI boundary: receives plain data only. No core/ or stages/ imports permitted.

"""
Render the simulation setup sidebar and return a settings dict.

The sidebar is split into:
  - SIMULATION SETUP (tier, spacecraft preset)
  - DETECTOR SETTINGS (area, band, integration time)
  - NOISE CONTROLS
  - ENVIRONMENT (central body, galactic potential)
  - DISPLAY (blind mode, animation speed)
  - RUN / RESET buttons

Returns a flat dict of all current settings. app.py is responsible for
reacting to setup-change events (see pending_reset pattern).
"""

from __future__ import annotations

import streamlit as st

import config


# ── Spacecraft preset descriptions ──────────────────────────────────────────

_PRESETS = [
    "Random deep space (default)",
    "Near a Sun-like star",
    "Galactic centre region",
    "Void between spiral arms",
    "Manual (GL / GB / Distance)",
]

_CENTRAL_BODIES = {
    "None (free space)": 0.0,
    "Earth-mass": 5.972e24,
    "Jupiter-mass": 1.898e27,
    "Solar-mass": config.M_SUN,
    "10× Solar": 10.0 * config.M_SUN,
}


def render() -> dict:
    """Render the full sidebar and return current settings as a plain dict.

    Never triggers filter reinitialisation directly — that responsibility
    belongs to app.py via the pending_reset session-state flag.
    """
    with st.sidebar:
        st.markdown(
            f'<h2 style="color:{config.COLOUR_ACCENT}; margin-bottom:0.3em;">'
            "XNAV Cold Start Simulator</h2>",
            unsafe_allow_html=True,
        )
        st.divider()

        # ── SIMULATION SETUP ────────────────────────────────────────────────
        st.subheader("SIMULATION SETUP")

        tier_options = list(config.ACCURACY_TIERS.keys())
        tier_labels = [
            f"{name}  (~{config.ACCURACY_TIERS[name]['expected_runtime_seconds']}s)"
            for name in tier_options
        ]
        tier_idx = st.radio(
            "Accuracy Tier",
            options=range(len(tier_options)),
            format_func=lambda i: tier_labels[i],
            index=tier_options.index(
                st.session_state.get("current_tier", config.DEFAULT_TIER)
            ),
        )
        selected_tier = tier_options[tier_idx]
        st.caption(config.ACCURACY_TIERS[selected_tier]["description"])

        st.markdown("**Spacecraft Position**")
        preset = st.selectbox("Preset scenario", _PRESETS, index=0)

        gl_manual = gb_manual = dist_manual = 0.0
        if preset == "Manual (GL / GB / Distance)":
            gl_manual = st.number_input("Galactic longitude GL (°)", 0.0, 360.0, 180.0, step=1.0)
            gb_manual = st.number_input("Galactic latitude GB (°)", -90.0, 90.0, 0.0, step=1.0)
            dist_manual = st.number_input("Distance from Sun (kpc)", 0.1, 30.0, 8.0, step=0.1)

        st.divider()

        # ── DETECTOR SETTINGS ───────────────────────────────────────────────
        st.subheader("DETECTOR SETTINGS")

        collecting_area_m2 = st.slider(
            "Collecting area (m²)",
            min_value=0.1, max_value=50.0,
            value=st.session_state.get("collecting_area_m2", 1.0),
            step=0.1,
            help="Larger = better photon statistics, faster profile matching",
        )
        freq_low_kev = st.slider(
            "Frequency band low (keV)", 0.5, 3.0,
            value=float(st.session_state.get("freq_low_kev", 1.0)), step=0.1,
        )
        freq_high_kev = st.slider(
            "Frequency band high (keV)", 3.0, 12.0,
            value=float(st.session_state.get("freq_high_kev", 10.0)), step=0.5,
        )
        integration_time_s = st.slider(
            "Integration time per pulsar (s)",
            min_value=100, max_value=10_000,
            value=int(st.session_state.get("integration_time_s", 1000)),
            step=100,
            help="Longer = less photon noise, slower per-iteration",
        )

        st.divider()

        # ── ADVANCED SETTINGS (collapsed by default) ─────────────────────
        with st.expander("Advanced settings", expanded=False):
            st.markdown(
                '<p style="color:#888; font-size:0.8em; margin:0 0 8px 0;">'
                'Noise controls, environment, and display options. '
                'Defaults work well for first runs.</p>',
                unsafe_allow_html=True,
            )

            st.subheader("NOISE CONTROLS")
            timing_noise_on = st.toggle("Timing noise", value=True)
            timing_noise_scale = 1.0
            if timing_noise_on:
                timing_noise_scale = st.slider(
                    "Timing noise multiplier", 0.1, 3.0, 1.0, step=0.1,
                    key="timing_noise_scale",
                )

            photon_noise_on = st.toggle("Photon noise", value=True)
            ism_turb_on = st.toggle("ISM turbulence", value=True)
            ism_turb_scale = 1.0
            if ism_turb_on:
                ism_turb_scale = st.slider(
                    "ISM turbulence multiplier", 0.1, 3.0, 1.0, step=0.1,
                    key="ism_turb_scale",
                )
            solar_wind = st.radio(
                "Solar wind", ["Quiet", "Moderate", "Active"], index=1, horizontal=True,
            )

            st.divider()
            st.subheader("ENVIRONMENT")
            central_body_name = st.selectbox("Central body", list(_CENTRAL_BODIES.keys()))
            central_body_mass_kg = _CENTRAL_BODIES[central_body_name]
            include_galactic_potential = st.checkbox(
                "Include galactic background potential", value=True,
            )

            st.divider()
            st.subheader("DISPLAY")
            blind_mode = st.toggle("Blind mode (hide true position)", value=False)
            if blind_mode:
                st.caption("🔒 True position hidden from all panels")
            animation_speed = st.radio(
                "Animation speed", ["Slow", "Normal", "Fast"], index=1, horizontal=True,
            )

        st.divider()

        # ── ACTION BUTTONS ──────────────────────────────────────────────────
        run_clicked = st.button(
            "▶  RUN SIMULATION",
            type="primary",
            use_container_width=True,
        )
        st.markdown('<div style="margin-top:4px;"></div>', unsafe_allow_html=True)
        reset_clicked = st.button(
            "↺  Reset",
            use_container_width=True,
            help="Clear all simulation state and start over",
        )

    return {
        # Setup
        "tier": selected_tier,
        "tier_config": config.ACCURACY_TIERS[selected_tier],
        "preset": preset,
        "gl_manual": gl_manual,
        "gb_manual": gb_manual,
        "dist_manual": dist_manual,
        # Detector
        "collecting_area_m2": collecting_area_m2,
        "freq_low_kev": freq_low_kev,
        "freq_high_kev": freq_high_kev,
        "integration_time_s": integration_time_s,
        # Noise
        "timing_noise_on": timing_noise_on,
        "timing_noise_scale": timing_noise_scale,
        "photon_noise_on": photon_noise_on,
        "ism_turb_on": ism_turb_on,
        "ism_turb_scale": ism_turb_scale,
        "solar_wind": solar_wind,
        # Environment
        "central_body_mass_kg": central_body_mass_kg,
        "central_body_name": central_body_name,
        "include_galactic_potential": include_galactic_potential,
        # Display
        "blind_mode": blind_mode,
        "animation_speed": animation_speed,
        # Buttons
        "run_clicked": run_clicked,
        "reset_clicked": reset_clicked,
    }
