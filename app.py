"""
app.py  ─  C3D Analyzer  ·  Main Entry Point
════════════════════════════════════════════
A professional biomechanics analysis platform built with Streamlit.
"""

import os
import sys
import io
import json
import zipfile
import tempfile
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ── Path setup ────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

# ── Page config (MUST be first Streamlit call) ─────────────────────────────
st.set_page_config(
    page_title="C3D Analyzer",
    page_icon="🦴",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help":     "https://github.com/",
        "Report a bug": "https://github.com/",
        "About":        "**C3D Analyzer** — Professional Biomechanics Analysis Platform",
    },
)

# ── Inject custom CSS ──────────────────────────────────────────────────────
css_path = BASE_DIR / "assets" / "style.css"
if css_path.exists():
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── Lazy imports (avoid error if ezc3d not installed at startup) ──────────
@st.cache_resource
def _check_deps():
    missing = []
    for pkg in ["ezc3d", "numpy", "pandas", "plotly", "scipy"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    return missing


# ── Import utils ─────────────────────────────────────────────────────────────
try:
    from utils.c3d_loader import (
        load_c3d, get_file_metadata,
        get_marker_data, get_analog_data, get_force_plate_data,
    )
    from utils.marker_analysis import (
        filter_marker_df, compute_velocity, compute_acceleration,
        compute_speed, compute_distance_between, compute_angle_3pt_safe,
        marker_stats, detect_gaps,
    )
    from utils.analog_analysis import (
        process_emg, emg_stats, compute_psd, compute_fft,
        compute_impulse, compute_loading_rate,
        detect_stance_phases, cop_path_length,
        detect_gait_events, compute_gait_metrics,
        lowpass_filter, bandpass_filter,
    )
    from utils.visualization import (
        PALETTE,
        plot_marker_trajectories, plot_marker_speed,
        plot_3d_trajectories, plot_stick_figure, plot_marker_heatmap,
        plot_marker_range_bar,
        plot_analog_channels, plot_grf, plot_cop,
        plot_emg, plot_psd,
        plot_stance_phases, plot_correlation_matrix,
    )
    UTILS_OK = True
except ImportError as e:
    UTILS_OK = False
    UTILS_ERR = str(e)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _css_badge(text: str, style: str = "blue") -> str:
    return f'<span class="badge badge-{style}">{text}</span>'


def _card(content_html: str) -> str:
    return f'<div class="analysis-card">{content_html}</div>'


def _section(icon: str, title: str, dot_color: str = "#0EA5E9"):
    st.markdown(
        f"""<div class="section-header">
              <div class="section-dot" style="background:{dot_color}"></div>
              <h2>{icon} {title}</h2>
           </div>""",
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner="Parsing C3D file…")
def _load_all(file_bytes: bytes):
    c3d  = load_c3d(file_bytes)
    meta = get_file_metadata(c3d)
    mdf  = get_marker_data(c3d)
    adf  = get_analog_data(c3d)
    fps  = get_force_plate_data(c3d, meta)
    return meta, mdf, adf, fps


def _safe_float(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

def render_sidebar():
    with st.sidebar:
        # Logo / Title
        st.markdown(
            """
            <div style="text-align:center; padding:1rem 0 0.5rem;">
              <div style="font-size:2.5rem; line-height:1;">🦴</div>
              <div style="font-weight:800; font-size:1.25rem;
                          color:#1E293B; letter-spacing:-0.02em;">
                C3D Analyzer
              </div>
              <div style="font-size:0.72rem; color:#94A3B8;
                          letter-spacing:0.08em; text-transform:uppercase;
                          margin-top:2px;">
                Biomechanics Platform
              </div>
            </div>
            <hr style="border:none;border-top:1px solid #E2E8F0;margin:0.75rem 0;">
            """,
            unsafe_allow_html=True,
        )

        # File uploader
        st.markdown("### 📂 Load File")
        uploaded = st.file_uploader(
            "Drop a C3D file here",
            type=["c3d"],
            label_visibility="collapsed",
        )

        if uploaded:
            st.success(f"✓ {uploaded.name}")
            st.session_state["file_name"]  = uploaded.name
            st.session_state["file_bytes"] = uploaded.read()

        st.markdown("---")

        # Navigation
        st.markdown("### 🗂️ Navigation")
        pages = [
            ("🏠", "Overview"),
            ("📍", "Markers"),
            ("📊", "Analog Signals"),
            ("⚡", "Force Plates"),
            ("🚶", "Gait Analysis"),
            ("🔬", "Signal Processing"),
            ("📐", "Joint Angles"),
            ("💾", "Export"),
        ]
        if "active_page" not in st.session_state:
            st.session_state["active_page"] = "Overview"

        for icon, name in pages:
            is_active = st.session_state["active_page"] == name
            btn_style = (
                "background:linear-gradient(135deg,#EFF6FF,#DBEAFE);"
                "border:1px solid #BFDBFE;color:#1D4ED8;font-weight:600;"
                if is_active else ""
            )
            if st.button(
                f"{icon}  {name}",
                key=f"nav_{name}",
                use_container_width=True,
                type="secondary" if not is_active else "primary",
            ):
                st.session_state["active_page"] = name
                st.rerun()

        st.markdown("---")
        st.markdown(
            '<div style="font-size:0.7rem;color:#94A3B8;text-align:center;">'
            "C3D Analyzer v1.0 · Built with Streamlit<br>"
            "Supports ezc3d · Plotly · SciPy"
            "</div>",
            unsafe_allow_html=True,
        )

    return st.session_state.get("active_page", "Overview")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════

def page_overview(meta: dict, mdf: pd.DataFrame, adf: pd.DataFrame, fps: list):
    _section("🏠", "File Overview")

    fn = st.session_state.get("file_name", "unknown.c3d")
    col_info, col_quick = st.columns([2, 1])

    with col_info:
        st.markdown(
            f"""<div class="analysis-card">
              <div style="font-size:0.75rem;color:#64748B;margin-bottom:4px;">
                FILE NAME</div>
              <div style="font-size:1.1rem;font-weight:700;color:#1E293B;">
                {fn}</div>
              <div style="margin-top:10px;display:flex;gap:8px;flex-wrap:wrap;">
                {_css_badge(meta["manufacturer"], "blue")}
                {_css_badge(meta["software"], "indigo")}
                {_css_badge(meta["point_unit"], "green")}
              </div>
            </div>""",
            unsafe_allow_html=True,
        )

    with col_quick:
        quality = "Good" if mdf.isnull().mean().mean() < 0.05 else \
                  "Fair" if mdf.isnull().mean().mean() < 0.15 else "Poor"
        q_color = {"Good": "#10B981", "Fair": "#F59E0B", "Poor": "#EF4444"}[quality]
        st.markdown(
            f"""<div class="analysis-card" style="text-align:center;">
              <div style="font-size:0.72rem;color:#64748B;
                          text-transform:uppercase;letter-spacing:.06em;">
                Data Quality</div>
              <div style="font-size:2rem;font-weight:800;color:{q_color};">
                {quality}</div>
            </div>""",
            unsafe_allow_html=True,
        )

    # Key metrics
    st.markdown("#### 📈 Key Parameters")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1: st.metric("Duration",   f"{meta['duration_s']:.2f} s")
    with c2: st.metric("Frame Rate", f"{meta['frame_rate']:.0f} Hz")
    with c3: st.metric("Frames",     f"{meta['n_frames']:,}")
    with c4: st.metric("Markers",    meta['n_markers'])
    with c5: st.metric("Analog Ch.", meta['n_analog'])
    with c6: st.metric("Force Plates", meta['n_force_plates'])

    st.markdown("---")

    # Two-column detail
    c_left, c_right = st.columns(2)

    with c_left:
        st.markdown("#### 📍 Markers")
        if meta["marker_labels"]:
            labels = meta["marker_labels"]
            html = "<div style='display:flex;flex-wrap:wrap;gap:6px;'>"
            for lbl in labels:
                html += f'<span class="badge badge-blue">{lbl}</span>'
            html += "</div>"
            st.markdown(html, unsafe_allow_html=True)
        else:
            st.info("No marker labels found.")

        if not mdf.empty:
            st.markdown("#### 🔍 Marker Data Quality")
            stats = marker_stats(mdf)
            gap_fig = plot_marker_heatmap(stats)
            st.plotly_chart(gap_fig, use_container_width=True, key="gap_heatmap")

    with c_right:
        st.markdown("#### 📊 Analog Channels")
        if meta["analog_labels"]:
            html = "<div style='display:flex;flex-wrap:wrap;gap:6px;'>"
            for lbl in meta["analog_labels"]:
                html += f'<span class="badge badge-indigo">{lbl}</span>'
            html += "</div>"
            st.markdown(html, unsafe_allow_html=True)
        else:
            st.info("No analog channels found.")

        st.markdown("#### 🗃️ Parameter Summary")
        param_data = {
            "Parameter": [
                "First Frame", "Last Frame", "Point Rate",
                "Analog Rate", "Point Unit", "Subjects",
            ],
            "Value": [
                meta["first_frame"], meta["last_frame"],
                f"{meta['point_rate']} Hz",
                f"{meta['analog_rate']} Hz",
                meta["point_unit"],
                ", ".join(meta["subjects"]) if meta["subjects"] else "—",
            ],
        }
        st.dataframe(
            pd.DataFrame(param_data),
            use_container_width=True,
            hide_index=True,
        )

    # Quick preview plots
    if not mdf.empty and meta["marker_labels"]:
        st.markdown("---")
        st.markdown("#### 🚀 Quick Preview")
        first_marker = meta["marker_labels"][0]
        col_a, col_b = st.columns(2)
        with col_a:
            fig = plot_marker_trajectories(mdf, first_marker)
            st.plotly_chart(fig, use_container_width=True, key="ov_traj")
        with col_b:
            if not adf.empty:
                first_chs = meta["analog_labels"][:4]
                fig2 = plot_analog_channels(adf, first_chs, "Analog Preview")
                st.plotly_chart(fig2, use_container_width=True, key="ov_analog")
            else:
                # 3D scatter of first 5 markers
                top5 = meta["marker_labels"][:5]
                fig2 = plot_3d_trajectories(mdf, top5)
                st.plotly_chart(fig2, use_container_width=True, key="ov_3d")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MARKERS
# ══════════════════════════════════════════════════════════════════════════════

def page_markers(meta: dict, mdf: pd.DataFrame):
    _section("📍", "Marker Analysis")

    if mdf.empty:
        st.warning("No marker data available in this file.")
        return

    fs = meta["point_rate"]
    markers = meta["marker_labels"]

    # ── Sidebar controls
    with st.expander("⚙️ Filter & Display Settings", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            apply_filter = st.toggle("Apply Butterworth Filter", value=True)
            cutoff = st.slider("Cutoff Freq (Hz)", 1.0, min(fs/2-1, 20.0), 6.0, 0.5)
        with col2:
            selected_markers = st.multiselect(
                "Select Markers",
                markers,
                default=markers[:min(6, len(markers))],
            )
        with col3:
            show_residual = st.toggle("Show Residual Quality", value=False)
            show_velocity = st.toggle("Show Velocity",          value=True)
            show_accel    = st.toggle("Show Acceleration",      value=False)

    if not selected_markers:
        st.info("Please select at least one marker.")
        return

    # Filter
    disp_df = filter_marker_df(mdf, cutoff, fs) if apply_filter else mdf

    # ── Stats table
    st.markdown("#### 📊 Marker Statistics")
    stats = marker_stats(disp_df[[(m, a) for m in selected_markers
                                  for a in ("X","Y","Z","Res")
                                  if (m, a) in disp_df.columns]])
    st.dataframe(stats.style.format("{:.2f}").background_gradient(
        subset=["gap_%"], cmap="YlOrRd", vmin=0, vmax=30),
        use_container_width=True)

    # ── Range of Motion bar
    st.markdown("#### 📐 Range of Motion")
    fig_rom = plot_marker_range_bar(stats)
    st.plotly_chart(fig_rom, use_container_width=True, key="rom_bar")

    st.markdown("---")
    st.markdown("#### 🕵️ Individual Marker Inspector")

    sel = st.selectbox("Choose Marker", selected_markers, key="marker_sel")

    tab_traj, tab_vel, tab_gap, tab_3d = st.tabs([
        "📈 Trajectory", "⚡ Velocity / Acceleration", "🕳️ Gap Analysis", "🌐 3-D View"
    ])

    with tab_traj:
        fig = plot_marker_trajectories(disp_df, sel,
                                       f"{'[Filtered] ' if apply_filter else ''}{sel}")
        st.plotly_chart(fig, use_container_width=True, key="traj_sel")

        if show_residual and (sel, "Res") in mdf.columns:
            res = mdf[(sel, "Res")].values
            import plotly.graph_objects as go
            fig_r = go.Figure(go.Scatter(
                x=mdf.index.values, y=res, mode="lines",
                line=dict(color="#F59E0B", width=1.5),
                name="Residual",
            ))
            fig_r.update_layout(height=200, margin=dict(l=10,r=10,t=30,b=10),
                                 title="Residual (tracking quality)",
                                 paper_bgcolor="rgba(0,0,0,0)",
                                 plot_bgcolor="#FAFCFF")
            st.plotly_chart(fig_r, use_container_width=True, key="residual_plot")

    with tab_vel:
        vel_df  = compute_velocity(disp_df, fs)
        acc_df  = compute_acceleration(vel_df, fs) if show_accel else None
        spd_df  = compute_speed(vel_df, [sel])

        col_v, col_a = st.columns(2)
        with col_v:
            # Component velocities
            import plotly.graph_objects as go
            fig_v = go.Figure()
            for i, ax in enumerate(["VX","VY","VZ"]):
                col_key = (sel, ax)
                if col_key in vel_df.columns:
                    fig_v.add_trace(go.Scatter(
                        x=vel_df.index.values, y=vel_df[col_key].values,
                        name=ax, line=dict(color=PALETTE[i], width=1.5),
                        mode="lines",
                    ))
            fig_v.update_layout(height=300, title="Velocity Components (mm/s)",
                                 paper_bgcolor="rgba(0,0,0,0)",
                                 plot_bgcolor="#FAFCFF",
                                 margin=dict(l=10,r=10,t=40,b=10))
            st.plotly_chart(fig_v, use_container_width=True, key="vel_comp")

        with col_a:
            spd_fig = plot_marker_speed(spd_df, [sel], f"{sel} — Speed")
            st.plotly_chart(spd_fig, use_container_width=True, key="spd_sel")

        if show_accel and acc_df is not None:
            import plotly.graph_objects as go
            fig_a = go.Figure()
            for i, ax in enumerate(["AX","AY","AZ"]):
                col_key = (sel, ax)
                if col_key in acc_df.columns:
                    fig_a.add_trace(go.Scatter(
                        x=acc_df.index.values, y=acc_df[col_key].values,
                        name=ax, line=dict(color=PALETTE[i], width=1.5, dash="dot"),
                        mode="lines",
                    ))
            fig_a.update_layout(height=280, title="Acceleration Components (mm/s²)",
                                  paper_bgcolor="rgba(0,0,0,0)",
                                  plot_bgcolor="#FAFCFF",
                                  margin=dict(l=10,r=10,t=40,b=10))
            st.plotly_chart(fig_a, use_container_width=True, key="acc_plot")

    with tab_gap:
        gaps = detect_gaps(mdf, sel)
        if not gaps:
            st.success(f"✅ **{sel}** has no data gaps — 100% valid frames.")
        else:
            st.warning(f"⚠️ Found **{len(gaps)}** gap(s) in marker **{sel}**.")
            gap_df = pd.DataFrame(gaps)
            gap_df["start_s"]     = gap_df["start_s"].round(4)
            gap_df["end_s"]       = gap_df["end_s"].round(4)
            gap_df["duration_s"]  = gap_df["duration_s"].round(4)
            st.dataframe(gap_df, use_container_width=True, hide_index=True)

    with tab_3d:
        top_n = st.slider("Number of markers to show", 2, min(20, len(markers)), 8,
                           key="3d_n")
        fig3d = plot_3d_trajectories(disp_df, selected_markers[:top_n])
        st.plotly_chart(fig3d, use_container_width=True, key="3d_traj")

    # ── Multi-marker comparison
    st.markdown("---")
    st.markdown("#### 🔄 Multi-Marker Speed Comparison")
    vel_all = compute_velocity(disp_df, fs)
    spd_all = compute_speed(vel_all, selected_markers)
    fig_spd = plot_marker_speed(spd_all, selected_markers, "Marker Speed Comparison")
    st.plotly_chart(fig_spd, use_container_width=True, key="multi_spd")

    # ── Distance analysis
    if len(selected_markers) >= 2:
        st.markdown("#### 📏 Inter-Marker Distance")
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            dm1 = st.selectbox("Marker 1", selected_markers, index=0, key="dm1")
        with col_d2:
            dm2 = st.selectbox("Marker 2", selected_markers, index=1, key="dm2")
        if dm1 != dm2:
            dist = compute_distance_between(disp_df, dm1, dm2)
            import plotly.graph_objects as go
            fig_dist = go.Figure(go.Scatter(
                x=disp_df.index.values, y=dist,
                mode="lines", line=dict(color=PALETTE[3], width=2),
                fill="tozeroy", fillcolor="rgba(245,158,11,0.1)",
                name=f"Distance {dm1}↔{dm2}",
            ))
            fig_dist.update_layout(height=300,
                                    title=f"Distance: {dm1} ↔ {dm2} (mm)",
                                    xaxis_title="Time (s)",
                                    yaxis_title="Distance (mm)",
                                    paper_bgcolor="rgba(0,0,0,0)",
                                    plot_bgcolor="#FAFCFF",
                                    margin=dict(l=10,r=10,t=40,b=10))
            c_stat1, c_stat2, c_stat3 = st.columns(3)
            c_stat1.metric("Mean Distance", f"{np.nanmean(dist):.2f} mm")
            c_stat2.metric("Max Distance",  f"{np.nanmax(dist):.2f} mm")
            c_stat3.metric("Min Distance",  f"{np.nanmin(dist):.2f} mm")
            st.plotly_chart(fig_dist, use_container_width=True, key="dist_plot")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ANALOG SIGNALS
# ══════════════════════════════════════════════════════════════════════════════

def page_analog(meta: dict, adf: pd.DataFrame):
    _section("📊", "Analog Signals", "#6366F1")

    if adf.empty:
        st.warning("No analog data available in this file.")
        return

    channels = meta["analog_labels"]
    fs       = meta["analog_rate"] or 1000.0

    with st.expander("⚙️ Channel & Filter Settings", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            sel_chs = st.multiselect("Channels", channels,
                                      default=channels[:min(8, len(channels))])
        with c2:
            time_range = st.slider(
                "Time Window (s)",
                0.0, float(adf.index[-1]),
                (0.0, float(min(adf.index[-1], 10.0))),
                key="analog_time",
            )
        with c3:
            norm_channels = st.toggle("Normalize (0–1)", value=False)
            downsample    = st.toggle("Downsample (×10)", value=False)

    if not sel_chs:
        st.info("Select at least one channel.")
        return

    # Slice time window
    mask   = (adf.index >= time_range[0]) & (adf.index <= time_range[1])
    sub_df = adf.loc[mask, [c for c in sel_chs if c in adf.columns]]
    if downsample:
        sub_df = sub_df.iloc[::10]
    if norm_channels:
        sub_df = (sub_df - sub_df.min()) / (sub_df.max() - sub_df.min() + 1e-12)

    # Time series
    st.markdown("#### 📈 Signal Overview")
    fig = plot_analog_channels(sub_df, sel_chs[:16], "Analog Signals")
    st.plotly_chart(fig, use_container_width=True, key="analog_overview")

    st.markdown("---")
    tab_single, tab_stats, tab_corr, tab_emg = st.tabs([
        "🔍 Channel Detail", "📊 Statistics", "🔗 Correlation", "🦾 EMG Processing"
    ])

    with tab_single:
        ch_sel = st.selectbox("Select Channel", sel_chs, key="ch_detail_sel")
        if ch_sel in adf.columns:
            raw = sub_df[ch_sel].values
            t   = sub_df.index.values
            import plotly.graph_objects as go
            fig_s = go.Figure()
            fig_s.add_trace(go.Scatter(
                x=t, y=raw, mode="lines",
                line=dict(color=PALETTE[0], width=1.5),
                name=ch_sel,
            ))
            fig_s.update_layout(
                height=300, title=ch_sel,
                xaxis_title="Time (s)",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#FAFCFF",
                margin=dict(l=10,r=10,t=40,b=10),
            )
            st.plotly_chart(fig_s, use_container_width=True, key="ch_single")

            # PSD
            freqs, psd = compute_psd(raw[~np.isnan(raw)], fs)
            fig_psd = plot_psd(freqs, psd, ch_sel)
            st.plotly_chart(fig_psd, use_container_width=True, key="ch_psd")

    with tab_stats:
        stats_rows = []
        for ch in sel_chs:
            if ch in adf.columns:
                v = adf[ch].dropna().values
                stats_rows.append({
                    "Channel": ch,
                    "Mean":    float(np.mean(v)),
                    "Std":     float(np.std(v)),
                    "Min":     float(np.min(v)),
                    "Max":     float(np.max(v)),
                    "Range":   float(np.max(v) - np.min(v)),
                    "RMS":     float(np.sqrt(np.mean(v**2))),
                    "Skew":    float(pd.Series(v).skew()),
                    "Kurtosis":float(pd.Series(v).kurt()),
                })
        if stats_rows:
            st.dataframe(
                pd.DataFrame(stats_rows).set_index("Channel").style.format("{:.4f}"),
                use_container_width=True,
            )

    with tab_corr:
        valid_chs = [c for c in sel_chs if c in adf.columns]
        if len(valid_chs) >= 2:
            # Align lengths by resampling sub_df
            fig_corr = plot_correlation_matrix(sub_df.fillna(0), valid_chs[:16])
            st.plotly_chart(fig_corr, use_container_width=True, key="corr_mat")
        else:
            st.info("Select ≥ 2 channels for correlation analysis.")

    with tab_emg:
        st.markdown("##### EMG Processing Pipeline")
        emg_ch = st.selectbox("EMG Channel", sel_chs, key="emg_ch")
        col_e1, col_e2, col_e3 = st.columns(3)
        with col_e1: hp_cut  = st.number_input("HP Cutoff (Hz)", 10.0, 100.0, 20.0)
        with col_e2: lp_env  = st.number_input("Envelope Cutoff (Hz)", 1.0, 50.0, 6.0)
        with col_e3: notch   = st.number_input("Notch Freq (Hz)", 0.0, 200.0, 50.0)

        if emg_ch in adf.columns:
            raw_emg   = adf[emg_ch].values
            emg_proc  = process_emg(raw_emg, fs, hp_cut, lp_env, notch)
            emg_t     = adf.index.values
            fig_emg   = plot_emg(emg_t, emg_proc, emg_ch)
            st.plotly_chart(fig_emg, use_container_width=True, key="emg_proc")

            stats_emg = emg_stats(emg_proc["filtered"], fs)
            col_e1, col_e2, col_e3, col_e4 = st.columns(4)
            col_e1.metric("RMS",         f"{stats_emg['rms']:.4f}")
            col_e2.metric("Peak",        f"{stats_emg['peak']:.4f}")
            col_e3.metric("Mean Freq",   f"{stats_emg['mean_freq']:.1f} Hz")
            col_e4.metric("Median Freq", f"{stats_emg['median_freq']:.1f} Hz")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: FORCE PLATES
# ══════════════════════════════════════════════════════════════════════════════

def page_force_plates(meta: dict, adf: pd.DataFrame, fps: list):
    _section("⚡", "Force Plate Analysis", "#10B981")

    if not fps:
        # Try to detect force-like channels from analog
        st.info("No force platform parameters detected in this file.")
        if not adf.empty:
            st.markdown("#### 🔍 Analog Channel Search")
            st.markdown("Looking for force-like channels (Fx, Fy, Fz, GRF)…")
            force_chs = [c for c in adf.columns
                         if any(k in c.lower() for k in ["fx","fy","fz","force","grf","grw"])]
            if force_chs:
                st.success(f"Found potential force channels: {force_chs}")
                fig = plot_analog_channels(adf, force_chs[:6], "Force-Like Channels")
                st.plotly_chart(fig, use_container_width=True, key="fp_manual")
            else:
                st.warning("No force channels detected in analog data.")
        return

    n_fp = meta["n_force_plates"]
    st.markdown(f"##### Detected **{n_fp}** force plate(s)")

    plate_idx = 0
    if n_fp > 1:
        plate_idx = st.radio("Select Plate", list(range(n_fp)),
                              format_func=lambda i: f"Plate {i+1}",
                              horizontal=True)

    fp = fps[plate_idx]
    if "Fz" not in fp:
        st.warning(f"Plate {plate_idx+1}: Could not map force channels.")
        return

    t  = fp["time_s"]
    fz = fp["Fz"]

    # ── GRF figure
    st.markdown("#### 📈 Ground Reaction Forces")
    fig_grf = plot_grf(t, fp, plate_idx)
    st.plotly_chart(fig_grf, use_container_width=True, key=f"grf_{plate_idx}")

    # ── Stance detection
    with st.expander("⚙️ Stance Detection Settings"):
        thresh = st.slider("Contact Threshold (N)", 1.0, 100.0, 10.0, key="fp_thresh")
        min_dur = st.slider("Min Stance Duration (s)", 0.01, 1.0, 0.1, key="fp_mindur")

    phases = detect_stance_phases(fz, t, threshold=thresh, min_duration_s=min_dur)

    col_p1, col_p2, col_p3, col_p4 = st.columns(4)
    with col_p1: st.metric("Stance Phases", len(phases))
    if phases:
        durations = [p["duration_s"] for p in phases]
        with col_p2: st.metric("Mean Duration", f"{np.mean(durations):.3f} s")
        with col_p3: st.metric("Peak Fz",       f"{max(p['peak_Fz'] for p in phases):.1f} N")
        total_impulse = compute_impulse(np.abs(fz), t)
        with col_p4: st.metric("Total Impulse", f"{total_impulse:.1f} N·s")

    fig_stance = plot_stance_phases(t, fz, phases)
    st.plotly_chart(fig_stance, use_container_width=True, key=f"stance_{plate_idx}")

    if phases:
        st.markdown("#### 📋 Stance Phase Details")
        ph_df = pd.DataFrame(phases)
        ph_df = ph_df.round(4)
        st.dataframe(ph_df, use_container_width=True, hide_index=True)

    # ── COP
    if "COPx" in fp and "COPy" in fp:
        st.markdown("---")
        st.markdown("#### 🎯 Centre of Pressure (COP)")
        cop_fig = plot_cop(fp["COPx"], fp["COPy"], fp.get("corners"))
        st.plotly_chart(cop_fig, use_container_width=True, key=f"cop_{plate_idx}")

        cop_len = cop_path_length(fp["COPx"], fp["COPy"])
        col_c1, col_c2 = st.columns(2)
        col_c1.metric("COP Path Length", f"{cop_len:.1f} mm")
        cop_valid = np.sum(~np.isnan(fp["COPx"]))
        col_c2.metric("COP Valid Points", f"{cop_valid:,}")

    # ── Loading rate per phase
    if phases:
        st.markdown("---")
        st.markdown("#### ⚡ Loading Analysis")
        lr_rows = []
        for ph in phases:
            seg_f = fz[ph["start_idx"]:ph["end_idx"]+1]
            seg_t = t[ph["start_idx"]:ph["end_idx"]+1]
            lr_rows.append({
                "Phase":        phases.index(ph) + 1,
                "Start (s)":    round(ph["start_s"], 3),
                "End (s)":      round(ph["end_s"], 3),
                "Duration (s)": round(ph["duration_s"], 3),
                "Peak Fz (N)":  round(ph["peak_Fz"], 1),
                "Impulse (N·s)": round(compute_impulse(np.abs(seg_f), seg_t), 2),
                "Loading Rate (N/s)": round(
                    compute_loading_rate(np.abs(seg_f), seg_t), 1),
            })
        st.dataframe(pd.DataFrame(lr_rows), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: GAIT ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def page_gait(meta: dict, mdf: pd.DataFrame, adf: pd.DataFrame, fps: list):
    _section("🚶", "Gait Analysis", "#F59E0B")

    if not fps and adf.empty:
        st.warning("No force plate or analog data available for gait analysis.")
        return

    st.markdown(
        """<div class="analysis-card">
          <b>🔬 Automated Gait Event Detection</b><br>
          Heel-strike (HS) and toe-off (TO) events are detected from the vertical
          GRF signal using a threshold-based approach. Spatiotemporal parameters
          are then computed from the event timing.
        </div>""",
        unsafe_allow_html=True,
    )

    # Choose force source
    fz_source = None
    fz_time   = None

    if fps and "Fz" in fps[0]:
        plate_choice = st.selectbox(
            "Force Source",
            [f"Force Plate {i+1}" for i in range(len(fps))
             if "Fz" in fps[i]],
            key="gait_fp",
        )
        idx = int(plate_choice.split()[-1]) - 1
        fz_source = fps[idx]["Fz"]
        fz_time   = fps[idx]["time_s"]
    elif not adf.empty:
        force_chs = [c for c in adf.columns
                     if any(k in c.lower() for k in ["fz","force","grw"])]
        if force_chs:
            ch = st.selectbox("Force Channel (Fz proxy)", force_chs, key="gait_ch")
            fz_source = adf[ch].values
            fz_time   = adf.index.values

    if fz_source is None:
        st.warning("Could not identify a vertical force signal for gait analysis.")
        return

    col_g1, col_g2, col_g3 = st.columns(3)
    with col_g1:
        contact_thresh = st.slider("Contact Threshold (N)", 1.0, 50.0, 10.0, key="g_thresh")
    with col_g2:
        body_mass = st.number_input("Body Mass (kg)", 20.0, 200.0, 70.0, key="g_mass")
    with col_g3:
        gait_speed = st.number_input("Approx. Speed (m/s)", 0.0, 5.0, 0.0, key="g_speed",
                                      help="Leave 0 if unknown")

    events = detect_gait_events(fz_source, fz_time, threshold=contact_thresh)
    metrics = compute_gait_metrics(events, body_mass_kg=body_mass)

    # Display events
    hs = np.array(events["heel_strike_times"])
    to = np.array(events["toe_off_times"])

    st.markdown("#### 📊 Gait Events")
    col_ev1, col_ev2, col_ev3, col_ev4 = st.columns(4)
    col_ev1.metric("Heel Strikes", len(hs))
    col_ev2.metric("Toe-Offs",     len(to))
    if "mean_stride_time_s" in metrics:
        col_ev3.metric("Mean Stride Time", f"{metrics['mean_stride_time_s']:.3f} s")
        col_ev4.metric("Cadence",          f"{metrics.get('cadence_steps_min',0):.1f} steps/min")

    if "mean_stance_s" in metrics:
        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        col_s1.metric("Mean Stance", f"{metrics['mean_stance_s']:.3f} s")
        col_s2.metric("Stance %",    f"{metrics.get('stance_pct',0):.1f}%")
        if "std_stride_time_s" in metrics:
            col_s3.metric("Stride Variability", f"{metrics['std_stride_time_s']*1000:.1f} ms")
        if gait_speed > 0 and "mean_stride_time_s" in metrics:
            stride_len = gait_speed * metrics["mean_stride_time_s"]
            col_s4.metric("Stride Length", f"{stride_len:.2f} m")

    # Force with event markers
    import plotly.graph_objects as go
    fig_ev = go.Figure()
    fig_ev.add_trace(go.Scatter(
        x=fz_time, y=np.abs(fz_source),
        mode="lines", line=dict(color=PALETTE[0], width=1.5),
        name="Fz", fill="tozeroy",
        fillcolor="rgba(14,165,233,0.08)",
    ))
    for ev_t in hs:
        fig_ev.add_vline(x=ev_t, line_color="#10B981", line_width=1.5,
                          line_dash="solid", annotation_text="HS",
                          annotation_font_size=8)
    for ev_t in to:
        fig_ev.add_vline(x=ev_t, line_color="#F43F5E", line_width=1.5,
                          line_dash="dash", annotation_text="TO",
                          annotation_font_size=8)
    fig_ev.update_layout(
        height=350, title="Gait Events on Vertical GRF",
        xaxis_title="Time (s)", yaxis_title="Fz (N)",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#FAFCFF",
        margin=dict(l=10,r=10,t=40,b=10),
        legend=dict(bgcolor="rgba(255,255,255,.9)"),
    )
    st.plotly_chart(fig_ev, use_container_width=True, key="gait_events")

    # Stride time variability
    if len(hs) >= 3:
        stride_times = np.diff(hs)
        fig_st = go.Figure(go.Bar(
            x=list(range(1, len(stride_times)+1)),
            y=stride_times,
            marker_color=PALETTE[2],
            name="Stride Time",
            text=np.round(stride_times, 3),
            textposition="outside",
        ))
        fig_st.add_hline(y=float(np.mean(stride_times)),
                          line_dash="dash", line_color="#F59E0B",
                          annotation_text="Mean")
        fig_st.update_layout(
            height=300, title="Stride-by-Stride Variability",
            xaxis_title="Stride #", yaxis_title="Stride Time (s)",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#FAFCFF",
            margin=dict(l=10,r=10,t=40,b=10),
        )
        st.plotly_chart(fig_st, use_container_width=True, key="stride_var")

    # Marker-based gait if available
    if not mdf.empty:
        st.markdown("---")
        st.markdown("#### 📍 Marker-Based Gait Kinematics")
        heel_markers = [m for m in meta["marker_labels"]
                        if any(k in m.upper() for k in ["HEEL","HEE","CALC","CAL"])]
        toe_markers  = [m for m in meta["marker_labels"]
                        if any(k in m.upper() for k in ["TOE","MTOE","MTP","META"])]

        if heel_markers:
            h_sel = st.selectbox("Heel Marker", heel_markers, key="g_heel")
            from utils.marker_analysis import filter_marker_df, compute_velocity, compute_speed
            fs_m = meta["point_rate"]
            filt_m = filter_marker_df(mdf, 6.0, fs_m)
            v_m = compute_velocity(filt_m, fs_m)
            s_m = compute_speed(v_m, [h_sel])
            fig_hs = plot_marker_speed(s_m, [h_sel], f"{h_sel} Heel Speed")
            st.plotly_chart(fig_hs, use_container_width=True, key="g_heel_spd")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: SIGNAL PROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def page_signal_processing(meta: dict, adf: pd.DataFrame, mdf: pd.DataFrame):
    _section("🔬", "Signal Processing", "#8B5CF6")

    st.markdown(
        """<div class="analysis-card">
          Interactive signal processing: apply filters, compute FFT/PSD,
          and compare raw vs filtered signals side-by-side.
        </div>""",
        unsafe_allow_html=True,
    )

    data_source = st.radio("Data Source", ["Analog", "Marker"], horizontal=True)

    if data_source == "Analog" and not adf.empty:
        channels = meta["analog_labels"]
        fs = meta["analog_rate"] or 1000.0
        ch = st.selectbox("Channel", channels, key="sp_ch")
        raw = adf[ch].values if ch in adf.columns else np.zeros(100)
        t   = adf.index.values
    elif data_source == "Marker" and not mdf.empty:
        markers = meta["marker_labels"]
        fs      = meta["point_rate"]
        m_sel   = st.selectbox("Marker", markers, key="sp_m")
        ax_sel  = st.radio("Axis", ["X","Y","Z"], horizontal=True)
        raw = mdf[(m_sel, ax_sel)].values if (m_sel, ax_sel) in mdf.columns else np.zeros(100)
        t   = mdf.index.values
        ch  = f"{m_sel}-{ax_sel}"
    else:
        st.warning("No data available for signal processing.")
        return

    # Filter controls
    st.markdown("#### ⚙️ Filter Configuration")
    col_f1, col_f2, col_f3, col_f4 = st.columns(4)
    with col_f1:
        ftype = st.selectbox("Filter Type",
                              ["Low-pass","High-pass","Band-pass","Notch"])
    with col_f2:
        order = st.slider("Filter Order", 1, 8, 4, key="sp_order")
    with col_f3:
        if ftype == "Band-pass":
            low_cut = st.number_input("Low Cutoff (Hz)", 0.1, fs/2-1, 20.0)
        elif ftype == "Notch":
            notch_f = st.number_input("Notch Freq (Hz)", 1.0, fs/2-1, 50.0)
        else:
            cutoff_f = st.number_input("Cutoff (Hz)", 0.1, fs/2-1,
                                        6.0 if ftype=="Low-pass" else 20.0)
    with col_f4:
        if ftype == "Band-pass":
            high_cut = st.number_input("High Cutoff (Hz)", 1.0, fs/2-1, 400.0)

    # Apply filter
    try:
        nan_mask = np.isnan(raw)
        clean    = raw.copy(); clean[nan_mask] = 0.0
        if ftype == "Low-pass":
            filtered = lowpass_filter(clean, cutoff_f, fs, order)
        elif ftype == "High-pass":
            filtered = highpass_filter(clean, cutoff_f, fs, order)
        elif ftype == "Band-pass":
            filtered = bandpass_filter(clean, low_cut, high_cut, fs, order)
        else:
            from utils.analog_analysis import notch_filter as nf
            filtered = nf(clean, notch_f, fs)
        filtered[nan_mask] = np.nan
    except Exception as e:
        st.error(f"Filter error: {e}")
        filtered = raw.copy()

    # Side-by-side
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Raw Signal", "Filtered Signal"])
    for col_i, (sig, name, color) in enumerate([
        (raw, "Raw", "#94A3B8"), (filtered, "Filtered", PALETTE[0])
    ], 1):
        fig.add_trace(go.Scatter(
            x=t, y=sig, mode="lines",
            line=dict(color=color, width=1.5), name=name,
        ), row=1, col=col_i)
    fig.update_layout(height=350, paper_bgcolor="rgba(0,0,0,0)",
                       plot_bgcolor="#FAFCFF",
                       margin=dict(l=10,r=10,t=50,b=10))
    fig.update_xaxes(gridcolor="#EEF2F7", linecolor="#CBD5E1")
    fig.update_yaxes(gridcolor="#EEF2F7", linecolor="#CBD5E1")
    st.plotly_chart(fig, use_container_width=True, key="sp_compare")

    # Spectral
    st.markdown("#### 🌊 Spectral Analysis")
    tab_fft, tab_psd = st.tabs(["FFT Amplitude Spectrum", "Welch PSD"])
    with tab_fft:
        col_fa, col_fb = st.columns(2)
        with col_fa:
            frq_raw, amp_raw = compute_fft(raw[~np.isnan(raw)], fs)
            fig_fa = go.Figure(go.Scatter(x=frq_raw, y=amp_raw, mode="lines",
                                           line=dict(color="#94A3B8", width=1.5),
                                           fill="tozeroy",
                                           fillcolor="rgba(148,163,184,.15)"))
            fig_fa.update_layout(title="Raw FFT", height=300, xaxis_title="Hz",
                                   paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#FAFCFF",
                                   margin=dict(l=10,r=10,t=40,b=10))
            st.plotly_chart(fig_fa, use_container_width=True, key="fft_raw")
        with col_fb:
            frq_f, amp_f = compute_fft(filtered[~np.isnan(filtered)], fs)
            fig_fb = go.Figure(go.Scatter(x=frq_f, y=amp_f, mode="lines",
                                           line=dict(color=PALETTE[0], width=1.5),
                                           fill="tozeroy",
                                           fillcolor="rgba(14,165,233,.15)"))
            fig_fb.update_layout(title="Filtered FFT", height=300, xaxis_title="Hz",
                                   paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#FAFCFF",
                                   margin=dict(l=10,r=10,t=40,b=10))
            st.plotly_chart(fig_fb, use_container_width=True, key="fft_filt")
    with tab_psd:
        frq_p, psd_p = compute_psd(filtered[~np.isnan(filtered)], fs)
        fig_psd = plot_psd(frq_p, psd_p, f"{ch} (filtered)")
        st.plotly_chart(fig_psd, use_container_width=True, key="psd_main")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: JOINT ANGLES
# ══════════════════════════════════════════════════════════════════════════════

def page_joint_angles(meta: dict, mdf: pd.DataFrame):
    _section("📐", "Joint Angle Analysis", "#0D9488")

    if mdf.empty:
        st.warning("No marker data available.")
        return

    markers = meta["marker_labels"]
    fs = meta["point_rate"]

    st.markdown(
        """<div class="analysis-card">
          Compute joint angles from 3-point definitions (proximal → joint → distal).
          Select three markers to compute the angle at the middle (vertex) marker.
        </div>""",
        unsafe_allow_html=True,
    )

    filt_df = filter_marker_df(mdf, 6.0, fs)

    # Allow multiple joint definitions
    n_joints = st.number_input("Number of Joints to Analyze", 1, 6, 2, key="n_joints")

    angle_results = {}
    for j in range(int(n_joints)):
        with st.expander(f"Joint {j+1} Definition", expanded=(j==0)):
            col_j1, col_j2, col_j3 = st.columns(3)
            with col_j1:
                p1 = st.selectbox("Proximal Marker", markers,
                                   index=min(0, len(markers)-1), key=f"j{j}_p1")
            with col_j2:
                vtx = st.selectbox("Vertex (Joint Center)", markers,
                                    index=min(1, len(markers)-1), key=f"j{j}_vtx")
            with col_j3:
                p2 = st.selectbox("Distal Marker", markers,
                                   index=min(2, len(markers)-1), key=f"j{j}_p2")

        if p1 != vtx and vtx != p2 and p1 != p2:
            angles = compute_angle_3pt_safe(filt_df, p1, vtx, p2)
            angle_results[f"Joint {j+1}: {p1}–{vtx}–{p2}"] = angles

    if angle_results:
        import plotly.graph_objects as go
        fig_ang = go.Figure()
        for i, (name, ang) in enumerate(angle_results.items()):
            fig_ang.add_trace(go.Scatter(
                x=filt_df.index.values, y=ang,
                name=name, mode="lines",
                line=dict(color=PALETTE[i % len(PALETTE)], width=2),
                hovertemplate=f"<b>{name}</b>: %{{y:.1f}}°<extra></extra>",
            ))
        fig_ang.update_layout(
            height=380, title="Joint Angles Over Time",
            xaxis_title="Time (s)", yaxis_title="Angle (°)",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#FAFCFF",
            margin=dict(l=10,r=10,t=40,b=10),
            legend=dict(bgcolor="rgba(255,255,255,.9)"),
        )
        st.plotly_chart(fig_ang, use_container_width=True, key="joint_angles")

        # Statistics
        st.markdown("#### 📊 Joint Angle Statistics")
        ang_stats = []
        for name, ang in angle_results.items():
            valid = ang[~np.isnan(ang)]
            if len(valid) > 0:
                ang_stats.append({
                    "Joint":    name,
                    "Mean (°)": round(float(np.mean(valid)), 2),
                    "Std (°)":  round(float(np.std(valid)),  2),
                    "Min (°)":  round(float(np.min(valid)),  2),
                    "Max (°)":  round(float(np.max(valid)),  2),
                    "ROM (°)":  round(float(np.max(valid) - np.min(valid)), 2),
                })
        if ang_stats:
            st.dataframe(
                pd.DataFrame(ang_stats).set_index("Joint").style.format("{:.2f}"),
                use_container_width=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: EXPORT
# ══════════════════════════════════════════════════════════════════════════════

def page_export(meta: dict, mdf: pd.DataFrame, adf: pd.DataFrame, fps: list):
    _section("💾", "Export Data", "#64748B")

    st.markdown(
        """<div class="analysis-card">
          Export all analyzed data in various formats. Select the data types
          you need and download a single ZIP archive.
        </div>""",
        unsafe_allow_html=True,
    )

    col_e1, col_e2 = st.columns(2)
    with col_e1:
        export_meta    = st.checkbox("✅ File Metadata (JSON)", value=True)
        export_markers = st.checkbox("✅ Marker Trajectories (CSV)", value=True)
        export_analog  = st.checkbox("✅ Analog Channels (CSV)",  value=True)
    with col_e2:
        export_fp      = st.checkbox("✅ Force Plate Data (CSV)", value=True)
        export_stats   = st.checkbox("✅ Marker Statistics (CSV)", value=True)
        export_xlsx    = st.checkbox("✅ Summary Excel Report",    value=True)

    fs_m = meta["point_rate"]
    fs_a = meta["analog_rate"] or 1000.0

    if st.button("⬇️  Generate & Download ZIP", type="primary"):
        with st.spinner("Preparing export…"):
            buf = io.BytesIO()
            fn_base = Path(st.session_state.get("file_name","data")).stem

            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:

                if export_meta:
                    meta_json = json.dumps(
                        {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                         for k, v in meta.items()},
                        indent=2,
                    )
                    zf.writestr(f"{fn_base}_metadata.json", meta_json)

                if export_markers and not mdf.empty:
                    csv_m = mdf.to_csv()
                    zf.writestr(f"{fn_base}_markers.csv", csv_m)

                if export_analog and not adf.empty:
                    csv_a = adf.to_csv()
                    zf.writestr(f"{fn_base}_analog.csv", csv_a)

                if export_fp and fps:
                    for i, fp in enumerate(fps):
                        fp_rows = {"time_s": fp.get("time_s", [])}
                        for k in ["Fx","Fy","Fz","Mx","My","Mz","GRF","COPx","COPy"]:
                            if k in fp:
                                fp_rows[k] = fp[k]
                        if len(fp_rows) > 1:
                            fp_df_out = pd.DataFrame(fp_rows)
                            zf.writestr(
                                f"{fn_base}_forceplate_{i+1}.csv",
                                fp_df_out.to_csv(index=False),
                            )

                if export_stats and not mdf.empty:
                    stats = marker_stats(mdf)
                    zf.writestr(f"{fn_base}_marker_stats.csv", stats.to_csv())

                if export_xlsx:
                    try:
                        import openpyxl
                        xl_buf = io.BytesIO()
                        with pd.ExcelWriter(xl_buf, engine="openpyxl") as writer:
                            pd.DataFrame([meta]).T.to_excel(
                                writer, sheet_name="Metadata")
                            if not mdf.empty:
                                marker_stats(mdf).to_excel(
                                    writer, sheet_name="Marker Stats")
                            if not adf.empty and len(adf.columns) > 0:
                                adf.describe().to_excel(
                                    writer, sheet_name="Analog Summary")
                        zf.writestr(f"{fn_base}_report.xlsx", xl_buf.getvalue())
                    except ImportError:
                        pass

                # README
                readme = textwrap.dedent(f"""
                C3D Analyzer Export
                ===================
                File: {fn_base}.c3d
                Duration: {meta['duration_s']:.3f} s
                Markers: {meta['n_markers']}
                Analog channels: {meta['n_analog']}
                Force plates: {meta['n_force_plates']}

                Contents
                --------
                *_metadata.json   – File parameters and header info
                *_markers.csv     – 3-D marker trajectories (X, Y, Z per marker)
                *_analog.csv      – Analog channel data
                *_forceplate_N.csv– Force plate N data (Fx,Fy,Fz,Mx,My,Mz,COP)
                *_marker_stats.csv– Per-marker statistics
                *_report.xlsx     – Summary Excel workbook

                Generated by C3D Analyzer v1.0
                """).strip()
                zf.writestr("README.txt", readme)

            buf.seek(0)
            st.download_button(
                label="📦 Download ZIP Archive",
                data=buf.getvalue(),
                file_name=f"{fn_base}_c3d_analysis.zip",
                mime="application/zip",
            )
            st.success("✅ Export ready! Click the button above to download.")


# ══════════════════════════════════════════════════════════════════════════════
# WELCOME / NO FILE
# ══════════════════════════════════════════════════════════════════════════════

def page_welcome():
    st.markdown(
        """
        <div style="text-align:center;padding:4rem 2rem;">
          <div style="font-size:5rem;line-height:1;margin-bottom:1rem;">🦴</div>
          <h1 style="font-size:2.5rem;font-weight:800;letter-spacing:-0.03em;
                     color:#1E293B;margin-bottom:0.5rem;">
            C3D Analyzer
          </h1>
          <p style="font-size:1.1rem;color:#64748B;max-width:500px;
                    margin:0 auto 2rem;">
            Professional biomechanics analysis platform.<br>
            Upload a <code style="background:#EEF2FF;padding:2px 6px;
            border-radius:4px;color:#4F46E5;">.c3d</code> file to begin.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_f1, col_f2, col_f3 = st.columns(3)
    features = [
        ("📍", "Marker Analysis",
         "3-D trajectories, velocity, acceleration, gap detection, ROM"),
        ("⚡", "Force Plates",
         "GRF components, COP, stance phases, loading rate, impulse"),
        ("🦾", "EMG Processing",
         "Bandpass filter, rectification, linear envelope, PSD"),
        ("🚶", "Gait Analysis",
         "HS/TO detection, stride time, cadence, stance phase %"),
        ("🔬", "Signal Processing",
         "Butterworth, notch, band-pass filters with spectral analysis"),
        ("💾", "Data Export",
         "CSV, Excel report, JSON metadata — all in one ZIP"),
    ]
    for i, (icon, title, desc) in enumerate(features):
        col = [col_f1, col_f2, col_f3][i % 3]
        with col:
            st.markdown(
                f"""<div class="analysis-card" style="text-align:center;">
                  <div style="font-size:2rem;">{icon}</div>
                  <div style="font-weight:700;color:#1E293B;margin:6px 0 4px;">
                    {title}</div>
                  <div style="font-size:0.82rem;color:#64748B;">{desc}</div>
                </div>""",
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    missing = _check_deps()
    if missing:
        st.error(f"Missing dependencies: {', '.join(missing)}\n\n"
                 f"Run: `pip install {' '.join(missing)}`")
        st.stop()

    if not UTILS_OK:
        st.error(f"Could not load utilities: {UTILS_ERR}")
        st.stop()

    active_page = render_sidebar()
    file_bytes  = st.session_state.get("file_bytes")

    if not file_bytes:
        page_welcome()
        return

    # Load data
    try:
        meta, mdf, adf, fps = _load_all(file_bytes)
    except Exception as e:
        st.error(f"❌ Failed to parse C3D file: {e}")
        st.exception(e)
        return

    # Route pages
    if active_page == "Overview":
        page_overview(meta, mdf, adf, fps)
    elif active_page == "Markers":
        page_markers(meta, mdf)
    elif active_page == "Analog Signals":
        page_analog(meta, adf)
    elif active_page == "Force Plates":
        page_force_plates(meta, adf, fps)
    elif active_page == "Gait Analysis":
        page_gait(meta, mdf, adf, fps)
    elif active_page == "Signal Processing":
        page_signal_processing(meta, adf, mdf)
    elif active_page == "Joint Angles":
        page_joint_angles(meta, mdf)
    elif active_page == "Export":
        page_export(meta, mdf, adf, fps)


if __name__ == "__main__":
    main()

