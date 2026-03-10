"""
visualization.py
────────────────
Plotly figure builders for the C3D Analyzer.
All figures use a consistent light theme.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Palette / theme ───────────────────────────────────────────────────────────

PALETTE = [
    "#0EA5E9", "#6366F1", "#10B981", "#F59E0B",
    "#F43F5E", "#8B5CF6", "#0D9488", "#EC4899",
    "#3B82F6", "#14B8A6", "#A855F7", "#EF4444",
]

LAYOUT_DEFAULTS = dict(
    font=dict(family="DM Sans, sans-serif", size=12, color="#1E293B"),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#FAFCFF",
    margin=dict(l=10, r=10, t=40, b=10),
    legend=dict(
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="#E2E8F0",
        borderwidth=1,
        font=dict(size=11),
    ),
    hoverlabel=dict(
        bgcolor="white",
        bordercolor="#CBD5E1",
        font_size=12,
        font_family="DM Sans, sans-serif",
    ),
)

AXIS_STYLE = dict(
    gridcolor="#EEF2F7",
    linecolor="#CBD5E1",
    zerolinecolor="#CBD5E1",
    showgrid=True,
)


def _apply_layout(fig: go.Figure, title: str = "", height: int = 400) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, weight="bold"), x=0.01),
        height=height,
        **LAYOUT_DEFAULTS,
    )
    fig.update_xaxes(**AXIS_STYLE)
    fig.update_yaxes(**AXIS_STYLE)
    return fig


# ── Marker Figures ────────────────────────────────────────────────────────────

def plot_marker_trajectories(df: pd.DataFrame, marker: str,
                              title: str = None) -> go.Figure:
    """X/Y/Z trajectory of a single marker over time."""
    t = df.index.values
    fig = go.Figure()
    for i, axis in enumerate(["X", "Y", "Z"]):
        col = (marker, axis)
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=t, y=df[col].values,
                name=f"{axis}-axis",
                line=dict(color=PALETTE[i], width=2),
                mode="lines",
                hovertemplate=f"<b>{axis}</b>: %{{y:.2f}} mm<br>t=%{{x:.3f}} s<extra></extra>",
            ))
    return _apply_layout(fig, title or f"Marker: {marker} — XYZ Trajectory", 380)


def plot_marker_speed(speed_df: pd.DataFrame, markers: list[str],
                      title: str = "Marker Speed") -> go.Figure:
    fig = go.Figure()
    for i, m in enumerate(markers):
        if m in speed_df.columns:
            fig.add_trace(go.Scatter(
                x=speed_df.index.values,
                y=speed_df[m].values,
                name=m,
                line=dict(color=PALETTE[i % len(PALETTE)], width=1.5),
                mode="lines",
                hovertemplate=f"<b>{m}</b>: %{{y:.1f}} mm/s<extra></extra>",
            ))
    fig.update_layout(yaxis_title="Speed (mm/s)", xaxis_title="Time (s)")
    return _apply_layout(fig, title, 360)


def plot_3d_trajectories(df: pd.DataFrame, markers: list[str]) -> go.Figure:
    """3-D scatter plot of marker trajectories."""
    fig = go.Figure()
    for i, m in enumerate(markers):
        try:
            x = df[(m, "X")]; y = df[(m, "Y")]; z = df[(m, "Z")]
        except KeyError:
            continue
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            name=m,
            mode="lines",
            line=dict(color=PALETTE[i % len(PALETTE)], width=2),
            hovertemplate=f"<b>{m}</b><br>X:%{{x:.1f}} Y:%{{y:.1f}} Z:%{{z:.1f}}<extra></extra>",
        ))
    fig.update_layout(
        height=550,
        scene=dict(
            xaxis=dict(backgroundcolor="#F8FAFF", gridcolor="#DDE4EF",
                       title="X (mm)"),
            yaxis=dict(backgroundcolor="#F8FAFF", gridcolor="#DDE4EF",
                       title="Y (mm)"),
            zaxis=dict(backgroundcolor="#F8FAFF", gridcolor="#DDE4EF",
                       title="Z (mm)"),
        ),
        **{k: v for k, v in LAYOUT_DEFAULTS.items() if k != "plot_bgcolor"},
    )
    return fig


def plot_stick_figure(df: pd.DataFrame, frame_idx: int,
                      segments: list[tuple] = None) -> go.Figure:
    """Single-frame 3-D stick figure."""
    markers = df.columns.get_level_values(0).unique().tolist()
    xs, ys, zs, lbls = [], [], [], []
    for m in markers:
        try:
            x = float(df.iloc[frame_idx][(m, "X")])
            y = float(df.iloc[frame_idx][(m, "Y")])
            z = float(df.iloc[frame_idx][(m, "Z")])
        except (KeyError, ValueError):
            continue
        if not any(np.isnan([x, y, z])):
            xs.append(x); ys.append(y); zs.append(z); lbls.append(m)

    fig = go.Figure()
    # Marker points
    fig.add_trace(go.Scatter3d(
        x=xs, y=ys, z=zs, text=lbls,
        mode="markers+text",
        textposition="top center",
        textfont=dict(size=8),
        marker=dict(size=5, color="#0EA5E9",
                    line=dict(color="#0369A1", width=1)),
        name="Markers",
    ))
    # Segments
    if segments:
        for m1, m2 in segments:
            if m1 in lbls and m2 in lbls:
                i1, i2 = lbls.index(m1), lbls.index(m2)
                fig.add_trace(go.Scatter3d(
                    x=[xs[i1], xs[i2]], y=[ys[i1], ys[i2]],
                    z=[zs[i1], zs[i2]],
                    mode="lines",
                    line=dict(color="#94A3B8", width=3),
                    showlegend=False,
                    hoverinfo="skip",
                ))
    fig.update_layout(
        height=500,
        scene=dict(
            xaxis=dict(backgroundcolor="#F8FAFF", gridcolor="#DDE4EF",
                       title="X (mm)"),
            yaxis=dict(backgroundcolor="#F8FAFF", gridcolor="#DDE4EF",
                       title="Y (mm)"),
            zaxis=dict(backgroundcolor="#F8FAFF", gridcolor="#DDE4EF",
                       title="Z (mm)"),
            aspectmode="data",
        ),
        **{k: v for k, v in LAYOUT_DEFAULTS.items() if k != "plot_bgcolor"},
    )
    return fig


def plot_marker_heatmap(stats_df: pd.DataFrame) -> go.Figure:
    """Heatmap of gap percentage per marker."""
    if "gap_%" not in stats_df.columns:
        return go.Figure()
    vals = stats_df["gap_%"].values.reshape(1, -1)
    fig = go.Figure(go.Heatmap(
        z=vals,
        x=stats_df.index.tolist(),
        y=["Gap %"],
        colorscale=[[0, "#D1FAE5"], [0.5, "#FEF3C7"], [1, "#FEE2E2"]],
        zmin=0, zmax=max(100, float(np.max(vals))),
        text=vals.round(1),
        texttemplate="%{text}%",
        hovertemplate="<b>%{x}</b><br>Gap: %{z:.1f}%<extra></extra>",
        showscale=True,
    ))
    fig.update_layout(height=120, margin=dict(l=10,r=10,t=20,b=10),
                      **{k:v for k,v in LAYOUT_DEFAULTS.items()
                         if k not in ("plot_bgcolor","paper_bgcolor")},
                      plot_bgcolor="rgba(0,0,0,0)",
                      paper_bgcolor="rgba(0,0,0,0)")
    return fig


# ── Analog / Force Figures ────────────────────────────────────────────────────

def plot_analog_channels(analog_df: pd.DataFrame, channels: list[str],
                          title: str = "Analog Channels") -> go.Figure:
    fig = go.Figure()
    for i, ch in enumerate(channels):
        if ch in analog_df.columns:
            fig.add_trace(go.Scatter(
                x=analog_df.index.values,
                y=analog_df[ch].values,
                name=ch,
                line=dict(color=PALETTE[i % len(PALETTE)], width=1.2),
                mode="lines",
                hovertemplate=f"<b>{ch}</b>: %{{y:.3f}}<br>t=%{{x:.4f}} s<extra></extra>",
            ))
    fig.update_layout(xaxis_title="Time (s)")
    return _apply_layout(fig, title, 380)


def plot_grf(time: np.ndarray, fp_data: dict, plate_idx: int = 0) -> go.Figure:
    """Ground Reaction Force components."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=["GRF Components (N)", "Resultant GRF (N)"],
                        vertical_spacing=0.12)
    t = fp_data.get("time_s", time)
    for comp, color, name in [
        ("Fx", PALETTE[3], "Fx (A/P)"),
        ("Fy", PALETTE[2], "Fy (M/L)"),
        ("Fz", PALETTE[0], "Fz (Vertical)"),
    ]:
        if comp in fp_data:
            fig.add_trace(go.Scatter(
                x=t, y=fp_data[comp], name=name,
                line=dict(color=color, width=2), mode="lines",
            ), row=1, col=1)
    if "GRF" in fp_data:
        fig.add_trace(go.Scatter(
            x=t, y=fp_data["GRF"], name="Resultant",
            line=dict(color=PALETTE[1], width=2.5), mode="lines",
            fill="tozeroy", fillcolor="rgba(99,102,241,0.1)",
        ), row=2, col=1)
    fig.update_layout(height=450, **LAYOUT_DEFAULTS)
    fig.update_xaxes(**AXIS_STYLE); fig.update_yaxes(**AXIS_STYLE)
    return fig


def plot_cop(cop_x: np.ndarray, cop_y: np.ndarray,
             corners=None) -> go.Figure:
    """COP trajectory plot."""
    fig = go.Figure()
    # Colour by time
    n = len(cop_x)
    colors = np.arange(n)
    mask = ~(np.isnan(cop_x) | np.isnan(cop_y))
    fig.add_trace(go.Scatter(
        x=cop_x[mask], y=cop_y[mask],
        mode="lines+markers",
        marker=dict(
            size=4,
            color=colors[mask],
            colorscale="Blues",
            showscale=True,
            colorbar=dict(title="Frame", thickness=12, len=0.7),
        ),
        line=dict(color="#94A3B8", width=1),
        name="COP Path",
        hovertemplate="COPx:%{x:.1f} COPy:%{y:.1f}<extra></extra>",
    ))
    # Force plate outline
    if corners is not None:
        try:
            c = np.array(corners)
            cx = list(c[0, :]) + [c[0, 0]]
            cy = list(c[1, :]) + [c[1, 0]]
            fig.add_trace(go.Scatter(
                x=cx, y=cy,
                mode="lines",
                line=dict(color="#CBD5E1", width=2, dash="dash"),
                name="Plate Boundary",
            ))
        except Exception:
            pass
    fig.update_layout(
        xaxis_title="COP-X (mm)", yaxis_title="COP-Y (mm)",
        yaxis_scaleanchor="x",
    )
    return _apply_layout(fig, "Centre of Pressure (COP) Trajectory", 450)


def plot_emg(time: np.ndarray, emg_data: dict, channel: str) -> go.Figure:
    """EMG processing result figure."""
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        subplot_titles=["Raw EMG", "Rectified", "Envelope / RMS"],
                        vertical_spacing=0.1)
    for row, key, color, name in [
        (1, "raw",       "#94A3B8", "Raw"),
        (2, "rectified", "#0EA5E9", "Rectified"),
        (3, "envelope",  "#10B981", "Envelope"),
    ]:
        if key in emg_data:
            fig.add_trace(go.Scatter(
                x=time, y=emg_data[key], name=name,
                line=dict(color=color, width=1.5), mode="lines",
            ), row=row, col=1)
    if "rms" in emg_data:
        fig.add_trace(go.Scatter(
            x=time, y=emg_data["rms"], name="RMS",
            line=dict(color="#F59E0B", width=2, dash="dot"),
        ), row=3, col=1)
    fig.update_layout(height=520, **LAYOUT_DEFAULTS, title_text=f"EMG: {channel}")
    fig.update_xaxes(**AXIS_STYLE); fig.update_yaxes(**AXIS_STYLE)
    return fig


def plot_psd(freqs: np.ndarray, psd: np.ndarray, channel: str) -> go.Figure:
    """Power spectral density."""
    fig = go.Figure(go.Scatter(
        x=freqs, y=10 * np.log10(psd + 1e-12),
        mode="lines",
        line=dict(color=PALETTE[1], width=2),
        fill="tozeroy",
        fillcolor="rgba(99,102,241,0.1)",
        hovertemplate="f: %{x:.1f} Hz<br>PSD: %{y:.1f} dB<extra></extra>",
    ))
    fig.update_layout(xaxis_title="Frequency (Hz)", yaxis_title="PSD (dB)")
    return _apply_layout(fig, f"Power Spectral Density — {channel}", 350)


def plot_stance_phases(time: np.ndarray, fz: np.ndarray,
                        phases: list[dict]) -> go.Figure:
    """Fz with stance phase shading."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time, y=fz, mode="lines",
        line=dict(color=PALETTE[0], width=2),
        name="Fz",
    ))
    for ph in phases:
        fig.add_vrect(
            x0=ph["start_s"], x1=ph["end_s"],
            fillcolor="rgba(16,185,129,0.15)",
            layer="below", line_width=0,
            annotation_text="Stance", annotation_position="top left",
            annotation_font_size=9,
        )
    fig.update_layout(xaxis_title="Time (s)", yaxis_title="Fz (N)")
    return _apply_layout(fig, "Vertical GRF with Stance Phases", 380)


# ── Summary ───────────────────────────────────────────────────────────────────

def plot_marker_range_bar(stats_df: pd.DataFrame) -> go.Figure:
    """Bar chart of marker range in X/Y/Z."""
    markers = stats_df.index.tolist()
    fig = go.Figure()
    for i, (col, label) in enumerate([
        ("X_range_mm", "X Range"),
        ("Y_range_mm", "Y Range"),
        ("Z_range_mm", "Z Range"),
    ]):
        if col in stats_df.columns:
            fig.add_trace(go.Bar(
                name=label, x=markers, y=stats_df[col],
                marker_color=PALETTE[i],
            ))
    fig.update_layout(barmode="group", xaxis_tickangle=-45,
                      yaxis_title="Range (mm)", xaxis_title="Marker")
    return _apply_layout(fig, "Marker Range of Motion", 420)


def plot_correlation_matrix(df: pd.DataFrame, labels: list[str]) -> go.Figure:
    """Pearson correlation heatmap for analog channels."""
    corr = df[labels].corr().values
    fig = go.Figure(go.Heatmap(
        z=corr,
        x=labels, y=labels,
        colorscale="RdBu",
        zmid=0,
        zmin=-1, zmax=1,
        text=np.round(corr, 2),
        texttemplate="%{text}",
        hovertemplate="%{x} × %{y}: %{z:.3f}<extra></extra>",
    ))
    fig.update_layout(height=max(350, 30 * len(labels)))
    return _apply_layout(fig, "Analog Channel Correlation Matrix",
                         max(350, 30 * len(labels)))
