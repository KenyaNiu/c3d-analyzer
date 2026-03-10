"""
marker_analysis.py
──────────────────
Kinematic computations on marker trajectory data.
"""

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks
from scipy.spatial.transform import Rotation


# ── Filtering ─────────────────────────────────────────────────────────────────

def butterworth_filter(data: np.ndarray, cutoff: float, fs: float,
                       order: int = 4, btype: str = "low") -> np.ndarray:
    """Zero-phase Butterworth filter (NaN-safe)."""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    if normal_cutoff >= 1.0:
        return data
    b, a = butter(order, normal_cutoff, btype=btype, analog=False)
    # Handle NaN by linear interpolation before filtering
    out = data.copy().astype(float)
    mask = np.isnan(out)
    if mask.all():
        return out
    idx = np.arange(len(out))
    out[mask] = np.interp(idx[mask], idx[~mask], out[~mask])
    out = filtfilt(b, a, out)
    out[mask] = np.nan
    return out


def filter_marker_df(df: pd.DataFrame, cutoff: float, fs: float,
                     order: int = 4) -> pd.DataFrame:
    """Apply Butterworth low-pass to every X/Y/Z column in marker DataFrame."""
    filtered = df.copy()
    for col in df.columns:
        if col[1] in ("X", "Y", "Z"):
            filtered[col] = butterworth_filter(df[col].values, cutoff, fs, order)
    return filtered


# ── Kinematics ────────────────────────────────────────────────────────────────

def compute_velocity(df: pd.DataFrame, fs: float) -> pd.DataFrame:
    """Central-difference velocity [unit/s] for each X/Y/Z channel."""
    vel = {}
    for col in df.columns:
        if col[1] in ("X", "Y", "Z"):
            v = np.gradient(df[col].values, 1.0 / fs)
            vel[(col[0], "V" + col[1])] = v
    return pd.DataFrame(vel, index=df.index)


def compute_acceleration(vel_df: pd.DataFrame, fs: float) -> pd.DataFrame:
    """Acceleration from velocity DataFrame."""
    acc = {}
    for col in vel_df.columns:
        a = np.gradient(vel_df[col].values, 1.0 / fs)
        axis = col[1].replace("V", "A")
        acc[(col[0], axis)] = a
    return pd.DataFrame(acc, index=vel_df.index)


def compute_speed(vel_df: pd.DataFrame, markers: list[str]) -> pd.DataFrame:
    """Resultant speed ‖v‖ for each marker."""
    speed = {}
    for m in markers:
        try:
            vx = vel_df[(m, "VX")].values
            vy = vel_df[(m, "VY")].values
            vz = vel_df[(m, "VZ")].values
            speed[m] = np.sqrt(vx**2 + vy**2 + vz**2)
        except KeyError:
            pass
    return pd.DataFrame(speed, index=vel_df.index)


def compute_distance_between(df: pd.DataFrame, m1: str, m2: str) -> np.ndarray:
    """Euclidean distance between two markers over time."""
    dx = df[(m1, "X")] - df[(m2, "X")]
    dy = df[(m1, "Y")] - df[(m2, "Y")]
    dz = df[(m1, "Z")] - df[(m2, "Z")]
    return np.sqrt(dx**2 + dy**2 + dz**2).values


def compute_angle_3pt(df: pd.DataFrame,
                      p1: str, vertex: str, p2: str) -> np.ndarray:
    """
    Joint angle (degrees) at `vertex` between vectors vertex→p1 and vertex→p2.
    """
    v1 = df[[p1]].loc[:, [(p1,"X"),(p1,"Y"),(p1,"Z")]].values - \
         df[[vertex]].loc[:, [(vertex,"X"),(vertex,"Y"),(vertex,"Z")]].values
    v2 = df[[p2]].loc[:, [(p2,"X"),(p2,"Y"),(p2,"Z")]].values - \
         df[[vertex]].loc[:, [(vertex,"X"),(vertex,"Y"),(vertex,"Z")]].values
    # Normalise
    n1 = np.linalg.norm(v1, axis=1, keepdims=True)
    n2 = np.linalg.norm(v2, axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        cos_a = np.clip(np.sum((v1/n1) * (v2/n2), axis=1), -1, 1)
    return np.degrees(np.arccos(cos_a))


def compute_angle_3pt_safe(df: pd.DataFrame,
                            p1: str, vertex: str, p2: str) -> np.ndarray:
    """Safe wrapper for compute_angle_3pt using column tuples."""
    axes = ["X","Y","Z"]
    try:
        v1 = np.column_stack([df[(p1, a)] - df[(vertex, a)] for a in axes])
        v2 = np.column_stack([df[(p2, a)] - df[(vertex, a)] for a in axes])
    except KeyError:
        return np.full(len(df), np.nan)
    n1 = np.linalg.norm(v1, axis=1, keepdims=True)
    n2 = np.linalg.norm(v2, axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        cos_a = np.clip(np.sum((v1 / (n1 + 1e-12)) * (v2 / (n2 + 1e-12)), axis=1), -1, 1)
    return np.degrees(np.arccos(cos_a))


# ── Summary statistics ────────────────────────────────────────────────────────

def marker_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Per-marker statistics: range, mean, std, gap_count."""
    rows = []
    markers = df.columns.get_level_values(0).unique().tolist()
    for m in markers:
        x = df[(m, "X")]; y = df[(m, "Y")]; z = df[(m, "Z")]
        valid = (~np.isnan(x)) & (~np.isnan(y)) & (~np.isnan(z))
        n_valid = int(valid.sum())
        n_total = len(x)
        gap_pct = 100.0 * (1 - n_valid / n_total) if n_total else 0
        # RMSE from mean trajectory
        rows.append({
            "Marker":       m,
            "X_mean_mm":    float(np.nanmean(x)),
            "Y_mean_mm":    float(np.nanmean(y)),
            "Z_mean_mm":    float(np.nanmean(z)),
            "X_range_mm":   float(np.nanmax(x) - np.nanmin(x)),
            "Y_range_mm":   float(np.nanmax(y) - np.nanmin(y)),
            "Z_range_mm":   float(np.nanmax(z) - np.nanmin(z)),
            "valid_frames": n_valid,
            "total_frames": n_total,
            "gap_%":        round(gap_pct, 2),
        })
    return pd.DataFrame(rows).set_index("Marker")


def detect_gaps(df: pd.DataFrame, marker: str) -> list[dict]:
    """Find continuous NaN gaps for a single marker."""
    x = df[(marker, "X")].values
    nan_mask = np.isnan(x)
    gaps = []
    in_gap = False
    start = 0
    times = df.index.values
    for i, v in enumerate(nan_mask):
        if v and not in_gap:
            in_gap = True; start = i
        elif not v and in_gap:
            in_gap = False
            gaps.append({
                "start_frame": start,
                "end_frame":   i - 1,
                "start_s":     times[start],
                "end_s":       times[i - 1],
                "duration_s":  times[i - 1] - times[start],
                "n_frames":    i - start,
            })
    if in_gap:
        gaps.append({
            "start_frame": start,
            "end_frame":   len(x) - 1,
            "start_s":     times[start],
            "end_s":       times[-1],
            "duration_s":  times[-1] - times[start],
            "n_frames":    len(x) - start,
        })
    return gaps
