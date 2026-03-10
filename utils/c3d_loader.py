"""
c3d_loader.py
─────────────
Core loader and caching layer for C3D files using ezc3d.
"""

import io
import numpy as np
import pandas as pd
import streamlit as st

try:
    import ezc3d
    EZC3D_AVAILABLE = True
except ImportError:
    EZC3D_AVAILABLE = False


# ── Public API ────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_c3d(file_bytes: bytes):
    """Load a C3D file from raw bytes and return the ezc3d object."""
    if not EZC3D_AVAILABLE:
        raise ImportError("ezc3d is not installed. Run: pip install ezc3d")
    tmp = io.BytesIO(file_bytes)
    # ezc3d needs a file path – write to a temp file
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".c3d", delete=False) as f:
        f.write(file_bytes)
        tmp_path = f.name
    try:
        c3d = ezc3d.c3d(tmp_path)
    finally:
        os.unlink(tmp_path)
    return c3d


def get_file_metadata(c3d) -> dict:
    """Extract header and parameter metadata."""
    header = c3d["header"]
    params = c3d["parameters"]

    # Safe key extraction helpers
    def pget(group, key, default=None):
        try:
            return params[group][key]["value"]
        except (KeyError, TypeError):
            return default

    # Subjects / trial info
    subjects = pget("SUBJECTS", "NAMES", [])
    trial    = pget("TRIAL",    "ACTUAL_START_FIELD", [None])[0]

    # Frame info
    n_frames   = header["nb_frames"]
    first_frame= header["first_frame"]
    last_frame = header["last_frame"]
    frame_rate = header["frame_rate"]
    duration   = n_frames / frame_rate if frame_rate else 0

    # Markers
    marker_labels = pget("POINT", "LABELS",  [])
    n_markers     = pget("POINT", "USED",    [0])[0]
    point_rate    = pget("POINT", "RATE",    [frame_rate])[0]
    point_unit    = pget("POINT", "UNITS",   ["mm"])[0] if pget("POINT", "UNITS") else "mm"

    # Analog
    analog_labels = pget("ANALOG", "LABELS", [])
    n_analog      = pget("ANALOG", "USED",   [0])[0]
    analog_rate   = pget("ANALOG", "RATE",   [0])[0]
    analog_units  = pget("ANALOG", "UNITS",  [])

    # Force plates
    fp_used = pget("FORCE_PLATFORM", "USED", [0])
    n_fp    = int(fp_used[0]) if fp_used else 0
    fp_type = pget("FORCE_PLATFORM", "TYPE", [])
    fp_corners = pget("FORCE_PLATFORM", "CORNERS", None)

    # Manufacturer
    manufacturer = pget("MANUFACTURER", "COMPANY",  ["Unknown"])[0] if pget("MANUFACTURER", "COMPANY") else "Unknown"
    software     = pget("MANUFACTURER", "SOFTWARE", ["Unknown"])[0] if pget("MANUFACTURER", "SOFTWARE") else "Unknown"

    return {
        "n_frames":       int(n_frames),
        "first_frame":    int(first_frame),
        "last_frame":     int(last_frame),
        "frame_rate":     float(frame_rate),
        "duration_s":     float(duration),
        "n_markers":      int(n_markers),
        "point_rate":     float(point_rate),
        "point_unit":     str(point_unit).strip(),
        "marker_labels":  [str(l).strip() for l in marker_labels if str(l).strip()],
        "n_analog":       int(n_analog),
        "analog_rate":    float(analog_rate) if analog_rate else 0.0,
        "analog_labels":  [str(l).strip() for l in analog_labels if str(l).strip()],
        "analog_units":   [str(u).strip() for u in analog_units],
        "n_force_plates": n_fp,
        "fp_type":        list(fp_type),
        "fp_corners":     fp_corners,
        "subjects":       [str(s).strip() for s in subjects],
        "manufacturer":   str(manufacturer),
        "software":       str(software),
    }


def get_marker_data(c3d) -> pd.DataFrame:
    """
    Return marker 3-D positions as a DataFrame.
    Shape: (n_frames,) index, columns = MultiIndex (marker, axis)
    """
    pts = c3d["data"]["points"]          # shape (4, n_markers, n_frames)
    params = c3d["parameters"]
    try:
        labels = [str(l).strip() for l in params["POINT"]["LABELS"]["value"]]
    except (KeyError, TypeError):
        labels = [f"M{i}" for i in range(pts.shape[1])]

    n_frames  = pts.shape[2]
    n_markers = pts.shape[1]
    # Trim labels to actual markers
    labels = labels[:n_markers]

    frame_rate = float(c3d["header"]["frame_rate"])
    times = np.arange(n_frames) / frame_rate

    cols = pd.MultiIndex.from_product([labels, ["X", "Y", "Z", "Res"]],
                                      names=["marker", "axis"])
    data = {}
    for i, lbl in enumerate(labels):
        data[(lbl, "X")]   = pts[0, i, :]
        data[(lbl, "Y")]   = pts[1, i, :]
        data[(lbl, "Z")]   = pts[2, i, :]
        data[(lbl, "Res")] = pts[3, i, :]   # residual (quality)

    df = pd.DataFrame(data, index=times)
    df.index.name = "time_s"
    # Replace invalid (residual == -1) with NaN
    for lbl in labels:
        mask = df[(lbl, "Res")] < 0
        df.loc[mask, (lbl, "X")] = np.nan
        df.loc[mask, (lbl, "Y")] = np.nan
        df.loc[mask, (lbl, "Z")] = np.nan
    return df


def get_analog_data(c3d) -> pd.DataFrame:
    """
    Return analog signals as a DataFrame.
    Shape: (n_analog_frames,) index, columns = channel labels
    """
    analog = c3d["data"]["analogs"]      # shape (1, n_channels, n_analog_frames)
    params = c3d["parameters"]
    try:
        labels = [str(l).strip() for l in params["ANALOG"]["LABELS"]["value"]]
    except (KeyError, TypeError):
        labels = [f"A{i}" for i in range(analog.shape[1])]

    n_ch     = analog.shape[1]
    n_af     = analog.shape[2]
    labels   = labels[:n_ch]

    try:
        analog_rate = float(params["ANALOG"]["RATE"]["value"][0])
    except (KeyError, TypeError, IndexError):
        analog_rate = float(c3d["header"]["frame_rate"])

    times = np.arange(n_af) / analog_rate
    df = pd.DataFrame(analog[0, :n_ch, :].T, index=times, columns=labels)
    df.index.name = "time_s"
    return df


def get_force_plate_data(c3d, metadata: dict) -> list[dict]:
    """
    Extract per-force-plate data (Fx, Fy, Fz, Mx, My, Mz, COPx, COPy).
    Returns a list of dicts, one per plate.
    """
    n_fp = metadata["n_force_plates"]
    if n_fp == 0:
        return []

    analog_df   = get_analog_data(c3d)
    analog_rate = metadata["analog_rate"]

    params = c3d["parameters"]

    def pget(group, key, default=None):
        try:
            return params[group][key]["value"]
        except (KeyError, TypeError):
            return default

    ch_per_plate = pget("FORCE_PLATFORM", "CHANNEL", None)
    corners      = pget("FORCE_PLATFORM", "CORNERS", None)  # (3, 4, n_fp)
    origin       = pget("FORCE_PLATFORM", "ORIGIN",  None)  # (3, n_fp)

    plates = []
    for p in range(n_fp):
        result = {"plate_index": p}
        result["corners"] = corners[:, :, p] if corners is not None else None
        result["origin"]  = origin[:, p]     if origin  is not None else None

        # Channel mapping – typically 6 or 8 channels per plate (0-based idx in c3d)
        if ch_per_plate is not None:
            chs = np.array(ch_per_plate).flatten()
            # Reshape by n_fp plates
            try:
                chs = chs.reshape(n_fp, -1)[p]
                ch_names = [analog_df.columns[int(c) - 1] for c in chs if 0 < int(c) <= len(analog_df.columns)]
                sub = analog_df[ch_names].copy()
                sub.columns = [f"Ch{i+1}" for i in range(len(ch_names))]
            except Exception:
                sub = pd.DataFrame()
        else:
            sub = pd.DataFrame()

        # Try to map Fx,Fy,Fz,Mx,My,Mz from known patterns
        fp_dict = _map_force_channels(sub, analog_df, p, params)
        result.update(fp_dict)
        plates.append(result)

    return plates


# ── Private helpers ───────────────────────────────────────────────────────────

def _map_force_channels(sub: pd.DataFrame, analog_df: pd.DataFrame,
                        plate_idx: int, params) -> dict:
    """Attempt to identify Fx/Fy/Fz/Mx/My/Mz columns."""
    result = {}
    n_ch = len(sub.columns)

    if n_ch >= 6:
        result["Fx"] = sub.iloc[:, 0].values
        result["Fy"] = sub.iloc[:, 1].values
        result["Fz"] = sub.iloc[:, 2].values
        result["Mx"] = sub.iloc[:, 3].values
        result["My"] = sub.iloc[:, 4].values
        result["Mz"] = sub.iloc[:, 5].values
        result["time_s"] = sub.index.values

        # Compute COP (centre of pressure)
        Fx = result["Fx"]; Fy = result["Fy"]; Fz = result["Fz"]
        Mx = result["Mx"]; My = result["My"]
        with np.errstate(divide="ignore", invalid="ignore"):
            cop_x = np.where(np.abs(Fz) > 10, -My / Fz, np.nan)
            cop_y = np.where(np.abs(Fz) > 10,  Mx / Fz, np.nan)
        result["COPx"] = cop_x
        result["COPy"] = cop_y

        # Resultant GRF
        result["GRF"] = np.sqrt(Fx**2 + Fy**2 + Fz**2)
    else:
        result["time_s"] = sub.index.values if len(sub) else np.array([])

    return result
