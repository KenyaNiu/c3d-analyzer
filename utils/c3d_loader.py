"""
c3d_loader.py
─────────────
Core loader and caching layer for C3D files.
Uses the pure-Python `c3d` package (no compilation needed, Python 3.14 compatible).
Outputs the same data structure as the previous ezc3d-based version so all
other modules (marker_analysis, analog_analysis, visualization) work unchanged.
"""

import io
import numpy as np
import pandas as pd
import streamlit as st

try:
    import c3d as c3d_lib
    C3D_AVAILABLE = True
except ImportError:
    C3D_AVAILABLE = False


# ── Internal reader ───────────────────────────────────────────────────────────

def _read_c3d(file_bytes: bytes) -> dict:
    """
    Parse a C3D file and return a dict that mimics the ezc3d structure:

    {
        "header":     { nb_frames, first_frame, last_frame, frame_rate },
        "parameters": { GROUP: { KEY: {"value": ...} } },
        "data": {
            "points":  ndarray (4, n_markers, n_frames)   – X Y Z residual
            "analogs": ndarray (1, n_channels, n_analog_frames)
        }
    }
    """
    reader = c3d_lib.Reader(io.BytesIO(file_bytes))
    hdr = reader.header

    # ── Parameter helpers ─────────────────────────────────────────────────────
    def _grp(name):
        return reader.groups.get(name.upper())

    def _par(group_name, param_name):
        g = _grp(group_name)
        if g is None:
            return None
        return g.params.get(param_name.upper())

    def _strings(group_name, param_name):
        p = _par(group_name, param_name)
        if p is None:
            return []
        try:
            return [s.strip() for s in p.string_array]
        except Exception:
            return []

    def _floats(group_name, param_name):
        p = _par(group_name, param_name)
        if p is None:
            return []
        for attr in ("float_array", "int16_array", "uint16_array"):
            try:
                a = getattr(p, attr, None)
                if a is not None and a.size:
                    return list(a.flatten().astype(float))
            except Exception:
                pass
        return []

    def _raw_array(group_name, param_name):
        p = _par(group_name, param_name)
        if p is None:
            return None
        for attr in ("float_array", "int16_array", "uint16_array", "int8_array"):
            try:
                a = getattr(p, attr, None)
                if a is not None and a.size:
                    return a.astype(float)
            except Exception:
                pass
        return None

    # ── Frame rate & header ───────────────────────────────────────────────────
    frame_rate  = float(hdr.frame_rate) if hdr.frame_rate else 100.0
    first_frame = int(hdr.first_frame)
    last_frame  = int(hdr.last_frame)

    # ── Read frames ───────────────────────────────────────────────────────────
    all_points  = []
    all_analogs = []
    for _i, points, analog in reader.read_frames():
        all_points.append(points)   # (n_markers, 5): x,y,z,residual,camera
        all_analogs.append(analog)  # (n_channels, samples_per_frame)

    n_frames = len(all_points)

    if n_frames > 0:
        pts_stack = np.stack(all_points, axis=0)    # (n_frames, n_markers, 5)
        pts_out = np.stack([
            pts_stack[:, :, 0].T,   # X  → (n_markers, n_frames)
            pts_stack[:, :, 1].T,   # Y
            pts_stack[:, :, 2].T,   # Z
            pts_stack[:, :, 3].T,   # residual (-1 = invalid)
        ], axis=0)                  # (4, n_markers, n_frames)

        ana_stack = np.concatenate(all_analogs, axis=1)  # (n_channels, total_analog_frames)
        ana_out   = ana_stack[np.newaxis, :, :]           # (1, n_channels, n_analog_frames)
    else:
        pts_out = np.zeros((4, 0, 0))
        ana_out = np.zeros((1, 0, 0))

    n_markers  = pts_out.shape[1]
    n_channels = ana_out.shape[1]

    # ── Labels ────────────────────────────────────────────────────────────────
    marker_labels = _strings("POINT",  "LABELS") or [f"M{i}" for i in range(n_markers)]
    analog_labels = _strings("ANALOG", "LABELS") or [f"A{i}" for i in range(n_channels)]

    point_rate   = _floats("POINT",  "RATE")  or [frame_rate]
    point_units  = _strings("POINT", "UNITS") or ["mm"]
    analog_rate  = _floats("ANALOG", "RATE")  or [frame_rate]
    analog_units = _strings("ANALOG","UNITS") or []

    fp_used    = _floats("FORCE_PLATFORM", "USED") or [0]
    fp_type    = _floats("FORCE_PLATFORM", "TYPE") or []
    fp_channel = _raw_array("FORCE_PLATFORM", "CHANNEL")
    fp_corners = _raw_array("FORCE_PLATFORM", "CORNERS")
    fp_origin  = _raw_array("FORCE_PLATFORM", "ORIGIN")

    def v(val):
        return {"value": val}

    return {
        "header": {
            "nb_frames":   n_frames,
            "first_frame": first_frame,
            "last_frame":  last_frame,
            "frame_rate":  frame_rate,
        },
        "parameters": {
            "POINT": {
                "LABELS": v(marker_labels),
                "USED":   v([n_markers]),
                "RATE":   v(point_rate),
                "UNITS":  v(point_units),
            },
            "ANALOG": {
                "LABELS": v(analog_labels),
                "USED":   v([n_channels]),
                "RATE":   v(analog_rate),
                "UNITS":  v(analog_units),
            },
            "FORCE_PLATFORM": {
                "USED":    v(fp_used),
                "TYPE":    v(fp_type),
                "CHANNEL": v(fp_channel),
                "CORNERS": v(fp_corners),
                "ORIGIN":  v(fp_origin),
            },
            "SUBJECTS": {
                "NAMES": v(_strings("SUBJECTS", "NAMES")),
            },
            "TRIAL": {
                "ACTUAL_START_FIELD": v([None]),
            },
            "MANUFACTURER": {
                "COMPANY":  v(_strings("MANUFACTURER", "COMPANY")  or ["Unknown"]),
                "SOFTWARE": v(_strings("MANUFACTURER", "SOFTWARE") or ["Unknown"]),
            },
        },
        "data": {
            "points":  pts_out,
            "analogs": ana_out,
        },
    }


# ── Public API ────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_c3d(file_bytes: bytes) -> dict:
    """Load a C3D file from raw bytes and return a structured dict."""
    if not C3D_AVAILABLE:
        raise ImportError(
            "The 'c3d' package is not installed.  Run: pip install c3d"
        )
    return _read_c3d(file_bytes)


def get_file_metadata(c3d) -> dict:
    header = c3d["header"]
    params = c3d["parameters"]

    def pget(group, key, default=None):
        try:
            return params[group][key]["value"]
        except (KeyError, TypeError):
            return default

    subjects    = pget("SUBJECTS", "NAMES", [])
    n_frames    = header["nb_frames"]
    first_frame = header["first_frame"]
    last_frame  = header["last_frame"]
    frame_rate  = header["frame_rate"]
    duration    = n_frames / frame_rate if frame_rate else 0

    marker_labels = pget("POINT", "LABELS",  [])
    n_markers     = pget("POINT", "USED",    [0])[0]
    point_rate    = pget("POINT", "RATE",    [frame_rate])[0]
    point_unit    = pget("POINT", "UNITS",   ["mm"])[0] if pget("POINT", "UNITS") else "mm"

    analog_labels = pget("ANALOG", "LABELS", [])
    n_analog      = pget("ANALOG", "USED",   [0])[0]
    analog_rate   = pget("ANALOG", "RATE",   [0])[0]
    analog_units  = pget("ANALOG", "UNITS",  [])

    fp_used    = pget("FORCE_PLATFORM", "USED", [0])
    n_fp       = int(fp_used[0]) if fp_used else 0
    fp_type    = pget("FORCE_PLATFORM", "TYPE", [])
    fp_corners = pget("FORCE_PLATFORM", "CORNERS", None)

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
    pts    = c3d["data"]["points"]      # (4, n_markers, n_frames)
    params = c3d["parameters"]
    try:
        labels = [str(l).strip() for l in params["POINT"]["LABELS"]["value"]]
    except (KeyError, TypeError):
        labels = [f"M{i}" for i in range(pts.shape[1])]

    n_frames  = pts.shape[2]
    n_markers = pts.shape[1]
    labels    = labels[:n_markers]

    frame_rate = float(c3d["header"]["frame_rate"])
    times      = np.arange(n_frames) / frame_rate

    data = {}
    for i, lbl in enumerate(labels):
        data[(lbl, "X")]   = pts[0, i, :]
        data[(lbl, "Y")]   = pts[1, i, :]
        data[(lbl, "Z")]   = pts[2, i, :]
        data[(lbl, "Res")] = pts[3, i, :]

    df = pd.DataFrame(data, index=times)
    df.index.name = "time_s"

    for lbl in labels:
        mask = df[(lbl, "Res")] < 0
        df.loc[mask, (lbl, "X")] = np.nan
        df.loc[mask, (lbl, "Y")] = np.nan
        df.loc[mask, (lbl, "Z")] = np.nan
    return df


def get_analog_data(c3d) -> pd.DataFrame:
    analog = c3d["data"]["analogs"]     # (1, n_channels, n_analog_frames)
    params = c3d["parameters"]
    try:
        labels = [str(l).strip() for l in params["ANALOG"]["LABELS"]["value"]]
    except (KeyError, TypeError):
        labels = [f"A{i}" for i in range(analog.shape[1])]

    n_ch   = analog.shape[1]
    n_af   = analog.shape[2]
    labels = labels[:n_ch]

    try:
        analog_rate = float(params["ANALOG"]["RATE"]["value"][0])
    except (KeyError, TypeError, IndexError):
        analog_rate = float(c3d["header"]["frame_rate"])

    times = np.arange(n_af) / analog_rate
    df    = pd.DataFrame(analog[0, :n_ch, :].T, index=times, columns=labels)
    df.index.name = "time_s"
    return df


def get_force_plate_data(c3d, metadata: dict) -> list[dict]:
    n_fp = metadata["n_force_plates"]
    if n_fp == 0:
        return []

    analog_df = get_analog_data(c3d)
    params    = c3d["parameters"]

    def pget(group, key, default=None):
        try:
            return params[group][key]["value"]
        except (KeyError, TypeError):
            return default

    ch_per_plate = pget("FORCE_PLATFORM", "CHANNEL", None)
    corners      = pget("FORCE_PLATFORM", "CORNERS", None)
    origin       = pget("FORCE_PLATFORM", "ORIGIN",  None)

    plates = []
    for p in range(n_fp):
        result = {"plate_index": p}
        result["corners"] = corners[:, :, p] if (corners is not None and hasattr(corners, 'ndim') and corners.ndim == 3) else None
        result["origin"]  = origin[:, p]     if (origin  is not None and hasattr(origin,  'ndim') and origin.ndim  == 2) else None

        if ch_per_plate is not None:
            chs = np.array(ch_per_plate).flatten()
            try:
                chs = chs.reshape(n_fp, -1)[p]
                ch_names = [analog_df.columns[int(c) - 1]
                            for c in chs if 0 < int(c) <= len(analog_df.columns)]
                sub = analog_df[ch_names].copy()
                sub.columns = [f"Ch{i+1}" for i in range(len(ch_names))]
            except Exception:
                sub = pd.DataFrame()
        else:
            sub = pd.DataFrame()

        fp_dict = _map_force_channels(sub, analog_df, p, params)
        result.update(fp_dict)
        plates.append(result)

    return plates


def _map_force_channels(sub: pd.DataFrame, analog_df: pd.DataFrame,
                        plate_idx: int, params) -> dict:
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

        Fx = result["Fx"]; Fy = result["Fy"]; Fz = result["Fz"]
        Mx = result["Mx"]; My = result["My"]
        with np.errstate(divide="ignore", invalid="ignore"):
            cop_x = np.where(np.abs(Fz) > 10, -My / Fz, np.nan)
            cop_y = np.where(np.abs(Fz) > 10,  Mx / Fz, np.nan)
        result["COPx"] = cop_x
        result["COPy"] = cop_y
        result["GRF"]  = np.sqrt(Fx**2 + Fy**2 + Fz**2)
    else:
        result["time_s"] = sub.index.values if len(sub) else np.array([])

    return result
