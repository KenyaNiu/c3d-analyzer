# 🦴 C3D Analyzer

**Professional Biomechanics Analysis Platform** built with Streamlit.

---

## Features

| Module | Capabilities |
|---|---|
| **Overview** | File metadata, marker labels, data quality heatmap, quick preview |
| **Marker Analysis** | XYZ trajectories, velocity, acceleration, speed, gap detection, ROM, inter-marker distance |
| **Analog Signals** | Multi-channel viewer, EMG pipeline, PSD, correlation matrix, statistics |
| **Force Plates** | GRF components (Fx/Fy/Fz), COP trajectory, stance detection, impulse, loading rate |
| **Gait Analysis** | Heel-strike / toe-off detection, stride time, cadence, stance %, variability |
| **Signal Processing** | Low/high/band-pass/notch Butterworth filters, FFT amplitude spectrum, Welch PSD |
| **Joint Angles** | 3-point angle computation, ROM statistics, multi-joint comparison |
| **Export** | CSV per data type, Excel summary, JSON metadata — all bundled in ZIP |

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note**: `ezc3d` has native binaries for Windows / macOS / Linux.
> If pip fails try: `conda install -c conda-forge ezc3d`

### 2. Run locally

```bash
streamlit run app.py
```

### 3. Deploy to Streamlit Cloud

1. Push this folder to a GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repo → set **Main file path** to `app.py`
4. Click **Deploy**

---

## Project Structure

```
c3d_analyzer/
├── app.py                   # Main Streamlit application (all pages)
├── requirements.txt         # Python dependencies
├── README.md
├── .streamlit/
│   └── config.toml          # Theme + server config
├── assets/
│   └── style.css            # Custom light-theme CSS
└── utils/
    ├── __init__.py
    ├── c3d_loader.py         # C3D parsing via ezc3d
    ├── marker_analysis.py    # Kinematics (velocity, acceleration, angles, gaps)
    ├── analog_analysis.py    # EMG, force plate, gait event detection
    └── visualization.py     # Plotly figure builders
```

---

## Supported C3D Data

- ✅ Marker trajectories (X, Y, Z, Residual)
- ✅ Analog channels (EMG, accelerometer, gyroscope, raw voltage)
- ✅ Force platform data (Fx, Fy, Fz, Mx, My, Mz, COP)
- ✅ File parameters (POINT, ANALOG, FORCE_PLATFORM, MANUFACTURER groups)

---

## Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | Web framework |
| `ezc3d` | C3D file I/O |
| `numpy` | Numerical computation |
| `pandas` | Data manipulation |
| `plotly` | Interactive visualization |
| `scipy` | Signal processing, filtering |
| `openpyxl` | Excel export |

---

## License

MIT — free to use, modify, and distribute.
