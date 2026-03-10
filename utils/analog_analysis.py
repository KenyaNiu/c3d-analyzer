"""
analog_analysis.py
──────────────────
Analysis of analog channels: EMG, force plates, accelerometers, etc.
"""

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, welch, find_peaks, iirnotch
from scipy.integrate import cumulative_trapezoid


# ── Filtering ─────────────────────────────────────────────────────────────────

def bandpass_filter(signal: np.ndarray, low: float, high: float,
                    fs: float, order: int = 4) -> np.ndarray:
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, signal)


def notch_filter(signal: np.ndarray, freq: float, fs: float,
                 quality: float = 30.0) -> np.ndarray:
    b, a = iirnotch(freq / (0.5 * fs), quality)
    return filtfilt(b, a, signal)


def lowpass_filter(signal: np.ndarray, cutoff: float, fs: float,
                   order: int = 4) -> np.ndarray:
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype="low")
    return filtfilt(b, a, signal)


def highpass_filter(signal: np.ndarray, cutoff: float, fs: float,
                    order: int = 4) -> np.ndarray:
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype="high")
    return filtfilt(b, a, signal)


# ── EMG Processing ────────────────────────────────────────────────────────────

def process_emg(signal: np.ndarray, fs: float,
                hp_cutoff: float = 20.0,
                lp_envelope: float = 6.0,
                notch_hz: float = 50.0) -> dict:
    """Full EMG processing pipeline."""
    # 1. Notch filter (50/60 Hz power line)
    denoised = notch_filter(signal, notch_hz, fs)
    # 2. High-pass to remove DC drift
    hp = highpass_filter(denoised, hp_cutoff, fs)
    # 3. Full-wave rectification
    rect = np.abs(hp)
    # 4. Linear envelope (low-pass)
    envelope = lowpass_filter(rect, lp_envelope, fs)
    # 5. RMS (50 ms window)
    win = max(1, int(0.05 * fs))
    rms = _rolling_rms(rect, win)
    return {
        "raw":      signal,
        "filtered": hp,
        "rectified": rect,
        "envelope": envelope,
        "rms":      rms,
    }


def emg_stats(signal: np.ndarray, fs: float) -> dict:
    """Descriptive stats for an EMG channel."""
    return {
        "mean_abs":   float(np.mean(np.abs(signal))),
        "rms":        float(np.sqrt(np.mean(signal**2))),
        "peak":       float(np.max(np.abs(signal))),
        "median_freq": float(_median_freq(signal, fs)),
        "mean_freq":   float(_mean_freq(signal, fs)),
        "std":         float(np.std(signal)),
    }


# ── Spectral Analysis ─────────────────────────────────────────────────────────

def compute_psd(signal: np.ndarray, fs: float, nperseg: int = 256) -> tuple:
    """Power spectral density via Welch method."""
    freqs, psd = welch(signal, fs=fs, nperseg=min(nperseg, len(signal)//2))
    return freqs, psd


def compute_fft(signal: np.ndarray, fs: float) -> tuple:
    """Single-sided FFT amplitude spectrum."""
    n = len(signal)
    fft_vals = np.fft.rfft(signal - np.mean(signal))
    freqs    = np.fft.rfftfreq(n, 1.0 / fs)
    amp      = (2.0 / n) * np.abs(fft_vals)
    return freqs, amp


# ── Force Plate Analysis ──────────────────────────────────────────────────────

def compute_impulse(force: np.ndarray, time: np.ndarray) -> float:
    """Impulse = ∫F dt (N·s)."""
    return float(cumulative_trapezoid(force, time, initial=0)[-1])


def compute_loading_rate(force: np.ndarray, time: np.ndarray,
                         threshold: float = 20.0) -> float:
    """
    Peak loading rate: maximum slope of the rising phase (N/s).
    Only computed above `threshold` N.
    """
    mask = force > threshold
    if mask.sum() < 5:
        return 0.0
    df_dt = np.gradient(force, time)
    return float(np.max(df_dt[mask]))


def detect_stance_phases(fz: np.ndarray, time: np.ndarray,
                          threshold: float = 10.0,
                          min_duration_s: float = 0.1) -> list[dict]:
    """Detect contact phases where |Fz| > threshold."""
    contact = np.abs(fz) > threshold
    phases = []
    in_phase = False
    start = 0
    fs = 1.0 / np.mean(np.diff(time)) if len(time) > 1 else 1.0
    min_frames = int(min_duration_s * fs)

    for i, c in enumerate(contact):
        if c and not in_phase:
            in_phase = True; start = i
        elif not c and in_phase:
            in_phase = False
            if i - start >= min_frames:
                phases.append({
                    "start_s":    time[start],
                    "end_s":      time[i-1],
                    "duration_s": time[i-1] - time[start],
                    "start_idx":  start,
                    "end_idx":    i - 1,
                    "peak_Fz":    float(np.max(np.abs(fz[start:i]))),
                })
    if in_phase and len(contact) - start >= min_frames:
        phases.append({
            "start_s":    time[start],
            "end_s":      time[-1],
            "duration_s": time[-1] - time[start],
            "start_idx":  start,
            "end_idx":    len(contact)-1,
            "peak_Fz":    float(np.max(np.abs(fz[start:]))),
        })
    return phases


def cop_path_length(cop_x: np.ndarray, cop_y: np.ndarray) -> float:
    """Total COP excursion length (mm)."""
    dx = np.diff(cop_x[~np.isnan(cop_x)])
    dy = np.diff(cop_y[~np.isnan(cop_y)])
    n = min(len(dx), len(dy))
    return float(np.sum(np.sqrt(dx[:n]**2 + dy[:n]**2)))


# ── Gait Analysis ─────────────────────────────────────────────────────────────

def detect_gait_events(fz: np.ndarray, time: np.ndarray,
                        threshold: float = 10.0) -> dict:
    """
    Detect heel-strike (HS) and toe-off (TO) events.
    Returns lists of event times.
    """
    contact = np.abs(fz) > threshold
    hs_indices = np.where(np.diff(contact.astype(int)) == 1)[0] + 1
    to_indices = np.where(np.diff(contact.astype(int)) == -1)[0] + 1
    return {
        "heel_strike_times": time[hs_indices].tolist() if len(hs_indices) else [],
        "toe_off_times":     time[to_indices].tolist() if len(to_indices) else [],
        "heel_strike_idx":   hs_indices.tolist(),
        "toe_off_idx":       to_indices.tolist(),
    }


def compute_gait_metrics(gait_events: dict, body_mass_kg: float = 70.0) -> dict:
    """
    Derive spatiotemporal gait metrics from detected events.
    """
    hs = np.array(gait_events["heel_strike_times"])
    to = np.array(gait_events["toe_off_times"])
    metrics = {}

    if len(hs) >= 2:
        stride_times = np.diff(hs)
        metrics["mean_stride_time_s"]  = float(np.mean(stride_times))
        metrics["std_stride_time_s"]   = float(np.std(stride_times))
        metrics["cadence_steps_min"]   = float(60.0 / np.mean(stride_times) * 2)

    if len(hs) >= 1 and len(to) >= 1:
        # Stance = HS to next TO
        stances = []
        for h in hs:
            later_to = to[to > h]
            if len(later_to) > 0:
                stances.append(later_to[0] - h)
        if stances:
            metrics["mean_stance_s"] = float(np.mean(stances))
            if "mean_stride_time_s" in metrics:
                metrics["stance_pct"] = float(np.mean(stances) / metrics["mean_stride_time_s"] * 100)

    return metrics


# ── Private helpers ───────────────────────────────────────────────────────────

def _rolling_rms(x: np.ndarray, win: int) -> np.ndarray:
    out = np.zeros_like(x)
    half = win // 2
    for i in range(len(x)):
        lo = max(0, i - half)
        hi = min(len(x), i + half + 1)
        out[i] = np.sqrt(np.mean(x[lo:hi]**2))
    return out


def _median_freq(signal: np.ndarray, fs: float) -> float:
    freqs, psd = welch(signal, fs=fs, nperseg=min(256, len(signal)//2))
    cum = np.cumsum(psd)
    idx = np.searchsorted(cum, cum[-1] / 2)
    return float(freqs[min(idx, len(freqs)-1)])


def _mean_freq(signal: np.ndarray, fs: float) -> float:
    freqs, psd = welch(signal, fs=fs, nperseg=min(256, len(signal)//2))
    return float(np.sum(freqs * psd) / (np.sum(psd) + 1e-12))
