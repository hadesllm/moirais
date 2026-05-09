"""Biomedical signal visualization backends.

Generates matplotlib figures for ECG, EEG, spectrograms, time-frequency
distributions, annotated signals, and decomposition plots.
All functions return matplotlib Figure objects.
"""

from __future__ import annotations

import numpy as np


def _get_plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_ecg_leads(
    signals: np.ndarray,
    fs: float = 500.0,
    lead_names: list[str] | None = None,
    duration: float | None = None,
) -> object:
    plt = _get_plt()
    if signals.ndim == 1:
        signals = signals.reshape(1, -1)
    n_leads, n_samples = signals.shape
    if lead_names is None:
        lead_names = [f"Lead {i + 1}" for i in range(n_leads)]
    t = np.arange(n_samples) / fs
    if duration:
        mask = t <= duration
        t = t[mask]
        signals = signals[:, mask]
    fig, axes = plt.subplots(n_leads, 1, figsize=(12, 2 * n_leads), sharex=True)
    if n_leads == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.plot(t, signals[i], "k-", linewidth=0.5)
        ax.set_ylabel(lead_names[i], fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(t[0], t[-1])
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("ECG", fontsize=12)
    plt.tight_layout()
    return fig


def plot_eeg_montage(
    channels: np.ndarray,
    fs: float = 256.0,
    channel_names: list[str] | None = None,
    duration: float | None = None,
) -> object:
    plt = _get_plt()
    if channels.ndim == 1:
        channels = channels.reshape(1, -1)
    n_ch, n_samples = channels.shape
    if channel_names is None:
        channel_names = [f"Ch{i + 1}" for i in range(n_ch)]
    t = np.arange(n_samples) / fs
    if duration:
        mask = t <= duration
        t = t[mask]
        channels = channels[:, mask]
    fig, ax = plt.subplots(figsize=(14, max(4, n_ch * 0.8)))
    spacing = np.std(channels) * 4 if np.std(channels) > 0 else 1
    for i in range(n_ch):
        offset = i * spacing
        ax.plot(t, channels[i] + offset, "b-", linewidth=0.5)
        ax.text(
            -0.01, offset, channel_names[i], transform=ax.get_yaxis_transform(), fontsize=7, ha="right", va="center"
        )
    ax.set_xlabel("Time (s)")
    ax.set_title("EEG Montage")
    ax.set_yticks([])
    ax.set_xlim(t[0], t[-1])
    plt.tight_layout()
    return fig


def plot_signal(
    x: np.ndarray,
    fs: float = 1.0,
    title: str = "Signal",
    xlabel: str = "Time (s)",
    ylabel: str = "Amplitude",
) -> object:
    plt = _get_plt()
    t = np.arange(len(x)) / fs
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, x, "b-", linewidth=0.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_filter_io(
    x_in: np.ndarray,
    x_out: np.ndarray,
    fs: float = 1.0,
    title: str = "Filter Input/Output",
) -> object:
    plt = _get_plt()
    t = np.arange(len(x_in)) / fs
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax1.plot(t, x_in, "b-", linewidth=0.8)
    ax1.set_ylabel("Input")
    ax1.set_title(title)
    ax1.grid(True, alpha=0.3)
    t_out = np.arange(len(x_out)) / fs
    ax2.plot(t_out, x_out, "r-", linewidth=0.8)
    ax2.set_ylabel("Output")
    ax2.set_xlabel("Time (s)")
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_spectrum(
    psd: np.ndarray,
    freqs: np.ndarray,
    log_scale: bool = True,
    title: str = "Power Spectrum",
) -> object:
    plt = _get_plt()
    fig, ax = plt.subplots(figsize=(10, 4))
    if log_scale:
        ax.semilogy(freqs, psd, "b-", linewidth=0.8)
        ax.set_ylabel("Power (dB)")
    else:
        ax.plot(freqs, psd, "b-", linewidth=0.8)
        ax.set_ylabel("Power")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_spectrogram(
    Sxx: np.ndarray,
    fs: float = 1.0,
    t: np.ndarray | None = None,
    f: np.ndarray | None = None,
    title: str = "Spectrogram",
) -> object:
    plt = _get_plt()
    fig, ax = plt.subplots(figsize=(10, 5))
    if t is None:
        t = np.arange(Sxx.shape[1])
    if f is None:
        f = np.arange(Sxx.shape[0])
    im = ax.pcolormesh(t, f, 10 * np.log10(np.abs(Sxx) + 1e-10), shading="auto", cmap="viridis")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="Power (dB)")
    plt.tight_layout()
    return fig


def plot_scalogram(
    coeffs: np.ndarray,
    scales: np.ndarray,
    fs: float = 1.0,
    title: str = "Wavelet Scalogram",
) -> object:
    plt = _get_plt()
    fig, ax = plt.subplots(figsize=(10, 5))
    t = np.arange(coeffs.shape[1]) / fs
    im = ax.pcolormesh(t, scales, np.abs(coeffs), shading="auto", cmap="jet")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Scale")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="|Coefficients|")
    plt.tight_layout()
    return fig


def plot_tfd(
    tfd: np.ndarray,
    t: np.ndarray,
    f: np.ndarray,
    title: str = "Time-Frequency Distribution",
) -> object:
    plt = _get_plt()
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.pcolormesh(t, f, np.abs(tfd), shading="auto", cmap="hot")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="Magnitude")
    plt.tight_layout()
    return fig


def plot_annotated_signal(
    x: np.ndarray,
    fs: float = 1.0,
    annotations: np.ndarray | None = None,
    title: str = "Annotated Signal",
    annotation_label: str = "Events",
) -> object:
    plt = _get_plt()
    t = np.arange(len(x)) / fs
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, x, "b-", linewidth=0.8)
    if annotations is not None and len(annotations) > 0:
        ann_times = annotations / fs
        ann_amps = x[np.clip(annotations.astype(int), 0, len(x) - 1)]
        ax.plot(ann_times, ann_amps, "rv", markersize=8, label=annotation_label)
        ax.legend()
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_imf_stack(
    imfs: list[np.ndarray],
    fs: float = 1.0,
    title: str = "IMF Decomposition",
) -> object:
    plt = _get_plt()
    n_imfs = len(imfs)
    fig, axes = plt.subplots(n_imfs, 1, figsize=(10, 2 * n_imfs), sharex=True)
    if n_imfs == 1:
        axes = [axes]
    for i, (ax, imf) in enumerate(zip(axes, imfs)):
        t = np.arange(len(imf)) / fs
        ax.plot(t, imf, "b-", linewidth=0.8)
        ax.set_ylabel(f"IMF {i + 1}", fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(title)
    plt.tight_layout()
    return fig


def plot_ar_poles(
    coeffs: np.ndarray,
    fs: float = 1.0,
    title: str = "AR Model Poles",
) -> object:
    plt = _get_plt()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    roots = np.roots(np.concatenate(([1], -coeffs)))
    theta = np.linspace(0, 2 * np.pi, 100)
    ax1.plot(np.cos(theta), np.sin(theta), "k--", linewidth=0.5)
    ax1.plot(np.real(roots), np.imag(roots), "rx", markersize=10)
    ax1.set_xlabel("Real")
    ax1.set_ylabel("Imaginary")
    ax1.set_title("Pole-Zero Diagram")
    ax1.set_aspect("equal")
    ax1.grid(True, alpha=0.3)
    n_freq = 512
    w = np.linspace(0, np.pi, n_freq)
    H = np.zeros(n_freq)
    for i, wi in enumerate(w):
        denom = 1.0
        for k, ak in enumerate(coeffs):
            denom -= ak * np.exp(-1j * (k + 1) * wi)
        H[i] = 1.0 / (np.abs(denom) ** 2 + 1e-10)
    f = w * fs / (2 * np.pi)
    ax2.plot(f, 10 * np.log10(H + 1e-10), "b-")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Power (dB)")
    ax2.set_title("AR Spectrum")
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.suptitle(title, y=1.02)
    return fig


def plot_eeg_bands(
    x: np.ndarray,
    fs: float = 256.0,
    title: str = "EEG Band Analysis",
) -> object:
    plt = _get_plt()
    from scipy.signal import butter, filtfilt

    bands = {
        "Delta (0.5-4 Hz)": (0.5, 4),
        "Theta (4-8 Hz)": (4, 8),
        "Alpha (8-13 Hz)": (8, 13),
        "Beta (13-30 Hz)": (13, 30),
        "Gamma (30-80 Hz)": (30, min(80, fs / 2 - 1)),
    }
    fig, axes = plt.subplots(len(bands) + 1, 1, figsize=(12, 2 * (len(bands) + 1)), sharex=True)
    t = np.arange(len(x)) / fs
    axes[0].plot(t, x, "k-", linewidth=0.5)
    axes[0].set_ylabel("Raw EEG")
    axes[0].set_title(title)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for i, (name, (low, high)) in enumerate(bands.items()):
        nyq = fs / 2
        lo = low / nyq
        hi = min(high / nyq, 0.99)
        if lo >= hi or lo <= 0:
            filtered = np.zeros_like(x)
        else:
            b, a = butter(4, [lo, hi], btype="band")
            filtered = filtfilt(b, a, x)
        axes[i + 1].plot(t, filtered, color=colors[i], linewidth=0.5)
        axes[i + 1].set_ylabel(name.split("(")[0].strip(), fontsize=8)
    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    return fig


def eeg_band_filter(
    x: np.ndarray,
    fs: float = 256.0,
    band: str = "alpha",
) -> np.ndarray:
    from scipy.signal import butter, filtfilt

    bands = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, min(80, fs / 2 - 1))}
    band = band.lower()
    if band not in bands:
        return x.copy()
    low, high = bands[band]
    nyq = fs / 2
    lo = low / nyq
    hi = min(high / nyq, 0.99)
    if lo >= hi or lo <= 0:
        return np.zeros_like(x)
    b, a = butter(4, [lo, hi], btype="band")
    return filtfilt(b, a, x)


def eeg_band_power(
    x: np.ndarray,
    fs: float = 256.0,
) -> dict:
    bands = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, min(80, fs / 2 - 1))}
    result = {}
    total = 0.0
    for name, (_low, _high) in bands.items():
        filtered = eeg_band_filter(x, fs, name)
        power = float(np.mean(filtered**2))
        result[name] = power
        total += power
    if total > 0:
        for name in bands:
            result[f"{name}_relative"] = result[name] / total
    result["total_power"] = total
    return result


def ecg_12lead_simulate(
    hr: float = 72.0,
    duration: float = 5.0,
    fs: float = 500.0,
) -> np.ndarray:
    n_samples = int(fs * duration)
    t = np.arange(n_samples) / fs
    rr = 60.0 / hr
    n_beats = int(duration / rr) + 1
    lead_gains = np.array([1.0, 1.2, 0.8, -0.6, 0.5, 0.9, 2.0, 1.5, 1.0, 0.7, 0.5, 0.3])
    signals = np.zeros((12, n_samples))
    for b in range(n_beats):
        beat_time = b * rr
        for lead in range(12):
            gain = lead_gains[lead]
            p_center = beat_time - 0.16
            q_center = beat_time - 0.04
            r_center = beat_time
            s_center = beat_time + 0.04
            t_center = beat_time + 0.22
            signals[lead] += gain * 0.15 * np.exp(-((t - p_center) ** 2) / (2 * 0.01**2))
            signals[lead] -= gain * 0.1 * np.exp(-((t - q_center) ** 2) / (2 * 0.005**2))
            signals[lead] += gain * 1.0 * np.exp(-((t - r_center) ** 2) / (2 * 0.008**2))
            signals[lead] -= gain * 0.15 * np.exp(-((t - s_center) ** 2) / (2 * 0.005**2))
            signals[lead] += gain * 0.3 * np.exp(-((t - t_center) ** 2) / (2 * 0.02**2))
    return signals


def hrv_metrics(
    rr_intervals: np.ndarray,
) -> dict:
    rr = np.asarray(rr_intervals, dtype=float)
    rr = rr[rr > 0]
    if len(rr) < 2:
        return {"sdnn": 0, "rmssd": 0, "pnn50": 0, "mean_rr": 0, "mean_hr": 0}
    sdnn = float(np.std(rr, ddof=1))
    diffs = np.diff(rr)
    rmssd = float(np.sqrt(np.mean(diffs**2)))
    pnn50 = float(np.sum(np.abs(diffs) > 0.05) / len(diffs) * 100) if len(diffs) > 0 else 0
    mean_rr = float(np.mean(rr))
    mean_hr = float(60.0 / mean_rr) if mean_rr > 0 else 0
    return {"sdnn": sdnn, "rmssd": rmssd, "pnn50": pnn50, "mean_rr": mean_rr, "mean_hr": mean_hr}


def respiratory_rate(
    x: np.ndarray,
    fs: float = 100.0,
) -> float:
    from scipy.signal import butter, filtfilt, find_peaks

    nyq = fs / 2
    low = 0.1 / nyq
    high = min(0.5 / nyq, 0.99)
    if low >= high or low <= 0:
        return 0.0
    # Use SOS (second-order sections) + sosfiltfilt instead of (b, a) +
    # filtfilt. Direct-form (b, a) becomes ill-conditioned at low cutoffs
    # (here 0.002 / 0.01 of Nyquist) and produces huge transients that
    # diverge between x86 and ARM BLAS implementations. SOS is stable.
    from scipy.signal import butter as _butter, sosfiltfilt
    sos = _butter(4, [low, high], btype="band", output="sos")
    filtered = sosfiltfilt(sos, x)
    peaks, _ = find_peaks(filtered, distance=int(fs * 1.5))
    if len(peaks) < 2:
        return 0.0
    intervals = np.diff(peaks) / fs
    return float(60.0 / np.mean(intervals))
