"""Event detection backends from Rangayyan & Krishnan Ch. 4.

Implements: threshold detection, template matching, derivative-based
detection, zero-crossing detection, amplitude envelope, and
morphological operators for biomedical signal event detection.
"""

from __future__ import annotations

import numpy as np


def threshold_detect(
    x: np.ndarray,
    threshold: float,
    *,
    min_distance: int = 1,
    direction: str = "above",
) -> np.ndarray:
    if direction == "above":
        mask = x > threshold
    elif direction == "below":
        mask = x < threshold
    else:
        mask = np.abs(x) > threshold

    indices = np.where(mask)[0]
    if len(indices) == 0:
        return indices

    if min_distance > 1:
        kept = [indices[0]]
        for idx in indices[1:]:
            if idx - kept[-1] >= min_distance:
                kept.append(idx)
        return np.array(kept)
    return indices


def derivative_detect(
    x: np.ndarray,
    fs: float = 1.0,
    *,
    threshold_factor: float = 0.5,
) -> np.ndarray:
    dx = np.diff(x) * fs
    threshold = threshold_factor * np.max(np.abs(dx))
    peaks = []
    for i in range(1, len(dx) - 1):
        if dx[i - 1] > 0 and dx[i] <= 0 and np.abs(dx[i - 1]) > threshold:
            peaks.append(i)
    return np.array(peaks, dtype=int)


def zero_crossing_rate(
    x: np.ndarray,
    frame_length: int | None = None,
) -> float | np.ndarray:
    if frame_length is None:
        signs = np.sign(x)
        crossings = np.sum(np.abs(np.diff(signs)) > 0)
        return crossings / (len(x) - 1)

    n_frames = len(x) // frame_length
    zcr = np.zeros(n_frames)
    for i in range(n_frames):
        frame = x[i * frame_length : (i + 1) * frame_length]
        signs = np.sign(frame)
        zcr[i] = np.sum(np.abs(np.diff(signs)) > 0) / (frame_length - 1)
    return zcr


def template_match(
    x: np.ndarray,
    template: np.ndarray,
    *,
    threshold: float = 0.7,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(template)
    correlations = np.zeros(len(x) - n + 1)
    template_norm = template - np.mean(template)
    t_std = np.std(template_norm)
    if t_std == 0:
        return np.array([], dtype=int), np.array([])

    for i in range(len(correlations)):
        seg = x[i : i + n]
        seg_norm = seg - np.mean(seg)
        s_std = np.std(seg_norm)
        if s_std == 0:
            continue
        correlations[i] = np.dot(seg_norm, template_norm) / (n * s_std * t_std)

    indices = np.where(correlations >= threshold)[0]
    return indices, correlations[indices]


def onset_detect(
    x: np.ndarray,
    fs: float,
    *,
    energy_window_ms: float = 20.0,
    threshold_factor: float = 3.0,
) -> np.ndarray:
    win = max(1, int(energy_window_ms * fs / 1000))
    energy = np.convolve(x**2, np.ones(win) / win, mode="same")
    baseline = np.median(energy)
    threshold = baseline * threshold_factor
    onsets = []
    above = False
    for i, e in enumerate(energy):
        if not above and e > threshold:
            onsets.append(i)
            above = True
        elif above and e < baseline:
            above = False
    return np.array(onsets, dtype=int)


def shannon_energy(x: np.ndarray) -> np.ndarray:
    x_sq = x**2
    x_sq = np.maximum(x_sq, 1e-12)
    return -x_sq * np.log(x_sq)


def teager_energy(x: np.ndarray) -> np.ndarray:
    psi = np.zeros_like(x)
    psi[1:-1] = x[1:-1] ** 2 - x[:-2] * x[2:]
    return psi


def hilbert_envelope(x: np.ndarray) -> np.ndarray:
    from scipy.signal import hilbert

    analytic = hilbert(x)
    return np.abs(analytic)


def pan_tompkins_qrs(ecg: np.ndarray, fs: float = 360.0) -> np.ndarray:
    from scipy.signal import butter, filtfilt

    nyq = fs / 2
    low = 5 / nyq
    high = min(15 / nyq, 0.99)
    b, a = butter(1, [low, high], btype="band")
    filtered = filtfilt(b, a, ecg)
    diff_signal = np.diff(filtered)
    squared = diff_signal**2
    win_size = int(0.15 * fs)
    if win_size < 1:
        win_size = 1
    kernel = np.ones(win_size) / win_size
    integrated = np.convolve(squared, kernel, mode="same")
    threshold = 0.5 * np.max(integrated)
    candidates = np.where(integrated > threshold)[0]
    if len(candidates) == 0:
        return np.array([], dtype=int)
    qrs = [candidates[0]]
    refractory = int(0.2 * fs)
    for c in candidates[1:]:
        if c - qrs[-1] > refractory:
            qrs.append(c)
    return np.array(qrs)


def dicrotic_notch_detect(pulse: np.ndarray, fs: float = 125.0) -> np.ndarray:
    d2 = np.diff(pulse, n=2)
    from scipy.signal import find_peaks

    peaks, _ = find_peaks(-d2, distance=int(0.1 * fs))
    systolic_end = int(0.3 * fs)
    notches = peaks[peaks > systolic_end]
    return notches


def t_wave_detect(ecg: np.ndarray, qrs_locs: np.ndarray, fs: float = 360.0) -> np.ndarray:
    t_peaks = []
    search_start = int(0.2 * fs)
    search_end = int(0.5 * fs)
    for loc in qrs_locs:
        start = loc + search_start
        end = loc + search_end
        if end > len(ecg):
            break
        seg = ecg[start:end]
        t_peaks.append(start + np.argmax(seg))
    return np.array(t_peaks)


def coherence_spectrum(
    x: np.ndarray, y: np.ndarray, fs: float = 1.0, nperseg: int = 256
) -> tuple[np.ndarray, np.ndarray]:
    from scipy.signal import coherence as sp_coherence

    f, Cxy = sp_coherence(x, y, fs=fs, nperseg=min(nperseg, len(x)))
    return f, Cxy


def cross_spectral_density(
    x: np.ndarray, y: np.ndarray, fs: float = 1.0, nperseg: int = 256
) -> tuple[np.ndarray, np.ndarray]:
    from scipy.signal import csd

    f, Pxy = csd(x, y, fs=fs, nperseg=min(nperseg, len(x)))
    return f, Pxy


def heart_rate_from_rr(rr_intervals: np.ndarray) -> np.ndarray:
    rr = np.asarray(rr_intervals, dtype=float)
    rr[rr == 0] = np.nan
    return 60.0 / rr


def homomorphic_filter(x: np.ndarray, cutoff: float = 0.1, fs: float = 1.0) -> np.ndarray:
    eps = 1e-10
    log_x = np.log(np.abs(x) + eps)
    X = np.fft.rfft(log_x)
    freqs = np.fft.rfftfreq(len(log_x), d=1.0 / fs)
    H = np.ones_like(freqs, dtype=float)
    H[freqs < cutoff] = 0.0
    filtered = np.fft.irfft(X * H, n=len(log_x))
    return np.exp(filtered)


def complex_cepstrum(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X = np.fft.fft(x)
    log_X = np.log(np.abs(X) + 1e-10) + 1j * np.unwrap(np.angle(X))
    cepstrum = np.real(np.fft.ifft(log_X))
    quefrency = np.arange(len(cepstrum))
    return cepstrum, quefrency
