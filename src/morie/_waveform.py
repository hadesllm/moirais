"""Waveform analysis backends from Rangayyan & Krishnan Ch. 5.

Implements: RMS, form factor, crest factor, shape factor,
waveform length, turns count, activity/mobility/complexity (Hjorth),
fractal dimension (Higuchi/Katz), amplitude histogram features,
and morphological waveform descriptors.
"""

from __future__ import annotations

import numpy as np


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x**2)))


def form_factor(x: np.ndarray) -> float:
    x_rms = rms(x)
    x_mean_abs = np.mean(np.abs(x))
    if x_mean_abs == 0:
        return 0.0
    return x_rms / x_mean_abs


def crest_factor(x: np.ndarray) -> float:
    x_rms = rms(x)
    if x_rms == 0:
        return 0.0
    return float(np.max(np.abs(x)) / x_rms)


def shape_factor(x: np.ndarray) -> float:
    x_mean_abs = np.mean(np.abs(x))
    x_mean_sq_root = np.mean(np.sqrt(np.abs(x))) ** 2
    if x_mean_sq_root == 0:
        return 0.0
    return x_mean_abs / x_mean_sq_root


def waveform_length(x: np.ndarray) -> float:
    return float(np.sum(np.abs(np.diff(x))))


def turns_count(
    x: np.ndarray,
    threshold: float = 0.0,
) -> int:
    dx = np.diff(x)
    count = 0
    for i in range(1, len(dx)):
        if dx[i - 1] * dx[i] < 0 and np.abs(dx[i - 1] - dx[i]) > threshold:
            count += 1
    return count


def hjorth_activity(x: np.ndarray) -> float:
    return float(np.var(x))


def hjorth_mobility(x: np.ndarray) -> float:
    activity = np.var(x)
    if activity == 0:
        return 0.0
    dx = np.diff(x)
    return float(np.sqrt(np.var(dx) / activity))


def hjorth_complexity(x: np.ndarray) -> float:
    mob_x = hjorth_mobility(x)
    if mob_x == 0:
        return 0.0
    dx = np.diff(x)
    mob_dx = hjorth_mobility(dx)
    return mob_dx / mob_x


def hjorth_parameters(x: np.ndarray) -> dict[str, float]:
    return {
        "activity": hjorth_activity(x),
        "mobility": hjorth_mobility(x),
        "complexity": hjorth_complexity(x),
    }


def myopulse_rate(
    x: np.ndarray,
    threshold: float | None = None,
) -> float:
    if threshold is None:
        threshold = 2 * np.std(x)
    return float(np.mean(np.abs(x) > threshold))


def willison_amplitude(
    x: np.ndarray,
    threshold: float | None = None,
) -> int:
    if threshold is None:
        threshold = np.std(x)
    return int(np.sum(np.abs(np.diff(x)) > threshold))


def slope_sign_changes(x: np.ndarray, threshold: float = 0.0) -> int:
    dx = np.diff(x)
    count = 0
    for i in range(1, len(dx)):
        if dx[i - 1] * dx[i] < 0 and np.abs(dx[i]) > threshold:
            count += 1
    return count


def signal_length_normalized(x: np.ndarray) -> float:
    return waveform_length(x) / len(x)


def amplitude_histogram(
    x: np.ndarray,
    n_bins: int = 50,
) -> dict[str, np.ndarray]:
    counts, edges = np.histogram(x, bins=n_bins)
    centers = (edges[:-1] + edges[1:]) / 2
    probs = counts / np.sum(counts)
    return {"counts": counts, "centers": centers, "probabilities": probs, "edges": edges}


def entropy_from_histogram(
    x: np.ndarray,
    n_bins: int = 50,
) -> float:
    hist = amplitude_histogram(x, n_bins)
    p = hist["probabilities"]
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def integrated_emg(x: np.ndarray) -> float:
    return float(np.sum(np.abs(x)))


def mean_absolute_value(x: np.ndarray) -> float:
    return float(np.mean(np.abs(x)))


def variance_ratio(
    x: np.ndarray,
    y: np.ndarray,
) -> float:
    vy = np.var(y)
    if vy == 0:
        return float("inf")
    return float(np.var(x) / vy)


def signal_arc_length(x: np.ndarray) -> float:
    return float(np.sum(np.sqrt(1 + np.diff(x) ** 2)))


def centroidal_time(x: np.ndarray, fs: float = 1.0) -> float:
    t = np.arange(len(x)) / fs
    energy = x**2
    total = np.sum(energy)
    if total == 0:
        return 0.0
    return float(np.sum(t * energy) / total)


def minimum_phase_correspondent(x: np.ndarray) -> np.ndarray:
    X = np.fft.fft(x)
    log_mag = np.log(np.abs(X) + 1e-10)
    cepstrum = np.real(np.fft.ifft(log_mag))
    n = len(cepstrum)
    window = np.zeros(n)
    window[0] = 1
    window[1 : n // 2] = 2
    if n % 2 == 0:
        window[n // 2] = 1
    min_phase_cepstrum = cepstrum * window
    min_phase_spectrum = np.exp(np.fft.fft(min_phase_cepstrum))
    return np.real(np.fft.ifft(min_phase_spectrum))


def higuchi_fd(x: np.ndarray, kmax: int = 10) -> float:
    n = len(x)
    lk = np.zeros(kmax)
    for k in range(1, kmax + 1):
        lm_sum = 0.0
        for m in range(1, k + 1):
            indices = np.arange(0, (n - m) // k) * k + m - 1
            if len(indices) < 2:
                continue
            seg = x[indices]
            length = np.sum(np.abs(np.diff(seg))) * (n - 1) / (((n - m) // k) * k)
            lm_sum += length
        lk[k - 1] = lm_sum / k
    valid = lk > 0
    if np.sum(valid) < 2:
        return 1.0
    log_k = np.log(1.0 / np.arange(1, kmax + 1))
    log_l = np.log(lk + 1e-15)
    slope, _ = np.polyfit(log_k[valid], log_l[valid], 1)
    return float(slope)


def box_counting_fd(x: np.ndarray, n_scales: int = 10) -> float:
    x_norm = x - np.min(x)
    rng = np.max(x_norm)
    if rng == 0:
        return 0.0
    x_norm = x_norm / rng
    n = len(x_norm)
    scales = np.logspace(0, np.log10(n / 2), n_scales, dtype=int)
    scales = np.unique(np.clip(scales, 1, n))
    counts = []
    for s in scales:
        n_boxes = 0
        for i in range(0, n, s):
            seg = x_norm[i : i + s]
            if len(seg) > 0:
                n_boxes += int(np.ceil((np.max(seg) - np.min(seg)) * n / s)) + 1
        counts.append(n_boxes)
    if len(counts) < 2:
        return 1.0
    log_s = np.log(1.0 / scales[: len(counts)])
    log_n = np.log(np.array(counts, dtype=float) + 1)
    slope, _ = np.polyfit(log_s, log_n, 1)
    return float(slope)


def ruler_fd(x: np.ndarray, n_rulers: int = 10) -> float:
    n = len(x)
    rulers = np.logspace(0, np.log10(n / 2), n_rulers, dtype=int)
    rulers = np.unique(np.clip(rulers, 1, n // 2))
    lengths = []
    for r in rulers:
        total = 0.0
        pos = 0
        while pos + r < n:
            total += np.sqrt(r**2 + (x[pos + r] - x[pos]) ** 2)
            pos += r
        lengths.append(total)
    if len(lengths) < 2:
        return 1.0
    log_r = np.log(rulers[: len(lengths)].astype(float))
    log_l = np.log(np.array(lengths) + 1e-15)
    slope, _ = np.polyfit(log_r, log_l, 1)
    return float(1 - slope)


def parzen_pdf(x: np.ndarray, bandwidth: float | None = None, n_points: int = 100) -> tuple[np.ndarray, np.ndarray]:
    if bandwidth is None:
        bandwidth = 1.06 * np.std(x) * len(x) ** (-0.2)
    if bandwidth <= 0:
        bandwidth = 0.1
    grid = np.linspace(np.min(x) - 3 * bandwidth, np.max(x) + 3 * bandwidth, n_points)
    density = np.zeros(n_points)
    for xi in x:
        density += np.exp(-0.5 * ((grid - xi) / bandwidth) ** 2)
    density /= len(x) * bandwidth * np.sqrt(2 * np.pi)
    return grid, density


def complex_demodulation(x: np.ndarray, fc: float, fs: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    t = np.arange(len(x)) / fs
    analytic = x * np.exp(-1j * 2 * np.pi * fc * t)
    from scipy.signal import butter, filtfilt

    nyq = fs / 2
    cutoff = min(fc * 0.5, nyq * 0.9) / nyq
    if cutoff <= 0 or cutoff >= 1:
        cutoff = 0.1
    b, a = butter(4, cutoff, btype="low")
    envelope = filtfilt(b, a, np.abs(analytic))
    phase = np.unwrap(np.angle(analytic))
    return envelope, phase


def qrs_waveform_features(beat: np.ndarray) -> dict:
    peak = np.argmax(np.abs(beat))
    amplitude = float(beat[peak])
    duration = len(beat)
    area = float(np.trapezoid(np.abs(beat)))
    slope_up = float(np.max(np.diff(beat[: peak + 1]))) if peak > 0 else 0.0
    slope_down = float(np.min(np.diff(beat[peak:]))) if peak < len(beat) - 1 else 0.0
    return {
        "amplitude": amplitude,
        "duration": duration,
        "area": area,
        "slope_up": slope_up,
        "slope_down": slope_down,
        "peak_index": int(peak),
    }


def baseline_corrected_correlation(x: np.ndarray, y: np.ndarray) -> float:
    x_c = x - np.mean(x)
    y_c = y - np.mean(y)
    num = np.sum(x_c * y_c)
    den = np.sqrt(np.sum(x_c**2) * np.sum(y_c**2))
    if den == 0:
        return 0.0
    return float(num / den)
