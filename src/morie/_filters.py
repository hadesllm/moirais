"""Signal filtering backends from Rangayyan & Krishnan Ch. 3.

Implements: moving average, Wiener filter, adaptive LMS/RLS/NLMS,
notch filter, comb filter, matched filter, median filter,
ensemble averaging, and synchronized averaging.
All operate on 1-D numpy arrays.
"""

from __future__ import annotations

import numpy as np


def moving_average(x: np.ndarray, window: int = 5) -> np.ndarray:
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="same")


def wiener_filter(
    x: np.ndarray,
    noise_psd: np.ndarray | None = None,
    *,
    noise_fraction: float = 0.1,
) -> np.ndarray:
    X = np.fft.rfft(x)
    Pxx = np.abs(X) ** 2
    if noise_psd is None:
        noise_psd = np.full_like(Pxx, noise_fraction * np.mean(Pxx))
    H = Pxx / (Pxx + noise_psd)
    return np.fft.irfft(H * X, n=len(x))


def lms_filter(
    x: np.ndarray,
    d: np.ndarray,
    order: int = 16,
    mu: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(x)
    w = np.zeros(order)
    y = np.zeros(n)
    e = np.zeros(n)
    for i in range(order, n):
        x_seg = x[i - order : i][::-1]
        y[i] = np.dot(w, x_seg)
        e[i] = d[i] - y[i]
        w += 2 * mu * e[i] * x_seg
    return y, e


def nlms_filter(
    x: np.ndarray,
    d: np.ndarray,
    order: int = 16,
    mu: float = 0.5,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(x)
    w = np.zeros(order)
    y = np.zeros(n)
    e = np.zeros(n)
    for i in range(order, n):
        x_seg = x[i - order : i][::-1]
        norm = np.dot(x_seg, x_seg) + eps
        y[i] = np.dot(w, x_seg)
        e[i] = d[i] - y[i]
        w += (2 * mu / norm) * e[i] * x_seg
    return y, e


def rls_filter(
    x: np.ndarray,
    d: np.ndarray,
    order: int = 16,
    lam: float = 0.99,
    delta: float = 100.0,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(x)
    w = np.zeros(order)
    P = delta * np.eye(order)
    y = np.zeros(n)
    e = np.zeros(n)
    for i in range(order, n):
        x_seg = x[i - order : i][::-1]
        y[i] = np.dot(w, x_seg)
        e[i] = d[i] - y[i]
        k = P @ x_seg / (lam + x_seg @ P @ x_seg)
        w += k * e[i]
        P = (P - np.outer(k, x_seg @ P)) / lam
    return y, e


def notch_filter(
    x: np.ndarray,
    freq: float,
    fs: float,
    q: float = 30.0,
) -> np.ndarray:
    from scipy.signal import filtfilt, iirnotch

    b, a = iirnotch(freq, q, fs)
    return filtfilt(b, a, x)


def comb_filter(
    x: np.ndarray,
    fundamental: float,
    fs: float,
    n_harmonics: int = 5,
    q: float = 30.0,
) -> np.ndarray:
    y = x.copy()
    for h in range(1, n_harmonics + 1):
        f = fundamental * h
        if f < fs / 2:
            y = notch_filter(y, f, fs, q)
    return y


def matched_filter(
    x: np.ndarray,
    template: np.ndarray,
) -> np.ndarray:
    h = template[::-1]
    return np.correlate(x, h, mode="same") / np.linalg.norm(h)


def median_filter(
    x: np.ndarray,
    kernel_size: int = 5,
) -> np.ndarray:
    from scipy.signal import medfilt

    return medfilt(x, kernel_size=kernel_size)


def ensemble_average(
    segments: np.ndarray,
) -> np.ndarray:
    return np.mean(segments, axis=0)


def synchronized_average(
    x: np.ndarray,
    trigger_indices: np.ndarray,
    window: int = 100,
) -> np.ndarray:
    segments = []
    for idx in trigger_indices:
        start = idx - window // 2
        end = idx + window // 2
        if start >= 0 and end <= len(x):
            segments.append(x[start:end])
    if not segments:
        return np.zeros(window)
    return np.mean(segments, axis=0)


def snr_estimate(
    signal: np.ndarray,
    noise: np.ndarray,
) -> float:
    ps = np.mean(signal**2)
    pn = np.mean(noise**2)
    if pn == 0:
        return float("inf")
    return 10 * np.log10(ps / pn)


def snr_improvement(
    x_noisy: np.ndarray,
    x_clean: np.ndarray,
    x_filtered: np.ndarray,
) -> float:
    noise_before = x_noisy - x_clean
    noise_after = x_filtered - x_clean
    snr_before = snr_estimate(x_clean, noise_before)
    snr_after = snr_estimate(x_clean, noise_after)
    return snr_after - snr_before


def turning_points_test(x: np.ndarray) -> dict:
    diffs = np.diff(x)
    turns = np.sum(diffs[:-1] * diffs[1:] < 0)
    n = len(x)
    expected = 2 * (n - 2) / 3
    variance = (16 * n - 29) / 90
    z = (turns - expected) / np.sqrt(variance)
    return {"turning_points": int(turns), "expected": expected, "z_statistic": z, "stationary": abs(z) < 1.96}


def coefficient_of_variation(x: np.ndarray) -> float:
    m = np.mean(x)
    if m == 0:
        return float("inf")
    return float(np.std(x) / abs(m))


def alpha_trimmed_mean_filter(x: np.ndarray, window: int = 5, alpha: float = 0.2) -> np.ndarray:
    n = len(x)
    y = np.zeros(n)
    half = window // 2
    trim = int(alpha * window)
    for i in range(n):
        start = max(0, i - half)
        end = min(n, i + half + 1)
        seg = np.sort(x[start:end])
        if trim > 0 and len(seg) > 2 * trim:
            seg = seg[trim:-trim]
        y[i] = np.mean(seg)
    return y


def hann_filter(x: np.ndarray, window: int = 5) -> np.ndarray:
    w = np.hanning(window)
    w /= w.sum()
    return np.convolve(x, w, mode="same")


def wiener_hopf_solve(Rxx: np.ndarray, rxd: np.ndarray) -> np.ndarray:
    return np.linalg.solve(Rxx, rxd)


def cross_correlation(x: np.ndarray, y: np.ndarray, max_lag: int | None = None) -> np.ndarray:
    if max_lag is None:
        max_lag = len(x) - 1
    result = np.correlate(x - np.mean(x), y - np.mean(y), mode="full")
    mid = len(result) // 2
    norm = np.sqrt(np.sum((x - np.mean(x)) ** 2) * np.sum((y - np.mean(y)) ** 2))
    if norm > 0:
        result = result / norm
    return result[mid - max_lag : mid + max_lag + 1]


def even_odd_decompose(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x_rev = x[::-1]
    x_even = (x + x_rev) / 2
    x_odd = (x - x_rev) / 2
    return x_even, x_odd
