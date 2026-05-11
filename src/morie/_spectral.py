"""Spectral analysis backends from Rangayyan & Krishnan Ch. 6.

Implements: periodogram, Bartlett PSD, spectral moments,
spectral power ratio, mean/median/edge frequencies,
spectral flatness, spectral entropy, coherence, and
autocorrelation from PSD.
"""

from __future__ import annotations

import numpy as np


def periodogram(
    x: np.ndarray,
    fs: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    N = len(x)
    X = np.fft.rfft(x)
    psd = (np.abs(X) ** 2) / (N * fs)
    psd[1:-1] *= 2
    freqs = np.fft.rfftfreq(N, d=1.0 / fs)
    return freqs, psd


def bartlett_psd(
    x: np.ndarray,
    fs: float = 1.0,
    n_segments: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    seg_len = len(x) // n_segments
    psd_sum = None
    for i in range(n_segments):
        seg = x[i * seg_len : (i + 1) * seg_len]
        f, p = periodogram(seg, fs)
        psd_sum = p if psd_sum is None else psd_sum + p
    return f, psd_sum / n_segments


def spectral_moment(
    psd: np.ndarray,
    freqs: np.ndarray,
    order: int = 0,
) -> float:
    df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
    return float(np.sum((freqs**order) * psd) * df)


def mean_frequency(
    psd: np.ndarray,
    freqs: np.ndarray,
) -> float:
    m0 = spectral_moment(psd, freqs, 0)
    if m0 == 0:
        return 0.0
    m1 = spectral_moment(psd, freqs, 1)
    return m1 / m0


def median_frequency(
    psd: np.ndarray,
    freqs: np.ndarray,
) -> float:
    df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
    cumulative = np.cumsum(psd * df)
    total = cumulative[-1]
    if total == 0:
        return 0.0
    idx = np.searchsorted(cumulative, total / 2)
    return float(freqs[min(idx, len(freqs) - 1)])


def spectral_edge_frequency(
    psd: np.ndarray,
    freqs: np.ndarray,
    pct: float = 0.95,
) -> float:
    df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
    cumulative = np.cumsum(psd * df)
    total = cumulative[-1]
    if total == 0:
        return 0.0
    idx = np.searchsorted(cumulative, pct * total)
    return float(freqs[min(idx, len(freqs) - 1)])


def spectral_power_ratio(
    psd: np.ndarray,
    freqs: np.ndarray,
    band1: tuple[float, float],
    band2: tuple[float, float],
) -> float:
    df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
    mask1 = (freqs >= band1[0]) & (freqs <= band1[1])
    mask2 = (freqs >= band2[0]) & (freqs <= band2[1])
    p1 = np.sum(psd[mask1]) * df
    p2 = np.sum(psd[mask2]) * df
    if p2 == 0:
        return float("inf")
    return float(p1 / p2)


def spectral_flatness(psd: np.ndarray) -> float:
    psd_pos = psd[psd > 0]
    if len(psd_pos) == 0:
        return 0.0
    geo_mean = np.exp(np.mean(np.log(psd_pos)))
    arith_mean = np.mean(psd_pos)
    if arith_mean == 0:
        return 0.0
    return float(geo_mean / arith_mean)


def spectral_entropy(
    psd: np.ndarray,
) -> float:
    total = np.sum(psd)
    if total == 0:
        return 0.0
    p = psd / total
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def psd_to_decibels(psd: np.ndarray) -> np.ndarray:
    return 10 * np.log10(np.maximum(psd, 1e-20))


def acf_from_psd(psd: np.ndarray) -> np.ndarray:
    return np.fft.irfft(psd)


def coherence(
    x: np.ndarray,
    y: np.ndarray,
    fs: float = 1.0,
    nperseg: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    from scipy.signal import coherence as sp_coherence

    f, cxy = sp_coherence(x, y, fs=fs, nperseg=nperseg)
    return f, cxy


def band_power(
    psd: np.ndarray,
    freqs: np.ndarray,
    f_low: float,
    f_high: float,
) -> float:
    df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
    mask = (freqs >= f_low) & (freqs <= f_high)
    return float(np.sum(psd[mask]) * df)


def fractal_dim_from_psd(psd: np.ndarray, freqs: np.ndarray) -> float:
    valid = (freqs > 0) & (psd > 0)
    if np.sum(valid) < 2:
        return 1.5
    log_f = np.log10(freqs[valid])
    log_p = np.log10(psd[valid])
    slope, _ = np.polyfit(log_f, log_p, 1)
    beta = -slope
    return float((5 - beta) / 2)


def spectral_kurtosis(psd: np.ndarray, freqs: np.ndarray) -> float:
    total = np.sum(psd)
    if total == 0:
        return 0.0
    p = psd / total
    f_mean = np.sum(freqs * p)
    m2 = np.sum((freqs - f_mean) ** 2 * p)
    m4 = np.sum((freqs - f_mean) ** 4 * p)
    if m2 == 0:
        return 0.0
    return float(m4 / m2**2)


def window_functions(N: int, wtype: str = "hamming") -> np.ndarray:
    wtype = wtype.lower()
    if wtype == "hamming":
        return np.hamming(N)
    elif wtype == "hanning" or wtype == "hann":
        return np.hanning(N)
    elif wtype == "blackman":
        return np.blackman(N)
    elif wtype == "triangular" or wtype == "bartlett":
        return np.bartlett(N)
    elif wtype == "kaiser":
        return np.kaiser(N, beta=14)
    elif wtype == "rectangular" or wtype == "boxcar":
        return np.ones(N)
    else:
        return np.hamming(N)


def fbm_synthesis(N: int, H: float = 0.5) -> np.ndarray:
    rng = np.random.default_rng()
    white = rng.standard_normal(N)
    freqs = np.fft.rfftfreq(N)
    freqs[0] = 1.0
    power_filter = freqs ** (-(H + 0.5))
    power_filter[0] = 0.0
    X = np.fft.rfft(white) * power_filter
    fbm = np.real(np.fft.irfft(X, n=N))
    return np.cumsum(fbm)


def welch_psd(
    x: np.ndarray, fs: float = 1.0, nperseg: int = 256, noverlap: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    from scipy.signal import welch as sp_welch

    f, Pxx = sp_welch(x, fs=fs, nperseg=min(nperseg, len(x)), noverlap=noverlap)
    return f, Pxx
