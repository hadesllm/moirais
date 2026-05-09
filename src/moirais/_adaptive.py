"""Adaptive and nonstationary analysis from Rangayyan & Krishnan Ch. 8.

Implements: STFT, inverse STFT, Wigner-Ville distribution,
Choi-Williams distribution, spectrogram, spectral error measure,
ACF-based distance, GLR change detection, and Kalman filter.
"""

from __future__ import annotations

import numpy as np


def stft(
    x: np.ndarray,
    window_size: int = 256,
    hop: int = 128,
    window: str = "hann",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(x)
    if window == "hann":
        w = np.hanning(window_size)
    elif window == "hamming":
        w = np.hamming(window_size)
    elif window == "blackman":
        w = np.blackman(window_size)
    else:
        w = np.ones(window_size)

    n_frames = (n - window_size) // hop + 1
    n_freq = window_size // 2 + 1
    S = np.zeros((n_freq, n_frames), dtype=complex)
    times = np.zeros(n_frames)

    for i in range(n_frames):
        start = i * hop
        frame = x[start : start + window_size] * w
        S[:, i] = np.fft.rfft(frame)
        times[i] = start + window_size / 2

    freqs = np.arange(n_freq) / window_size
    return S, times, freqs


def istft(
    S: np.ndarray,
    window_size: int = 256,
    hop: int = 128,
    window: str = "hann",
) -> np.ndarray:
    n_freq, n_frames = S.shape
    if window == "hann":
        w = np.hanning(window_size)
    elif window == "hamming":
        w = np.hamming(window_size)
    else:
        w = np.ones(window_size)

    n = window_size + (n_frames - 1) * hop
    x = np.zeros(n)
    w_sum = np.zeros(n)

    for i in range(n_frames):
        start = i * hop
        frame = np.fft.irfft(S[:, i], n=window_size) * w
        x[start : start + window_size] += frame
        w_sum[start : start + window_size] += w**2

    w_sum[w_sum < 1e-8] = 1
    return x / w_sum


def spectrogram(
    x: np.ndarray,
    window_size: int = 256,
    hop: int = 128,
    window: str = "hann",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    S, times, freqs = stft(x, window_size, hop, window)
    return np.abs(S) ** 2, times, freqs


def wigner_ville(
    x: np.ndarray,
) -> np.ndarray:
    n = len(x)
    analytic = np.fft.ifft(np.fft.fft(x) * 2 * (np.arange(n) < n // 2))
    W = np.zeros((n, n))
    for t in range(n):
        tau_max = min(t, n - 1 - t, n // 2 - 1)
        for tau in range(-tau_max, tau_max + 1):
            W[t, (tau + n) % n] = np.real(analytic[t + tau] * np.conj(analytic[t - tau]))
    return np.fft.fft(W, axis=1).real


def choi_williams(
    x: np.ndarray,
    sigma: float = 1.0,
) -> np.ndarray:
    n = len(x)
    analytic = np.fft.ifft(np.fft.fft(x) * 2 * (np.arange(n) < n // 2))
    C = np.zeros((n, n))
    for t in range(n):
        tau_max = min(t, n - 1 - t, n // 2 - 1)
        for tau in range(-tau_max, tau_max + 1):
            kernel = 1.0 if tau == 0 else np.exp(-sigma * tau**2)
            C[t, (tau + n) % n] = kernel * np.real(analytic[t + tau] * np.conj(analytic[t - tau]))
    return np.fft.fft(C, axis=1).real


def spectral_error_measure(
    psd1: np.ndarray,
    psd2: np.ndarray,
) -> float:
    psd1 = np.maximum(psd1, 1e-20)
    psd2 = np.maximum(psd2, 1e-20)
    ratio = psd1 / psd2
    return float(np.mean(ratio + 1 / ratio - 2))


def acf_distance(
    x1: np.ndarray,
    x2: np.ndarray,
    max_lag: int = 20,
) -> float:
    from moirais._armodel import autocorrelation

    r1 = autocorrelation(x1, max_lag)
    r2 = autocorrelation(x2, max_lag)
    return float(np.sqrt(np.sum((r1 - r2) ** 2)))


def glr_change_detect(
    x: np.ndarray,
    min_segment: int = 20,
) -> tuple[int, float]:
    n = len(x)
    best_split = min_segment
    best_glr = -np.inf
    total_var = np.var(x)
    if total_var == 0:
        return n // 2, 0.0

    for t in range(min_segment, n - min_segment):
        var1 = np.var(x[:t])
        var2 = np.var(x[t:])
        var1 = max(var1, 1e-20)
        var2 = max(var2, 1e-20)
        glr = -t * np.log(var1) - (n - t) * np.log(var2) + n * np.log(total_var)
        if glr > best_glr:
            best_glr = glr
            best_split = t

    return best_split, float(best_glr)


def kalman_filter(
    y: np.ndarray,
    A: np.ndarray | float = 1.0,
    H: np.ndarray | float = 1.0,
    Q: np.ndarray | float = 0.01,
    R: np.ndarray | float = 1.0,
    x0: float = 0.0,
    P0: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(y)
    is_scalar = np.isscalar(A)

    if is_scalar:
        x_est = np.zeros(n)
        P_est = np.zeros(n)
        x = x0
        P = P0
        for i in range(n):
            x_pred = A * x
            P_pred = A * P * A + Q
            K = P_pred * H / (H * P_pred * H + R)
            x = x_pred + K * (y[i] - H * x_pred)
            P = (1 - K * H) * P_pred
            x_est[i] = x
            P_est[i] = P
        return x_est, P_est

    dim = A.shape[0]
    x_est = np.zeros((n, dim))
    P_est = np.zeros((n, dim, dim))
    x = np.atleast_1d(np.float64(x0)) if np.isscalar(x0) else np.array(x0, dtype=float)
    P = np.eye(dim) * P0 if np.isscalar(P0) else np.array(P0, dtype=float)
    Q = np.eye(dim) * Q if np.isscalar(Q) else np.array(Q, dtype=float)
    R = np.atleast_2d(np.float64(R)) if np.isscalar(R) else np.array(R, dtype=float)
    H = np.atleast_2d(np.float64(H)) if np.isscalar(H) else np.array(H, dtype=float)

    for i in range(n):
        x_pred = A @ x
        P_pred = A @ P @ A.T + Q
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)
        yi = np.atleast_1d(y[i])
        x = x_pred + K @ (yi - H @ x_pred)
        P = (np.eye(dim) - K @ H) @ P_pred
        x_est[i] = x
        P_est[i] = P
    return x_est, P_est


def rls_lattice_filter(
    x: np.ndarray,
    d: np.ndarray,
    order: int = 8,
    lam: float = 0.99,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(x)
    y = np.zeros(n)
    e = np.zeros(n)
    f = np.zeros((order + 1, n))
    b = np.zeros((order + 1, n))
    f[0, :] = x
    b[0, :] = x
    kappa = np.zeros(order)
    for i in range(1, n):
        for m in range(min(order, i)):
            start = max(0, i - order)
            seg_b = b[m, start:i]
            den = lam * (np.dot(seg_b, seg_b) + 1e-8)
            seg_f = f[m, start:i]
            seg_b_prev = b[m, max(0, start - 1) : i - 1]
            min_len = min(len(seg_f), len(seg_b_prev))
            if min_len > 0 and den > 0:
                kappa[m] = np.dot(seg_f[:min_len], seg_b_prev[:min_len]) / den
            else:
                kappa[m] = 0
            f[m + 1, i] = f[m, i] - kappa[m] * b[m, i - 1]
            b[m + 1, i] = b[m, i - 1] - kappa[m] * f[m, i]
        y[i] = f[0, i]
        e[i] = d[i] - y[i]
    return y, e


def cwt_compute(
    x: np.ndarray,
    scales: np.ndarray | None = None,
    wavelet: str = "morlet",
    fs: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(x)
    if scales is None:
        scales = np.arange(1, min(n // 2, 64) + 1)
    coeffs = np.zeros((len(scales), n))
    for i, s in enumerate(scales):
        half = min(int(5 * s), n // 2)
        t_wavelet = np.arange(-half, half + 1) / fs
        if wavelet == "morlet":
            omega0 = 5.0
            psi = np.exp(1j * omega0 * t_wavelet / s) * np.exp(-(t_wavelet**2) / (2 * s**2))
            psi = np.real(psi) / np.sqrt(s)
        else:
            psi = (1 - (t_wavelet / s) ** 2) * np.exp(-(t_wavelet**2) / (2 * s**2)) / np.sqrt(s)
        coeffs[i, :] = np.convolve(x, psi, mode="same")[:n]
    freqs = fs / (2 * np.pi * scales) if wavelet == "morlet" else fs / scales
    return coeffs, scales, freqs


def smoothed_pseudo_wvd(
    x: np.ndarray,
    fs: float = 1.0,
    t_smooth: int = 11,
    f_smooth: int = 11,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from scipy.signal import windows

    n = len(x)
    analytic = np.fft.ifft(np.fft.fft(x) * (2 * (np.arange(n) < n // 2)))
    n_freq = n
    tfd = np.zeros((n_freq, n))
    t_win = windows.hamming(min(t_smooth, n))
    for ti in range(n):
        half = len(t_win) // 2
        for tau_idx, tau in enumerate(range(-half, half + 1)):
            t1 = ti + tau
            t2 = ti - tau
            if 0 <= t1 < n and 0 <= t2 < n:
                val = analytic[t1] * np.conj(analytic[t2]) * t_win[tau_idx]
                for fi in range(n_freq):
                    tfd[fi, ti] += np.real(val * np.exp(-1j * 4 * np.pi * fi * tau / n_freq))
    t = np.arange(n) / fs
    f = np.arange(n_freq) * fs / (2 * n_freq)
    return tfd, t, f


def lacunarity(
    x: np.ndarray,
    box_sizes: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    binary = (x > np.mean(x)).astype(float)
    n = len(binary)
    if box_sizes is None:
        box_sizes = np.unique(np.logspace(0, np.log10(n // 2), 15).astype(int))
    lac = np.zeros(len(box_sizes))
    for i, r in enumerate(box_sizes):
        if r >= n:
            lac[i] = 1.0
            continue
        masses = np.array([np.sum(binary[j : j + r]) for j in range(n - r + 1)])
        mean_m = np.mean(masses)
        if mean_m == 0:
            lac[i] = 1.0
        else:
            lac[i] = np.mean(masses**2) / mean_m**2
    return lac, box_sizes


def bispectrum(
    x: np.ndarray,
    fs: float = 1.0,
    nfft: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    X = np.fft.fft(x, n=nfft)
    n_freq = nfft // 2
    bispec = np.zeros((n_freq, n_freq), dtype=complex)
    for f1 in range(n_freq):
        for f2 in range(f1, min(n_freq, nfft - f1)):
            bispec[f1, f2] = X[f1] * X[f2] * np.conj(X[f1 + f2])
            bispec[f2, f1] = bispec[f1, f2]
    freqs = np.fft.fftfreq(nfft, d=1 / fs)[:n_freq]
    return bispec, freqs
