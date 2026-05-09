"""Autoregressive modeling backends from Rangayyan & Krishnan Ch. 7.

Implements: Yule-Walker, Burg, covariance AR methods,
Levinson-Durbin recursion, AR spectrum, AR prediction,
optimal model order selection, cepstral coefficients,
reflection coefficients, and point process modeling.
"""

from __future__ import annotations

import numpy as np


def autocorrelation(x: np.ndarray, max_lag: int | None = None) -> np.ndarray:
    n = len(x)
    if max_lag is None:
        max_lag = n - 1
    x_centered = x - np.mean(x)
    r = np.correlate(x_centered, x_centered, mode="full")
    r = r[n - 1 : n + max_lag] / n
    return r


def levinson_durbin(
    r: np.ndarray,
    order: int,
) -> tuple[np.ndarray, float]:
    a = np.zeros(order)
    e = r[0]
    for i in range(order):
        lam = r[i + 1]
        for j in range(i):
            lam -= a[j] * r[i - j]
        k = lam / e
        a_new = np.zeros(i + 1)
        a_new[i] = k
        for j in range(i):
            a_new[j] = a[j] - k * a[i - 1 - j]
        a[: i + 1] = a_new
        e *= 1 - k * k
    return a, e


def ar_yule_walker(
    x: np.ndarray,
    order: int = 10,
) -> tuple[np.ndarray, float]:
    r = autocorrelation(x, max_lag=order)
    return levinson_durbin(r, order)


def ar_burg(
    x: np.ndarray,
    order: int = 10,
) -> tuple[np.ndarray, float]:
    n = len(x)
    ef = x.copy().astype(float)
    eb = x.copy().astype(float)
    a = np.zeros(order)
    e = np.mean(x**2)

    for i in range(order):
        ef_prev = ef[1 : n - i].copy()
        eb_prev = eb[: n - i - 1].copy()
        num = -2 * np.dot(ef_prev, eb_prev)
        den = np.dot(ef_prev, ef_prev) + np.dot(eb_prev, eb_prev)
        k = num / den if den != 0 else 0

        a_new = np.zeros(i + 1)
        a_new[i] = k
        for j in range(i):
            a_new[j] = a[j] + k * a[i - 1 - j]
        a[: i + 1] = a_new

        ef_new = ef_prev + k * eb_prev
        eb_new = eb_prev + k * ef_prev
        ef[1 : n - i] = ef_new
        eb[: n - i - 1] = eb_new
        e *= 1 - k * k

    return a, e


def ar_covariance(
    x: np.ndarray,
    order: int = 10,
) -> tuple[np.ndarray, float]:
    n = len(x)
    x = x.astype(float)
    R = np.zeros((order, order))
    r = np.zeros(order)
    for i in range(order):
        for j in range(order):
            for k in range(order, n):
                R[i, j] += x[k - i - 1] * x[k - j - 1]
        for k in range(order, n):
            r[i] += x[k] * x[k - i - 1]

    try:
        a = np.linalg.solve(R, r)
    except np.linalg.LinAlgError:
        a = np.linalg.lstsq(R, r, rcond=None)[0]

    residual = x[order:].copy()
    for i in range(order):
        residual -= a[i] * x[order - i - 1 : n - i - 1]
    sigma2 = float(np.mean(residual**2))
    return a, sigma2


def ar_spectrum(
    a: np.ndarray,
    sigma2: float,
    fs: float = 1.0,
    n_points: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    freqs = np.linspace(0, fs / 2, n_points)
    order = len(a)
    psd = np.zeros(n_points)
    for i, f in enumerate(freqs):
        w = 2 * np.pi * f / fs
        denom = 1.0
        for k in range(order):
            denom -= a[k] * np.exp(-1j * w * (k + 1))
        psd[i] = sigma2 / (np.abs(denom) ** 2 * fs)
    return freqs, psd


def ar_predict(
    x: np.ndarray,
    a: np.ndarray,
    n_ahead: int = 1,
) -> np.ndarray:
    order = len(a)
    predicted = np.zeros(n_ahead)
    buffer = list(x[-order:])
    for i in range(n_ahead):
        val = sum(a[k] * buffer[-(k + 1)] for k in range(order))
        predicted[i] = val
        buffer.append(val)
    return predicted


def optimal_ar_order(
    x: np.ndarray,
    max_order: int = 30,
    criterion: str = "aic",
) -> tuple[int, np.ndarray]:
    n = len(x)
    scores = np.zeros(max_order)
    for p in range(1, max_order + 1):
        _, sigma2 = ar_burg(x, p)
        sigma2 = max(sigma2, 1e-20)
        if criterion == "aic":
            scores[p - 1] = n * np.log(sigma2) + 2 * p
        elif criterion == "bic":
            scores[p - 1] = n * np.log(sigma2) + p * np.log(n)
        elif criterion == "fpe":
            scores[p - 1] = sigma2 * (n + p + 1) / (n - p - 1)
        else:
            scores[p - 1] = n * np.log(sigma2) + 2 * p
    best_order = int(np.argmin(scores)) + 1
    return best_order, scores


def cepstral_coefficients(
    a: np.ndarray,
    n_coeffs: int = 13,
) -> np.ndarray:
    p = len(a)
    c = np.zeros(n_coeffs)
    for n in range(n_coeffs):
        if n == 0:
            c[0] = a[0] if p > 0 else 0
        elif n < p:
            c[n] = a[n]
            for k in range(1, n):
                c[n] += (k / n) * c[k] * a[n - k - 1]
        else:
            for k in range(1, n):
                if n - k - 1 < p:
                    c[n] += (k / n) * c[k] * a[n - k - 1]
    return c


def reflection_coefficients(a: np.ndarray) -> np.ndarray:
    p = len(a)
    a_curr = a.copy()
    k = np.zeros(p)
    for i in range(p - 1, -1, -1):
        k[i] = a_curr[i]
        if abs(k[i]) >= 1:
            break
        a_prev = np.zeros(i)
        for j in range(i):
            a_prev[j] = (a_curr[j] - k[i] * a_curr[i - 1 - j]) / (1 - k[i] ** 2)
        a_curr[:i] = a_prev
    return k


def parcor_coefficients(
    x: np.ndarray,
    order: int = 10,
) -> np.ndarray:
    a, _ = ar_burg(x, order)
    return reflection_coefficients(a)
