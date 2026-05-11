# morie.fn — function file (hadesllm/morie)
"""Partial autocorrelation function."""

from __future__ import annotations

import numpy as np

from ._containers import TimeSeriesResult


def partial_acf(x, *, max_lag: int = 20) -> TimeSeriesResult:
    """Partial ACF via Durbin-Levinson recursion.

    Parameters
    ----------
    x : array-like
        Time series observations.
    max_lag : int
        Maximum lag to compute. Default 20.

    Returns
    -------
    TimeSeriesResult
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    max_lag = min(max_lag, n - 1)
    # Compute ACF first
    x_c = x - x.mean()
    var = np.sum(x_c**2) / n
    acf_v = np.array([np.sum(x_c[: n - k] * x_c[k:]) / (n * var) if var > 0 else 0 for k in range(max_lag + 1)])
    # Durbin-Levinson recursion
    pacf_v = np.zeros(max_lag + 1)
    pacf_v[0] = 1.0
    if max_lag >= 1:
        pacf_v[1] = acf_v[1]
    phi = np.zeros((max_lag + 1, max_lag + 1))
    if max_lag >= 1:
        phi[1, 1] = acf_v[1]
    for k in range(2, max_lag + 1):
        num = acf_v[k] - sum(phi[k - 1, j] * acf_v[k - j] for j in range(1, k))
        den = 1 - sum(phi[k - 1, j] * acf_v[j] for j in range(1, k))
        phi[k, k] = num / den if abs(den) > 1e-10 else 0
        for j in range(1, k):
            phi[k, j] = phi[k - 1, j] - phi[k, k] * phi[k - 1, k - j]
        pacf_v[k] = phi[k, k]
    ci = 1.96 / np.sqrt(n)
    return TimeSeriesResult(
        name="PACF",
        values=pacf_v,
        lags=np.arange(max_lag + 1),
        ci_upper=np.full(max_lag + 1, ci),
        ci_lower=np.full(max_lag + 1, -ci),
    )


pacf = partial_acf


def cheatsheet() -> str:
    return "partial_acf({}) -> Partial autocorrelation function."
