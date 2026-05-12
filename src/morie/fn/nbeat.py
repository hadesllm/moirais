# morie.fn — function file (hadesllm/morie)
"""N-BEATS-style basis-expansion forecasting (Oreshkin et al. 2020)."""
from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["nbeats_basis"]


def nbeats_basis(x, horizon=1, n_trend=3, n_season=5, period=12):
    r"""Polynomial (trend) + Fourier (seasonality) basis fit, N-BEATS-style.

    .. math::

        \hat y_t = \sum_{k=0}^{P} \theta^{trend}_k\, t^k +
                   \sum_{j=1}^{H}\bigl[\theta^{s,c}_j\cos(2\pi j t/m) +
                                       \theta^{s,s}_j\sin(2\pi j t/m)\bigr]

    Coefficients are ordinary-least-squares on the historical window;
    forecasts at horizon ``H`` are obtained by extrapolation.

    Parameters
    ----------
    x : array-like
        Univariate history.
    horizon : int, default 1
        Forecast horizon.
    n_trend : int, default 3
        Polynomial-trend degree.
    n_season : int, default 5
        Number of Fourier harmonics for seasonality.
    period : int, default 12
        Seasonal period (samples per cycle).

    Returns
    -------
    RichResult
        keys: ``forecast``, ``fitted``, ``trend``, ``seasonal``,
        ``theta_trend``, ``theta_seasonal``, ``r2``, ``n``, ``horizon``,
        ``method``.

    References
    ----------
    Oreshkin BN, Carpov D, Chapados N, Bengio Y (2020). N-BEATS:
    Neural Basis Expansion Analysis for Interpretable Time Series
    Forecasting. *ICLR*.
    """
    y = np.asarray(x, dtype=float).ravel()
    n = y.size
    if n < n_trend + 2 * n_season + 2:
        raise ValueError(
            f"Need at least P+2H+2 obs (={n_trend + 2*n_season + 2}); have {n}.")

    t = np.arange(n, dtype=float)
    T_cols = [t ** k for k in range(n_trend + 1)]
    S_cols = []
    for j in range(1, n_season + 1):
        S_cols.append(np.sin(2 * np.pi * j * t / period))
        S_cols.append(np.cos(2 * np.pi * j * t / period))
    X = np.column_stack(T_cols + S_cols)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ coef

    # Forecast.
    tf = np.arange(n, n + horizon, dtype=float)
    Tf = [tf ** k for k in range(n_trend + 1)]
    Sf = []
    for j in range(1, n_season + 1):
        Sf.append(np.sin(2 * np.pi * j * tf / period))
        Sf.append(np.cos(2 * np.pi * j * tf / period))
    Xf = np.column_stack(Tf + Sf)
    forecast = Xf @ coef

    theta_trend = coef[: n_trend + 1]
    theta_season = coef[n_trend + 1:]
    trend = np.column_stack(T_cols) @ theta_trend
    seasonal = np.column_stack(S_cols) @ theta_season
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum((y - fitted) ** 2)) / ss_tot if ss_tot > 0 else np.nan
    return RichResult(payload={
        "forecast": forecast,
        "fitted": fitted,
        "trend": trend,
        "seasonal": seasonal,
        "theta_trend": theta_trend,
        "theta_seasonal": theta_season,
        "r2": float(r2),
        "n": int(n), "horizon": int(horizon),
        "method": (f"N-BEATS basis: polynomial(P={n_trend}) + "
                   f"Fourier(H={n_season}, period={period})"),
    })


def cheatsheet():
    return "nbeat: N-BEATS basis-expansion forecasting (Oreshkin et al. 2020)."
