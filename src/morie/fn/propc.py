# morie.fn — function file (hadesllm/morie)
"""Prophet-style additive decomposition (Taylor & Letham 2018)."""
from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["prophet_components"]


def prophet_components(x, period=12):
    r"""Decompose a series into trend + seasonality (+ noise).

    .. math::

        y(t) = g(t) + s(t) + \epsilon(t)

    The trend ``g(t)`` is a piecewise-linear (logistic-style here:
    plain linear) least-squares fit; seasonality ``s(t)`` is a 5-term
    Fourier series at the supplied period; the residual ``ε(t)`` is the
    noise component.  This implementation does NOT include a holiday
    component (``h(t)``) — pass exogenous indicators if you need that.

    Parameters
    ----------
    x : array-like
        Univariate time series.
    period : int, default 12
        Seasonal period (monthly = 12).

    Returns
    -------
    RichResult
        keys: ``trend``, ``seasonal``, ``residual``, ``slope``,
        ``intercept``, ``fourier_terms``, ``period``, ``n``, ``method``.

    References
    ----------
    Taylor SJ, Letham B (2018). Forecasting at Scale. *Am. Statistician*
    72(1), 37-45.
    """
    y = np.asarray(x, dtype=float).ravel()
    n = y.size
    if n < max(2 * period, 6):
        raise ValueError(f"Need >=max(2*period, 6) obs, got {n}.")

    t = np.arange(n, dtype=float)
    # Linear trend.
    A = np.column_stack([np.ones(n), t])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    intercept, slope = coef
    trend = A @ coef
    detrended = y - trend

    # Fourier seasonality (5 harmonic pairs).
    K = 5
    F = []
    for k in range(1, K + 1):
        F.append(np.sin(2 * np.pi * k * t / period))
        F.append(np.cos(2 * np.pi * k * t / period))
    F = np.column_stack(F)
    fcoef, *_ = np.linalg.lstsq(F, detrended, rcond=None)
    seasonal = F @ fcoef
    residual = detrended - seasonal

    return RichResult(payload={
        "trend": trend,
        "seasonal": seasonal,
        "residual": residual,
        "slope": float(slope),
        "intercept": float(intercept),
        "fourier_terms": fcoef,
        "period": int(period),
        "n": int(n),
        "method": "Prophet-style linear-trend + Fourier(K=5) seasonality",
    })


def cheatsheet():
    return "propc: Prophet-style trend/seasonality decomposition (Taylor & Letham 2018)."
