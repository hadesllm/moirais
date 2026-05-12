# morie.fn -- function file (hadesllm/morie)
"""Unobserved-components model -- trend + seasonal + irregular (Harvey 1989)."""
from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["unobserved_components"]


def unobserved_components(x, period=12, trend="local linear"):
    r"""Structural time-series decomposition.

    .. math::

        y_t = \mu_t + \gamma_t + \epsilon_t

    where :math:`\mu_t` is a (local-level or local-linear) trend,
    :math:`\gamma_t` is a stochastic seasonal component summing to zero
    over one period, and :math:`\epsilon_t` is irregular.

    Parameters
    ----------
    x : array-like
        Univariate time series.
    period : int, default 12
        Seasonal period; pass ``0`` to omit the seasonal component.
    trend : {"local level", "local linear"}, default "local linear"
        Trend specification.

    Returns
    -------
    RichResult
        keys: ``trend``, ``seasonal``, ``irregular``, ``loglik``, ``n``,
        ``period``, ``method``.

    References
    ----------
    Harvey AC (1989). *Forecasting, Structural Time Series Models and
    the Kalman Filter*. Cambridge UP.
    """
    y = np.asarray(x, dtype=float).ravel()
    n = y.size
    if n < max(2 * period, 6):
        raise ValueError(f"Need at least max(2*period, 6) obs, got {n}.")

    try:
        from statsmodels.tsa.statespace.structural import UnobservedComponents
        kwargs = dict(level=trend, freq_seasonal=None)
        if period and period > 1:
            kwargs["seasonal"] = period
        mod = UnobservedComponents(y, **kwargs)
        fit = mod.fit(disp=False)
        mu = np.asarray(fit.level["smoothed"])
        season = (np.asarray(fit.seasonal["smoothed"])
                  if period and period > 1 else np.zeros(n))
        irr = y - mu - season
        return RichResult(payload={
            "trend": mu, "seasonal": season, "irregular": irr,
            "loglik": float(fit.llf), "n": int(n),
            "period": int(period),
            "method": "Structural UCM via statsmodels",
        })
    except Exception:
        pass

    # Pure-NumPy fallback: classical additive decomposition via centred
    # moving average for the trend and per-phase mean for seasonality.
    if period > 1:
        w = period
        kernel = np.ones(w) / w
        if w % 2 == 0:
            # 2x w-period MA -- pad symmetrically.
            ma = np.convolve(y, np.ones(w + 1) / (w + 1), mode="same")
        else:
            ma = np.convolve(y, kernel, mode="same")
        detrended = y - ma
        season = np.zeros(n)
        for k in range(period):
            idx = np.arange(k, n, period)
            season[idx] = np.nanmean(detrended[idx])
        season -= np.nanmean(season)
        mu = ma
    else:
        mu = np.convolve(y, np.ones(5) / 5, mode="same")
        season = np.zeros(n)
    irr = y - mu - season
    return RichResult(payload={
        "trend": mu, "seasonal": season, "irregular": irr,
        "loglik": np.nan, "n": int(n), "period": int(period),
        "method": "Additive trend+seasonal decomposition (numpy)",
    })


def cheatsheet():
    return "ucmod: Unobserved-components decomposition (Harvey 1989)."
