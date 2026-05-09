"""Time-lag cross-correlation analysis."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def time_lag_analysis(
    exposure_ts: np.ndarray | list,
    outcome_ts: np.ndarray | list,
    *,
    max_lag: int = 10,
) -> DescriptiveResult:
    """
    Cross-correlation between an exposure and outcome time series.

    Parameters
    ----------
    exposure_ts : array-like
        Exposure time series.
    outcome_ts : array-like
        Outcome time series (same length).
    max_lag : int
        Maximum lag to evaluate.

    Returns
    -------
    DescriptiveResult
        extra has 'lags', 'correlations', 'best_lag', 'best_corr'.

    References
    ----------
    Peng, R. D., et al. (2005). Seasonal analyses of air pollution and
    mortality in 100 US cities. *Am J Epidemiol*, 161(6), 585-594.
    """
    exp = np.asarray(exposure_ts, dtype=float)
    out = np.asarray(outcome_ts, dtype=float)
    if len(exp) != len(out):
        raise ValueError("Time series must be same length.")
    n = len(exp)
    if n < max_lag + 2:
        raise ValueError("Series too short for given max_lag.")

    exp_c = exp - np.mean(exp)
    out_c = out - np.mean(out)
    denom = np.sqrt(np.sum(exp_c**2) * np.sum(out_c**2))

    lags = list(range(-max_lag, max_lag + 1))
    corrs = []
    for lag in lags:
        if lag >= 0:
            r = np.sum(exp_c[: n - lag] * out_c[lag:]) / denom if denom > 0 else 0
        else:
            r = np.sum(exp_c[-lag:] * out_c[: n + lag]) / denom if denom > 0 else 0
        corrs.append(float(r))

    best_idx = int(np.argmax(np.abs(corrs)))
    best_lag = lags[best_idx]
    best_corr = corrs[best_idx]

    return DescriptiveResult(
        name="time_lag_analysis",
        value=float(best_corr),
        extra={
            "lags": lags,
            "correlations": corrs,
            "best_lag": best_lag,
            "best_corr": best_corr,
        },
    )


tlag = time_lag_analysis


def cheatsheet() -> str:
    return "time_lag_analysis({}) -> Time-lag cross-correlation analysis."
