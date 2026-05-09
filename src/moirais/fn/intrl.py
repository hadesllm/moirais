# moirais.fn — function file (hadesllm/moirais)
"""Interrupted time series analysis (ITS)."""

from __future__ import annotations

import numpy as np

from ._containers import ESRes


def interrupted_time_series(y: list[float] | np.ndarray, intervention_point: int, cdf=None) -> ESRes:
    """Segmented regression for interrupted time series analysis.

    Fits: Y_t = beta0 + beta1*time + beta2*intervention + beta3*time_after

    Parameters
    ----------
    y : array-like of float
        Outcome time series.
    intervention_point : int
        Index at which intervention begins.

    Returns
    -------
    ESRes

    References
    ----------
    Wagner, A. K. et al. (2002). Segmented regression analysis of
    interrupted time series studies in medication use research.
    Journal of Clinical Pharmacy and Therapeutics, 27(4), 299-309.
    """
    y_arr = np.asarray(y, dtype=float)
    n = len(y_arr)
    if intervention_point < 1 or intervention_point >= n - 1:
        raise ValueError("intervention_point must be in [1, n-2]")

    t = np.arange(n, dtype=float)
    D = (t >= intervention_point).astype(float)
    t_after = np.maximum(t - intervention_point, 0) * D

    X = np.column_stack([np.ones(n), t, D, t_after])
    beta = np.linalg.lstsq(X, y_arr, rcond=None)[0]

    y_hat = X @ beta
    residuals = y_arr - y_hat
    sse = float(np.sum(residuals**2))
    mse = sse / (n - 4)

    XtX_inv = np.linalg.inv(X.T @ X)
    se_beta = np.sqrt(np.diag(XtX_inv) * mse)

    from scipy import stats as st
    t_stats = beta / se_beta
    p_values = 2 * (1 - st.t.cdf(np.abs(t_stats), n - 4))

    return ESRes(
        measure="ITS",
        estimate=float(beta[2]),
        se=float(se_beta[2]),
        extra={
            "intercept": float(beta[0]),
            "pre_trend": float(beta[1]),
            "level_change": float(beta[2]),
            "slope_change": float(beta[3]),
            "se": se_beta.tolist(),
            "t_statistics": t_stats.tolist(),
            "p_values": p_values.tolist(),
            "r_squared": float(1 - sse / np.sum((y_arr - y_arr.mean()) ** 2)),
            "intervention_point": intervention_point,
        },
    )


intrl = interrupted_time_series


def cheatsheet() -> str:
    return "interrupted_time_series({}) -> Segmented regression ITS analysis."
