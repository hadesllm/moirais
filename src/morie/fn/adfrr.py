# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Augmented Dickey-Fuller unit root test."""

import numpy as np
from scipy import stats

from ._containers import DescriptiveResult


def adf_test(y: np.ndarray, max_lag: int | None = None, cdf=None) -> DescriptiveResult:
    r"""
    Augmented Dickey-Fuller test for a unit root.

    Tests H0: series has a unit root vs H1: stationary.

    .. math::

        \\Delta y_t = \\alpha + \\gamma y_{t-1}
        + \\sum_{i=1}^{p} \\beta_i \\Delta y_{t-i} + \\varepsilon_t

    :param y: 1-D time series.
    :param max_lag: Maximum augmentation lags. Default int(12*(n/100)^{1/4}).
    :return: DescriptiveResult with test statistic and approximate p-value.
    :raises ValueError: If series too short.

    References
    ----------
    Dickey D.A. & Fuller W.A. (1979). Distribution of the estimators
    for autoregressive time series with a unit root. *JASA*, 74, 427-431.
    """
    y = np.asarray(y, dtype=float).ravel()
    n = len(y)
    if max_lag is None:
        max_lag = int(12 * (n / 100) ** 0.25)
    max_lag = min(max_lag, n // 3)
    if n < max_lag + 5:
        raise ValueError(f"Need at least {max_lag + 5} observations, got {n}.")
    dy = np.diff(y)
    best_aic = np.inf
    best_lag = 0
    best_stat = 0.0
    best_se = 1.0
    for lag in range(max_lag + 1):
        T = n - 1 - lag
        dep = dy[lag:]
        regs = [np.ones(T), y[lag : n - 1]]
        for i in range(1, lag + 1):
            regs.append(dy[lag - i : n - 1 - i])
        X = np.column_stack(regs)
        beta = np.linalg.lstsq(X, dep, rcond=None)[0]
        resid = dep - X @ beta
        sig2 = float(np.sum(resid ** 2) / (T - len(beta)))
        aic = T * np.log(np.sum(resid ** 2) / T) + 2 * len(beta)
        if aic < best_aic:
            best_aic = aic
            best_lag = lag
            gamma_hat = beta[1]
            XtX_inv = np.linalg.inv(X.T @ X)
            best_se = float(np.sqrt(max(sig2 * XtX_inv[1, 1], 1e-20)))
            best_stat = float(gamma_hat / best_se)
    crit_1 = -3.43
    crit_5 = -2.86
    crit_10 = -2.57
    if best_stat < crit_1:
        approx_p = 0.005
    elif best_stat < crit_5:
        approx_p = 0.03
    elif best_stat < crit_10:
        approx_p = 0.07
    else:
        approx_p = min(1.0, stats.norm.cdf(best_stat) * 2)
    return DescriptiveResult(
        name="adf_test",
        value=best_stat,
        extra={
            "statistic": best_stat,
            "p_value": approx_p,
            "lags_used": best_lag,
            "critical_values": {"1%": crit_1, "5%": crit_5, "10%": crit_10},
            "n": n,
        },
    )


adfrr = adf_test


def cheatsheet() -> str:
    return "adf_test({}) -> Augmented Dickey-Fuller unit root test."
