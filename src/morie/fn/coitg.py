# morie.fn -- function file (hadesllm/morie)
"""Engle-Granger cointegration test."""

import numpy as np
from scipy import stats

from ._containers import DescriptiveResult


def eg_coint(y1: np.ndarray, y2: np.ndarray, max_lag: int | None = None, cdf=None) -> DescriptiveResult:
    """
    Engle-Granger two-step cointegration test.

    Step 1: Regress y1 on y2 and a constant.
    Step 2: ADF test on the residuals.

    :param y1: 1-D first series.
    :param y2: 1-D second series (same length).
    :param max_lag: Max ADF augmentation lags. Default int(12*(n/100)^{1/4}).
    :return: DescriptiveResult with ADF stat on residuals, beta, p-value.
    :raises ValueError: On length mismatch or insufficient data.

    References
    ----------
    Engle R.F. & Granger C.W.J. (1987). Co-integration and error
    correction: Representation, estimation, and testing.
    *Econometrica*, 55(2), 251-276.
    """
    y1 = np.asarray(y1, dtype=float).ravel()
    y2 = np.asarray(y2, dtype=float).ravel()
    if len(y1) != len(y2):
        raise ValueError(f"Series must have same length, got {len(y1)} and {len(y2)}.")
    n = len(y1)
    if n < 20:
        raise ValueError(f"Need at least 20 observations, got {n}.")
    if max_lag is None:
        max_lag = int(12 * (n / 100) ** 0.25)
    X = np.column_stack([np.ones(n), y2])
    beta = np.linalg.lstsq(X, y1, rcond=None)[0]
    resid = y1 - X @ beta
    dy = np.diff(resid)
    best_aic = np.inf
    best_stat = 0.0
    for lag in range(max_lag + 1):
        T = n - 1 - lag
        dep = dy[lag:]
        regs = [resid[lag : n - 1]]
        for i in range(1, lag + 1):
            regs.append(dy[lag - i : n - 1 - i])
        Xr = np.column_stack(regs)
        b = np.linalg.lstsq(Xr, dep, rcond=None)[0]
        e = dep - Xr @ b
        sig2 = float(np.sum(e ** 2) / (T - len(b)))
        aic = T * np.log(np.sum(e ** 2) / T) + 2 * len(b)
        if aic < best_aic:
            best_aic = aic
            se = float(np.sqrt(max(sig2 * np.linalg.inv(Xr.T @ Xr)[0, 0], 1e-20)))
            best_stat = float(b[0] / se)
    crit = {"1%": -3.90, "5%": -3.34, "10%": -3.04}
    if best_stat < crit["1%"]:
        approx_p = 0.005
    elif best_stat < crit["5%"]:
        approx_p = 0.03
    elif best_stat < crit["10%"]:
        approx_p = 0.07
    else:
        approx_p = min(1.0, stats.norm.cdf(best_stat) * 2)
    return DescriptiveResult(
        name="eg_coint",
        value=best_stat,
        extra={
            "adf_statistic": best_stat,
            "p_value": approx_p,
            "beta": beta.tolist(),
            "critical_values": crit,
            "n": n,
        },
    )


coitg = eg_coint


def cheatsheet() -> str:
    return "eg_coint({}) -> Engle-Granger cointegration test."
