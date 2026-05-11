# morie.fn — function file (hadesllm/morie)
"""Johansen cointegration test."""

import numpy as np

from ._containers import DescriptiveResult


def johansen_test(Y: np.ndarray, lags: int = 1) -> DescriptiveResult:
    """
    Johansen trace test for cointegration rank.

    :param Y: (n, k) array of k time series.
    :param lags: Number of lagged differences. Default 1.
    :return: DescriptiveResult with trace statistics per rank, eigenvalues.
    :raises ValueError: If insufficient data.

    References
    ----------
    Johansen S. (1991). Estimation and hypothesis testing of
    cointegration vectors in Gaussian vector autoregressive models.
    *Econometrica*, 59(6), 1551-1580.
    """
    Y = np.asarray(Y, dtype=float)
    if Y.ndim == 1:
        raise ValueError("Y must be 2-D with at least 2 columns.")
    n, k = Y.shape
    if n < lags + k + 5:
        raise ValueError(f"Need at least {lags + k + 5} observations, got {n}.")
    dY = np.diff(Y, axis=0)
    T = n - 1 - lags
    dep = dY[lags:]
    Y_lag = Y[lags : n - 1]
    regs = [np.ones((T, 1))]
    for lag in range(1, lags + 1):
        regs.append(dY[lags - lag : n - 1 - lag])
    Z = np.hstack(regs)
    beta_dep = np.linalg.lstsq(Z, dep, rcond=None)[0]
    R0 = dep - Z @ beta_dep
    beta_lag = np.linalg.lstsq(Z, Y_lag, rcond=None)[0]
    R1 = Y_lag - Z @ beta_lag
    S00 = R0.T @ R0 / T
    S11 = R1.T @ R1 / T
    S01 = R0.T @ R1 / T
    S10 = S01.T
    try:
        S11_inv = np.linalg.inv(S11)
    except np.linalg.LinAlgError:
        S11_inv = np.linalg.pinv(S11)
    try:
        S00_inv = np.linalg.inv(S00)
    except np.linalg.LinAlgError:
        S00_inv = np.linalg.pinv(S00)
    M = S00_inv @ S01 @ S11_inv @ S10
    eigenvalues = np.sort(np.real(np.linalg.eigvals(M)))[::-1]
    eigenvalues = np.clip(eigenvalues, 1e-15, 1 - 1e-15)
    trace_stats = []
    for r in range(k):
        tr = -T * np.sum(np.log(1 - eigenvalues[r:]))
        trace_stats.append(float(tr))
    crit_5 = {0: 15.41, 1: 3.76}
    return DescriptiveResult(
        name="johansen_test",
        value=float(trace_stats[0]) if trace_stats else 0.0,
        extra={
            "trace_statistics": trace_stats,
            "eigenvalues": eigenvalues.tolist(),
            "critical_values_5pct": crit_5,
            "lags": lags,
            "k": k,
            "n": n,
        },
    )


johcg = johansen_test


def cheatsheet() -> str:
    return "johansen_test({}) -> Johansen cointegration trace test."
