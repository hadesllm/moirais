# morie.fn — function file (hadesllm/morie)
"""KPSS stationarity test."""

from __future__ import annotations

import numpy as np

from ._containers import TestResult


def kpss_test(x, *, n_lags: int | None = None) -> TestResult:
    """KPSS test. H0: series is stationary (opposite of ADF!).

    Parameters
    ----------
    x : array-like
        Time series observations.
    n_lags : int, optional
        Number of lags for Bartlett kernel. Default: ceil(sqrt(n)).

    Returns
    -------
    TestResult
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 10:
        raise ValueError("Need at least 10 observations")
    if n_lags is None:
        n_lags = int(np.ceil(np.sqrt(n)))
    # Detrend (constant only)
    e = x - x.mean()
    S = np.cumsum(e)
    # Long-run variance (Bartlett kernel)
    gamma0 = float(np.sum(e**2) / n)
    lrv = gamma0
    for j in range(1, n_lags + 1):
        w = 1 - j / (n_lags + 1)
        gamma_j = float(np.sum(e[j:] * e[:-j]) / n)
        lrv += 2 * w * gamma_j
    kpss_stat = float(np.sum(S**2) / (n**2 * lrv)) if lrv > 0 else float("inf")
    # Approximate p-value (critical values: 0.347@10%, 0.463@5%, 0.739@1%)
    if kpss_stat > 0.739:
        p_approx = 0.01
    elif kpss_stat > 0.463:
        p_approx = 0.05
    elif kpss_stat > 0.347:
        p_approx = 0.10
    else:
        p_approx = 0.15
    return TestResult(
        test_name="KPSS",
        statistic=kpss_stat,
        p_value=p_approx,
        n=n,
        method=f"KPSS test (lags={n_lags})",
        extra={"n_lags": n_lags, "long_run_variance": lrv},
    )


kpss = kpss_test


def cheatsheet() -> str:
    return "kpss_test({}) -> KPSS stationarity test."
