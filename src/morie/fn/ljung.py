# morie.fn -- function file (hadesllm/morie)
"""Ljung-Box test for autocorrelation."""

from __future__ import annotations

import numpy as np
from scipy import stats as _st

from ._containers import TestResult


def ljung_box(x, cdf=None, *, n_lags: int = 10) -> TestResult:
    """Ljung-Box Q-test for residual autocorrelation.

    Parameters
    ----------
    x : array-like
        Residuals or time series.
    n_lags : int
        Number of lags to test. Default 10.

    Returns
    -------
    TestResult
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    x_c = x - x.mean()
    var = np.sum(x_c**2) / n
    if var == 0:
        return TestResult(test_name="Ljung-Box", statistic=0.0, p_value=1.0, df=float(n_lags), n=n)
    acf_vals = np.array([np.sum(x_c[: n - k] * x_c[k:]) / (n * var) for k in range(1, n_lags + 1)])
    Q = float(n * (n + 2) * np.sum(acf_vals**2 / np.arange(n - 1, n - n_lags - 1, -1)))
    p_val = float(1 - _st.chi2.cdf(Q, n_lags))
    return TestResult(
        test_name="Ljung-Box",
        statistic=Q,
        p_value=p_val,
        df=float(n_lags),
        n=n,
        method=f"Ljung-Box Q (lags={n_lags})",
    )


ljung = ljung_box


def cheatsheet() -> str:
    return "ljung_box({}) -> Ljung-Box test for autocorrelation."
