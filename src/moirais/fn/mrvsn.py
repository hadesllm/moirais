# moirais.fn — function file (hadesllm/moirais)
"""Granger causality test. 'Choice is an illusion.' -- The Merovingian"""

from __future__ import annotations

import numpy as np

from ._containers import TestResult


def granger_causality(x: np.ndarray | list, y: np.ndarray | list, cdf=None, *, max_lag: int = 4) -> TestResult:
    """Bivariate Granger causality test: does *x* Granger-cause *y*?

    Compares a restricted AR model of *y* on its own lags to an
    unrestricted model that also includes lags of *x*, using an F-test.

    Parameters
    ----------
    x, y : array-like
        Time series of equal length (T observations).
    max_lag : int
        Maximum lag order (must be >= 1).

    Returns
    -------
    TestResult
        F-statistic and p-value for the null H0: x does NOT Granger-cause y.

    References
    ----------
    Granger, C. W. J. (1969). Investigating causal relations by econometric
    models and cross-spectral methods. Econometrica, 37(3), 424-438.
    """
    from scipy import stats as _st

    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if x_arr.ndim != 1 or y_arr.ndim != 1:
        raise ValueError("x and y must be 1-D")
    if len(x_arr) != len(y_arr):
        raise ValueError("x and y must have the same length")
    T = len(y_arr)
    if max_lag < 1:
        raise ValueError("max_lag must be >= 1")
    if 2 * max_lag + 1 >= T:
        raise ValueError("Not enough observations for the given max_lag")

    dep = y_arr[max_lag:]
    n = len(dep)

    Z_r = np.column_stack([y_arr[max_lag - k - 1 : T - k - 1] for k in range(max_lag)])
    Z_r = np.column_stack([np.ones(n), Z_r])

    Z_u = np.column_stack(
        [
            Z_r,
            *[x_arr[max_lag - k - 1 : T - k - 1].reshape(-1, 1) for k in range(max_lag)],
        ]
    )

    def _ssr(Z, dep):
        beta, *_ = np.linalg.lstsq(Z, dep, rcond=None)
        resid = dep - Z @ beta
        return float(np.sum(resid**2))

    ssr_r = _ssr(Z_r, dep)
    ssr_u = _ssr(Z_u, dep)

    df_num = max_lag
    df_den = n - Z_u.shape[1]
    if df_den <= 0:
        raise ValueError("Not enough observations for F-test degrees of freedom")

    f_stat = ((ssr_r - ssr_u) / df_num) / (ssr_u / df_den)
    p_val = float(1 - _st.f.cdf(f_stat, df_num, df_den))

    return TestResult(
        test_name="Granger Causality",
        statistic=float(f_stat),
        p_value=p_val,
        df=float(df_num),
        method=f"F-test (lags={max_lag})",
        n=T,
        extra={"max_lag": max_lag, "ssr_restricted": ssr_r, "ssr_unrestricted": ssr_u, "df_den": df_den},
    )


mrvsn = granger_causality


def cheatsheet() -> str:
    return "granger_causality({}) -> Granger causality test. 'Choice is an illusion.' -- The Mero"
