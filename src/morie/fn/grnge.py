# morie.fn — function file (hadesllm/morie)
"""Granger causality test."""

from __future__ import annotations

import numpy as np
from scipy import stats as _st

from ._containers import TestResult

__all__ = ["grnge", "granger_test"]


def granger_test(y1, y2, maxlag: int = 4, cdf=None) -> TestResult:
    """Granger causality test: does y2 Granger-cause y1?

    Tests H0: y2 does not improve forecasts of y1 when past y1 is already
    included (i.e. all lag coefficients of y2 in the unrestricted VAR are
    jointly zero).

    The F-statistic compares the restricted model (y1 ~ lagged y1 + constant)
    with the unrestricted model (y1 ~ lagged y1 + lagged y2 + constant).

    Parameters
    ----------
    y1 : array-like
        Dependent variable (n,).
    y2 : array-like
        Potential cause (n,).
    maxlag : int
        Number of lags for both variables.  Default 4.

    Returns
    -------
    TestResult
        statistic: F-statistic.
        p_value: p-value from F(maxlag, n - 2*maxlag - 1) distribution.
        df: float(maxlag) (numerator df).
        extra['df_denom']: denominator degrees of freedom.
        extra['maxlag']: maxlag used.
        extra['rss_restricted']: RSS of restricted model.
        extra['rss_unrestricted']: RSS of unrestricted model.

    References
    ----------
    Granger C.W.J. (1969). Investigating causal relations by econometric
    models and cross-spectral methods.
    Econometrica, 37(3), 424-438.
    """
    y1 = np.asarray(y1, dtype=float).ravel()
    y2 = np.asarray(y2, dtype=float).ravel()
    n = len(y1)
    if len(y2) != n:
        raise ValueError("y1 and y2 must have the same length.")
    if maxlag < 1:
        raise ValueError(f"maxlag must be >= 1, got {maxlag}.")
    if n < 2 * maxlag + 3:
        raise ValueError(
            f"Need n >= 2*maxlag + 3 = {2*maxlag + 3}; got n={n}."
        )

    y_target = y1[maxlag:]
    n_use = len(y_target)

    # Build lagged columns helper.
    def _lag_cols(series, p):
        return np.column_stack([series[p - j : n - j] for j in range(1, p + 1)])

    # Restricted: y1 ~ constant + lagged y1.
    X_res = np.column_stack([np.ones(n_use), _lag_cols(y1, maxlag)])
    try:
        b_res, _, _, _ = np.linalg.lstsq(X_res, y_target, rcond=None)
        rss_res = float(np.sum((y_target - X_res @ b_res) ** 2))
    except np.linalg.LinAlgError:
        rss_res = float(np.sum((y_target - y_target.mean()) ** 2))

    # Unrestricted: y1 ~ constant + lagged y1 + lagged y2.
    X_unres = np.column_stack([X_res, _lag_cols(y2, maxlag)])
    try:
        b_unres, _, _, _ = np.linalg.lstsq(X_unres, y_target, rcond=None)
        rss_unres = float(np.sum((y_target - X_unres @ b_unres) ** 2))
    except np.linalg.LinAlgError:
        rss_unres = rss_res

    df_num = maxlag
    df_den = n_use - X_unres.shape[1]
    if df_den <= 0 or rss_unres <= 0.0:
        f_stat = 0.0
        p_val = 1.0
    else:
        f_stat = float(((rss_res - rss_unres) / df_num) / (rss_unres / df_den))
        p_val = float(1.0 - _st.f.cdf(f_stat, df_num, df_den))

    return TestResult(
        test_name="Granger Causality",
        statistic=float(f_stat),
        p_value=float(p_val),
        df=float(df_num),
        n=n,
        method=f"Granger F-test (maxlag={maxlag})",
        extra={
            "df_denom": df_den,
            "maxlag": maxlag,
            "rss_restricted": rss_res,
            "rss_unrestricted": rss_unres,
        },
    )


grnge = granger_test


def cheatsheet() -> str:
    return "granger_test(y1, y2, maxlag=4) -> Granger causality test (H0: y2 does not cause y1)."
