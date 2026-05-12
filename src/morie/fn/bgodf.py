# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Breusch-Godfrey test for serial correlation."""

from __future__ import annotations

import numpy as np
from scipy import stats as _st

from ._containers import TestResult

__all__ = ["bgodf", "bg_test"]


def bg_test(y, lags: int = 1, x=None, cdf=None) -> TestResult:
    """Breusch-Godfrey LM test for serial correlation.

    Tests H0: no serial correlation up to order *lags* in the OLS residuals.
    The LM statistic is n·R² from the auxiliary regression of residuals on
    the original regressors and lagged residuals.  Under H0, LM ~ chi²(lags).

    Parameters
    ----------
    y : array-like
        Dependent variable (n,).  If *x* is None the series is regressed on a
        constant to obtain residuals.
    lags : int
        Number of lag orders to test.  Default 1.
    x : array-like or None
        Exogenous regressors (n, k).  If None only a constant is used.

    Returns
    -------
    TestResult
        statistic: LM = n·R².
        p_value: right-tail chi²(lags) p-value.
        df: float(lags).
        extra['r_squared']: R² from auxiliary regression.

    References
    ----------
    Breusch T.S. (1978). Testing for autocorrelation in dynamic linear models.
    Australian Economic Papers, 17(31), 334-355.

    Godfrey L.G. (1978). Testing against general autoregressive and moving
    average error models when the regressors include lagged dependent variables.
    Econometrica, 46(6), 1293-1301.
    """
    y = np.asarray(y, dtype=float).ravel()
    n = len(y)
    if lags < 1:
        raise ValueError(f"lags must be >= 1, got {lags}.")
    if n < lags + 3:
        raise ValueError(f"Need n >= lags + 3; got n={n}, lags={lags}.")

    if x is None:
        X_mat = np.ones((n, 1))
    else:
        x_arr = np.asarray(x, dtype=float)
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(-1, 1)
        X_mat = np.column_stack([np.ones(n), x_arr])

    # Step 1: OLS residuals from original model.
    try:
        beta0, _, _, _ = np.linalg.lstsq(X_mat, y, rcond=None)
        resid = y - X_mat @ beta0
    except np.linalg.LinAlgError:
        resid = y - np.mean(y)

    # Step 2: Auxiliary regression -- resid on X_mat + lagged residuals.
    # Pad lagged residuals with zeros for the first *lags* positions
    # (standard Breusch-Godfrey convention).
    lag_cols = []
    for j in range(1, lags + 1):
        col = np.zeros(n)
        col[j:] = resid[: n - j]
        lag_cols.append(col)

    X_aux = np.column_stack([X_mat] + lag_cols)
    try:
        beta_aux, _, _, _ = np.linalg.lstsq(X_aux, resid, rcond=None)
        resid_aux = resid - X_aux @ beta_aux
    except np.linalg.LinAlgError:
        resid_aux = resid

    ss_res = float(np.sum(resid_aux ** 2))
    ss_tot = float(np.sum((resid - resid.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0
    r2 = float(np.clip(r2, 0.0, 1.0))

    lm_stat = float(n * r2)
    p_val = float(1.0 - _st.chi2.cdf(lm_stat, lags))

    return TestResult(
        test_name="Breusch-Godfrey",
        statistic=lm_stat,
        p_value=p_val,
        df=float(lags),
        n=n,
        method=f"BG LM test (lags={lags})",
        extra={"r_squared": r2, "lags": lags},
    )


bgodf = bg_test


def cheatsheet() -> str:
    return "bg_test(y, lags=1, x=None) -> Breusch-Godfrey serial correlation LM test."
