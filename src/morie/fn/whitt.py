"""White's test for heteroskedasticity."""

from __future__ import annotations

import numpy as np
from scipy import stats as _st

from ._containers import DescriptiveResult


def white_test(
    y: np.ndarray,
    X: np.ndarray,
    *,
    add_intercept: bool = True,
) -> DescriptiveResult:
    r"""White's general test for heteroskedasticity.

    Regresses squared OLS residuals on all predictors, their squares,
    and cross-products.  Under :math:`H_0` (homoscedasticity),
    :math:`nR^2 \\sim \\chi^2_q`.

    Parameters
    ----------
    y : (n,) response
    X : (n, p) predictors
    add_intercept : bool

    Returns
    -------
    DescriptiveResult
        ``value`` is the nR^2 test statistic.

    References
    ----------
    White, H. (1980). A heteroskedasticity-consistent covariance matrix
    estimator and a direct test for heteroskedasticity. *Econometrica*,
    48(4), 817--838.
    """
    y = np.asarray(y, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, p = X.shape

    X_ols = np.column_stack([np.ones(n), X]) if add_intercept else X
    beta = np.linalg.lstsq(X_ols, y, rcond=None)[0]
    e = y - X_ols @ beta
    e_sq = e ** 2

    aux_cols = [np.ones(n)]
    for j in range(p):
        aux_cols.append(X[:, j])
    for j in range(p):
        aux_cols.append(X[:, j] ** 2)
    for j in range(p):
        for k in range(j + 1, p):
            aux_cols.append(X[:, j] * X[:, k])
    Z = np.column_stack(aux_cols)
    q = Z.shape[1] - 1

    gamma = np.linalg.lstsq(Z, e_sq, rcond=None)[0]
    e_sq_hat = Z @ gamma
    ss_res = float(np.sum((e_sq - e_sq_hat) ** 2))
    ss_tot = float(np.sum((e_sq - np.mean(e_sq)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    test_stat = n * r2
    p_value = float(_st.chi2.sf(test_stat, df=q))

    return DescriptiveResult(
        name="White Test",
        value=float(test_stat),
        extra={
            "p_value": p_value,
            "df": q,
            "r2_auxiliary": r2,
            "n": n,
            "reject_H0": p_value < 0.05,
        },
    )


whitt = white_test


def cheatsheet() -> str:
    return "white_test({}) -> White's test for heteroskedasticity."
