"""White's test for heteroscedasticity."""

from typing import Union

import numpy as np
import scipy.stats as stats

from ._containers import TestResult


def white_test(
    residuals: Union[list, np.ndarray],
    X: Union[list, np.ndarray],
) -> TestResult:
    """
    White's general test for heteroscedasticity.

    Unlike Breusch-Pagan, White's test does not assume a specific functional
    form. The auxiliary regression includes all regressors, their squares,
    and cross-products.

    .. math::

        LM = n \\cdot R^2_{e^2 \\sim X,\\, X^2,\\, X \\times X}

    :param residuals: OLS residuals (1-D array, length n).
    :param X: Design matrix (n x k), **without** intercept.
    :return: TestResult with chi2 statistic, p_value, df.
    :raises ValueError: If dimensions are inconsistent.

    References
    ----------
    White, H. (1980). A heteroskedasticity-consistent covariance matrix
        estimator and a direct test for heteroskedasticity. Econometrica,
        48(4), 817-838.
    """
    e = np.asarray(residuals, dtype=float).ravel()
    Xm = np.asarray(X, dtype=float)
    if Xm.ndim == 1:
        Xm = Xm.reshape(-1, 1)

    n = len(e)
    if Xm.shape[0] != n:
        raise ValueError(f"X has {Xm.shape[0]} rows but residuals has {n} elements.")

    k = Xm.shape[1]

    # Build auxiliary regressors: X, X^2, cross-products
    terms = [np.ones((n, 1)), Xm]

    # Squared terms
    terms.append(Xm**2)

    # Cross-products
    for i in range(k):
        for j in range(i + 1, k):
            terms.append((Xm[:, i] * Xm[:, j]).reshape(-1, 1))

    Z = np.hstack(terms)
    df = Z.shape[1] - 1  # exclude intercept

    # Dependent variable: squared residuals
    e2 = e**2

    # Auxiliary OLS
    beta, _, _, _ = np.linalg.lstsq(Z, e2, rcond=None)
    fitted = Z @ beta
    ss_reg = np.sum((fitted - np.mean(e2)) ** 2)
    ss_tot = np.sum((e2 - np.mean(e2)) ** 2)
    r2 = ss_reg / ss_tot if ss_tot > 0 else 0.0

    chi2_stat = float(n * r2)
    p_value = float(stats.chi2.sf(chi2_stat, df=df))

    return TestResult(
        test_name="White",
        statistic=chi2_stat,
        p_value=p_value,
        df=df,
        method="White's test for heteroscedasticity",
        n=n,
    )


white = white_test


def cheatsheet() -> str:
    return "white_test({}) -> White's test for heteroscedasticity."
