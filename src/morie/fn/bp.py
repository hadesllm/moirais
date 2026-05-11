# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Breusch-Pagan test for heteroscedasticity."""

from typing import Union

import numpy as np
import scipy.stats as stats

from ._containers import TestResult


def breusch_pagan_test(
    residuals: Union[list, np.ndarray],
    X: Union[list, np.ndarray],
) -> TestResult:
    """
    Breusch-Pagan Lagrange Multiplier test for heteroscedasticity.

    Tests H0: error variances are all equal (homoscedasticity) against
    H1: error variances are a multiplicative function of one or more
    variables in *X*.

    .. math::

        LM = n \\cdot R^2_{e^2 \\sim X}

    where :math:`e^2` are the squared residuals from the original model
    and :math:`R^2` is from the auxiliary regression of :math:`e^2` on *X*.

    :param residuals: OLS residuals (1-D array, length n).
    :param X: Design matrix (n x k) from the original regression.
    :return: TestResult with LM statistic, p_value, df=k.
    :raises ValueError: If dimensions are inconsistent.

    References
    ----------
    Breusch, T. S., & Pagan, A. R. (1979). A simple test for
        heteroscedasticity and random coefficient variation. Econometrica,
        47(5), 1287-1294.
    """
    e = np.asarray(residuals, dtype=float).ravel()
    Xm = np.asarray(X, dtype=float)
    if Xm.ndim == 1:
        Xm = Xm.reshape(-1, 1)

    n = len(e)
    if Xm.shape[0] != n:
        raise ValueError(f"X has {Xm.shape[0]} rows but residuals has {n} elements.")

    # Squared residuals, normalised
    e2 = e**2
    e2_norm = e2 / np.mean(e2)

    # Add intercept if not present
    if not np.all(Xm[:, 0] == 1):
        Xm_aug = np.column_stack([np.ones(n), Xm])
    else:
        Xm_aug = Xm

    k = Xm_aug.shape[1] - 1  # df = number of regressors (excl intercept)

    # Auxiliary OLS: e2_norm ~ Xm_aug
    beta, res, _, _ = np.linalg.lstsq(Xm_aug, e2_norm, rcond=None)
    fitted = Xm_aug @ beta
    ss_reg = np.sum((fitted - np.mean(e2_norm)) ** 2)
    ss_tot = np.sum((e2_norm - np.mean(e2_norm)) ** 2)
    r2 = ss_reg / ss_tot if ss_tot > 0 else 0.0

    lm_stat = float(n * r2)
    p_value = float(stats.chi2.sf(lm_stat, df=k))

    return TestResult(
        test_name="Breusch-Pagan",
        statistic=lm_stat,
        p_value=p_value,
        df=k,
        method="Breusch-Pagan LM test",
        n=n,
    )


bp = breusch_pagan_test


def cheatsheet() -> str:
    return "breusch_pagan_test({}) -> Breusch-Pagan test for heteroscedasticity."
