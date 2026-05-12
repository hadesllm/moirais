# morie.fn — function file (hadesllm/morie)
"""Ramsey RESET specification test."""

from typing import Union

import numpy as np
import scipy.stats as stats

from ._containers import TestResult


def ramsey_reset_test(
    y: Union[list, np.ndarray],
    X: Union[list, np.ndarray],
    fitted: Union[list, np.ndarray],
    *,
    power: int = 3,
) -> TestResult:
    r"""
    Ramsey RESET (Regression Equation Specification Error Test).

    Tests for omitted non-linearities by augmenting the original model
    with powers of the fitted values and testing their joint significance.

    .. math::

        y = X\\beta + \\hat{y}^2 \\gamma_2 + \\hat{y}^3 \\gamma_3 + \\varepsilon

    H0: :math:`\\gamma_2 = \\gamma_3 = 0` (no misspecification).

    :param y: Response variable (length n).
    :param X: Original design matrix (n x k), should include intercept.
    :param fitted: Fitted values from the original OLS (length n).
    :param power: Highest power of fitted values to include. Default 3.
    :return: TestResult with F statistic, p_value, df=(q, n-k-q).
    :raises ValueError: If dimensions are inconsistent.

    References
    ----------
    Ramsey, J. B. (1969). Tests for specification errors in classical linear
        least-squares regression analysis. Journal of the Royal Statistical
        Society B, 31(2), 350-371.
    """
    y_arr = np.asarray(y, dtype=float).ravel()
    X_arr = np.asarray(X, dtype=float)
    y_hat = np.asarray(fitted, dtype=float).ravel()

    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)

    n = len(y_arr)
    if X_arr.shape[0] != n or len(y_hat) != n:
        raise ValueError("y, X, and fitted must have the same number of rows.")

    k = X_arr.shape[1]

    # Augmented regressors: y_hat^2, y_hat^3, ...
    aug_cols = [y_hat**p for p in range(2, power + 1)]
    q = len(aug_cols)
    X_aug = np.column_stack([X_arr] + aug_cols)

    # Restricted model (original)
    _, res_r, _, _ = np.linalg.lstsq(X_arr, y_arr, rcond=None)
    ssr_r = float(np.sum((y_arr - X_arr @ np.linalg.lstsq(X_arr, y_arr, rcond=None)[0]) ** 2))

    # Unrestricted model (augmented)
    beta_u = np.linalg.lstsq(X_aug, y_arr, rcond=None)[0]
    ssr_u = float(np.sum((y_arr - X_aug @ beta_u) ** 2))

    df1 = q
    df2 = n - k - q

    if df2 <= 0 or ssr_u <= 0:
        return TestResult(
            test_name="RESET",
            statistic=float("nan"),
            p_value=float("nan"),
            df=df1,
            method="Ramsey RESET (insufficient df)",
            n=n,
        )

    F = ((ssr_r - ssr_u) / df1) / (ssr_u / df2)
    p_value = float(stats.f.sf(F, df1, df2))

    return TestResult(
        test_name="RESET",
        statistic=float(F),
        p_value=p_value,
        df=df1,
        method=f"Ramsey RESET (powers 2..{power})",
        n=n,
        extra={"df1": df1, "df2": df2, "ssr_restricted": ssr_r, "ssr_unrestricted": ssr_u},
    )


reset = ramsey_reset_test


def cheatsheet() -> str:
    return "ramsey_reset_test({}) -> Ramsey RESET specification test."
