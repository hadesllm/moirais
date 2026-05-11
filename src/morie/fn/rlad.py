# morie.fn — function file (hadesllm/morie)
"""Least absolute deviations regression. 'The unexamined statistic is not worth reporting. — adapted from Socrates'"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linprog

from ._containers import DescriptiveResult


def lad_regression(X: np.ndarray, y: np.ndarray) -> DescriptiveResult:
    """
    Least Absolute Deviations (LAD / L1) regression via linear programming.

    Minimises :math:`\\sum_i |y_i - X_i \\beta|` which is more robust to
    outliers than OLS.

    :param X: (n, p) design matrix (without intercept; added automatically).
    :type X: numpy.ndarray
    :param y: (n,) response vector.
    :type y: numpy.ndarray
    :return: DescriptiveResult with coefficients and residuals.
    :rtype: DescriptiveResult
    :raises ValueError: If X and y have incompatible shapes.

    References
    ----------
    Bloomfield P. & Steiger W.L. (1983). *Least Absolute Deviations:
    Theory, Applications, and Algorithms*. Birkhauser.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, p = X.shape
    if len(y) != n:
        raise ValueError(f"X has {n} rows but y has {len(y)} elements.")
    X_aug = np.column_stack([np.ones(n), X])
    k = X_aug.shape[1]
    c = np.concatenate([np.zeros(k), np.ones(n), np.ones(n)])
    A_eq = np.column_stack([X_aug, np.eye(n), -np.eye(n)])
    bounds = [(None, None)] * k + [(0, None)] * (2 * n)
    res = linprog(c, A_eq=A_eq, b_eq=y, bounds=bounds, method="highs")
    coefs = res.x[:k]
    residuals = y - X_aug @ coefs
    return DescriptiveResult(
        name="lad_regression",
        value=float(np.sum(np.abs(residuals))),
        extra={
            "coefficients": coefs,
            "residuals": residuals,
            "mad": float(np.median(np.abs(residuals))),
        },
    )


rlad = lad_regression


def cheatsheet() -> str:
    return "lad_regression({}) -> Least absolute deviations regression. 'The dark side clouds "
