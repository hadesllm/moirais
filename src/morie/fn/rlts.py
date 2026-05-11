# morie.fn — function file (hadesllm/morie)
"""Least trimmed squares regression. 'There is always a bigger fish.' -- Qui-Gon Jinn"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def lts_regression(
    X: np.ndarray, y: np.ndarray, h: int | None = None, n_subsets: int = 500, seed: int = 42
) -> DescriptiveResult:
    """
    Least Trimmed Squares (LTS) regression.

    Minimises :math:`\\sum_{i=1}^{h} r_{(i)}^2` where :math:`r_{(i)}^2`
    are the *h* smallest squared residuals. Default
    :math:`h = \\lfloor(n + p + 1)/2\\rfloor`.

    :param X: (n, p) design matrix (intercept added automatically).
    :type X: numpy.ndarray
    :param y: (n,) response vector.
    :type y: numpy.ndarray
    :param h: Number of observations to keep. Default floor((n+p+1)/2).
    :type h: int or None
    :param n_subsets: Random subsets to sample. Default 500.
    :type n_subsets: int
    :param seed: Random seed. Default 42.
    :type seed: int
    :return: DescriptiveResult with coefficients and trimmed sum of squares.
    :rtype: DescriptiveResult

    References
    ----------
    Rousseeuw P.J. & Leroy A.M. (1987). *Robust Regression and Outlier
    Detection*. Wiley.
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
    if h is None:
        h = (n + k + 1) // 2
    rng = np.random.default_rng(seed)
    best_obj = np.inf
    best_coefs = np.zeros(k)
    for _ in range(n_subsets):
        idx = rng.choice(n, size=k, replace=False)
        try:
            coefs = np.linalg.solve(X_aug[idx], y[idx])
        except np.linalg.LinAlgError:
            continue
        resid2 = (y - X_aug @ coefs) ** 2
        sorted_r2 = np.sort(resid2)
        obj = float(sorted_r2[:h].sum())
        if obj < best_obj:
            best_obj = obj
            best_coefs = coefs
    residuals = y - X_aug @ best_coefs
    return DescriptiveResult(
        name="lts_regression",
        value=best_obj,
        extra={
            "coefficients": best_coefs,
            "trimmed_ss": best_obj,
            "h": h,
            "residuals": residuals,
        },
    )


rlts = lts_regression


def cheatsheet() -> str:
    return "lts_regression({}) -> Least trimmed squares regression. 'There is always a bigger "
