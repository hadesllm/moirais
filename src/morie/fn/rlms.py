# morie.fn — function file (hadesllm/morie)
"""Least median of squares regression. 'The road up and the road down are the same thing. — Heraclitus' -- Chirrut Imwe"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def lms_regression(X: np.ndarray, y: np.ndarray, n_subsets: int = 500, seed: int = 42) -> DescriptiveResult:
    """
    Least Median of Squares (LMS) regression.

    Minimises :math:`\\mathrm{median}_i (y_i - X_i \\beta)^2` by
    sampling random p-subsets and computing exact fits.

    :param X: (n, p) design matrix (intercept added automatically).
    :type X: numpy.ndarray
    :param y: (n,) response vector.
    :type y: numpy.ndarray
    :param n_subsets: Number of random subsets to try. Default 500.
    :type n_subsets: int
    :param seed: Random seed. Default 42.
    :type seed: int
    :return: DescriptiveResult with best coefficients and median residual.
    :rtype: DescriptiveResult

    References
    ----------
    Rousseeuw P.J. (1984). Least median of squares regression. *Journal
    of the American Statistical Association*, 79(388), 871-880.
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
    rng = np.random.default_rng(seed)
    best_med = np.inf
    best_coefs = np.zeros(k)
    for _ in range(n_subsets):
        idx = rng.choice(n, size=k, replace=False)
        Xs = X_aug[idx]
        ys = y[idx]
        try:
            coefs = np.linalg.solve(Xs, ys)
        except np.linalg.LinAlgError:
            continue
        resid2 = (y - X_aug @ coefs) ** 2
        med = float(np.median(resid2))
        if med < best_med:
            best_med = med
            best_coefs = coefs
    residuals = y - X_aug @ best_coefs
    return DescriptiveResult(
        name="lms_regression",
        value=best_med,
        extra={
            "coefficients": best_coefs,
            "median_sq_residual": best_med,
            "residuals": residuals,
        },
    )


rlms = lms_regression


def cheatsheet() -> str:
    return "Nature does not hurry, yet everything is accomplished. — Lao Tzu"
