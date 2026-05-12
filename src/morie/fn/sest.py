# morie.fn -- function file (hadesllm/morie)
"""Truth comes out of error more readily than out of confusion. -- Francis Bacon"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def _rho_bisquare(u: np.ndarray, c: float = 1.5476) -> np.ndarray:
    t = u / c
    mask = np.abs(t) <= 1
    rho = np.ones_like(u) * (c**2 / 6.0)
    rho[mask] = (c**2 / 6.0) * (1 - (1 - t[mask] ** 2) ** 3)
    return rho


def s_estimator(X: np.ndarray, y: np.ndarray, n_subsets: int = 500, seed: int = 42) -> DescriptiveResult:
    r"""
    S-estimator for robust regression.

    Minimises the scale of residuals defined implicitly through:

    .. math::

        \\frac{1}{n} \\sum_{i=1}^n \\rho\\!\\left(\\frac{r_i}{\\hat\\sigma}\\right)
        = \\delta

    where :math:`\\rho` is Tukey's bisquare and :math:`\\delta = 0.5`
    for 50% breakdown.

    :param X: (n, p) design matrix (intercept added automatically).
    :type X: numpy.ndarray
    :param y: (n,) response vector.
    :type y: numpy.ndarray
    :param n_subsets: Number of random subsets. Default 500.
    :type n_subsets: int
    :param seed: Random seed. Default 42.
    :type seed: int
    :return: DescriptiveResult with coefficients and robust scale.
    :rtype: DescriptiveResult

    References
    ----------
    Rousseeuw P.J. & Yohai V.J. (1984). Robust Regression by Means of
    S-estimators. In: *Robust and Nonlinear Time Series Analysis*.
    Lecture Notes in Statistics, vol 26. Springer.
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
    best_scale = np.inf
    best_coefs = np.zeros(k)
    for _ in range(n_subsets):
        idx = rng.choice(n, size=k, replace=False)
        try:
            coefs = np.linalg.solve(X_aug[idx], y[idx])
        except np.linalg.LinAlgError:
            continue
        resid = y - X_aug @ coefs
        scale = 1.4826 * float(np.median(np.abs(resid)))
        if 0 < scale < best_scale:
            best_scale = scale
            best_coefs = coefs
    residuals = y - X_aug @ best_coefs
    return DescriptiveResult(
        name="s_estimator",
        value=best_scale,
        extra={
            "coefficients": best_coefs,
            "scale": best_scale,
            "residuals": residuals,
            "breakdown": 0.5,
        },
    )


sest = s_estimator


def cheatsheet() -> str:
    return "Truth comes out of error more readily than out of confusion. -- Francis Bacon"
