# morie.fn — function file (hadesllm/morie)
"""MM-estimator for robust regression. 'So this is how liberty dies.' -- Padme Amidala"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def _tukey_bisquare(r: np.ndarray, c: float) -> np.ndarray:
    u = r / c
    mask = np.abs(u) <= 1
    w = np.zeros_like(r)
    w[mask] = (1 - u[mask] ** 2) ** 2
    return w


def mm_estimator(
    X: np.ndarray, y: np.ndarray, efficiency: float = 0.95, max_iter: int = 50, tol: float = 1e-6, seed: int = 42
) -> DescriptiveResult:
    """
    MM-estimator for robust regression.

    Combines a high-breakdown S-estimate initial fit with an efficient
    M-estimate reweighting step using Tukey's bisquare weight function.

    :param X: (n, p) design matrix (intercept added automatically).
    :type X: numpy.ndarray
    :param y: (n,) response vector.
    :type y: numpy.ndarray
    :param efficiency: Asymptotic efficiency target (0.85 or 0.95). Default 0.95.
    :type efficiency: float
    :param max_iter: Maximum IRLS iterations. Default 50.
    :type max_iter: int
    :param tol: Convergence tolerance. Default 1e-6.
    :type tol: float
    :param seed: Random seed for initial S-estimate. Default 42.
    :type seed: int
    :return: DescriptiveResult with coefficients and scale.
    :rtype: DescriptiveResult

    References
    ----------
    Yohai V.J. (1987). High breakdown-point and high efficiency robust
    estimates for regression. *Annals of Statistics*, 15(2), 642-656.
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
    c_val = 4.685 if efficiency >= 0.95 else 3.443
    rng = np.random.default_rng(seed)
    best_scale = np.inf
    best_coefs = np.zeros(k)
    for _ in range(200):
        idx = rng.choice(n, size=k, replace=False)
        try:
            coefs = np.linalg.solve(X_aug[idx], y[idx])
        except np.linalg.LinAlgError:
            continue
        resid = y - X_aug @ coefs
        scale = 1.4826 * float(np.median(np.abs(resid)))
        if scale < best_scale and scale > 1e-12:
            best_scale = scale
            best_coefs = coefs
    coefs = best_coefs.copy()
    scale = best_scale
    for _ in range(max_iter):
        resid = y - X_aug @ coefs
        w = _tukey_bisquare(resid / scale, c_val)
        W = np.diag(w)
        try:
            coefs_new = np.linalg.solve(X_aug.T @ W @ X_aug, X_aug.T @ W @ y)
        except np.linalg.LinAlgError:
            break
        if np.max(np.abs(coefs_new - coefs)) < tol:
            coefs = coefs_new
            break
        coefs = coefs_new
    residuals = y - X_aug @ coefs
    return DescriptiveResult(
        name="mm_estimator",
        value=float(scale),
        extra={
            "coefficients": coefs,
            "scale": float(scale),
            "residuals": residuals,
            "efficiency": efficiency,
        },
    )


mmest = mm_estimator


def cheatsheet() -> str:
    return "_tukey_bisquare({}) -> MM-estimator for robust regression. 'So this is how liberty "
