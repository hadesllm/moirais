"""Spatial GEE — generalized estimating equations (Schabenberger & Gotway Ch 7)."""

import numpy as np
from scipy.spatial.distance import cdist


def spgee(
    coords: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    *,
    family: str = "gaussian",
    range_param: float = 1.0,
    max_iter: int = 25,
    tol: float = 1e-6,
) -> dict:
    """
    Fit a spatial GEE model with exponential working correlation.

    GEE provides consistent regression coefficient estimates even when
    the spatial correlation structure is mis-specified, via the sandwich
    variance estimator.

    :param coords: Observation coordinates (n, 2).
    :param X: Design matrix (n, p).
    :param y: Response vector (n,).
    :param family: ``'gaussian'`` or ``'binomial'``.
    :param range_param: Spatial working correlation range.
    :param max_iter: Maximum iterations.
    :param tol: Convergence tolerance.
    :return: dict with ``coefficients``, ``robust_se``, ``fitted``, ``sandwich_cov``.
    :raises ValueError: If shapes incompatible.

    References
    ----------
    Albert, P. S. & McShane, L. M. (1995). A generalized estimating
    equations approach for spatially correlated binary data.
    *Biometrics*, 51(2), 627-638.

    Schabenberger & Gotway (2005), Ch. 7.
    """
    coords = np.asarray(coords, dtype=float)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n = int(y) if y.ndim == 0 else len(y)
    if coords.shape[0] != n or X.shape[0] != n:
        raise ValueError("coords, X, and y must have same number of rows.")

    D = cdist(coords, coords)
    R = np.exp(-D / (range_param + 1e-12))
    np.fill_diagonal(R, 1.0)
    R_inv = np.linalg.inv(R + 1e-6 * np.eye(n))

    def _inv_link(eta):
        if family == "gaussian":
            return eta
        return 1.0 / (1.0 + np.exp(-eta))

    def _var_func(mu):
        if family == "gaussian":
            return np.ones_like(mu)
        return mu * (1 - mu) + 1e-12

    beta = np.linalg.lstsq(X, y, rcond=None)[0]

    for _ in range(max_iter):
        eta = X @ beta
        mu = _inv_link(eta)
        v = _var_func(mu)
        A_half = np.diag(np.sqrt(v))
        A_half_inv = np.diag(1.0 / np.sqrt(v))
        V_inv = A_half_inv @ R_inv @ A_half_inv
        S = X.T @ V_inv @ (y - mu)
        H = X.T @ V_inv @ X
        try:
            delta = np.linalg.solve(H, S)
        except np.linalg.LinAlgError:
            delta = np.linalg.lstsq(H, S, rcond=None)[0]
        beta = beta + delta
        if np.max(np.abs(delta)) < tol:
            break

    eta = X @ beta
    mu = _inv_link(eta)
    resid = y - mu

    H_inv = np.linalg.inv(H + 1e-8 * np.eye(len(beta)))
    B = X.T @ np.diag(resid ** 2) @ X
    sandwich = H_inv @ B @ H_inv
    robust_se = np.sqrt(np.diag(sandwich))

    return {
        "coefficients": beta,
        "robust_se": robust_se,
        "sandwich_cov": sandwich,
        "fitted": mu,
        "residuals": resid,
        "family": family,
        "n": n,
        "p": X.shape[1],
    }


spgee_fn = spgee


def cheatsheet() -> str:
    return "spgee({}) -> Spatial GEE with sandwich variance."
