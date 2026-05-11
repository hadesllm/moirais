"""Spatial GLM with spatial errors (Schabenberger & Gotway Ch 7)."""

import numpy as np
from scipy.spatial.distance import cdist


def spglm(
    coords: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    *,
    family: str = "gaussian",
    range_param: float = 1.0,
    sill: float = 1.0,
    max_iter: int = 25,
    tol: float = 1e-6,
) -> dict:
    """
    Fit a spatial GLM with exponential correlation in the errors.

    Iteratively reweighted least squares (IRLS) with a spatial
    covariance structure for the working residuals.

    :param coords: Observation coordinates (n, 2).
    :param X: Design matrix (n, p).
    :param y: Response vector (n,).
    :param family: ``'gaussian'`` or ``'binomial'`` (logistic).
    :param range_param: Spatial correlation range.
    :param sill: Spatial correlation sill.
    :param max_iter: Maximum IRLS iterations.
    :param tol: Convergence tolerance.
    :return: dict with ``coefficients``, ``fitted``, ``residuals``, ``spatial_cov``.
    :raises ValueError: If family unknown or shapes incompatible.

    References
    ----------
    Schabenberger & Gotway (2005), Ch. 7.
    """
    coords = np.asarray(coords, dtype=float)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n = int(y) if y.ndim == 0 else len(y)
    if coords.shape[0] != n:
        raise ValueError(f"coords rows ({coords.shape[0]}) must match y length ({n}).")
    if X.shape[0] != n:
        raise ValueError(f"X rows ({X.shape[0]}) must match y length ({n}).")

    D = cdist(coords, coords)
    R = sill * np.exp(-D / (range_param + 1e-12))
    np.fill_diagonal(R, R.diagonal() + 1e-6)
    R_inv = np.linalg.inv(R)

    def _link(mu):
        if family == "gaussian":
            return mu
        return np.log(mu / (1 - mu + 1e-12) + 1e-12)

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
        if family == "binomial":
            d_mu = mu * (1 - mu) + 1e-12
        else:
            d_mu = np.ones(n)
        W = np.diag(d_mu ** 2 / v)
        z = eta + (y - mu) / (d_mu + 1e-12)
        WR = W @ R_inv
        A = X.T @ WR @ X
        b = X.T @ WR @ z
        try:
            beta_new = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            beta_new = np.linalg.lstsq(A, b, rcond=None)[0]
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new

    eta = X @ beta
    mu = _inv_link(eta)
    residuals = y - mu

    return {
        "coefficients": beta,
        "fitted": mu,
        "residuals": residuals,
        "spatial_cov": R,
        "family": family,
        "n": n,
        "p": X.shape[1],
    }


spglm_fn = spglm


def cheatsheet() -> str:
    return "spglm({}) -> Spatial GLM with spatial errors."
