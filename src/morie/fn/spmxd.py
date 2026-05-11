"""Spatial mixed model — random spatial effects (Schabenberger & Gotway Ch 7)."""

import numpy as np
from scipy.spatial.distance import cdist


def spmxd(
    coords: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    *,
    range_param: float = 1.0,
    sill: float = 1.0,
    nugget: float = 0.1,
    max_iter: int = 50,
    tol: float = 1e-6,
) -> dict:
    """
    Fit a spatial linear mixed model with spatially correlated random effects.

    .. math::

        y = X\\beta + u + \\varepsilon

    where :math:`u \\sim N(0, \\sigma^2 R(\\phi))` and
    :math:`\\varepsilon \\sim N(0, \\tau^2 I)`.

    :param coords: Observation coordinates (n, 2).
    :param X: Fixed-effects design matrix (n, p).
    :param y: Response vector (n,).
    :param range_param: Spatial correlation range.
    :param sill: Variance of spatial random effect.
    :param nugget: Nugget (residual) variance.
    :param max_iter: Maximum EM iterations.
    :param tol: Convergence tolerance.
    :return: dict with ``beta``, ``random_effects``, ``fitted``, ``log_likelihood``.
    :raises ValueError: If shapes incompatible.

    References
    ----------
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

    sigma2 = sill
    tau2 = nugget

    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    prev_ll = -np.inf

    for _ in range(max_iter):
        V = sigma2 * R + tau2 * np.eye(n)
        try:
            V_inv = np.linalg.inv(V)
        except np.linalg.LinAlgError:
            V_inv = np.linalg.pinv(V)

        A = X.T @ V_inv @ X
        try:
            beta = np.linalg.solve(A, X.T @ V_inv @ y)
        except np.linalg.LinAlgError:
            beta = np.linalg.lstsq(A, X.T @ V_inv @ y, rcond=None)[0]

        resid = y - X @ beta
        u = sigma2 * R @ V_inv @ resid

        r = resid - u
        sigma2_new = float(u @ np.linalg.solve(R + 1e-8 * np.eye(n), u)) / n
        tau2_new = float(r @ r) / n
        sigma2 = max(sigma2_new, 1e-8)
        tau2 = max(tau2_new, 1e-8)

        sign, logdet = np.linalg.slogdet(V)
        ll = -0.5 * (n * np.log(2 * np.pi) + logdet + float(resid @ V_inv @ resid))
        if abs(ll - prev_ll) < tol:
            break
        prev_ll = ll

    return {
        "beta": beta,
        "random_effects": u,
        "fitted": X @ beta + u,
        "residuals": y - X @ beta - u,
        "sigma2": sigma2,
        "tau2": tau2,
        "log_likelihood": float(ll),
        "n": n,
        "p": X.shape[1],
    }


spmxd_fn = spmxd


def cheatsheet() -> str:
    return "spmxd({}) -> Spatial linear mixed model."
