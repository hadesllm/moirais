"""Penalized quasi-likelihood spatial GLMM."""

from __future__ import annotations

import numpy as np

from ._containers import SpatialResult


def pql_spatial_glmm(
    Z: np.ndarray,
    X: np.ndarray,
    coords: np.ndarray,
    family: str = "poisson",
    cov_params: dict | None = None,
    max_iter: int = 20,
) -> SpatialResult:
    r"""PQL estimation for a spatial GLMM.

    Parameters
    ----------
    Z : np.ndarray
        Response, shape ``(n,)``.
    X : np.ndarray
        Design matrix, shape ``(n, p)``.
    coords : np.ndarray
        Observation coordinates, shape ``(n, 2)``.
    family : str
        ``"poisson"`` or ``"binomial"``.
    cov_params : dict, optional
        ``{"sill", "range", "nugget"}``.
    max_iter : int
        PQL iterations.

    Returns
    -------
    SpatialResult
        ``statistic`` is the first fixed-effect coefficient.
        ``extra`` has ``beta``, ``random_effects``, ``sigma2``.

    References
    ----------
    Schabenberger & Gotway (2005), Ch. 6.

    .. epigraph::

    """
    Z = np.asarray(Z, dtype=np.float64).ravel()
    X = np.asarray(X, dtype=np.float64)
    coords = np.asarray(coords, dtype=np.float64)
    n = len(Z)
    params = cov_params or {"sill": 1.0, "range": 1.0, "nugget": 0.1}
    sill = params.get("sill", 1.0)
    rng = params.get("range", 1.0)
    nug = params.get("nugget", 0.1)

    dist = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1))
    G = sill * np.exp(-dist / rng) + nug * np.eye(n)

    beta = np.zeros(X.shape[1])
    u = np.zeros(n)

    for _ in range(max_iter):
        eta = X @ beta + u
        if family == "poisson":
            mu = np.exp(eta)
            mu = np.maximum(mu, 1e-10)
            w = mu
        else:
            mu = 1.0 / (1.0 + np.exp(-eta))
            mu = np.clip(mu, 1e-10, 1 - 1e-10)
            w = mu * (1 - mu)

        y_tilde = eta + (Z - mu) / np.maximum(w, 1e-10)
        R_inv = np.diag(w)
        V = G + np.diag(1.0 / np.maximum(w, 1e-10))
        V_inv = np.linalg.inv(V)
        XtViX = X.T @ V_inv @ X
        beta_new = np.linalg.solve(XtViX, X.T @ V_inv @ y_tilde)
        resid = y_tilde - X @ beta_new
        u = G @ V_inv @ resid
        beta = beta_new

    sigma2 = float(np.var(Z - mu))

    return SpatialResult(
        name="pql_spatial_glmm",
        statistic=float(beta[0]),
        p_value=None,
        extra={"beta": beta, "random_effects": u, "sigma2": sigma2},
    )


sgpql = pql_spatial_glmm


def cheatsheet() -> str:
    return "pql_spatial_glmm({}) -> Penalized quasi-likelihood spatial GLMM."
