"""Single-index model estimation."""

from __future__ import annotations

import numpy as np


def simod(
    y: np.ndarray,
    X: np.ndarray,
    *,
    bandwidth: float | None = None,
    kernel: str = "gaussian",
    max_iter: int = 100,
    tol: float = 1e-6,
) -> dict:
    r"""
    Single-index model estimation via iterative least-squares.

    The single-index model is:

    .. math::

        E[Y \mid X] = g(X^\top \beta)

    where :math:`g(\cdot)` is an unknown smooth link function and
    :math:`\beta` is a finite-dimensional index parameter (identified up
    to scale, so :math:`\|\beta\| = 1`).

    Uses an iterative procedure:
    1. Given current :math:`\beta`, form index :math:`z_i = X_i^\top \beta`.
    2. Estimate :math:`g` via Nadaraya-Watson on :math:`(z, Y)`.
    3. Update :math:`\beta` by OLS of :math:`Y` on :math:`\hat{g}'(z) X`.
    4. Normalize :math:`\beta`.

    Parameters
    ----------
    y : np.ndarray
        Response vector (n,).
    X : np.ndarray
        Covariate matrix (n, p).
    bandwidth : float or None
        Bandwidth for NW regression. If None, Silverman's rule.
    kernel : str
        Kernel: ``'gaussian'``, ``'epanechnikov'``, ``'uniform'``.
    max_iter : int
        Maximum iterations. Default 100.
    tol : float
        Convergence tolerance on :math:`\|\beta^{(k+1)} - \beta^{(k)}\|`.

    Returns
    -------
    dict
        Keys: ``beta`` (index coefficients, unit norm), ``g_hat`` (fitted
        link values), ``index`` (X @ beta), ``converged``, ``n_iter``,
        ``bandwidth``, ``n_obs``.

    References
    ----------
    Ichimura, H. (1993). Semiparametric least squares (SLS) and weighted SLS
        estimation of single-index models. Journal of Econometrics, 58, 71-120.
    Horowitz, J. L. (2009). Semiparametric and Nonparametric Methods in
        Econometrics. Springer. Chapter 4.
    """
    y = np.asarray(y, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, p = X.shape
    if y.shape[0] != n:
        raise ValueError(f"y length {y.shape[0]} != X rows {n}.")
    if n < max(10, 2 * p):
        raise ValueError(f"Need at least {max(10, 2*p)} observations, got {n}.")

    from morie.fn.nwker import _silverman_bw, nwker

    beta = np.ones(p) / np.sqrt(p)
    converged = False
    n_iter = 0

    for it in range(max_iter):
        z = X @ beta
        bw = bandwidth if bandwidth is not None else _silverman_bw(z)
        nw_result = nwker(z, y, bandwidth=bw, kernel=kernel)
        g_hat = nw_result["y_hat"]

        eps = 1e-8
        z_plus = z + eps
        g_plus = nwker(z_plus, y, x_eval=z_plus, bandwidth=bw, kernel=kernel)["y_hat"]
        g_prime = (g_plus - g_hat) / eps

        gp_X = g_prime[:, None] * X
        XtX = gp_X.T @ gp_X
        if np.linalg.matrix_rank(XtX) < p:
            break
        beta_new = np.linalg.solve(XtX, gp_X.T @ y)
        norm = np.linalg.norm(beta_new)
        if norm < 1e-15:
            break
        beta_new = beta_new / norm
        if beta_new[0] < 0:
            beta_new = -beta_new

        change = np.linalg.norm(beta_new - beta)
        beta = beta_new
        n_iter = it + 1
        if change < tol:
            converged = True
            break

    z = X @ beta
    bw = bandwidth if bandwidth is not None else _silverman_bw(z)
    g_hat = nwker(z, y, bandwidth=bw, kernel=kernel)["y_hat"]

    return {
        "beta": beta.tolist(),
        "g_hat": g_hat,
        "index": z,
        "converged": converged,
        "n_iter": n_iter,
        "bandwidth": bw,
        "n_obs": n,
    }


simod_fn = simod


def cheatsheet() -> str:
    return "simod({y, X}) -> Single-index model estimation."
