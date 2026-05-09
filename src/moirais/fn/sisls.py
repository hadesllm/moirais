"""Single-index model via iterative SLS (Ichimura 1993)."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def sisls(
    y: np.ndarray,
    X: np.ndarray,
    *,
    bandwidth: float | None = None,
    kernel: str = "gaussian",
    max_iter: int = 50,
    tol: float = 1e-6,
) -> dict:
    r"""
    Single-index model via semiparametric least squares (Ichimura 1993).

    Iteratively estimates :math:`\beta` in :math:`E[Y|X] = G(X'\beta)`
    by alternating between:

    1. Fixing :math:`\beta`, estimate :math:`G` by kernel regression of
       :math:`Y` on :math:`X'\beta`.
    2. Fixing :math:`G`, update :math:`\beta` by NLS.

    Parameters
    ----------
    y : np.ndarray
        Response (n,).
    X : np.ndarray
        Covariates (n, p).
    bandwidth : float or None
        Kernel bandwidth. If None, Silverman's rule.
    kernel : str
        Kernel function.
    max_iter : int
        Maximum iterations.
    tol : float
        Convergence tolerance on beta.

    Returns
    -------
    dict
        ``beta`` (normalised), ``index``, ``g_hat`` (estimated link
        values), ``n_iter``, ``converged``, ``n_obs``.

    References
    ----------
    Ichimura, H. (1993). Semiparametric least squares (SLS) and weighted
        SLS estimation of single-index models. JoE, 58, 71-120.
    Horowitz (2009). Ch 2.
    """
    y = np.asarray(y, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, p = X.shape
    if y.shape[0] != n:
        raise ValueError(f"y length {y.shape[0]} != X rows {n}.")
    if p < 2:
        raise ValueError("Need p >= 2 covariates.")

    from moirais.fn.nwker import _get_kernel, _silverman_bw

    k_fn = _get_kernel(kernel)

    beta = np.zeros(p)
    beta[0] = 1.0

    converged = False
    n_iter = 0

    for iteration in range(max_iter):
        idx_vals = X @ beta

        if bandwidth is None:
            h = _silverman_bw(idx_vals)
        else:
            h = bandwidth

        diff = idx_vals[:, None] - idx_vals[None, :]
        W = k_fn(diff / h)
        np.fill_diagonal(W, 0.0)
        denom = W.sum(axis=1)
        denom = np.where(denom < 1e-15, 1.0, denom)
        g_hat = (W @ y) / denom

        def obj(b):
            b_norm = b / (np.linalg.norm(b) + 1e-15)
            idx_v = X @ b_norm
            d = idx_v[:, None] - idx_v[None, :]
            Wk = k_fn(d / h)
            np.fill_diagonal(Wk, 0.0)
            den = Wk.sum(axis=1)
            den = np.where(den < 1e-15, 1.0, den)
            g = (Wk @ y) / den
            return float(np.mean((y - g) ** 2))

        res = minimize(obj, beta, method="L-BFGS-B",
                       options={"maxiter": 20, "ftol": 1e-10})
        beta_new = res.x / (np.linalg.norm(res.x) + 1e-15)
        n_iter = iteration + 1

        if np.linalg.norm(beta_new - beta) < tol:
            converged = True
            beta = beta_new
            break
        beta = beta_new

    idx_final = X @ beta

    if bandwidth is None:
        h = _silverman_bw(idx_final)
    else:
        h = bandwidth
    diff = idx_final[:, None] - idx_final[None, :]
    W = k_fn(diff / h)
    np.fill_diagonal(W, 0.0)
    denom = W.sum(axis=1)
    denom = np.where(denom < 1e-15, 1.0, denom)
    g_hat = (W @ y) / denom

    return {
        "beta": beta.tolist(),
        "index": idx_final.tolist(),
        "g_hat": g_hat.tolist(),
        "n_iter": n_iter,
        "converged": converged,
        "n_obs": n,
    }


sisls_fn = sisls


def cheatsheet() -> str:
    return "sisls({y, X}) -> Single-index via iterative SLS (Ichimura 1993)."
