"""Single-index minimum distance estimator."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def simin(
    y: np.ndarray,
    X: np.ndarray,
    *,
    bandwidth: float | None = None,
    kernel: str = "gaussian",
    n_slices: int = 10,
) -> dict:
    r"""
    Single-index model via minimum distance estimation.

    Estimates :math:`\beta` by minimising a distance criterion between
    a nonparametric estimator of :math:`E[Y|X'\beta]` and slice averages:

    .. math::

        \hat{\beta} = \arg\min_{\|\beta\|=1}
        \sum_{s=1}^S n_s (\bar{Y}_s - \hat{G}(\bar{v}_s))^2

    where slices partition the index :math:`v = X'\beta`.

    Parameters
    ----------
    y : np.ndarray
        Response (n,).
    X : np.ndarray
        Covariates (n, p), p >= 2.
    bandwidth : float or None
        Kernel bandwidth.
    kernel : str
        Kernel function.
    n_slices : int
        Number of slices for the distance criterion.

    Returns
    -------
    dict
        ``beta``, ``index``, ``g_hat``, ``min_distance``, ``n_obs``.

    References
    ----------
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

    def obj(b):
        b_norm = b / (np.linalg.norm(b) + 1e-15)
        idx = X @ b_norm
        h = bandwidth if bandwidth is not None else _silverman_bw(idx)

        quantiles = np.linspace(0, 100, n_slices + 1)
        edges = np.percentile(idx, quantiles)

        dist = 0.0
        for s in range(n_slices):
            mask = (idx >= edges[s]) & (idx < edges[s + 1])
            if s == n_slices - 1:
                mask = (idx >= edges[s]) & (idx <= edges[s + 1])
            ns = mask.sum()
            if ns < 1:
                continue
            y_bar = y[mask].mean()
            v_bar = idx[mask].mean()

            diff = v_bar - idx
            w = k_fn(diff / h)
            sw = w.sum()
            if sw < 1e-15:
                continue
            g_hat = (w * y).sum() / sw
            dist += ns * (y_bar - g_hat) ** 2

        return dist

    b0 = np.zeros(p)
    b0[0] = 1.0
    res = minimize(obj, b0, method="L-BFGS-B",
                   options={"maxiter": 100, "ftol": 1e-8})
    beta = res.x / (np.linalg.norm(res.x) + 1e-15)

    idx_final = X @ beta
    h = bandwidth if bandwidth is not None else _silverman_bw(idx_final)
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
        "min_distance": float(res.fun),
        "n_obs": n,
    }


simin_fn = simin


def cheatsheet() -> str:
    return "simin({y, X}) -> Single-index minimum distance estimator."
