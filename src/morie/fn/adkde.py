# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Adaptive kernel density estimation (variable bandwidth)."""

from __future__ import annotations

import numpy as np


def adkde(
    x: np.ndarray,
    x_eval: np.ndarray | None = None,
    *,
    bandwidth: float | None = None,
    alpha: float = 0.5,
    kernel: str = "gaussian",
    n_grid: int = 256,
) -> dict:
    r"""
    Adaptive (variable-bandwidth) kernel density estimation.

    Two-stage Abramson (1982) estimator:

    1. Compute a pilot density :math:`\tilde{f}` with fixed bandwidth.
    2. Set local bandwidth :math:`h_i = h \cdot \lambda_i` where
       :math:`\lambda_i = (\tilde{f}(X_i) / g)^{-\alpha}` and
       :math:`g = \exp(n^{-1}\sum \log \tilde{f}(X_i))`.

    Parameters
    ----------
    x : np.ndarray
        Data vector (n,).
    x_eval : np.ndarray or None
        Evaluation grid. If None, auto-generated.
    bandwidth : float or None
        Global pilot bandwidth. If None, Silverman's rule.
    alpha : float
        Sensitivity parameter in [0, 1]. Default 0.5 (square-root law).
    kernel : str
        ``'gaussian'``, ``'epanechnikov'``, ``'uniform'``.
    n_grid : int
        Grid size when ``x_eval`` is None.

    Returns
    -------
    dict
        ``x_eval``, ``density``, ``local_bandwidths``, ``bandwidth``,
        ``alpha``, ``n_obs``.

    References
    ----------
    Abramson, I. (1982). On bandwidth variation in kernel estimates -- a
        square root law. Annals of Statistics, 10, 1217-1223.
    Horowitz (2009). Ch 2.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = x.shape[0]
    if n < 5:
        raise ValueError("Need at least 5 observations.")
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be in [0, 1].")

    from morie.fn.nwker import _get_kernel, _silverman_bw

    k_fn = _get_kernel(kernel)
    if bandwidth is None:
        bandwidth = _silverman_bw(x)

    u_pilot = (x[:, None] - x[None, :]) / bandwidth
    pilot = k_fn(u_pilot).mean(axis=1) / bandwidth
    pilot = np.maximum(pilot, 1e-15)

    log_g = np.mean(np.log(pilot))
    g = np.exp(log_g)
    lambdas = (pilot / g) ** (-alpha)
    local_bw = bandwidth * lambdas

    if x_eval is None:
        lo = x.min() - 3 * bandwidth
        hi = x.max() + 3 * bandwidth
        x_eval = np.linspace(lo, hi, n_grid)
    else:
        x_eval = np.asarray(x_eval, dtype=float).ravel()

    density = np.zeros(len(x_eval))
    for i in range(n):
        u = (x_eval - x[i]) / local_bw[i]
        density += k_fn(u) / local_bw[i]
    density /= n

    return {
        "x_eval": x_eval.tolist(),
        "density": density.tolist(),
        "local_bandwidths": local_bw.tolist(),
        "bandwidth": float(bandwidth),
        "alpha": alpha,
        "n_obs": n,
    }


adkde_fn = adkde


def cheatsheet() -> str:
    return "adkde({x}) -> Adaptive KDE with variable bandwidth (Abramson)."
