# moirais.fn — function file (hadesllm/moirais)
"""Nonparametric quantile regression."""

from __future__ import annotations

import numpy as np


def npqnt(
    x: np.ndarray,
    y: np.ndarray,
    *,
    tau: float = 0.5,
    x_eval: np.ndarray | None = None,
    bandwidth: float | None = None,
    kernel: str = "gaussian",
    n_grid: int = 200,
) -> dict:
    r"""
    Nonparametric (kernel) conditional quantile regression.

    Estimates :math:`Q_\tau(Y \mid X = x)` by inverting the kernel-estimated
    conditional CDF:

    .. math::

        \hat{Q}_\tau(x) = \inf\{q : \hat{F}(q \mid x) \ge \tau\}

    where

    .. math::

        \hat{F}(q \mid x) = \frac{\sum_i K_h(x - X_i)\,
        \mathbf{1}(Y_i \le q)}{\sum_i K_h(x - X_i)}

    Parameters
    ----------
    x : np.ndarray
        Predictor values (n,).
    y : np.ndarray
        Response values (n,).
    tau : float
        Quantile level in (0, 1). Default 0.5 (median).
    x_eval : np.ndarray or None
        Evaluation points. Defaults to ``x``.
    bandwidth : float or None
        Kernel bandwidth. If None, Silverman's rule.
    kernel : str
        Kernel: ``'gaussian'``, ``'epanechnikov'``, ``'uniform'``.
    n_grid : int
        Grid size for y-axis when inverting CDF. Default 200.

    Returns
    -------
    dict
        Keys: ``x_eval``, ``quantiles`` (estimated conditional quantiles),
        ``tau``, ``bandwidth``, ``kernel``, ``n_obs``.

    Raises
    ------
    ValueError
        If tau not in (0, 1) or inputs invalid.

    References
    ----------
    Yu, K. & Jones, M. C. (1998). Local linear quantile regression.
        JASA, 93(441), 228-237.
    Horowitz, J. L. (2009). Semiparametric and Nonparametric Methods in
        Econometrics. Springer. Chapter 5.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have same length.")
    n = x.shape[0]
    if n < 4:
        raise ValueError("Need at least 4 observations.")
    if not 0 < tau < 1:
        raise ValueError(f"tau must be in (0, 1), got {tau}.")

    if x_eval is None:
        x_eval = x.copy()
    else:
        x_eval = np.asarray(x_eval, dtype=float).ravel()

    from moirais.fn.nwker import _get_kernel, _silverman_bw
    if bandwidth is None:
        bandwidth = _silverman_bw(x)
    k_fn = _get_kernel(kernel)

    y_grid = np.linspace(y.min(), y.max(), n_grid)
    m = len(x_eval)
    quantiles = np.empty(m)

    u = (x_eval[:, None] - x[None, :]) / bandwidth
    w = k_fn(u)
    w_sum = w.sum(axis=1, keepdims=True)
    w_sum = np.where(w_sum == 0, 1.0, w_sum)
    w_norm = w / w_sum

    for j in range(m):
        indicators = (y[None, :] <= y_grid[:, None]).astype(float)
        cond_cdf = (w_norm[j][None, :] * indicators).sum(axis=1)
        idx = np.searchsorted(cond_cdf, tau)
        if idx >= n_grid:
            quantiles[j] = y_grid[-1]
        elif idx == 0:
            quantiles[j] = y_grid[0]
        else:
            f0 = cond_cdf[idx - 1]
            f1 = cond_cdf[idx]
            frac = (tau - f0) / max(f1 - f0, 1e-15)
            quantiles[j] = y_grid[idx - 1] + frac * (y_grid[idx] - y_grid[idx - 1])

    return {
        "x_eval": x_eval,
        "quantiles": quantiles,
        "tau": tau,
        "bandwidth": bandwidth,
        "kernel": kernel,
        "n_obs": n,
    }


npqnt_fn = npqnt


def cheatsheet() -> str:
    return "npqnt({x, y, tau}) -> Nonparametric conditional quantile regression."
