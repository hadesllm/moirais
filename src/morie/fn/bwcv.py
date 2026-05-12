# morie.fn -- function file (hadesllm/morie)
"""Bandwidth selection via leave-one-out cross-validation."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar


def bwcv(
    x: np.ndarray,
    y: np.ndarray,
    *,
    kernel: str = "gaussian",
    bw_range: tuple[float, float] | None = None,
    method: str = "regression",
) -> dict:
    r"""
    Bandwidth selection via leave-one-out cross-validation.

    For **regression** (Nadaraya-Watson), minimizes the LOO-CV criterion:

    .. math::

        \text{CV}(h) = \frac{1}{n} \sum_{i=1}^{n}
        \left(Y_i - \hat{m}_{-i}(X_i)\right)^2

    For **density**, minimizes the integrated squared error criterion:

    .. math::

        \text{CV}(h) = \int \hat{f}_h^2(x)\,dx
        - \frac{2}{n} \sum_{i=1}^{n} \hat{f}_{h,-i}(X_i)

    Parameters
    ----------
    x : np.ndarray
        1-d predictor values (n,).
    y : np.ndarray
        1-d response values (n,). Ignored when ``method='density'``.
    kernel : str
        Kernel: ``'gaussian'``, ``'epanechnikov'``, or ``'uniform'``.
    bw_range : tuple or None
        (min_bw, max_bw) search interval. If None, auto-set from data.
    method : str
        ``'regression'`` (NW-CV) or ``'density'`` (LSCV).

    Returns
    -------
    dict
        Keys: ``bandwidth`` (optimal), ``cv_score`` (minimized criterion),
        ``method``, ``kernel``, ``n_obs``.

    References
    ----------
    Hardle, W. (1990). Applied Nonparametric Regression. Cambridge.
    Horowitz, J. L. (2009). Semiparametric and Nonparametric Methods in
        Econometrics. Springer. Chapter 2.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    n = x.shape[0]
    if n < 4:
        raise ValueError("Need at least 4 observations for CV bandwidth selection.")

    valid_methods = {"regression", "density"}
    if method not in valid_methods:
        raise ValueError(f"Unknown method '{method}'. Choose from {valid_methods}.")

    from morie.fn.nwker import _get_kernel, _silverman_bw
    k_fn = _get_kernel(kernel)
    h_ref = _silverman_bw(x)

    if bw_range is None:
        bw_range = (h_ref * 0.1, h_ref * 5.0)

    if method == "regression":
        if x.shape[0] != y.shape[0]:
            raise ValueError("x and y must have same length.")

        def cv_obj(h):
            if h <= 0:
                return 1e30
            diff = x[:, None] - x[None, :]
            u = diff / h
            w = k_fn(u)
            np.fill_diagonal(w, 0.0)
            denom = w.sum(axis=1)
            denom = np.where(denom == 0, 1.0, denom)
            y_hat_loo = (w * y[None, :]).sum(axis=1) / denom
            return float(np.mean((y - y_hat_loo) ** 2))
    else:
        def cv_obj(h):
            if h <= 0:
                return 1e30
            diff = x[:, None] - x[None, :]
            u = diff / h
            w = k_fn(u)
            f_all = w.sum(axis=1) / (n * h)
            np.fill_diagonal(w, 0.0)
            f_loo = w.sum(axis=1) / ((n - 1) * h)
            return float(np.mean(f_all ** 2) - 2 * np.mean(f_loo))

    result = minimize_scalar(cv_obj, bounds=bw_range, method="bounded")
    return {
        "bandwidth": float(result.x),
        "cv_score": float(result.fun),
        "method": method,
        "kernel": kernel,
        "n_obs": n,
    }


bwcv_fn = bwcv


def cheatsheet() -> str:
    return "bwcv({x, y}) -> Bandwidth selection via leave-one-out cross-validation."
