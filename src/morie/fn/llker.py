# morie.fn — function file (hadesllm/morie)
"""Local linear kernel regression estimator."""

from __future__ import annotations

import numpy as np


def llker(
    x: np.ndarray,
    y: np.ndarray,
    x_eval: np.ndarray | None = None,
    *,
    bandwidth: float | None = None,
    kernel: str = "gaussian",
) -> dict:
    r"""
    Local linear kernel regression estimator.

    Fits a weighted linear regression at each evaluation point, solving:

    .. math::

        \hat{m}(x) = e_1^\top
        \left(\mathbf{X}_x^\top \mathbf{W}_x \mathbf{X}_x\right)^{-1}
        \mathbf{X}_x^\top \mathbf{W}_x \mathbf{Y}

    where :math:`\mathbf{X}_x` has rows :math:`(1, X_i - x)` and
    :math:`\mathbf{W}_x = \text{diag}\{K_h(X_i - x)\}`. The local linear
    estimator adapts to boundary bias better than Nadaraya-Watson.

    Parameters
    ----------
    x : np.ndarray
        1-d array of predictor values (n,).
    y : np.ndarray
        1-d array of response values (n,).
    x_eval : np.ndarray or None
        Points at which to evaluate. Defaults to ``x``.
    bandwidth : float or None
        Kernel bandwidth. If None, uses Silverman's rule of thumb.
    kernel : str
        Kernel function: ``'gaussian'``, ``'epanechnikov'``, or ``'uniform'``.

    Returns
    -------
    dict
        Keys: ``x_eval``, ``y_hat`` (fitted values), ``slope`` (local slopes),
        ``bandwidth``, ``kernel``, ``n_obs``.

    Raises
    ------
    ValueError
        If inputs have mismatched lengths or unknown kernel.

    References
    ----------
    Fan, J. & Gijbels, I. (1996). Local Polynomial Modelling and Its
        Applications. Chapman & Hall.
    Horowitz, J. L. (2009). Semiparametric and Nonparametric Methods in
        Econometrics. Springer. Chapter 2.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"x and y must have same length, got {x.shape[0]} and {y.shape[0]}.")
    n = x.shape[0]
    if n < 3:
        raise ValueError("Need at least 3 observations for local linear fit.")

    valid_kernels = {"gaussian", "epanechnikov", "uniform"}
    if kernel not in valid_kernels:
        raise ValueError(f"Unknown kernel '{kernel}'. Choose from {valid_kernels}.")

    if x_eval is None:
        x_eval = x.copy()
    else:
        x_eval = np.asarray(x_eval, dtype=float).ravel()

    if bandwidth is None:
        from morie.fn.nwker import _silverman_bw
        bandwidth = _silverman_bw(x)

    if bandwidth <= 0:
        raise ValueError(f"Bandwidth must be positive, got {bandwidth}.")

    from morie.fn.nwker import _get_kernel
    k_fn = _get_kernel(kernel)

    m = len(x_eval)
    y_hat = np.empty(m)
    slope = np.empty(m)

    for j in range(m):
        dx = x - x_eval[j]
        u = dx / bandwidth
        w = k_fn(u)
        sw = w.sum()
        if sw < 1e-15:
            y_hat[j] = np.nan
            slope[j] = np.nan
            continue
        s1 = (w * dx).sum()
        s2 = (w * dx ** 2).sum()
        denom = sw * s2 - s1 ** 2
        if abs(denom) < 1e-15:
            y_hat[j] = (w * y).sum() / sw
            slope[j] = 0.0
        else:
            a0 = (s2 * (w * y).sum() - s1 * (w * dx * y).sum()) / denom
            a1 = (sw * (w * dx * y).sum() - s1 * (w * y).sum()) / denom
            y_hat[j] = a0
            slope[j] = a1

    return {
        "x_eval": x_eval,
        "y_hat": y_hat,
        "slope": slope,
        "bandwidth": bandwidth,
        "kernel": kernel,
        "n_obs": n,
    }


llker_fn = llker


def cheatsheet() -> str:
    return "llker({x, y}) -> Local linear kernel regression estimator."
