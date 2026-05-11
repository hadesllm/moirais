# morie.fn — function file (hadesllm/morie)
"""Local linear regression."""

from __future__ import annotations

import numpy as np


def llreg(
    x: np.ndarray,
    y: np.ndarray,
    x_eval: np.ndarray | None = None,
    *,
    bandwidth: float | None = None,
    kernel: str = "gaussian",
) -> dict:
    r"""
    Local linear regression estimator (Fan 1992).

    At each evaluation point :math:`x_0`, fits a weighted least squares
    line :math:`\alpha + \beta(x - x_0)` with kernel weights
    :math:`K_h(x_i - x_0)`:

    .. math::

        \hat{m}(x_0) = e_1' (X_0' W_0 X_0)^{-1} X_0' W_0 y

    Avoids the boundary bias of Nadaraya-Watson.

    Parameters
    ----------
    x : np.ndarray
        Predictor (n,).
    y : np.ndarray
        Response (n,).
    x_eval : np.ndarray or None
        Evaluation points. Defaults to ``x``.
    bandwidth : float or None
        If None, Silverman's rule.
    kernel : str
        ``'gaussian'``, ``'epanechnikov'``, ``'uniform'``.

    Returns
    -------
    dict
        ``x_eval``, ``y_hat`` (fitted), ``slope`` (local slopes),
        ``bandwidth``, ``n_obs``.

    References
    ----------
    Fan, J. (1992). Design-adaptive nonparametric regression. JASA, 87, 998-1004.
    Horowitz (2009). Appendix A.19.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    n = x.shape[0]
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"x and y must match, got {x.shape[0]} and {y.shape[0]}.")
    if n < 3:
        raise ValueError("Need at least 3 observations.")

    from morie.fn.nwker import _get_kernel, _silverman_bw

    k_fn = _get_kernel(kernel)
    if bandwidth is None:
        bandwidth = _silverman_bw(x)

    if x_eval is None:
        x_eval = x.copy()
    else:
        x_eval = np.asarray(x_eval, dtype=float).ravel()

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
        s2 = (w * dx**2).sum()
        det = sw * s2 - s1**2
        if abs(det) < 1e-15:
            y_hat[j] = (w * y).sum() / sw
            slope[j] = 0.0
        else:
            y_hat[j] = ((s2 * w - s1 * w * dx) @ y) / det
            slope[j] = ((sw * w * dx - s1 * w) @ y) / det

    return {
        "x_eval": x_eval.tolist(),
        "y_hat": y_hat.tolist(),
        "slope": slope.tolist(),
        "bandwidth": float(bandwidth),
        "n_obs": n,
    }


llreg_fn = llreg


def cheatsheet() -> str:
    return "llreg({x, y}) -> Local linear regression (Fan 1992)."
