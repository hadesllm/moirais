# morie.fn -- function file (hadesllm/morie)
"""Local polynomial regression (degree p)."""

from __future__ import annotations

import numpy as np


def lpreg(
    x: np.ndarray,
    y: np.ndarray,
    x_eval: np.ndarray | None = None,
    *,
    degree: int = 2,
    bandwidth: float | None = None,
    kernel: str = "gaussian",
) -> dict:
    r"""
    Local polynomial regression of degree *p* (Fan & Gijbels 1996).

    At each evaluation point :math:`x_0`, fits

    .. math::

        \min_{\beta} \sum_{i=1}^n K_h(x_i - x_0)
        \left(y_i - \sum_{j=0}^{p} \beta_j (x_i - x_0)^j\right)^2

    Parameters
    ----------
    x, y : np.ndarray
        Predictor and response (n,).
    x_eval : np.ndarray or None
        Evaluation points.
    degree : int
        Polynomial degree p >= 0.
    bandwidth : float or None
        If None, Silverman's rule.
    kernel : str
        ``'gaussian'``, ``'epanechnikov'``, ``'uniform'``.

    Returns
    -------
    dict
        ``x_eval``, ``y_hat``, ``coefficients`` (list of arrays per
        eval point), ``degree``, ``bandwidth``, ``n_obs``.

    References
    ----------
    Fan, J. & Gijbels, I. (1996). Local Polynomial Modelling. Chapman & Hall.
    Horowitz (2009). Appendix A.19.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    n = x.shape[0]
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have same length.")
    if n < degree + 2:
        raise ValueError(f"Need at least {degree + 2} observations for degree {degree}.")
    if degree < 0:
        raise ValueError("degree must be >= 0.")

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
    coefs = []

    for j in range(m):
        dx = x - x_eval[j]
        u = dx / bandwidth
        w = k_fn(u)
        W = np.diag(w)
        X_local = np.column_stack([dx**p for p in range(degree + 1)])
        XtW = X_local.T @ W
        XtWX = XtW @ X_local
        try:
            beta = np.linalg.solve(XtWX, XtW @ y)
        except np.linalg.LinAlgError:
            beta = np.linalg.lstsq(XtWX, XtW @ y, rcond=None)[0]
        y_hat[j] = beta[0]
        coefs.append(beta.tolist())

    return {
        "x_eval": x_eval.tolist(),
        "y_hat": y_hat.tolist(),
        "coefficients": coefs,
        "degree": degree,
        "bandwidth": float(bandwidth),
        "n_obs": n,
    }


lpreg_fn = lpreg


def cheatsheet() -> str:
    return "lpreg({x, y}) -> Local polynomial regression (degree p)."
