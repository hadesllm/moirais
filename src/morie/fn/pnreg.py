# morie.fn -- function file (hadesllm/morie)
"""Penalized regression (ridge kernel)."""

from __future__ import annotations

import numpy as np


def pnreg(
    x: np.ndarray,
    y: np.ndarray,
    x_eval: np.ndarray | None = None,
    *,
    bandwidth: float | None = None,
    penalty: float = 1.0,
    kernel: str = "gaussian",
) -> dict:
    r"""
    Penalized (ridge) kernel regression.

    Combines kernel smoothing with Tikhonov-type regularization:

    .. math::

        \hat{m}(x_0) = \mathbf{e}_1'
        (\mathbf{X}'\mathbf{W}\mathbf{X} + \lambda \mathbf{I})^{-1}
        \mathbf{X}'\mathbf{W}\mathbf{y}

    where :math:`\mathbf{W} = \mathrm{diag}(K_h(x_i - x_0))`.

    Parameters
    ----------
    x, y : np.ndarray
        Predictor and response (n,).
    x_eval : np.ndarray or None
        Evaluation points.
    bandwidth : float or None
        Kernel bandwidth. If None, Silverman's rule.
    penalty : float
        Ridge penalty lambda > 0.
    kernel : str
        ``'gaussian'``, ``'epanechnikov'``, ``'uniform'``.

    Returns
    -------
    dict
        ``x_eval``, ``y_hat``, ``bandwidth``, ``penalty``, ``n_obs``.

    References
    ----------
    Horowitz (2009). Ch 3 (penalized methods).
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    n = x.shape[0]
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have same length.")
    if n < 3:
        raise ValueError("Need at least 3 observations.")
    if penalty < 0:
        raise ValueError("penalty must be >= 0.")

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

    for j in range(m):
        dx = x - x_eval[j]
        u = dx / bandwidth
        w = k_fn(u)
        X_loc = np.column_stack([np.ones(n), dx])
        W = np.diag(w)
        XtW = X_loc.T @ W
        A = XtW @ X_loc + penalty * np.eye(2)
        try:
            beta = np.linalg.solve(A, XtW @ y)
        except np.linalg.LinAlgError:
            beta = np.linalg.lstsq(A, XtW @ y, rcond=None)[0]
        y_hat[j] = beta[0]

    return {
        "x_eval": x_eval.tolist(),
        "y_hat": y_hat.tolist(),
        "bandwidth": float(bandwidth),
        "penalty": float(penalty),
        "n_obs": n,
    }


pnreg_fn = pnreg


def cheatsheet() -> str:
    return "pnreg({x, y}) -> Penalized (ridge) kernel regression."
