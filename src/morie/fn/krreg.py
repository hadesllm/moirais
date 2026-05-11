# morie.fn — function file (hadesllm/morie)
"""Kernel ridge regression."""

from __future__ import annotations

import numpy as np


def krreg(
    x: np.ndarray,
    y: np.ndarray,
    x_eval: np.ndarray | None = None,
    *,
    bandwidth: float | None = None,
    penalty: float = 1.0,
    kernel: str = "gaussian",
) -> dict:
    r"""
    Kernel ridge regression (KRR).

    Solves the dual representation:

    .. math::

        \hat{\alpha} = (K + \lambda I)^{-1} y

    where :math:`K_{ij} = K_h(x_i - x_j)` is the Gram matrix.
    Prediction: :math:`\hat{m}(x_0) = \sum_i \alpha_i K_h(x_0 - x_i)`.

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
        ``x_eval``, ``y_hat``, ``alpha`` (dual coefficients),
        ``bandwidth``, ``penalty``, ``n_obs``.

    References
    ----------
    Saunders, C. et al. (1998). Ridge regression learning algorithm in
        dual variables. ICML.
    Horowitz (2009). Ch 3.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    n = x.shape[0]
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have same length.")
    if n < 3:
        raise ValueError("Need at least 3 observations.")
    if penalty <= 0:
        raise ValueError("penalty must be > 0.")

    from morie.fn.nwker import _get_kernel, _silverman_bw

    k_fn = _get_kernel(kernel)
    if bandwidth is None:
        bandwidth = _silverman_bw(x)

    diff = x[:, None] - x[None, :]
    K_mat = k_fn(diff / bandwidth)

    alpha = np.linalg.solve(K_mat + penalty * np.eye(n), y)

    if x_eval is None:
        x_eval = x.copy()
    else:
        x_eval = np.asarray(x_eval, dtype=float).ravel()

    K_eval = k_fn((x_eval[:, None] - x[None, :]) / bandwidth)
    y_hat = K_eval @ alpha

    return {
        "x_eval": x_eval.tolist(),
        "y_hat": y_hat.tolist(),
        "alpha": alpha.tolist(),
        "bandwidth": float(bandwidth),
        "penalty": float(penalty),
        "n_obs": n,
    }


krreg_fn = krreg


def cheatsheet() -> str:
    return "krreg({x, y}) -> Kernel ridge regression."
