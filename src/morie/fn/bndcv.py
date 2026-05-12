# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Bandwidth selection via leave-one-out cross-validation."""

from __future__ import annotations

import numpy as np


def bndcv(
    x: np.ndarray,
    *,
    h_grid: np.ndarray | None = None,
    kernel: str = "gaussian",
    n_grid: int = 30,
) -> dict:
    r"""
    Bandwidth selection for KDE via leave-one-out cross-validation (LSCV).

    Minimises the integrated squared error proxy:

    .. math::

        \text{LSCV}(h) = \int \hat{f}_h^2(x)\,dx
        - \frac{2}{n}\sum_{i=1}^{n} \hat{f}_{-i,h}(X_i)

    Parameters
    ----------
    x : np.ndarray
        Data vector (n,).
    h_grid : np.ndarray or None
        Candidate bandwidths. If None, a log-spaced grid is used.
    kernel : str
        ``'gaussian'``, ``'epanechnikov'``, or ``'uniform'``.
    n_grid : int
        Number of candidate bandwidths when ``h_grid`` is None.

    Returns
    -------
    dict
        ``h_opt`` (optimal bandwidth), ``cv_scores`` (LSCV values),
        ``h_grid`` (bandwidths evaluated), ``n_obs``.

    References
    ----------
    Silverman, B. W. (1986). Density Estimation. Ch. 3.4.3.
    Horowitz (2009). Appendix A.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = x.shape[0]
    if n < 5:
        raise ValueError("Need at least 5 observations.")

    from morie.fn.nwker import _get_kernel, _silverman_bw

    k_fn = _get_kernel(kernel)
    h_rot = _silverman_bw(x)

    if h_grid is None:
        h_grid = np.exp(np.linspace(np.log(h_rot * 0.1), np.log(h_rot * 3), n_grid))

    cv_scores = np.empty(len(h_grid))
    for idx, h in enumerate(h_grid):
        diff = x[:, None] - x[None, :]
        u = diff / h
        K = k_fn(u) / h
        np.fill_diagonal(K, 0.0)
        f_loo = K.sum(axis=1) / (n - 1)

        conv = k_fn(diff / (h * np.sqrt(2))) / (h * np.sqrt(2))
        int_f2 = conv.mean()

        cv_scores[idx] = int_f2 - 2.0 * f_loo.mean()

    best = int(np.argmin(cv_scores))
    return {
        "h_opt": float(h_grid[best]),
        "cv_scores": cv_scores.tolist(),
        "h_grid": h_grid.tolist(),
        "n_obs": n,
    }


bndcv_fn = bndcv


def cheatsheet() -> str:
    return "bndcv({x}) -> LOO cross-validation bandwidth for KDE."
