# morie.fn -- function file (hadesllm/morie)
"""Multivariate product kernel density estimator."""

from __future__ import annotations

import numpy as np
from ._richresult import RichResult

__all__ = ["mkdft"]


def mkdft(
    data: np.ndarray,
    x_eval: np.ndarray | None = None,
    *,
    bw: np.ndarray | float | None = None,
    n_grid: int = 50,
) -> dict:
    r"""
    Multivariate product kernel density estimator.

    Uses a product of univariate Gaussian kernels:

    .. math::

        \hat{f}(\mathbf{x}) = \frac{1}{n \prod_{j=1}^{d} h_j}
        \sum_{i=1}^{n} \prod_{j=1}^{d}
        K\!\left(\frac{x_j - X_{ij}}{h_j}\right)

    with independent bandwidths per dimension.

    Parameters
    ----------
    data : np.ndarray
        2-d array of shape (n, d).
    x_eval : np.ndarray or None
        Evaluation points of shape (m, d). If None, a grid is formed
        from per-dimension linspaces.
    bw : array-like, float, or None
        Per-dimension bandwidths. If scalar, used for all dims.
        Silverman's rule per dimension if None.
    n_grid : int
        Per-dimension grid size when *x_eval* is None.

    Returns
    -------
    dict
        ``x_eval`` (m, d), ``density`` (m,), ``bw`` (d,).

    References
    ----------
    Scott, D. W. (1992). *Multivariate Density Estimation*. Wiley. Chapter 6.
    """
    data = np.asarray(data, dtype=float)
    if data.ndim == 1:
        data = data[:, None]
    n, d = data.shape
    if n < 2:
        raise ValueError("Need at least 2 observations.")

    if bw is None:
        bw_vec = np.empty(d)
        for j in range(d):
            col = data[:, j]
            sigma = np.std(col, ddof=1)
            iqr = np.subtract(*np.percentile(col, [75, 25]))
            s = min(sigma, iqr / 1.349) if iqr > 0 else sigma
            bw_vec[j] = max(0.9 * s * n ** (-1.0 / (d + 4)), 1e-10)
    elif np.isscalar(bw):
        bw_vec = np.full(d, float(bw))
    else:
        bw_vec = np.asarray(bw, dtype=float).ravel()
        if len(bw_vec) != d:
            raise ValueError(f"bw length {len(bw_vec)} != data dim {d}.")

    if np.any(bw_vec <= 0):
        raise ValueError("All bandwidths must be positive.")

    if x_eval is None:
        grids_1d = []
        for j in range(d):
            lo = data[:, j].min() - 3 * bw_vec[j]
            hi = data[:, j].max() + 3 * bw_vec[j]
            grids_1d.append(np.linspace(lo, hi, n_grid))
        mesh = np.meshgrid(*grids_1d, indexing="ij")
        x_eval = np.column_stack([m.ravel() for m in mesh])
    else:
        x_eval = np.asarray(x_eval, dtype=float)
        if x_eval.ndim == 1:
            x_eval = x_eval[:, None]

    m = x_eval.shape[0]
    density = np.zeros(m)
    norm_const = 1.0 / (np.prod(bw_vec) * (2 * np.pi) ** (d / 2.0))
    for i in range(n):
        diff = (x_eval - data[i]) / bw_vec
        density += np.exp(-0.5 * np.sum(diff ** 2, axis=1))
    density *= norm_const / n

    return RichResult(payload={"x_eval": x_eval, "density": density, "bw": bw_vec})


def cheatsheet() -> str:
    return "mkdft({data}) -> Multivariate product kernel density."
