# moirais.fn — function file (hadesllm/moirais)
"""Bandwidth selection via least-squares cross-validation (LSCV)."""

from __future__ import annotations

import numpy as np
from ._richresult import RichResult

__all__ = ["kbwcv"]


def kbwcv(
    data: np.ndarray,
    *,
    bw_range: tuple[float, float] | None = None,
    n_bw: int = 50,
) -> dict:
    r"""
    Bandwidth selection via least-squares cross-validation (LSCV).

    Minimises the integrated squared error proxy:

    .. math::

        \mathrm{LSCV}(h) = \int \hat{f}_h^2(x)\,dx
        - \frac{2}{n} \sum_{i=1}^{n} \hat{f}_{-i,h}(X_i)

    where :math:`\hat{f}_{-i,h}` is the leave-one-out density.

    Parameters
    ----------
    data : np.ndarray
        1-d array of observations.
    bw_range : tuple or None
        ``(bw_min, bw_max)`` search interval. If None, a sensible range
        is derived from the data.
    n_bw : int
        Number of candidate bandwidths to evaluate.

    Returns
    -------
    dict
        ``bw_opt``, ``bw_grid``, ``lscv_scores``.

    References
    ----------
    Rudemo, M. (1982). Empirical choice of histograms and kernel density
        estimators. *Scandinavian Journal of Statistics*, 9(2), 65-78.

    Bowman, A. W. (1984). An alternative method of cross-validation for
        the smoothing of density estimates. *Biometrika*, 71(2), 353-360.
    """
    data = np.asarray(data, dtype=float).ravel()
    n = data.shape[0]
    if n < 3:
        raise ValueError("Need at least 3 observations for LSCV.")

    sigma = np.std(data, ddof=1)
    iqr = np.subtract(*np.percentile(data, [75, 25]))
    s = min(sigma, iqr / 1.349) if iqr > 0 else sigma
    s = max(s, 1e-10)
    h_rot = 0.9 * s * n ** (-0.2)

    if bw_range is None:
        bw_range = (h_rot * 0.1, h_rot * 3.0)

    bw_grid = np.linspace(bw_range[0], bw_range[1], n_bw)
    bw_grid = bw_grid[bw_grid > 0]

    diffs = data[:, None] - data[None, :]

    scores = np.empty(len(bw_grid))
    for idx, h in enumerate(bw_grid):
        sq2h = np.sqrt(2.0) * h
        term1 = np.exp(-0.5 * (diffs / sq2h) ** 2).sum() / (n ** 2 * sq2h * np.sqrt(np.pi))

        loo_vals = np.exp(-0.5 * (diffs / h) ** 2) / (h * np.sqrt(2 * np.pi))
        np.fill_diagonal(loo_vals, 0.0)
        term2 = loo_vals.sum() / (n * (n - 1))

        scores[idx] = term1 - 2.0 * term2

    best = int(np.argmin(scores))
    return RichResult(payload={"bw_opt": float(bw_grid[best]), "bw_grid": bw_grid, "lscv_scores": scores})


def cheatsheet() -> str:
    return "kbwcv({data}) -> LSCV bandwidth selection for KDE."
