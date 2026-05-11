# morie.fn — function file (hadesllm/morie)
"""Partial dependence plot (PDP) values.

Computes the marginal effect of one or two features by averaging
the model predictions over the joint distribution of all other features.

References
----------
Friedman, J. H. (2001). Greedy function approximation: A gradient
boosting machine. *Annals of Statistics*, 29(5), 1189-1232.

Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of
Statistical Learning* (2nd ed.). Springer. Section 10.13.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

__all__ = ["pdplt"]


def pdplt(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    X: np.ndarray,
    feature_idx: int | tuple[int, int],
    *,
    grid_points: int = 50,
    grid_values: np.ndarray | None = None,
    percentile_range: tuple[float, float] = (5.0, 95.0),
) -> dict[str, Any]:
    r"""Compute partial dependence values for one or two features.

    For a single feature :math:`j`:

    .. math::

        \bar{f}_j(x_j) = \frac{1}{n} \sum_{i=1}^{n}
            \hat{f}(x_j, \mathbf{x}_{i,-j})

    For two features :math:`j, k` (2-D PDP):

    .. math::

        \bar{f}_{jk}(x_j, x_k) = \frac{1}{n} \sum_{i=1}^{n}
            \hat{f}(x_j, x_k, \mathbf{x}_{i,-jk})

    Parameters
    ----------
    predict_fn : Callable
        Model prediction function: ``(n, p) -> (n,)``.
    X : np.ndarray
        Background data, shape ``(n, p)``.
    feature_idx : int or (int, int)
        Feature index (or pair) to compute PDP for.
    grid_points : int
        Grid resolution per feature.
    grid_values : np.ndarray, optional
        Explicit 1-D grid (single feature) or 2-D grid (two features).
    percentile_range : tuple
        Percentile range for automatic grid construction.

    Returns
    -------
    dict
        ``pdp`` (shape ``(G,)`` or ``(G1, G2)`` for 2D),
        ``grid`` (1-D or tuple of 1-D arrays),
        ``feature_idx``, ``n``, ``method``.

    References
    ----------
    Friedman (2001). Annals of Statistics, 29(5), 1189-1232.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be 2-D (n, p).")
    n, p = X.shape
    lo, hi = percentile_range

    if isinstance(feature_idx, (list, tuple)) and len(feature_idx) == 2:
        j1, j2 = int(feature_idx[0]), int(feature_idx[1])
        if not (0 <= j1 < p and 0 <= j2 < p):
            raise ValueError("feature_idx values must be in [0, p-1].")
        g1 = np.linspace(np.percentile(X[:, j1], lo), np.percentile(X[:, j1], hi), grid_points)
        g2 = np.linspace(np.percentile(X[:, j2], lo), np.percentile(X[:, j2], hi), grid_points)
        pdp = np.empty((grid_points, grid_points))
        for i1, v1 in enumerate(g1):
            for i2, v2 in enumerate(g2):
                X_mod = X.copy()
                X_mod[:, j1] = v1
                X_mod[:, j2] = v2
                pdp[i1, i2] = predict_fn(X_mod).mean()
        return {
            "pdp": pdp,
            "grid": (g1, g2),
            "feature_idx": (j1, j2),
            "n": n,
            "method": "PDP-2D",
        }

    j = int(feature_idx)
    if not (0 <= j < p):
        raise ValueError(f"feature_idx must be in [0, {p - 1}].")

    if grid_values is not None:
        grid = np.asarray(grid_values, dtype=float)
    else:
        grid = np.linspace(
            np.percentile(X[:, j], lo),
            np.percentile(X[:, j], hi),
            grid_points,
        )

    pdp = np.empty(len(grid))
    for g, val in enumerate(grid):
        X_mod = X.copy()
        X_mod[:, j] = val
        pdp[g] = predict_fn(X_mod).mean()

    return {
        "pdp": pdp,
        "grid": grid,
        "feature_idx": j,
        "n": n,
        "method": "PDP",
    }


def cheatsheet() -> str:
    return "pdplt(predict_fn, X, feature_idx) -> Partial dependence values (Friedman 2001)."
