# morie.fn -- function file (hadesllm/morie)
"""Individual Conditional Expectation (ICE) curves.

ICE plots visualise how the predicted outcome of any model changes as
a feature is varied, holding all other features fixed.  The average
of ICE curves is the partial dependence function.

References
----------
Goldstein, A., Kapelner, A., Bleich, J., & Pitkin, E. (2015).
Peeking inside the black box: Visualizing statistical learning with
plots of individual conditional expectation.
*Journal of Computational and Graphical Statistics*, 24(1), 44-65.

Friedman, J. H. (2001). Greedy function approximation: A gradient
boosting machine. *Annals of Statistics*, 29(5), 1189-1232.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

__all__ = ["icerc"]


def icerc(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    X: np.ndarray,
    feature_idx: int,
    *,
    grid_points: int = 50,
    grid_values: np.ndarray | None = None,
    center: bool = True,
) -> dict[str, Any]:
    r"""Compute Individual Conditional Expectation (ICE) curves.

    For feature :math:`j` with grid :math:`\{v_1, \ldots, v_G\}`,
    the ICE curve for observation :math:`i` is:

    .. math::

        \text{ICE}_i(v) = \hat{f}(v, X_{i,-j})

    where :math:`X_{i,-j}` denotes all features except :math:`j` held
    fixed at their observed values.

    If ``center=True``, curves are centred at the first grid point
    (c-ICE): :math:`\Delta_i(v) = \text{ICE}_i(v) - \text{ICE}_i(v_1)`.

    Parameters
    ----------
    predict_fn : Callable
        A function mapping ``(n_samples, p)`` array to predictions
        ``(n_samples,)``.
    X : np.ndarray
        Background dataset, shape ``(n, p)``.
    feature_idx : int
        Column index of the feature to vary.
    grid_points : int
        Number of evenly-spaced grid values (ignored if grid_values
        provided).
    grid_values : np.ndarray, optional
        Explicit grid values for the feature.
    center : bool
        If True, return centred ICE (c-ICE).

    Returns
    -------
    dict
        ``ice`` (shape ``(n, G)``), ``pdp`` (mean of ICE, shape
        ``(G,)``), ``grid`` (shape ``(G,)``),
        ``feature_idx``, ``n``, ``G``, ``method``.

    References
    ----------
    Goldstein et al. (2015). JCGS, 24(1), 44-65.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be 2-D (n, p).")
    n, p = X.shape
    if not (0 <= feature_idx < p):
        raise ValueError(f"feature_idx must be in [0, {p - 1}].")

    if grid_values is not None:
        grid = np.asarray(grid_values, dtype=float)
    else:
        fmin = float(X[:, feature_idx].min())
        fmax = float(X[:, feature_idx].max())
        grid = np.linspace(fmin, fmax, grid_points)
    G = len(grid)

    ice = np.empty((n, G))
    for g, val in enumerate(grid):
        X_mod = X.copy()
        X_mod[:, feature_idx] = val
        ice[:, g] = predict_fn(X_mod)

    if center:
        ice = ice - ice[:, [0]]

    pdp = ice.mean(axis=0)

    return {
        "ice": ice,
        "pdp": pdp,
        "grid": grid,
        "feature_idx": feature_idx,
        "n": n,
        "G": G,
        "method": "c-ICE" if center else "ICE",
    }


def cheatsheet() -> str:
    return "icerc(predict_fn, X, feature_idx) -> ICE curves (Goldstein et al. 2015, JCGS)."
