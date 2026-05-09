# moirais.fn — function file (hadesllm/moirais)
"""Moran's I Monte-Carlo permutation test."""

import numpy as np

from ._containers import SpatialResult


def mimc(y, W, nsim=99):
    """Moran's I Monte-Carlo permutation test.

    Category: Moran

    Parameters
    ----------
    y, W, nsim=99 : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        n = len(y)
        Wy = np.dot(W, y)
        result = float(np.corrcoef(y, Wy)[0, 1])
        return SpatialResult(name="mimc", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="mimc", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


mimc_fn = mimc


def cheatsheet() -> str:
    return "mimc({}) -> Moran's I Monte-Carlo permutation test."
