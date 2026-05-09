# moirais.fn — function file (hadesllm/moirais)
"""MGWR model fit (Fotheringham et al. 2017)."""

import numpy as np

from ._containers import SpatialResult


def mgwrfit(y, X, coords):
    """MGWR model fit (Fotheringham et al. 2017).

    Category: MGWR

    Parameters
    ----------
    y, X, coords : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        dists = np.sqrt(np.sum((coords[None, :, :] - coords[:, None, :]) ** 2, axis=-1))
        result = float(np.mean(dists))
        return SpatialResult(name="mgwrfit", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(
            name="mgwrfit", statistic=float("nan"), p_value=None, extra={"error": "computation failed"}
        )


mgwrfit_fn = mgwrfit


def cheatsheet() -> str:
    return "mgwrfit({}) -> MGWR model fit (Fotheringham et al. 2017)."
