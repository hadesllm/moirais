"""Moran's I of spatially filtered residuals."""

import numpy as np

from ._containers import SpatialResult


def sfmi(resid_f, W):
    """Moran's I of spatially filtered residuals.

    Category: SFilter

    Parameters
    ----------
    resid_f, W : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        n = len(resid_f)
        Wresid = np.dot(W, resid_f)
        result = float(np.dot(resid_f, Wresid) / (np.dot(resid_f, resid_f) + 1e-12))
        return SpatialResult(name="sfmi", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="sfmi", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


sfmi_fn = sfmi


def cheatsheet() -> str:
    return "sfmi({}) -> Moran's I of spatially filtered residuals."
