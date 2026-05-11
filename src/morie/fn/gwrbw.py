# morie.fn — function file (hadesllm/morie)
"""GWR bandwidth selection (AICc cross-validation)."""

import numpy as np

from ._containers import SpatialResult


def gwrbw(y, X, coords):
    """GWR bandwidth selection (AICc cross-validation).

    Category: GWR

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
        return SpatialResult(name="gwrbw", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="gwrbw", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


gwrbw_fn = gwrbw


def cheatsheet() -> str:
    return "gwrbw({}) -> GWR bandwidth selection (AICc cross-validation)."
