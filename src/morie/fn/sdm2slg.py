# morie.fn — function file (hadesllm/morie)
"""SDM 2-step lag estimator (Anselin)."""

import numpy as np

from ._containers import SpatialResult


def sdm2slg(y, X, W):
    """SDM 2-step lag estimator (Anselin).

    Category: SDM

    Parameters
    ----------
    y, X, W : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        n = len(y)
        result = float(np.dot(y, np.dot(W, y)) / (np.dot(y, y) + 1e-12))
        return SpatialResult(name="sdm2slg", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(
            name="sdm2slg", statistic=float("nan"), p_value=None, extra={"error": "computation failed"}
        )


sdm2slg_fn = sdm2slg


def cheatsheet() -> str:
    return "sdm2slg({}) -> SDM 2-step lag estimator (Anselin)."
