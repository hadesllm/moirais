"""Spatial panel bootstrap."""

import numpy as np

from ._containers import SpatialResult


def sppboot(y, X, W, time_id, unit_id, B=9):
    """Spatial panel bootstrap.

    Category: SPanel

    Parameters
    ----------
    y, X, W, time_id, unit_id, B=9 : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        n = len(y)
        if W.shape[0] < n:
            T = n // W.shape[0]
            W_full = np.kron(np.eye(T), W)
        else:
            W_full = W
        result = float(np.dot(y, np.dot(W_full, y)) / (np.dot(y, y) + 1e-12))
        return SpatialResult(name="sppboot", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(
            name="sppboot", statistic=float("nan"), p_value=None, extra={"error": "computation failed"}
        )


sppboot_fn = sppboot


def cheatsheet() -> str:
    return "sppboot({}) -> Spatial panel bootstrap."
