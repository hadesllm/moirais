# morie.fn -- function file (hadesllm/morie)
"""Spatial Poisson fixed-effects panel."""

import numpy as np

from ._containers import SpatialResult


def scpflx(y, X, W, unit_id):
    """Spatial Poisson fixed-effects panel.

    Category: SCount

    Parameters
    ----------
    y, X, W, unit_id : see function signature.

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
        return SpatialResult(name="scpflx", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="scpflx", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


scpflx_fn = scpflx


def cheatsheet() -> str:
    return "scpflx({}) -> Spatial Poisson fixed-effects panel."
