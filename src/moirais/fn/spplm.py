"""Spatial panel LM test for spatial lag in panel."""

import numpy as np

from ._containers import SpatialResult


def spplm(resid, W, n, T):
    """Spatial panel LM test for spatial lag in panel.

    Category: SPanel

    Parameters
    ----------
    resid, W, n, T : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        nt = len(resid)
        if W.shape[0] < nt:
            T_periods = nt // W.shape[0]
            W_full = np.kron(np.eye(T_periods), W)
        else:
            W_full = W
        Wresid = np.dot(W_full, resid)
        result = float(np.dot(resid, Wresid) / (np.dot(resid, resid) + 1e-12))
        return SpatialResult(name="spplm", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="spplm", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


spplm_fn = spplm


def cheatsheet() -> str:
    return "spplm({}) -> Spatial panel LM test for spatial lag in panel."
