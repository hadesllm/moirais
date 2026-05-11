# morie.fn — function file (hadesllm/morie)
"""Gravity model variance function."""

import numpy as np

from ._containers import SpatialResult


def igravvf(flows_hat, phi=1.0):
    """Gravity model variance function.

    Category: Gravity

    Parameters
    ----------
    flows_hat, phi=1.0 : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        flows_hat = np.asarray(flows_hat, dtype=float)
        result = float(np.mean(flows_hat**phi))
        return SpatialResult(name="igravvf", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(
            name="igravvf", statistic=float("nan"), p_value=None, extra={"error": "computation failed"}
        )


igravvf_fn = igravvf


def cheatsheet() -> str:
    return "igravvf({}) -> Gravity model variance function."
