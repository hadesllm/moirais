# morie.fn -- function file (hadesllm/morie)
"""GNS Wald test on rho and lambda."""

import numpy as np

from ._containers import SpatialResult


def gnswald(params, vcov):
    """GNS Wald test on rho and lambda.

    Category: GNS

    Parameters
    ----------
    params, vcov : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        result = float(params @ np.linalg.solve(vcov + 1e-8 * np.eye(len(vcov)), params))
        return SpatialResult(name="gnswald", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(
            name="gnswald", statistic=float("nan"), p_value=None, extra={"error": "computation failed"}
        )


gnswald_fn = gnswald


def cheatsheet() -> str:
    return "gnswald({}) -> GNS Wald test on rho and lambda."
