# morie.fn -- function file (hadesllm/morie)
"""SAC Wald test on rho and lambda jointly."""

import numpy as np

from ._containers import SpatialResult


def sacwald(params, vcov):
    """SAC Wald test on rho and lambda jointly.

    Category: SAC

    Parameters
    ----------
    params, vcov : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        result = float(params @ np.linalg.solve(vcov + 1e-8 * np.eye(len(vcov)), params))
        return SpatialResult(name="sacwald", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(
            name="sacwald", statistic=float("nan"), p_value=None, extra={"error": "computation failed"}
        )


sacwald_fn = sacwald


def cheatsheet() -> str:
    return "sacwald({}) -> SAC Wald test on rho and lambda jointly."
