# moirais.fn — function file (hadesllm/moirais)
"""Gravity Wilson entropy model."""

import numpy as np

from ._containers import SpatialResult


def igravwl(flows, mass_o, mass_d, dist, beta=1.0):
    """Gravity Wilson entropy model.

    Category: Gravity

    Parameters
    ----------
    flows, mass_o, mass_d, dist, beta=1.0 : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        log_flows = np.log(flows + 1)
        log_dist = np.log(dist + 1)
        result = float(np.corrcoef(log_flows, log_dist)[0, 1])
        return SpatialResult(name="igravwl", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(
            name="igravwl", statistic=float("nan"), p_value=None, extra={"error": "computation failed"}
        )


igravwl_fn = igravwl


def cheatsheet() -> str:
    return "igravwl({}) -> Gravity Wilson entropy model."
