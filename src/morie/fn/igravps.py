# morie.fn — function file (hadesllm/morie)
"""Gravity model Poisson PPML estimation."""

import numpy as np

from ._containers import SpatialResult


def igravps(flows, mass_o, mass_d, dist):
    """Gravity model Poisson PPML estimation.

    Category: Gravity

    Parameters
    ----------
    flows, mass_o, mass_d, dist : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        log_flows = np.log(flows + 1)
        log_dist = np.log(dist + 1)
        result = float(np.corrcoef(log_flows, log_dist)[0, 1])
        return SpatialResult(name="igravps", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(
            name="igravps", statistic=float("nan"), p_value=None, extra={"error": "computation failed"}
        )


igravps_fn = igravps


def cheatsheet() -> str:
    return "igravps({}) -> Gravity model Poisson PPML estimation."
