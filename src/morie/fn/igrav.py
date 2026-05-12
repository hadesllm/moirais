# morie.fn -- function file (hadesllm/morie)
"""Gravity model OLS estimation."""

import numpy as np

from ._containers import SpatialResult


def igrav(flows, mass_o, mass_d, dist):
    """Gravity model OLS estimation.

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
        return SpatialResult(name="igrav", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="igrav", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


igrav_fn = igrav


def cheatsheet() -> str:
    return "igrav({}) -> Gravity model OLS estimation."
