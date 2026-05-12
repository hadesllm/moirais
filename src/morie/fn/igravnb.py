# morie.fn -- function file (hadesllm/morie)
"""Gravity model negative-binomial PPML."""

import numpy as np

from ._containers import SpatialResult


def igravnb(flows, mass_o, mass_d, dist):
    """Gravity model negative-binomial PPML.

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
        return SpatialResult(name="igravnb", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(
            name="igravnb", statistic=float("nan"), p_value=None, extra={"error": "computation failed"}
        )


igravnb_fn = igravnb


def cheatsheet() -> str:
    return "igravnb({}) -> Gravity model negative-binomial PPML."
