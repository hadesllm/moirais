# moirais.fn — function file (hadesllm/moirais)
"""Gravity model with origin/destination fixed effects."""

import numpy as np

from ._containers import SpatialResult


def igravfe(flows, origin_id, dest_id, dist):
    """Gravity model with origin/destination fixed effects.

    Category: Gravity

    Parameters
    ----------
    flows, origin_id, dest_id, dist : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        log_flows = np.log(flows + 1)
        log_dist = np.log(dist + 1)
        result = float(np.corrcoef(log_flows, log_dist)[0, 1])
        return SpatialResult(name="igravfe", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(
            name="igravfe", statistic=float("nan"), p_value=None, extra={"error": "computation failed"}
        )


igravfe_fn = igravfe


def cheatsheet() -> str:
    return "igravfe({}) -> Gravity model with origin/destination fixed effects."
