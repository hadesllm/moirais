# morie.fn -- function file (hadesllm/morie)
"""Gravity model calibration (beta estimation)."""

import numpy as np

from ._containers import SpatialResult


def igravcl(flows, mass_o, mass_d, dist):
    """Gravity model calibration (beta estimation).

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
        return SpatialResult(name="igravcl", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(
            name="igravcl", statistic=float("nan"), p_value=None, extra={"error": "computation failed"}
        )


igravcl_fn = igravcl


def cheatsheet() -> str:
    return "igravcl({}) -> Gravity model calibration (beta estimation)."
