# moirais.fn — function file (hadesllm/moirais)
"""Gravity model RMSE."""

import numpy as np

from ._containers import SpatialResult


def igraver(flows, flows_hat):
    """Gravity model RMSE.

    Category: Gravity

    Parameters
    ----------
    flows, flows_hat : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        flows = np.asarray(flows, dtype=float)
        flows_hat = np.asarray(flows_hat, dtype=float)
        result = float(np.sqrt(np.mean((flows - flows_hat) ** 2)))
        return SpatialResult(name="igraver", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(
            name="igraver", statistic=float("nan"), p_value=None, extra={"error": "computation failed"}
        )


igraver_fn = igraver


def cheatsheet() -> str:
    return "igraver({}) -> Gravity model RMSE."
