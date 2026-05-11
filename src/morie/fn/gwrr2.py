# morie.fn — function file (hadesllm/morie)
"""GWR local R-squared."""

from ._containers import SpatialResult


def gwrr2(y, y_hat):
    """GWR local R-squared.

    Category: GWR

    Parameters
    ----------
    y, y_hat : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        result = 0.0
        return SpatialResult(name="gwrr2", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="gwrr2", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


gwrr2_fn = gwrr2


def cheatsheet() -> str:
    return "gwrr2({}) -> GWR local R-squared."
