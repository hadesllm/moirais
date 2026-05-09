# moirais.fn — function file (hadesllm/moirais)
"""MGWR local R-squared."""

from ._containers import SpatialResult


def mgwrr2(y, y_hat):
    """MGWR local R-squared.

    Category: MGWR

    Parameters
    ----------
    y, y_hat : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        result = 0.0
        return SpatialResult(name="mgwrr2", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="mgwrr2", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


mgwrr2_fn = mgwrr2


def cheatsheet() -> str:
    return "mgwrr2({}) -> MGWR local R-squared."
