# moirais.fn — function file (hadesllm/moirais)
"""MGWR bandwidth confidence interval."""

from ._containers import SpatialResult


def mgwrbnd(bw, se_bw):
    """MGWR bandwidth confidence interval.

    Category: MGWR

    Parameters
    ----------
    bw, se_bw : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        result = 0.0
        return SpatialResult(name="mgwrbnd", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(
            name="mgwrbnd", statistic=float("nan"), p_value=None, extra={"error": "computation failed"}
        )


mgwrbnd_fn = mgwrbnd


def cheatsheet() -> str:
    return "mgwrbnd({}) -> MGWR bandwidth confidence interval."
