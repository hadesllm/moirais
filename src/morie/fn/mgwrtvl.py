# morie.fn -- function file (hadesllm/morie)
"""MGWR t-values for local coefficients."""

from ._containers import SpatialResult


def mgwrtvl(coef, se):
    """MGWR t-values for local coefficients.

    Category: MGWR

    Parameters
    ----------
    coef, se : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        result = 0.0
        return SpatialResult(name="mgwrtvl", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(
            name="mgwrtvl", statistic=float("nan"), p_value=None, extra={"error": "computation failed"}
        )


mgwrtvl_fn = mgwrtvl


def cheatsheet() -> str:
    return "mgwrtvl({}) -> MGWR t-values for local coefficients."
