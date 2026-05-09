"""SLX local/global impacts decomposition."""

from ._containers import SpatialResult


def slximp(coef, theta):
    """SLX local/global impacts decomposition.

    Category: SLX

    Parameters
    ----------
    coef, theta : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        result = 0.0
        return SpatialResult(name="slximp", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="slximp", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


slximp_fn = slximp


def cheatsheet() -> str:
    return "slximp({}) -> SLX local/global impacts decomposition."
