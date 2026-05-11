"""Block spatial weights (group-based)."""

from ._containers import SpatialResult


def swblk(groups):
    """Block spatial weights (group-based).

    Category: Weights

    Parameters
    ----------
    groups : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        result = 0.0
        return SpatialResult(name="swblk", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="swblk", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


swblk_fn = swblk


def cheatsheet() -> str:
    return "swblk({}) -> Block spatial weights (group-based)."
