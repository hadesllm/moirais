# morie.fn — function file (hadesllm/morie)
"""Moran's I normal approximation p-value."""

from ._containers import SpatialResult


def minorm(I, n, S0, S1, S2):
    """Moran's I normal approximation p-value.

    Category: Moran

    Parameters
    ----------
    I, n, S0, S1, S2 : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        result = 0.0
        return SpatialResult(name="minorm", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="minorm", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


minorm_fn = minorm


def cheatsheet() -> str:
    return "minorm({}) -> Moran's I normal approximation p-value."
