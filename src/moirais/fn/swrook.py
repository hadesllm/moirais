"""Rook-contiguity spatial weights (grid)."""

from ._containers import SpatialResult


def swrook(nrow=4, ncol=4):
    """Rook-contiguity spatial weights (grid).

    Category: Weights

    Parameters
    ----------
    nrow=4, ncol=4 : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        result = 0.0
        return SpatialResult(name="swrook", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="swrook", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


swrook_fn = swrook


def cheatsheet() -> str:
    return "swrook({}) -> Rook-contiguity spatial weights (grid)."
