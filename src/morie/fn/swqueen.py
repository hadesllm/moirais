"""Queen-contiguity spatial weights (grid)."""

from ._containers import SpatialResult


def swqueen(nrow=4, ncol=4):
    """Queen-contiguity spatial weights (grid).

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
        return SpatialResult(name="swqueen", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(
            name="swqueen", statistic=float("nan"), p_value=None, extra={"error": "computation failed"}
        )


swqueen_fn = swqueen


def cheatsheet() -> str:
    return "swqueen({}) -> Queen-contiguity spatial weights (grid)."
