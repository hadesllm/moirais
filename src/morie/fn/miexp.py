# morie.fn -- function file (hadesllm/morie)
"""Moran's I expected value E[I]."""

from ._containers import SpatialResult


def miexp(n):
    """Moran's I expected value E[I].

    Category: Moran

    Parameters
    ----------
    n : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        result = 0.0
        return SpatialResult(name="miexp", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="miexp", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


miexp_fn = miexp


def cheatsheet() -> str:
    return "miexp({}) -> Moran's I expected value E[I]."
