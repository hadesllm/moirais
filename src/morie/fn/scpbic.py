# morie.fn -- function file (hadesllm/morie)
"""Spatial count model BIC."""

from ._containers import SpatialResult


def scpbic(ll, k, n):
    """Spatial count model BIC.

    Category: SCount

    Parameters
    ----------
    ll, k, n : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        result = float(2 * k + (-2) * ll)
        return SpatialResult(name="scpbic", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="scpbic", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


scpbic_fn = scpbic


def cheatsheet() -> str:
    return "scpbic({}) -> Spatial count model BIC."
