# morie.fn -- function file (hadesllm/morie)
"""Spatial count model AIC."""

from ._containers import SpatialResult


def scpaic(ll, k, n):
    """Spatial count model AIC.

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
        return SpatialResult(name="scpaic", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="scpaic", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


scpaic_fn = scpaic


def cheatsheet() -> str:
    return "scpaic({}) -> Spatial count model AIC."
