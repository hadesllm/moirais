"""Spatial panel AIC."""

from ._containers import SpatialResult


def sppaic(ll, k, n):
    """Spatial panel AIC.

    Category: SPanel

    Parameters
    ----------
    ll, k, n : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        result = float(2 * k + (-2) * ll)
        return SpatialResult(name="sppaic", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="sppaic", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


sppaic_fn = sppaic


def cheatsheet() -> str:
    return "sppaic({}) -> Spatial panel AIC."
