"""Spatial panel BIC."""

from ._containers import SpatialResult


def sppbic(ll, k, n):
    """Spatial panel BIC.

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
        return SpatialResult(name="sppbic", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="sppbic", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


sppbic_fn = sppbic


def cheatsheet() -> str:
    return "sppbic({}) -> Spatial panel BIC."
