"""Spatial probit BIC."""

from ._containers import SpatialResult


def spprbic(ll, k, n):
    """Spatial probit BIC.

    Category: SProbit

    Parameters
    ----------
    ll, k, n : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        result = float(2 * k + (-2) * ll)
        return SpatialResult(name="spprbic", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(
            name="spprbic", statistic=float("nan"), p_value=None, extra={"error": "computation failed"}
        )


spprbic_fn = spprbic


def cheatsheet() -> str:
    return "spprbic({}) -> Spatial probit BIC."
