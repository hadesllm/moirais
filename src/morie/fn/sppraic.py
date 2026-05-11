"""Spatial probit AIC."""

from ._containers import SpatialResult


def sppraic(ll, k, n):
    """Spatial probit AIC.

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
        return SpatialResult(name="sppraic", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(
            name="sppraic", statistic=float("nan"), p_value=None, extra={"error": "computation failed"}
        )


sppraic_fn = sppraic


def cheatsheet() -> str:
    return "sppraic({}) -> Spatial probit AIC."
