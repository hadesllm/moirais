"""Spatial logit AIC."""

from ._containers import SpatialResult


def splgaic(ll, k, n):
    """Spatial logit AIC.

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
        return SpatialResult(name="splgaic", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(
            name="splgaic", statistic=float("nan"), p_value=None, extra={"error": "computation failed"}
        )


splgaic_fn = splgaic


def cheatsheet() -> str:
    return "splgaic({}) -> Spatial logit AIC."
