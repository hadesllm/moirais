"""Spatial logit BIC."""

from ._containers import SpatialResult


def splgbic(ll, k, n):
    """Spatial logit BIC.

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
        return SpatialResult(name="splgbic", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(
            name="splgbic", statistic=float("nan"), p_value=None, extra={"error": "computation failed"}
        )


splgbic_fn = splgbic


def cheatsheet() -> str:
    return "splgbic({}) -> Spatial logit BIC."
