# moirais.fn — function file (hadesllm/moirais)
"""Gravity model BIC."""

from ._containers import SpatialResult


def igrbic(ll, k, n):
    """Gravity model BIC.

    Category: Gravity

    Parameters
    ----------
    ll, k, n : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        result = float(2 * k + (-2) * ll)
        return SpatialResult(name="igrbic", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="igrbic", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


igrbic_fn = igrbic


def cheatsheet() -> str:
    return "igrbic({}) -> Gravity model BIC."
