# morie.fn — function file (hadesllm/morie)
"""SAR Akaike information criterion."""

from ._containers import SpatialResult


def saraic(ll, k, n):
    """SAR Akaike information criterion.

    Category: SAR

    Parameters
    ----------
    ll, k, n : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        result = float(2 * k + (-2) * ll)
        return SpatialResult(name="saraic", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="saraic", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


saraic_fn = saraic


def cheatsheet() -> str:
    return "saraic({}) -> SAR Akaike information criterion."
