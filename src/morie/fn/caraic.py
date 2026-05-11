# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""CAR Akaike information criterion."""

from ._containers import SpatialResult


def caraic(ll, k, n):
    """CAR Akaike information criterion.

    Category: CAR

    Parameters
    ----------
    ll, k, n : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        result = float(2 * k + (-2) * ll)
        return SpatialResult(name="caraic", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="caraic", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


caraic_fn = caraic


def cheatsheet() -> str:
    return "caraic({}) -> CAR Akaike information criterion."
