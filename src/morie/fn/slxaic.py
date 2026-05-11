"""SLX Akaike information criterion."""

from ._containers import SpatialResult


def slxaic(ll, k, n):
    """SLX Akaike information criterion.

    Category: SLX

    Parameters
    ----------
    ll, k, n : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        result = float(2 * k + (-2) * ll)
        return SpatialResult(name="slxaic", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="slxaic", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


slxaic_fn = slxaic


def cheatsheet() -> str:
    return "slxaic({}) -> SLX Akaike information criterion."
