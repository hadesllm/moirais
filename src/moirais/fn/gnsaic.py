# moirais.fn — function file (hadesllm/moirais)
"""GNS Akaike information criterion."""

from ._containers import SpatialResult


def gnsaic(ll, k, n):
    """GNS Akaike information criterion.

    Category: GNS

    Parameters
    ----------
    ll, k, n : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        result = float(2 * k + (-2) * ll)
        return SpatialResult(name="gnsaic", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="gnsaic", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


gnsaic_fn = gnsaic


def cheatsheet() -> str:
    return "gnsaic({}) -> GNS Akaike information criterion."
