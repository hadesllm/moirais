# morie.fn -- function file (hadesllm/morie)
"""SDM Akaike information criterion."""

from ._containers import SpatialResult


def sdmaic(ll, k, n):
    """SDM Akaike information criterion.

    Category: SDM

    Parameters
    ----------
    ll, k, n : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        result = float(2 * k + (-2) * ll)
        return SpatialResult(name="sdmaic", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="sdmaic", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


sdmaic_fn = sdmaic


def cheatsheet() -> str:
    return "sdmaic({}) -> SDM Akaike information criterion."
