# morie.fn — function file (hadesllm/morie)
"""SEM Akaike information criterion."""

from ._containers import SpatialResult


def semaic(ll, k, n):
    """SEM Akaike information criterion.

    Category: SEM

    Parameters
    ----------
    ll, k, n : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        result = float(2 * k + (-2) * ll)
        return SpatialResult(name="semaic", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="semaic", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


semaic_fn = semaic


def cheatsheet() -> str:
    return "semaic({}) -> SEM Akaike information criterion."
