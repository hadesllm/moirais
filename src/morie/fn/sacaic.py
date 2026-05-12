# morie.fn -- function file (hadesllm/morie)
"""SAC Akaike information criterion."""

from ._containers import SpatialResult


def sacaic(ll, k, n):
    """SAC Akaike information criterion.

    Category: SAC

    Parameters
    ----------
    ll, k, n : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        result = float(2 * k + (-2) * ll)
        return SpatialResult(name="sacaic", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="sacaic", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


sacaic_fn = sacaic


def cheatsheet() -> str:
    return "sacaic({}) -> SAC Akaike information criterion."
