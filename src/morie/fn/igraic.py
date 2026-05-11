# morie.fn — function file (hadesllm/morie)
"""Gravity model AIC."""

from ._containers import SpatialResult


def igraic(ll, k, n):
    """Gravity model AIC.

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
        return SpatialResult(name="igraic", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="igraic", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


igraic_fn = igraic


def cheatsheet() -> str:
    return "igraic({}) -> Gravity model AIC."
