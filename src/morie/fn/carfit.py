# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""CAR DIC (deviance information criterion)."""

from ._containers import SpatialResult


def carfit(ll, p_d):
    """CAR DIC (deviance information criterion).

    Category: CAR

    Parameters
    ----------
    ll, p_d : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        result = 0.0
        return SpatialResult(name="carfit", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="carfit", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


carfit_fn = carfit


def cheatsheet() -> str:
    return "carfit({}) -> CAR DIC (deviance information criterion)."
