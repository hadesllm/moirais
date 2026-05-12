# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""CAR Bayesian information criterion."""

from ._containers import SpatialResult


def carbic(ll, k, n):
    """CAR Bayesian information criterion.

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
        return SpatialResult(name="carbic", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="carbic", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


carbic_fn = carbic


def cheatsheet() -> str:
    return "carbic({}) -> CAR Bayesian information criterion."
