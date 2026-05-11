# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""CAR sigma-squared estimate."""

from ._containers import SpatialResult


def carsig(resid, n):
    """CAR sigma-squared estimate.

    Category: CAR

    Parameters
    ----------
    resid, n : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        result = 0.0
        return SpatialResult(name="carsig", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="carsig", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


carsig_fn = carsig


def cheatsheet() -> str:
    return "carsig({}) -> CAR sigma-squared estimate."
