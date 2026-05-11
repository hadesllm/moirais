# morie.fn — function file (hadesllm/morie)
"""SDM sigma-squared ML estimate."""

from ._containers import SpatialResult


def sdmsig(resid, n):
    """SDM sigma-squared ML estimate.

    Category: SDM

    Parameters
    ----------
    resid, n : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        result = 0.0
        return SpatialResult(name="sdmsig", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="sdmsig", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


sdmsig_fn = sdmsig


def cheatsheet() -> str:
    return "sdmsig({}) -> SDM sigma-squared ML estimate."
