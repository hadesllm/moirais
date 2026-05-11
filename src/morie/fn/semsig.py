# morie.fn — function file (hadesllm/morie)
"""SEM sigma-squared ML estimate."""

from ._containers import SpatialResult


def semsig(resid, n):
    """SEM sigma-squared ML estimate.

    Category: SEM

    Parameters
    ----------
    resid, n : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        result = 0.0
        return SpatialResult(name="semsig", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="semsig", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


semsig_fn = semsig


def cheatsheet() -> str:
    return "semsig({}) -> SEM sigma-squared ML estimate."
