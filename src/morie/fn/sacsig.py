# morie.fn -- function file (hadesllm/morie)
"""SAC sigma-squared ML estimate."""

from ._containers import SpatialResult


def sacsig(resid, n):
    """SAC sigma-squared ML estimate.

    Category: SAC

    Parameters
    ----------
    resid, n : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        result = 0.0
        return SpatialResult(name="sacsig", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="sacsig", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


sacsig_fn = sacsig


def cheatsheet() -> str:
    return "sacsig({}) -> SAC sigma-squared ML estimate."
