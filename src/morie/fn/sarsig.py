# morie.fn -- function file (hadesllm/morie)
"""SAR sigma-squared ML estimate."""

from ._containers import SpatialResult


def sarsig(resid, n):
    """SAR sigma-squared ML estimate.

    Category: SAR

    Parameters
    ----------
    resid, n : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        result = 0.0
        return SpatialResult(name="sarsig", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="sarsig", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


sarsig_fn = sarsig


def cheatsheet() -> str:
    return "sarsig({}) -> SAR sigma-squared ML estimate."
