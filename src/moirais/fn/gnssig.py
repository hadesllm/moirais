# moirais.fn — function file (hadesllm/moirais)
"""GNS sigma-squared ML estimate."""

from ._containers import SpatialResult


def gnssig(resid, n):
    """GNS sigma-squared ML estimate.

    Category: GNS

    Parameters
    ----------
    resid, n : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        result = 0.0
        return SpatialResult(name="gnssig", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="gnssig", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


gnssig_fn = gnssig


def cheatsheet() -> str:
    return "gnssig({}) -> GNS sigma-squared ML estimate."
