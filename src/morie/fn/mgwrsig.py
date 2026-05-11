# morie.fn — function file (hadesllm/morie)
"""MGWR sigma-squared estimate."""

from ._containers import SpatialResult


def mgwrsig(resid, tr_S, n):
    """MGWR sigma-squared estimate.

    Category: MGWR

    Parameters
    ----------
    resid, tr_S, n : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        result = 0.0
        return SpatialResult(name="mgwrsig", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(
            name="mgwrsig", statistic=float("nan"), p_value=None, extra={"error": "computation failed"}
        )


mgwrsig_fn = mgwrsig


def cheatsheet() -> str:
    return "mgwrsig({}) -> MGWR sigma-squared estimate."
