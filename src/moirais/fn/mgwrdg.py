# moirais.fn — function file (hadesllm/moirais)
"""MGWR diagnostic summary."""

from ._containers import SpatialResult


def mgwrdg(ll, tr_S, n, k):
    """MGWR diagnostic summary.

    Category: MGWR

    Parameters
    ----------
    ll, tr_S, n, k : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        result = float(2 * k + (-2) * ll)
        return SpatialResult(name="mgwrdg", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="mgwrdg", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


mgwrdg_fn = mgwrdg


def cheatsheet() -> str:
    return "mgwrdg({}) -> MGWR diagnostic summary."
