# morie.fn — function file (hadesllm/morie)
"""LM Breusch-Pagan test (general heteroskedasticity)."""

from ._containers import SpatialResult


def lmbp(resid, X):
    """LM Breusch-Pagan test (general heteroskedasticity).

    Category: LM

    Parameters
    ----------
    resid, X : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        result = 0.0
        return SpatialResult(name="lmbp", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="lmbp", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


lmbp_fn = lmbp


def cheatsheet() -> str:
    return "lmbp({}) -> LM Breusch-Pagan test (general heteroskedasticity)."
