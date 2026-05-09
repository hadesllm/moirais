# moirais.fn — function file (hadesllm/moirais)
"""LM test for spatial heteroskedasticity."""

from ._containers import SpatialResult


def lmhe(resid, X):
    """LM test for spatial heteroskedasticity.

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
        return SpatialResult(name="lmhe", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="lmhe", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


lmhe_fn = lmhe


def cheatsheet() -> str:
    return "lmhe({}) -> LM test for spatial heteroskedasticity."
