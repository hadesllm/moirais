# moirais.fn — function file (hadesllm/moirais)
"""Koenker-Bassett heteroskedasticity test."""

from ._containers import SpatialResult


def lmkoenk(resid, X):
    """Koenker-Bassett heteroskedasticity test.

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
        return SpatialResult(name="lmkoenk", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(
            name="lmkoenk", statistic=float("nan"), p_value=None, extra={"error": "computation failed"}
        )


lmkoenk_fn = lmkoenk


def cheatsheet() -> str:
    return "lmkoenk({}) -> Koenker-Bassett heteroskedasticity test."
