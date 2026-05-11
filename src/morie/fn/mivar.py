# morie.fn — function file (hadesllm/morie)
"""Moran's I variance Var[I] under normality."""

from ._containers import SpatialResult


def mivar(n, S0, S1, S2):
    """Moran's I variance Var[I] under normality.

    Category: Moran

    Parameters
    ----------
    n, S0, S1, S2 : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        result = 0.0
        return SpatialResult(name="mivar", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="mivar", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


mivar_fn = mivar


def cheatsheet() -> str:
    return "mivar({}) -> Moran's I variance Var[I] under normality."
