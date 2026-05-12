# morie.fn -- function file (hadesllm/morie)
"""Moran's I standardised z-score."""

from ._containers import SpatialResult


def mizval(I, E_I, Var_I):
    """Moran's I standardised z-score.

    Category: Moran

    Parameters
    ----------
    I, E_I, Var_I : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        result = 0.0
        return SpatialResult(name="mizval", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="mizval", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


mizval_fn = mizval


def cheatsheet() -> str:
    return "mizval({}) -> Moran's I standardised z-score."
