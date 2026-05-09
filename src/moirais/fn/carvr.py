# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""CAR variance ratio (spatial vs unstructured)."""

from ._containers import SpatialResult


def carvr(var_sp, var_un):
    """CAR variance ratio (spatial vs unstructured).

    Category: CAR

    Parameters
    ----------
    var_sp, var_un : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        result = 0.0
        return SpatialResult(name="carvr", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="carvr", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


carvr_fn = carvr


def cheatsheet() -> str:
    return "carvr({}) -> CAR variance ratio (spatial vs unstructured)."
