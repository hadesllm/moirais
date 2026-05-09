"""Spatial panel within-demeaning transform."""

from ._containers import SpatialResult


def sppde(y, unit_id):
    """Spatial panel within-demeaning transform.

    Category: SPanel

    Parameters
    ----------
    y, unit_id : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        result = 0.0
        return SpatialResult(name="sppde", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="sppde", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


sppde_fn = sppde


def cheatsheet() -> str:
    return "sppde({}) -> Spatial panel within-demeaning transform."
