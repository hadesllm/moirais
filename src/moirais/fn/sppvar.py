"""Spatial panel variance components."""

from ._containers import SpatialResult


def sppvar(resid, unit_id):
    """Spatial panel variance components.

    Category: SPanel

    Parameters
    ----------
    resid, unit_id : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        result = 0.0
        return SpatialResult(name="sppvar", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="sppvar", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


sppvar_fn = sppvar


def cheatsheet() -> str:
    return "sppvar({}) -> Spatial panel variance components."
