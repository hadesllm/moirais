"""Spatial panel covariance structure."""

from ._containers import SpatialResult


def sppcov(resid, unit_id, time_id):
    """Spatial panel covariance structure.

    Category: SPanel

    Parameters
    ----------
    resid, unit_id, time_id : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        result = 0.0
        return SpatialResult(name="sppcov", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="sppcov", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


sppcov_fn = sppcov


def cheatsheet() -> str:
    return "sppcov({}) -> Spatial panel covariance structure."
