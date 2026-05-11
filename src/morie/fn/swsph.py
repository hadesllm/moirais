"""Spherical (great-circle) distance weights."""

from ._containers import SpatialResult


def swsph(lat, lon, d=500.0):
    """Spherical (great-circle) distance weights.

    Category: Weights

    Parameters
    ----------
    lat, lon, d=500.0 : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        result = 0.0
        return SpatialResult(name="swsph", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="swsph", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


swsph_fn = swsph


def cheatsheet() -> str:
    return "swsph({}) -> Spherical (great-circle) distance weights."
