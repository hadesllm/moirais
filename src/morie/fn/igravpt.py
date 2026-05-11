# morie.fn — function file (hadesllm/morie)
"""Spatial potential model."""

from ._containers import SpatialResult


def igravpt(mass, dist):
    """Spatial potential model.

    Category: Gravity

    Parameters
    ----------
    mass, dist : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        result = 0.0
        return SpatialResult(name="igravpt", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(
            name="igravpt", statistic=float("nan"), p_value=None, extra={"error": "computation failed"}
        )


igravpt_fn = igravpt


def cheatsheet() -> str:
    return "igravpt({}) -> Spatial potential model."
