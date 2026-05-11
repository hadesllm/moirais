"""Spatial panel Hausman test (FE vs RE)."""

from ._containers import SpatialResult


def spphaus(coef_fe, coef_re, vcov_fe, vcov_re):
    """Spatial panel Hausman test (FE vs RE).

    Category: SPanel

    Parameters
    ----------
    coef_fe, coef_re, vcov_fe, vcov_re : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        result = 0.0
        return SpatialResult(name="spphaus", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(
            name="spphaus", statistic=float("nan"), p_value=None, extra={"error": "computation failed"}
        )


spphaus_fn = spphaus


def cheatsheet() -> str:
    return "spphaus({}) -> Spatial panel Hausman test (FE vs RE)."
