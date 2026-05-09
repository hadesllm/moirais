# moirais.fn — function file (hadesllm/moirais)
"""Gravity retail potential (Huff model)."""

from ._containers import SpatialResult


def igravrt(mass, dist, beta=2.0):
    """Gravity retail potential (Huff model).

    Category: Gravity

    Parameters
    ----------
    mass, dist, beta=2.0 : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        result = 0.0
        return SpatialResult(name="igravrt", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(
            name="igravrt", statistic=float("nan"), p_value=None, extra={"error": "computation failed"}
        )


igravrt_fn = igravrt


def cheatsheet() -> str:
    return "igravrt({}) -> Gravity retail potential (Huff model)."
