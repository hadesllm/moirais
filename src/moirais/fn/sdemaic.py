# moirais.fn — function file (hadesllm/moirais)
"""SDEM Akaike information criterion."""

from ._containers import SpatialResult


def sdemaic(ll, k, n):
    """SDEM Akaike information criterion.

    Category: SDEM

    Parameters
    ----------
    ll, k, n : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        result = float(2 * k + (-2) * ll)
        return SpatialResult(name="sdemaic", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(
            name="sdemaic", statistic=float("nan"), p_value=None, extra={"error": "computation failed"}
        )


sdemaic_fn = sdemaic


def cheatsheet() -> str:
    return "sdemaic({}) -> SDEM Akaike information criterion."
