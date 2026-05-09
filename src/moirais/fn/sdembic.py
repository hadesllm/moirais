# moirais.fn — function file (hadesllm/moirais)
"""SDEM Bayesian information criterion."""

from ._containers import SpatialResult


def sdembic(ll, k, n):
    """SDEM Bayesian information criterion.

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
        return SpatialResult(name="sdembic", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(
            name="sdembic", statistic=float("nan"), p_value=None, extra={"error": "computation failed"}
        )


sdembic_fn = sdembic


def cheatsheet() -> str:
    return "sdembic({}) -> SDEM Bayesian information criterion."
