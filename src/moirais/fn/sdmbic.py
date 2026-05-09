# moirais.fn — function file (hadesllm/moirais)
"""SDM Bayesian information criterion."""

from ._containers import SpatialResult


def sdmbic(ll, k, n):
    """SDM Bayesian information criterion.

    Category: SDM

    Parameters
    ----------
    ll, k, n : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        result = float(2 * k + (-2) * ll)
        return SpatialResult(name="sdmbic", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="sdmbic", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


sdmbic_fn = sdmbic


def cheatsheet() -> str:
    return "sdmbic({}) -> SDM Bayesian information criterion."
