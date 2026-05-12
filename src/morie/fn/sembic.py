# morie.fn -- function file (hadesllm/morie)
"""SEM Bayesian information criterion."""

from ._containers import SpatialResult


def sembic(ll, k, n):
    """SEM Bayesian information criterion.

    Category: SEM

    Parameters
    ----------
    ll, k, n : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        result = float(2 * k + (-2) * ll)
        return SpatialResult(name="sembic", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="sembic", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


sembic_fn = sembic


def cheatsheet() -> str:
    return "sembic({}) -> SEM Bayesian information criterion."
