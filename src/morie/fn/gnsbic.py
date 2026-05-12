# morie.fn -- function file (hadesllm/morie)
"""GNS Bayesian information criterion."""

from ._containers import SpatialResult


def gnsbic(ll, k, n):
    """GNS Bayesian information criterion.

    Category: GNS

    Parameters
    ----------
    ll, k, n : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        result = float(2 * k + (-2) * ll)
        return SpatialResult(name="gnsbic", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="gnsbic", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


gnsbic_fn = gnsbic


def cheatsheet() -> str:
    return "gnsbic({}) -> GNS Bayesian information criterion."
