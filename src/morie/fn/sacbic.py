# morie.fn -- function file (hadesllm/morie)
"""SAC Bayesian information criterion."""

from ._containers import SpatialResult


def sacbic(ll, k, n):
    """SAC Bayesian information criterion.

    Category: SAC

    Parameters
    ----------
    ll, k, n : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        result = float(2 * k + (-2) * ll)
        return SpatialResult(name="sacbic", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="sacbic", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


sacbic_fn = sacbic


def cheatsheet() -> str:
    return "sacbic({}) -> SAC Bayesian information criterion."
