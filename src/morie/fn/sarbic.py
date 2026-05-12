# morie.fn -- function file (hadesllm/morie)
"""SAR Bayesian information criterion."""

from ._containers import SpatialResult


def sarbic(ll, k, n):
    """SAR Bayesian information criterion.

    Category: SAR

    Parameters
    ----------
    ll, k, n : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        result = float(2 * k + (-2) * ll)
        return SpatialResult(name="sarbic", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="sarbic", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


sarbic_fn = sarbic


def cheatsheet() -> str:
    return "sarbic({}) -> SAR Bayesian information criterion."
