"""SLX Bayesian information criterion."""

from ._containers import SpatialResult


def slxbic(ll, k, n):
    """SLX Bayesian information criterion.

    Category: SLX

    Parameters
    ----------
    ll, k, n : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        result = float(2 * k + (-2) * ll)
        return SpatialResult(name="slxbic", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="slxbic", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


slxbic_fn = slxbic


def cheatsheet() -> str:
    return "slxbic({}) -> SLX Bayesian information criterion."
