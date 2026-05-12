# morie.fn -- function file (hadesllm/morie)
"""SEM likelihood-ratio test vs OLS."""

from ._containers import SpatialResult


def semlrt(ll_sem, ll_ols, df=1):
    """SEM likelihood-ratio test vs OLS.

    Category: SEM

    Parameters
    ----------
    ll_sem, ll_ols, df=1 : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        result = float(2 * (ll_sem - ll_ols))
        return SpatialResult(name="semlrt", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="semlrt", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


semlrt_fn = semlrt


def cheatsheet() -> str:
    return "semlrt({}) -> SEM likelihood-ratio test vs OLS."
