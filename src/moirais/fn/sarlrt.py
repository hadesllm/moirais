# moirais.fn — function file (hadesllm/moirais)
"""SAR likelihood-ratio test vs OLS."""

from ._containers import SpatialResult


def sarlrt(ll_sar, ll_ols, df=1):
    """SAR likelihood-ratio test vs OLS.

    Category: SAR

    Parameters
    ----------
    ll_sar, ll_ols, df=1 : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        result = float(2 * (ll_sar - ll_ols))
        return SpatialResult(name="sarlrt", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="sarlrt", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


sarlrt_fn = sarlrt


def cheatsheet() -> str:
    return "sarlrt({}) -> SAR likelihood-ratio test vs OLS."
