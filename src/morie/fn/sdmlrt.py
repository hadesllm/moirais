# morie.fn — function file (hadesllm/morie)
"""SDM likelihood-ratio test vs SAR."""

from ._containers import SpatialResult


def sdmlrt(ll_sdm, ll_sar, df=2):
    """SDM likelihood-ratio test vs SAR.

    Category: SDM

    Parameters
    ----------
    ll_sdm, ll_sar, df=2 : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        result = float(2 * (ll_sdm - ll_sar))
        return SpatialResult(name="sdmlrt", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="sdmlrt", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


sdmlrt_fn = sdmlrt


def cheatsheet() -> str:
    return "sdmlrt({}) -> SDM likelihood-ratio test vs SAR."
