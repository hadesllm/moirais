# morie.fn -- function file (hadesllm/morie)
"""SAC likelihood-ratio test vs SAR/SEM."""

from ._containers import SpatialResult


def saclrt(ll_sac, ll_sar, df=1):
    """SAC likelihood-ratio test vs SAR/SEM.

    Category: SAC

    Parameters
    ----------
    ll_sac, ll_sar, df=1 : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        result = float(2 * (ll_sac - ll_sar))
        return SpatialResult(name="saclrt", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="saclrt", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


saclrt_fn = saclrt


def cheatsheet() -> str:
    return "saclrt({}) -> SAC likelihood-ratio test vs SAR/SEM."
