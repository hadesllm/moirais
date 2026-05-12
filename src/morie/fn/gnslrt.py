# morie.fn -- function file (hadesllm/morie)
"""GNS likelihood-ratio test vs SDM."""

from ._containers import SpatialResult


def gnslrt(ll_gns, ll_sdm, df=1):
    """GNS likelihood-ratio test vs SDM.

    Category: GNS

    Parameters
    ----------
    ll_gns, ll_sdm, df=1 : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        result = float(2 * (ll_gns - ll_sdm))
        return SpatialResult(name="gnslrt", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="gnslrt", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


gnslrt_fn = gnslrt


def cheatsheet() -> str:
    return "gnslrt({}) -> GNS likelihood-ratio test vs SDM."
