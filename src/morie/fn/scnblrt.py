# morie.fn — function file (hadesllm/morie)
"""Spatial NB likelihood-ratio test."""

from ._containers import SpatialResult


def scnblrt(ll_nb, ll_pois, df=1):
    """Spatial NB likelihood-ratio test.

    Category: SCount

    Parameters
    ----------
    ll_nb, ll_pois, df=1 : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        result = float(2 * (ll_nb - ll_pois))
        return SpatialResult(name="scnblrt", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(
            name="scnblrt", statistic=float("nan"), p_value=None, extra={"error": "computation failed"}
        )


scnblrt_fn = scnblrt


def cheatsheet() -> str:
    return "scnblrt({}) -> Spatial NB likelihood-ratio test."
