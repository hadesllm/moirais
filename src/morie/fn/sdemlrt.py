# morie.fn -- function file (hadesllm/morie)
"""SDEM likelihood-ratio test vs SEM."""

from ._containers import SpatialResult


def sdemlrt(ll_sdem, ll_sem, df=2):
    """SDEM likelihood-ratio test vs SEM.

    Category: SDEM

    Parameters
    ----------
    ll_sdem, ll_sem, df=2 : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        result = float(2 * (ll_sdem - ll_sem))
        return SpatialResult(name="sdemlrt", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(
            name="sdemlrt", statistic=float("nan"), p_value=None, extra={"error": "computation failed"}
        )


sdemlrt_fn = sdemlrt


def cheatsheet() -> str:
    return "sdemlrt({}) -> SDEM likelihood-ratio test vs SEM."
