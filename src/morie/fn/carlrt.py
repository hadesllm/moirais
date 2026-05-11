# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""CAR likelihood-ratio test."""

from ._containers import SpatialResult


def carlrt(ll_car, ll_null, df=1):
    """CAR likelihood-ratio test.

    Category: CAR

    Parameters
    ----------
    ll_car, ll_null, df=1 : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        result = float(2 * (ll_car - ll_null))
        return SpatialResult(name="carlrt", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="carlrt", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


carlrt_fn = carlrt


def cheatsheet() -> str:
    return "carlrt({}) -> CAR likelihood-ratio test."
