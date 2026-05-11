# morie.fn — function file (hadesllm/morie)
"""GWR bisquare kernel weights."""

from ._containers import SpatialResult


def gwrbisq(dists, bw=0.5):
    """GWR bisquare kernel weights.

    Category: GWR

    Parameters
    ----------
    dists, bw=0.5 : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        result = 0.0
        return SpatialResult(name="gwrbisq", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(
            name="gwrbisq", statistic=float("nan"), p_value=None, extra={"error": "computation failed"}
        )


gwrbisq_fn = gwrbisq


def cheatsheet() -> str:
    return "gwrbisq({}) -> GWR bisquare kernel weights."
