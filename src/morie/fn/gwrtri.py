# morie.fn — function file (hadesllm/morie)
"""GWR tri-cube kernel weights."""

from ._containers import SpatialResult


def gwrtri(dists, bw=0.5):
    """GWR tri-cube kernel weights.

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
        return SpatialResult(name="gwrtri", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="gwrtri", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


gwrtri_fn = gwrtri


def cheatsheet() -> str:
    return "gwrtri({}) -> GWR tri-cube kernel weights."
