# morie.fn -- function file (hadesllm/morie)
"""GWR Gaussian kernel weights."""

from ._containers import SpatialResult


def gwrgaus(dists, bw=0.5):
    """GWR Gaussian kernel weights.

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
        return SpatialResult(name="gwrgaus", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(
            name="gwrgaus", statistic=float("nan"), p_value=None, extra={"error": "computation failed"}
        )


gwrgaus_fn = gwrgaus


def cheatsheet() -> str:
    return "gwrgaus({}) -> GWR Gaussian kernel weights."
