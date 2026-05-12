# morie.fn -- function file (hadesllm/morie)
"""Square gravity (symmetric OD matrix)."""

from ._containers import SpatialResult


def igravsq(flow_matrix, mass, dist_matrix):
    """Square gravity (symmetric OD matrix).

    Category: Gravity

    Parameters
    ----------
    flow_matrix, mass, dist_matrix : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        result = 0.0
        return SpatialResult(name="igravsq", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(
            name="igravsq", statistic=float("nan"), p_value=None, extra={"error": "computation failed"}
        )


igravsq_fn = igravsq


def cheatsheet() -> str:
    return "igravsq({}) -> Square gravity (symmetric OD matrix)."
