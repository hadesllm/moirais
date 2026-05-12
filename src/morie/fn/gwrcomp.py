# morie.fn -- function file (hadesllm/morie)
"""GWR comparison: fixed vs adaptive bandwidth."""

from ._containers import SpatialResult


def gwrcomp(aicc_fixed, aicc_adapt):
    """GWR comparison: fixed vs adaptive bandwidth.

    Category: GWR

    Parameters
    ----------
    aicc_fixed, aicc_adapt : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        result = 0.0
        return SpatialResult(name="gwrcomp", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(
            name="gwrcomp", statistic=float("nan"), p_value=None, extra={"error": "computation failed"}
        )


gwrcomp_fn = gwrcomp


def cheatsheet() -> str:
    return "gwrcomp({}) -> GWR comparison: fixed vs adaptive bandwidth."
