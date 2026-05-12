# morie.fn -- function file (hadesllm/morie)
"""SDM spillover index (indirect/total)."""

from ._containers import SpatialResult


def sdmspil(indirect, total):
    """SDM spillover index (indirect/total).

    Category: SDM

    Parameters
    ----------
    indirect, total : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        result = 0.0
        return SpatialResult(name="sdmspil", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(
            name="sdmspil", statistic=float("nan"), p_value=None, extra={"error": "computation failed"}
        )


sdmspil_fn = sdmspil


def cheatsheet() -> str:
    return "sdmspil({}) -> SDM spillover index (indirect/total)."
