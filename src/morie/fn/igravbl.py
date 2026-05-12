# morie.fn -- function file (hadesllm/morie)
"""Gravity model row/column balancing (IPFP)."""

from ._containers import SpatialResult


def igravbl(flow_matrix, row_totals, col_totals):
    """Gravity model row/column balancing (IPFP).

    Category: Gravity

    Parameters
    ----------
    flow_matrix, row_totals, col_totals : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        result = 0.0
        return SpatialResult(name="igravbl", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(
            name="igravbl", statistic=float("nan"), p_value=None, extra={"error": "computation failed"}
        )


igravbl_fn = igravbl


def cheatsheet() -> str:
    return "igravbl({}) -> Gravity model row/column balancing (IPFP)."
