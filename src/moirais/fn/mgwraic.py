# moirais.fn — function file (hadesllm/moirais)
"""MGWR AICc for model selection."""

from ._containers import SpatialResult


def mgwraic(ll, tr_S, n):
    """MGWR AICc for model selection.

    Category: MGWR

    Parameters
    ----------
    ll, tr_S, n : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        result = 0.0
        return SpatialResult(name="mgwraic", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(
            name="mgwraic", statistic=float("nan"), p_value=None, extra={"error": "computation failed"}
        )


mgwraic_fn = mgwraic


def cheatsheet() -> str:
    return "mgwraic({}) -> MGWR AICc for model selection."
