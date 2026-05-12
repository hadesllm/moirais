# morie.fn -- function file (hadesllm/morie)
"""GWR corrected AIC (AICc) for model selection."""

from ._containers import SpatialResult


def gwraicc(ll, k, n):
    """GWR corrected AIC (AICc) for model selection.

    Category: GWR

    Parameters
    ----------
    ll, k, n : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        result = float(2 * k + (-2) * ll)
        return SpatialResult(name="gwraicc", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(
            name="gwraicc", statistic=float("nan"), p_value=None, extra={"error": "computation failed"}
        )


gwraicc_fn = gwraicc


def cheatsheet() -> str:
    return "gwraicc({}) -> GWR corrected AIC (AICc) for model selection."
