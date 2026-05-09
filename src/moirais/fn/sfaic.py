# moirais.fn — function file (hadesllm/moirais)
"""Spatial filtering AIC model selection."""

from ._containers import SpatialResult


def sfaic(ll, k, n):
    """Spatial filtering AIC model selection.

    Category: SFilter

    Parameters
    ----------
    ll, k, n : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        result = float(2 * k + (-2) * ll)
        return SpatialResult(name="sfaic", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="sfaic", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


sfaic_fn = sfaic


def cheatsheet() -> str:
    return "sfaic({}) -> Spatial filtering AIC model selection."
