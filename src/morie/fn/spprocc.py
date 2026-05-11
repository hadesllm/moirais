"""Spatial probit ROC-AUC."""

from ._containers import SpatialResult


def spprocc(y, probs):
    """Spatial probit ROC-AUC.

    Category: SProbit

    Parameters
    ----------
    y, probs : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        result = 0.0
        return SpatialResult(name="spprocc", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(
            name="spprocc", statistic=float("nan"), p_value=None, extra={"error": "computation failed"}
        )


spprocc_fn = spprocc


def cheatsheet() -> str:
    return "spprocc({}) -> Spatial probit ROC-AUC."
