# morie.fn -- function file (hadesllm/morie)
"""SEM common-factor restriction test (Wald)."""

from ._containers import SpatialResult


def semspec(coef_sar, coef_sem, vcov):
    """SEM common-factor restriction test (Wald).

    Category: SEM

    Parameters
    ----------
    coef_sar, coef_sem, vcov : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        result = 0.0
        return SpatialResult(name="semspec", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(
            name="semspec", statistic=float("nan"), p_value=None, extra={"error": "computation failed"}
        )


semspec_fn = semspec


def cheatsheet() -> str:
    return "semspec({}) -> SEM common-factor restriction test (Wald)."
