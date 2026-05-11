# morie.fn — function file (hadesllm/morie)
"""SAR Wald test on spatial lag parameter."""

from ._containers import SpatialResult


def sarwald(rho, se_rho):
    """SAR Wald test on spatial lag parameter.

    Category: SAR

    Parameters
    ----------
    rho, se_rho : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        result = float((rho / se_rho) ** 2)
        return SpatialResult(name="sarwald", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(
            name="sarwald", statistic=float("nan"), p_value=None, extra={"error": "computation failed"}
        )


sarwald_fn = sarwald


def cheatsheet() -> str:
    return "sarwald({}) -> SAR Wald test on spatial lag parameter."
