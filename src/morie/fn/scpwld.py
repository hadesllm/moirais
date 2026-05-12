# morie.fn -- function file (hadesllm/morie)
"""Spatial Poisson Wald test on rho."""

from ._containers import SpatialResult


def scpwld(rho, se_rho):
    """Spatial Poisson Wald test on rho.

    Category: SCount

    Parameters
    ----------
    rho, se_rho : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        result = float((rho / se_rho) ** 2)
        return SpatialResult(name="scpwld", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="scpwld", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


scpwld_fn = scpwld


def cheatsheet() -> str:
    return "scpwld({}) -> Spatial Poisson Wald test on rho."
