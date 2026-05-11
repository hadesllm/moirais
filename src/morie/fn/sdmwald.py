# morie.fn — function file (hadesllm/morie)
"""SDM Wald test on rho."""

from ._containers import SpatialResult


def sdmwald(rho, se_rho):
    """SDM Wald test on rho.

    Category: SDM

    Parameters
    ----------
    rho, se_rho : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        result = float((rho / se_rho) ** 2)
        return SpatialResult(name="sdmwald", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(
            name="sdmwald", statistic=float("nan"), p_value=None, extra={"error": "computation failed"}
        )


sdmwald_fn = sdmwald


def cheatsheet() -> str:
    return "sdmwald({}) -> SDM Wald test on rho."
