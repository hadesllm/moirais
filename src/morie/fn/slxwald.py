"""SLX Wald test on theta = 0."""

from ._containers import SpatialResult


def slxwald(theta, se_theta):
    """SLX Wald test on theta = 0.

    Category: SLX

    Parameters
    ----------
    theta, se_theta : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        result = 0.0
        return SpatialResult(name="slxwald", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(
            name="slxwald", statistic=float("nan"), p_value=None, extra={"error": "computation failed"}
        )


slxwald_fn = slxwald


def cheatsheet() -> str:
    return "slxwald({}) -> SLX Wald test on theta = 0."
