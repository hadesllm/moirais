# moirais.fn — function file (hadesllm/moirais)
"""SEM Wald test on spatial error parameter lambda."""

from ._containers import SpatialResult


def semwald(lam, se_lam):
    """SEM Wald test on spatial error parameter lambda.

    Category: SEM

    Parameters
    ----------
    lam, se_lam : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        result = float((lam / se_lam) ** 2)
        return SpatialResult(name="semwald", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(
            name="semwald", statistic=float("nan"), p_value=None, extra={"error": "computation failed"}
        )


semwald_fn = semwald


def cheatsheet() -> str:
    return "semwald({}) -> SEM Wald test on spatial error parameter lambda."
