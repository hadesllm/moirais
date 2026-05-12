# morie.fn -- function file (hadesllm/morie)
"""SDEM Wald test on lambda."""

from ._containers import SpatialResult


def sdemwld(lam, se_lam):
    """SDEM Wald test on lambda.

    Category: SDEM

    Parameters
    ----------
    lam, se_lam : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        result = float((lam / se_lam) ** 2)
        return SpatialResult(name="sdemwld", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(
            name="sdemwld", statistic=float("nan"), p_value=None, extra={"error": "computation failed"}
        )


sdemwld_fn = sdemwld


def cheatsheet() -> str:
    return "sdemwld({}) -> SDEM Wald test on lambda."
