# morie.fn — function file (hadesllm/morie)
"""DALY = YLL + YLD calculation."""

from ._containers import ESRes


def daly_calc(
    yll: float,
    yld: float,
) -> ESRes:
    """Compute disability-adjusted life years.

    .. math::

        DALY = YLL + YLD

    Parameters
    ----------
    yll : float
    yld : float

    Returns
    -------
    ESRes
    """
    if yll < 0 or yld < 0:
        raise ValueError("YLL and YLD must be non-negative")

    total = yll + yld
    pct_yll = (yll / total * 100) if total > 0 else 0.0

    return ESRes(
        measure="DALY",
        estimate=float(total),
        extra={"yll": float(yll), "yld": float(yld), "pct_yll": float(pct_yll)},
    )


cddaly = daly_calc


def cheatsheet() -> str:
    return "daly_calc({}) -> DALY = YLL + YLD calculation."
