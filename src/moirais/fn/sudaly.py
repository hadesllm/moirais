"""DALYs from substance use."""

from ._containers import ESRes


def substance_daly(
    yll: float,
    yld: float,
) -> ESRes:
    """Compute disability-adjusted life years from substance use.

    .. math::

        DALY = YLL + YLD

    Parameters
    ----------
    yll : float
        Years of life lost.
    yld : float
        Years lived with disability.

    Returns
    -------
    ESRes
    """
    if yll < 0 or yld < 0:
        raise ValueError("YLL and YLD must be non-negative")

    total = yll + yld
    pct_yll = (yll / total * 100) if total > 0 else 0.0

    return ESRes(
        measure="substance_DALY",
        estimate=float(total),
        extra={"yll": float(yll), "yld": float(yld), "pct_yll": float(pct_yll), "pct_yld": float(100 - pct_yll)},
    )


sudaly = substance_daly


def cheatsheet() -> str:
    return "substance_daly({}) -> DALYs from substance use."
