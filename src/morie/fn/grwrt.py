# morie.fn — function file (hadesllm/morie)
"""Population growth rate."""

import numpy as np

from ._containers import ESRes


def population_growth_rate(
    pop_start: int,
    pop_end: int,
    years: float,
) -> ESRes:
    r"""Compute annual population growth rate.

    .. math::

        r = \\frac{\\ln(P_1 / P_0)}{t}

    Parameters
    ----------
    pop_start, pop_end : int
    years : float

    Returns
    -------
    ESRes
    """
    if pop_start <= 0 or pop_end <= 0:
        raise ValueError("Populations must be positive")
    if years <= 0:
        raise ValueError("years must be positive")

    r = np.log(pop_end / pop_start) / years
    pct = (np.exp(r) - 1) * 100

    return ESRes(
        measure="population_growth_rate",
        estimate=float(r),
        extra={"annual_pct": float(pct), "pop_start": pop_start, "pop_end": pop_end, "years": float(years)},
    )


grwrt = population_growth_rate


def cheatsheet() -> str:
    return "population_growth_rate({}) -> Population growth rate."
