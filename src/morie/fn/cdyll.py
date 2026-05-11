# morie.fn — function file (hadesllm/morie)
"""Years of life lost (YLL)."""

import numpy as np

from ._containers import ESRes


def years_life_lost(
    deaths: int | np.ndarray,
    life_expectancy_remaining: float | np.ndarray,
) -> ESRes:
    """Compute YLL = sum of remaining life expectancy at age of death.

    Parameters
    ----------
    deaths : int or array
        Number of deaths (or array of individual death counts per age).
    life_expectancy_remaining : float or array
        Remaining LE at age of death.

    Returns
    -------
    ESRes
    """
    d = np.atleast_1d(np.asarray(deaths, dtype=float))
    le = np.atleast_1d(np.asarray(life_expectancy_remaining, dtype=float))

    if d.shape != le.shape:
        raise ValueError("deaths and life_expectancy_remaining must match in shape")

    yll = float(np.sum(d * le))

    return ESRes(
        measure="YLL",
        estimate=yll,
        extra={"total_deaths": int(np.sum(d)), "mean_le_remaining": float(np.mean(le))},
    )


cdyll = years_life_lost


def cheatsheet() -> str:
    return "years_life_lost({}) -> Years of life lost (YLL)."
