"""Years of life lost with discounting and age weighting (GBD method)."""

from __future__ import annotations

import numpy as np

from ._containers import ESRes


def years_of_life_lost_std(
    deaths: list[int] | np.ndarray,
    ages_at_death: list[float] | np.ndarray,
    life_expectancy: list[float] | np.ndarray | float = 80.0,
    discount_rate: float = 0.03,
    age_weight: bool = False,
    beta: float = 0.04,
    C: float = 0.1658,
) -> ESRes:
    """Years of life lost with discounting and optional age weighting.

    Extends the simple YLL in ``yll.py`` with GBD-style discounting
    and age-weighting functions.

    .. math::

        YLL_i = \\frac{1 - e^{-r L_i}}{r}

    where :math:`r` is the discount rate and :math:`L_i` is years lost.

    Parameters
    ----------
    deaths : array-like of int
        Number of deaths per age group.
    ages_at_death : array-like of float
        Mean age at death per group.
    life_expectancy : float or array-like
        Standard life expectancy (scalar or per group).
    discount_rate : float, default 0.03
        Discount rate (0 = no discounting).
    age_weight : bool, default False
        Apply GBD age-weighting function.
    beta : float, default 0.04
        Age-weight parameter.
    C : float, default 0.1658
        Age-weight constant.

    Returns
    -------
    ESRes

    References
    ----------
    Murray, C. J. L. (1994). Quantifying the burden of disease: the
    technical basis for disability-adjusted life years. Bulletin of
    the World Health Organization, 72(3), 429-445.
    """
    d = np.asarray(deaths, dtype=float)
    a = np.asarray(ages_at_death, dtype=float)
    if np.isscalar(life_expectancy):
        le = np.full_like(a, life_expectancy)
    else:
        le = np.asarray(life_expectancy, dtype=float)

    if len(d) != len(a) or len(d) != len(le):
        raise ValueError("All arrays must have the same length")

    L = np.maximum(le - a, 0)
    r = discount_rate

    total_yll = 0.0
    for i in range(len(d)):
        if r > 0:
            yll_i = (1 - np.exp(-r * L[i])) / r
        else:
            yll_i = L[i]

        if age_weight:
            yll_i *= C * a[i] * np.exp(-beta * a[i])

        total_yll += d[i] * yll_i

    return ESRes(
        measure="YLL_std",
        estimate=float(total_yll),
        n=int(np.sum(d)),
        extra={
            "discount_rate": r,
            "age_weighted": age_weight,
            "years_remaining": L.tolist(),
        },
    )


yllst = years_of_life_lost_std


def cheatsheet() -> str:
    return "years_of_life_lost_std({}) -> YLL with discounting and age weighting."
