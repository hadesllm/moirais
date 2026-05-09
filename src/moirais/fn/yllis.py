"""Years of life lost (YLL) computation."""

from __future__ import annotations

from typing import Any

import numpy as np


def years_of_life_lost(
    ages_at_death: np.ndarray,
    life_expectancy: float | np.ndarray = 80.0,
    *,
    discount_rate: float = 0.03,
    age_weight: bool = False,
) -> dict[str, Any]:
    """Compute years of life lost (YLL) with optional discounting.

    .. math::

        YLL_i = L_i \\cdot e^{-r \\cdot 0} \\frac{1 - e^{-r L_i}}{r}

    where L_i = remaining life expectancy at death, r = discount rate.
    Without discounting: YLL_i = L_i.

    Parameters
    ----------
    ages_at_death : array_like
        Ages at death for each individual.
    life_expectancy : float or array_like, default 80.0
        Standard life expectancy (scalar or per-individual).
    discount_rate : float, default 0.03
        Annual discount rate. Set to 0 for undiscounted.
    age_weight : bool, default False
        If True, apply WHO age-weighting function C * x * exp(-beta*x)
        with C=0.1658, beta=0.04.

    Returns
    -------
    dict
        Keys: 'total_yll', 'mean_yll', 'yll_per_death', 'n_deaths'.

    References
    ----------
    Murray, C. J. L. (1994). Quantifying the burden of disease: the
    technical basis for disability-adjusted life years. Bulletin of the
    WHO, 72(3), 429-445.
    """
    ages = np.asarray(ages_at_death, dtype=float)
    if isinstance(life_expectancy, (int, float)):
        le = np.full_like(ages, life_expectancy)
    else:
        le = np.asarray(life_expectancy, dtype=float)
        if le.shape != ages.shape:
            raise ValueError("life_expectancy must match ages_at_death shape.")

    remaining = np.maximum(le - ages, 0.0)

    r = discount_rate
    if r > 0:
        yll = np.where(remaining > 0, (1 - np.exp(-r * remaining)) / r, 0.0)
    else:
        yll = remaining.copy()

    if age_weight:
        C = 0.1658
        beta = 0.04
        w = C * ages * np.exp(-beta * ages)
        yll = yll * w

    return {
        "total_yll": float(np.sum(yll)),
        "mean_yll": float(np.mean(yll)) if len(yll) > 0 else 0.0,
        "yll_per_death": yll,
        "n_deaths": len(ages),
    }


yllis = years_of_life_lost


def cheatsheet() -> str:
    return "years_of_life_lost({}) -> Years of life lost (YLL)."
