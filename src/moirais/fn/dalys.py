# moirais.fn — function file (hadesllm/moirais)
"""Disability-adjusted life years (DALY) computation."""

from __future__ import annotations

from typing import Any

import numpy as np


def disability_adjusted_life_years(
    yll_total: float,
    yld_total: float,
) -> dict[str, Any]:
    """Compute disability-adjusted life years (DALY = YLL + YLD).

    .. math::

        DALY = YLL + YLD

    Parameters
    ----------
    yll_total : float
        Total years of life lost due to premature mortality.
    yld_total : float
        Total years lived with disability.

    Returns
    -------
    dict
        Keys: 'daly', 'yll', 'yld', 'pct_yll', 'pct_yld'.

    References
    ----------
    Murray, C. J. L. & Lopez, A. D. (1996). *The Global Burden of
    Disease*. Harvard School of Public Health/WHO/World Bank.
    """
    if yll_total < 0 or yld_total < 0:
        raise ValueError("YLL and YLD must be non-negative.")

    daly = yll_total + yld_total
    pct_yll = (yll_total / daly * 100) if daly > 0 else 0.0
    pct_yld = (yld_total / daly * 100) if daly > 0 else 0.0

    return {
        "daly": float(daly),
        "yll": float(yll_total),
        "yld": float(yld_total),
        "pct_yll": float(pct_yll),
        "pct_yld": float(pct_yld),
    }


def yld_from_prevalence(
    prevalence: float,
    disability_weight: float,
    duration: float = 1.0,
    *,
    discount_rate: float = 0.03,
) -> dict[str, Any]:
    """Compute years lived with disability (YLD) from prevalence.

    .. math::

        YLD = P \\times DW \\times D

    Parameters
    ----------
    prevalence : float
        Number of prevalent cases (or prevalence proportion * population).
    disability_weight : float
        Disability weight (0 = perfect health, 1 = death).
    duration : float, default 1.0
        Average duration of disability (years).
    discount_rate : float, default 0.03
        Annual discount rate.

    Returns
    -------
    dict
        Keys: 'yld', 'yld_discounted'.
    """
    if not (0 <= disability_weight <= 1):
        raise ValueError("disability_weight must be in [0, 1].")
    if prevalence < 0 or duration < 0:
        raise ValueError("prevalence and duration must be non-negative.")

    yld = prevalence * disability_weight * duration

    r = discount_rate
    if r > 0 and duration > 0:
        yld_disc = prevalence * disability_weight * (1 - np.exp(-r * duration)) / r
    else:
        yld_disc = yld

    return {
        "yld": float(yld),
        "yld_discounted": float(yld_disc),
    }


dalys = disability_adjusted_life_years


def cheatsheet() -> str:
    return "disability_adjusted_life_years({}) -> DALY = YLL + YLD."
