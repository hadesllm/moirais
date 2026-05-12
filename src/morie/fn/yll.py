"""Years of Life Lost (YLL)."""

from __future__ import annotations

from typing import Any, Union

import numpy as np


def years_of_life_lost(
    ages_at_death: Union[list, np.ndarray],
    *,
    life_expectancy: float = 80.0,
) -> dict[str, Any]:
    r"""
    Compute Years of Life Lost (YLL).

    .. math::

        \\text{YLL} = \\sum_{i=1}^{n} \\max(0,\\; e^* - a_i)

    where e* is the reference life expectancy and a_i is age at death.

    :param ages_at_death: Array of ages at death.
    :param life_expectancy: Reference life expectancy (default 80).
    :return: Dictionary with total_yll, mean_yll, n.
    :raises ValueError: If ages array is empty or life_expectancy <= 0.

    References
    ----------
    WHO (2024). *Global Health Estimates: Methods and Data Sources*.
    World Health Organization.

    Murray, C. J. L. (1994). Quantifying the burden of disease: the
    technical basis for disability-adjusted life years. *Bulletin of the
    WHO*, 72(3), 429--445.
    """
    ages = np.asarray(ages_at_death, dtype=float)
    if len(ages) == 0:
        raise ValueError("ages_at_death must not be empty.")
    if life_expectancy <= 0:
        raise ValueError("life_expectancy must be positive.")

    yll_per_death = np.maximum(0.0, life_expectancy - ages)
    total = float(np.sum(yll_per_death))
    n = len(ages)

    return {
        "total_yll": total,
        "mean_yll": total / n,
        "n": n,
    }


yll = years_of_life_lost


def cheatsheet() -> str:
    return "years_of_life_lost({}) -> Years of Life Lost (YLL)."
