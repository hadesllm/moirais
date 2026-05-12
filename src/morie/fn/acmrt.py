# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Age-cause-specific mortality rates."""

from __future__ import annotations

import numpy as np

from ._containers import ESRes


def age_cause_mortality(
    deaths: list[list[int]] | np.ndarray,
    populations: list[int] | np.ndarray,
    age_labels: list[str] | None = None,
    cause_labels: list[str] | None = None,
    per: int = 100000,
) -> ESRes:
    """Compute age-cause-specific mortality rates.

    Parameters
    ----------
    deaths : 2D array-like (n_ages x n_causes)
        Deaths by age group (rows) and cause (columns).
    populations : array-like of int
        Population per age group.
    age_labels : list[str], optional
        Labels for age groups.
    cause_labels : list[str], optional
        Labels for causes.
    per : int, default 100000
        Rate denominator (per 100,000).

    Returns
    -------
    ESRes
        estimate is overall crude rate; extra has the rate matrix.

    References
    ----------
    Preston, S. H. et al. (2001). *Demography: Measuring and Modeling
    Population Processes*. Blackwell, Ch. 2.
    """
    d = np.asarray(deaths, dtype=float)
    p = np.asarray(populations, dtype=float)

    if d.ndim == 1:
        d = d.reshape(-1, 1)
    if d.shape[0] != len(p):
        raise ValueError("deaths rows must match populations length")
    if np.any(p <= 0):
        raise ValueError("populations must be positive")

    rates = (d.T / p * per).T
    total_deaths = float(np.sum(d))
    total_pop = float(np.sum(p))
    crude_rate = total_deaths / total_pop * per

    return ESRes(
        measure="age_cause_mortality",
        estimate=float(crude_rate),
        n=int(total_deaths),
        extra={
            "rates": rates.tolist(),
            "per": per,
            "age_labels": age_labels,
            "cause_labels": cause_labels,
        },
    )


acmrt = age_cause_mortality


def cheatsheet() -> str:
    return "age_cause_mortality({}) -> Age-cause-specific mortality rates."
