# morie.fn -- function file (hadesllm/morie)
"""Age-adjusted chronic disease prevalence."""

import numpy as np
import scipy.stats as stats

from ._containers import ESRes


def chronic_disease_prevalence(
    n_cases: int | np.ndarray,
    population: int | np.ndarray,
    age_groups: np.ndarray | None = None,
    std_pop: np.ndarray | None = None,
    confidence: float = 0.95,
) -> ESRes:
    """Age-adjusted chronic disease prevalence via direct standardization.

    If *age_groups* and *std_pop* are provided, computes age-adjusted rate.
    Otherwise returns crude prevalence.

    Parameters
    ----------
    n_cases : int or array
        Cases per age group (or total).
    population : int or array
        Population per age group (or total).
    age_groups : array or None
        Age group labels.
    std_pop : array or None
        Standard population weights.
    confidence : float

    Returns
    -------
    ESRes
    """
    if std_pop is not None and age_groups is not None:
        c = np.asarray(n_cases, dtype=float)
        p = np.asarray(population, dtype=float)
        w = np.asarray(std_pop, dtype=float)
        rates = c / p
        adj_rate = float(np.sum(rates * w) / np.sum(w))
        se = float(np.sqrt(np.sum((w / np.sum(w)) ** 2 * rates * (1 - rates) / p)))
        z = stats.norm.ppf((1 + confidence) / 2)
        return ESRes(
            measure="age_adjusted_prevalence",
            estimate=adj_rate,
            ci_lower=float(max(0, adj_rate - z * se)),
            ci_upper=float(adj_rate + z * se),
            se=float(se),
            n=int(np.sum(p)),
        )

    total_cases = int(np.sum(n_cases))
    total_pop = int(np.sum(population))
    if total_pop <= 0:
        raise ValueError("population must be positive")
    crude = total_cases / total_pop
    z = stats.norm.ppf((1 + confidence) / 2)
    se = np.sqrt(crude * (1 - crude) / total_pop)

    return ESRes(
        measure="crude_prevalence",
        estimate=float(crude),
        ci_lower=float(max(0, crude - z * se)),
        ci_upper=float(crude + z * se),
        se=float(se),
        n=total_pop,
    )


cdprv = chronic_disease_prevalence


def cheatsheet() -> str:
    return "chronic_disease_prevalence({}) -> Age-adjusted chronic disease prevalence."
