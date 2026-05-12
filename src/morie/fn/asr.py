# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Age-standardized rate (direct method)."""

import numpy as np
import scipy.stats as stats

from ._containers import ESRes


def age_standardized_rate(
    counts: np.ndarray,
    population: np.ndarray,
    standard_pop: np.ndarray,
    confidence: float = 0.95,
    per: float = 100_000.0,
) -> ESRes:
    r"""Age-standardized rate using the direct method.

    .. math::

        ASR = \\frac{\\sum_i w_i \\cdot r_i}{\\sum_i w_i}

    where :math:`r_i` is the age-specific rate and :math:`w_i` is the
    standard population weight for age group *i*.

    Parameters
    ----------
    counts : array-like
        Observed event counts per age group.
    population : array-like
        Population at risk per age group.
    standard_pop : array-like
        Standard population weights per age group.
    confidence : float, default 0.95
        Confidence level.
    per : float, default 100000
        Multiplier (rate per X).

    Returns
    -------
    ESRes

    References
    ----------
    Breslow, N. E. & Day, N. E. (1987). Statistical Methods in Cancer
    Research, Vol. 2. IARC Scientific Publications No. 82.
    """
    counts = np.asarray(counts, dtype=float)
    population = np.asarray(population, dtype=float)
    standard_pop = np.asarray(standard_pop, dtype=float)

    if len(counts) != len(population) or len(counts) != len(standard_pop):
        raise ValueError("counts, population, and standard_pop must have equal length")
    if np.any(population <= 0):
        raise ValueError("population values must be positive")

    rates = counts / population
    w = standard_pop / np.sum(standard_pop)
    asr_val = np.sum(w * rates) * per

    var_asr = np.sum((w**2) * counts / (population**2)) * per**2
    se = np.sqrt(var_asr)
    z = stats.norm.ppf((1 + confidence) / 2)

    return ESRes(
        measure="ASR",
        estimate=float(asr_val),
        ci_lower=float(asr_val - z * se),
        ci_upper=float(asr_val + z * se),
        se=float(se),
        n=int(np.sum(counts)),
    )


asr = age_standardized_rate


def cheatsheet() -> str:
    return "age_standardized_rate({}) -> Age-standardized rate (direct method)."
