# morie.fn — function file (hadesllm/morie)
"""Indirect age-adjustment via Standardized Mortality Ratio (SMR)."""

import numpy as np
import scipy.stats as stats

from ._containers import ESRes


def indirect_age_adjustment(
    observed_deaths: np.ndarray,
    expected_rates: np.ndarray,
    population: np.ndarray,
    confidence: float = 0.95,
) -> ESRes:
    r"""Indirect age-standardization via the Standardized Mortality Ratio.

    .. math::

        SMR = \\frac{O}{E} = \\frac{\\sum d_i}{\\sum R_i \\cdot n_i}

    where :math:`d_i` are observed deaths, :math:`R_i` are reference rates,
    and :math:`n_i` are study population sizes per stratum.

    Parameters
    ----------
    observed_deaths : array-like
        Observed deaths per stratum.
    expected_rates : array-like
        Reference/standard rates per stratum.
    population : array-like
        Study population per stratum.
    confidence : float, default 0.95
        Confidence level for Byar's CI.

    Returns
    -------
    ESRes
        estimate = SMR. extra contains O, E.

    References
    ----------
    Breslow, N. E. & Day, N. E. (1987). Statistical Methods in Cancer
    Research, Vol. 2. IARC Scientific Publications No. 82.
    """
    obs = np.asarray(observed_deaths, dtype=float)
    er = np.asarray(expected_rates, dtype=float)
    pop = np.asarray(population, dtype=float)

    if len(obs) != len(er) or len(obs) != len(pop):
        raise ValueError("All arrays must have equal length")

    O = float(np.sum(obs))
    E = float(np.sum(er * pop))

    if E <= 0:
        raise ValueError("Expected deaths (E) must be positive")

    smr = O / E

    z = stats.norm.ppf((1 + confidence) / 2)
    ci_lo = (np.sqrt(O) - z * 0.5) ** 2 / E if O > 0 else 0.0
    ci_hi = (np.sqrt(O) + z * 0.5) ** 2 / E

    return ESRes(
        measure="SMR",
        estimate=float(smr),
        ci_lower=float(ci_lo),
        ci_upper=float(ci_hi),
        n=int(O),
        extra={"O": O, "E": E},
    )


iar = indirect_age_adjustment


def cheatsheet() -> str:
    return "indirect_age_adjustment({}) -> Indirect age-adjustment via Standardized Mortality Ratio (SM"
