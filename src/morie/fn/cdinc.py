# morie.fn — function file (hadesllm/morie)
"""Person-time incidence rate with CI."""

import scipy.stats as stats

from ._containers import ESRes


def incidence_rate(
    n_new_cases: int,
    person_time: float,
    confidence: float = 0.95,
) -> ESRes:
    """Person-time incidence rate with exact Poisson CI.

    .. math::

        IR = \\frac{d}{PT}

    Parameters
    ----------
    n_new_cases : int
    person_time : float
        Total person-time at risk.
    confidence : float

    Returns
    -------
    ESRes
    """
    if person_time <= 0:
        raise ValueError("person_time must be positive")
    if n_new_cases < 0:
        raise ValueError("n_new_cases must be non-negative")

    rate = n_new_cases / person_time
    alpha = 1 - confidence
    ci_lo = stats.chi2.ppf(alpha / 2, 2 * n_new_cases) / (2 * person_time) if n_new_cases > 0 else 0.0
    ci_hi = stats.chi2.ppf(1 - alpha / 2, 2 * (n_new_cases + 1)) / (2 * person_time)

    return ESRes(
        measure="incidence_rate",
        estimate=float(rate),
        ci_lower=float(ci_lo),
        ci_upper=float(ci_hi),
        n=n_new_cases,
        extra={"person_time": float(person_time)},
    )


cdinc = incidence_rate


def cheatsheet() -> str:
    return "incidence_rate({}) -> Person-time incidence rate with CI."
