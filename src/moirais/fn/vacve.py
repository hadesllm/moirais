"""Real-world vaccine effectiveness."""

import numpy as np
import scipy.stats as stats

from ._containers import ESRes


def vaccine_effectiveness(
    cases_vacc: int,
    cases_unvacc: int,
    pop_vacc: int,
    pop_unvacc: int,
    confidence: float = 0.95,
) -> ESRes:
    """Compute real-world vaccine effectiveness.

    VE = 1 - (attack rate vaccinated / attack rate unvaccinated).

    Parameters
    ----------
    cases_vacc, cases_unvacc : int
    pop_vacc, pop_unvacc : int
    confidence : float

    Returns
    -------
    ESRes
    """
    if pop_vacc <= 0 or pop_unvacc <= 0:
        raise ValueError("Populations must be positive")

    ar_v = cases_vacc / pop_vacc
    ar_u = cases_unvacc / pop_unvacc
    rr = ar_v / ar_u if ar_u > 0 else 0
    ve = (1 - rr) * 100

    se_log = np.sqrt(1 / max(cases_vacc, 1) - 1 / pop_vacc + 1 / max(cases_unvacc, 1) - 1 / pop_unvacc)
    z = stats.norm.ppf((1 + confidence) / 2)
    rr_lo = rr * np.exp(-z * se_log)
    rr_hi = rr * np.exp(z * se_log)

    return ESRes(
        measure="vaccine_effectiveness_pct",
        estimate=float(ve),
        ci_lower=float((1 - rr_hi) * 100),
        ci_upper=float((1 - rr_lo) * 100),
        n=pop_vacc + pop_unvacc,
        extra={"rr": float(rr), "ar_vacc": float(ar_v), "ar_unvacc": float(ar_u)},
    )


vacve = vaccine_effectiveness


def cheatsheet() -> str:
    return "vaccine_effectiveness({}) -> Real-world vaccine effectiveness."
