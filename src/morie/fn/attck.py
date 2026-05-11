# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Attack rate and secondary attack rate."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats as _st


def attack_rate(
    cases: int,
    population: int,
    *,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Compute attack rate (cumulative incidence proportion) with CI.

    .. math::

        AR = \\frac{\\text{cases}}{\\text{population at risk}}

    Parameters
    ----------
    cases : int
        Number of cases during the epidemic period.
    population : int
        Population at risk at the start of the period.
    alpha : float, default 0.05
        Significance level for the Wilson score CI.

    Returns
    -------
    dict
        Keys: 'attack_rate', 'ci_lower', 'ci_upper',
              'cases', 'population'.
    """
    if cases < 0 or population <= 0:
        raise ValueError("cases >= 0 and population > 0 required.")
    if cases > population:
        raise ValueError("cases cannot exceed population.")

    ar = cases / population
    z = _st.norm.ppf(1 - alpha / 2)
    denom = 1 + z**2 / population
    center = ar + z**2 / (2 * population)
    half = z * np.sqrt(ar * (1 - ar) / population + z**2 / (4 * population**2))
    ci_lo = max(0.0, (center - half) / denom)
    ci_hi = min(1.0, (center + half) / denom)

    return {
        "attack_rate": float(ar),
        "ci_lower": float(ci_lo),
        "ci_upper": float(ci_hi),
        "cases": cases,
        "population": population,
    }


def secondary_attack_rate(
    secondary_cases: int,
    contacts: int,
    *,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Compute secondary attack rate (SAR) with exact binomial CI.

    .. math::

        SAR = \\frac{\\text{secondary cases}}{\\text{total contacts}}

    Parameters
    ----------
    secondary_cases : int
        Number of secondary cases among contacts.
    contacts : int
        Total number of contacts exposed.
    alpha : float, default 0.05
        Significance level for exact Clopper-Pearson CI.

    Returns
    -------
    dict
        Keys: 'sar', 'ci_lower', 'ci_upper', 'secondary_cases', 'contacts'.

    References
    ----------
    Halloran, M. E. (2001). Secondary attack rate. In *Encyclopedia of
    Biostatistics*. Wiley.
    """
    if secondary_cases < 0 or contacts <= 0:
        raise ValueError("secondary_cases >= 0 and contacts > 0 required.")
    if secondary_cases > contacts:
        raise ValueError("secondary_cases cannot exceed contacts.")

    sar = secondary_cases / contacts
    ci_lo = _st.beta.ppf(alpha / 2, secondary_cases, contacts - secondary_cases + 1) if secondary_cases > 0 else 0.0
    ci_hi = _st.beta.ppf(1 - alpha / 2, secondary_cases + 1, contacts - secondary_cases) if secondary_cases < contacts else 1.0

    return {
        "sar": float(sar),
        "ci_lower": float(ci_lo),
        "ci_upper": float(ci_hi),
        "secondary_cases": secondary_cases,
        "contacts": contacts,
    }


attck = attack_rate


def cheatsheet() -> str:
    return "attack_rate({}) -> Attack rate and secondary_attack_rate({})."
