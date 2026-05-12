# morie.fn -- function file (hadesllm/morie)
"""Infection fatality rate (IFR) with seroprevalence adjustment."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats as _st


def infection_fatality_rate(
    deaths: int,
    population: int,
    seroprevalence: float,
    *,
    sensitivity: float = 1.0,
    specificity: float = 1.0,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Estimate infection fatality rate (IFR) using seroprevalence.

    The IFR corrects the CFR by estimating total infections (including
    asymptomatic/undetected) from seroprevalence surveys.

    .. math::

        \\text{Adjusted seroprevalence} = \\frac{p_{\\text{raw}} - (1 - Sp)}{Se - (1 - Sp)}

        \\text{IFR} = \\frac{D}{N \\times p_{\\text{adj}}}

    Parameters
    ----------
    deaths : int
        Cumulative deaths.
    population : int
        Population from which seroprevalence was estimated.
    seroprevalence : float
        Estimated seroprevalence (0 to 1).
    sensitivity : float, default 1.0
        Serological test sensitivity.
    specificity : float, default 1.0
        Serological test specificity.
    alpha : float, default 0.05
        Significance level for CI.

    Returns
    -------
    dict
        Keys: 'ifr', 'ci_lower', 'ci_upper', 'adjusted_seroprevalence',
              'estimated_infections'.

    References
    ----------
    Meyerowitz-Katz, G. & Merone, L. (2020). A systematic review and
    meta-analysis of published research data on COVID-19 infection
    fatality rates. International Journal of Infectious Diseases, 101,
    138-148.
    """
    if deaths < 0:
        raise ValueError("deaths must be non-negative.")
    if population <= 0:
        raise ValueError("population must be positive.")
    if not (0 <= seroprevalence <= 1):
        raise ValueError("seroprevalence must be in [0, 1].")
    if not (0 < sensitivity <= 1):
        raise ValueError("sensitivity must be in (0, 1].")
    if not (0 < specificity <= 1):
        raise ValueError("specificity must be in (0, 1].")

    denom = sensitivity - (1.0 - specificity)
    if denom <= 0:
        raise ValueError("sensitivity + specificity must exceed 1.")

    adj_sero = (seroprevalence - (1.0 - specificity)) / denom
    adj_sero = np.clip(adj_sero, 0.0, 1.0)

    est_infections = population * adj_sero

    ifr = deaths / est_infections if est_infections > 0 else np.inf

    se_sero = np.sqrt(seroprevalence * (1 - seroprevalence) / population)
    z = _st.norm.ppf(1 - alpha / 2)
    sero_lo = max(0.0, seroprevalence - z * se_sero)
    sero_hi = min(1.0, seroprevalence + z * se_sero)

    adj_lo = max(0.0, (sero_lo - (1 - specificity)) / denom)
    adj_hi = min(1.0, (sero_hi - (1 - specificity)) / denom)

    ifr_lo = deaths / (population * adj_hi) if adj_hi > 0 else 0.0
    ifr_hi = deaths / (population * adj_lo) if adj_lo > 0 else np.inf

    return {
        "ifr": float(ifr),
        "ci_lower": float(min(ifr_lo, ifr_hi)),
        "ci_upper": float(max(ifr_lo, ifr_hi)),
        "adjusted_seroprevalence": float(adj_sero),
        "estimated_infections": float(est_infections),
        "deaths": deaths,
        "population": population,
    }


ifris = infection_fatality_rate


def cheatsheet() -> str:
    return "infection_fatality_rate({}) -> IFR from seroprevalence data."
