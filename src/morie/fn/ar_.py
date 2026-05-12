# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Attack Rate (cumulative incidence)."""

from __future__ import annotations

import math
from typing import Any


def attack_rate(
    cases: int,
    population: int,
    *,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """
    Compute the attack rate (cumulative incidence) with Wilson score CI.

    .. math::

        \\text{AR} = \\frac{\\text{cases}}{\\text{population at risk}}

    Used in outbreak settings to quantify the proportion of a population
    that becomes ill during a defined time period.

    :param cases: Number of cases.
    :param population: Population at risk.
    :param alpha: Significance level (default 0.05).
    :return: Dictionary with rate, se, ci_lower, ci_upper.
    :raises ValueError: If cases > population or inputs invalid.

    References
    ----------
    Rothman, K. J., Greenland, S., & Lash, T. L. (2008). *Modern
    Epidemiology* (3rd ed.). Lippincott Williams & Wilkins.
    """
    from scipy import stats as _st

    if population <= 0:
        raise ValueError("population must be positive.")
    if cases < 0:
        raise ValueError("cases must be non-negative.")
    if cases > population:
        raise ValueError("cases cannot exceed population.")

    p = cases / population
    se = math.sqrt(p * (1 - p) / population) if population > 0 else 0.0

    # Wilson score CI
    z = _st.norm.ppf(1 - alpha / 2)
    denom = 1 + z**2 / population
    centre = (p + z**2 / (2 * population)) / denom
    margin = z * math.sqrt(p * (1 - p) / population + z**2 / (4 * population**2)) / denom

    return {
        "rate": float(p),
        "se": float(se),
        "ci_lower": float(max(0.0, centre - margin)),
        "ci_upper": float(min(1.0, centre + margin)),
    }


ar_ = attack_rate


def cheatsheet() -> str:
    return "attack_rate({}) -> Attack Rate (cumulative incidence)."
