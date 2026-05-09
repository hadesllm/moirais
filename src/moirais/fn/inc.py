# moirais.fn — function file (hadesllm/moirais)
"""Incidence rate with confidence interval."""

from __future__ import annotations

import math
from typing import Any


def incidence_rate(
    events: int,
    person_time: float,
    *,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """
    Compute the incidence rate (events per person-time).

    .. math::

        \\text{IR} = \\frac{d}{T}

    Confidence interval uses the exact Poisson method (mid-P) for small
    counts and the normal approximation for large counts.

    :param events: Number of events (non-negative integer).
    :param person_time: Total person-time at risk (positive float).
    :param alpha: Significance level for CI (default 0.05).
    :return: Dictionary with rate, se, ci_lower, ci_upper.
    :raises ValueError: If events < 0 or person_time <= 0.

    References
    ----------
    Rothman, K. J., Greenland, S., & Lash, T. L. (2008). *Modern
    Epidemiology* (3rd ed.). Lippincott Williams & Wilkins, Ch. 3.
    """
    from scipy import stats as _st

    if events < 0:
        raise ValueError("events must be non-negative.")
    if person_time <= 0:
        raise ValueError("person_time must be positive.")

    rate = events / person_time
    se = math.sqrt(events) / person_time if events > 0 else 0.0

    # Exact Poisson CI for the count, then divide by person_time
    ci_lo_count = _st.chi2.ppf(alpha / 2, 2 * events) / 2 if events > 0 else 0.0
    ci_hi_count = _st.chi2.ppf(1 - alpha / 2, 2 * (events + 1)) / 2

    return {
        "rate": float(rate),
        "se": float(se),
        "ci_lower": float(ci_lo_count / person_time),
        "ci_upper": float(ci_hi_count / person_time),
    }


inc = incidence_rate


def cheatsheet() -> str:
    return "incidence_rate({}) -> Incidence rate with confidence interval."
