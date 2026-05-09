# moirais.fn — function file (hadesllm/moirais)
"""Secondary Attack Rate (SAR)."""

from __future__ import annotations

import math
from typing import Any


def secondary_attack_rate(
    secondary_cases: int,
    total_contacts: int,
    *,
    index_cases: int = 0,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """
    Compute the Secondary Attack Rate (SAR).

    .. math::

        \\text{SAR} = \\frac{\\text{secondary cases}}
            {\\text{total contacts} - \\text{index cases}}

    Used in outbreak investigation to quantify transmissibility.
    CI via Wilson score interval.

    :param secondary_cases: Number of secondary cases.
    :param total_contacts: Total number of contacts (including index cases).
    :param index_cases: Number of index (primary) cases to exclude from
        denominator (default 0).
    :param alpha: Significance level (default 0.05).
    :return: Dictionary with sar, se, ci_lower, ci_upper.
    :raises ValueError: If secondary_cases > susceptible contacts.

    References
    ----------
    Halloran, M. E. (2001). Secondary attack rate. In *Encyclopedia of
    Biostatistics*. Wiley.
    """
    from scipy import stats as _st

    susceptible = total_contacts - index_cases
    if susceptible <= 0:
        raise ValueError("Susceptible contacts (total - index) must be positive.")
    if secondary_cases < 0:
        raise ValueError("secondary_cases must be non-negative.")
    if secondary_cases > susceptible:
        raise ValueError("secondary_cases cannot exceed susceptible contacts.")

    p = secondary_cases / susceptible
    se = math.sqrt(p * (1 - p) / susceptible) if susceptible > 0 else 0.0

    # Wilson score CI
    z = _st.norm.ppf(1 - alpha / 2)
    denom = 1 + z**2 / susceptible
    centre = (p + z**2 / (2 * susceptible)) / denom
    margin = z * math.sqrt(p * (1 - p) / susceptible + z**2 / (4 * susceptible**2)) / denom

    return {
        "sar": float(p),
        "se": float(se),
        "ci_lower": float(max(0.0, centre - margin)),
        "ci_upper": float(min(1.0, centre + margin)),
    }


sar = secondary_attack_rate


def cheatsheet() -> str:
    return "secondary_attack_rate({}) -> Secondary Attack Rate (SAR)."
