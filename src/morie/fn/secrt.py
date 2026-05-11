# morie.fn — function file (hadesllm/morie)
"""Secondary attack rate."""

from __future__ import annotations

import scipy.stats as stats

from ._containers import ESRes


def secondary_attack_rate(
    secondary_cases: int,
    contacts: int,
    confidence: float = 0.95,
) -> ESRes:
    """Secondary attack rate with exact binomial CI.

    SAR = secondary cases / total susceptible contacts exposed
    to a primary case.

    Parameters
    ----------
    secondary_cases : int
        Number of secondary infections among contacts.
    contacts : int
        Total susceptible contacts exposed.
    confidence : float, default 0.95
        Confidence level.

    Returns
    -------
    ESRes

    References
    ----------
    Halloran, M. E. (2001). Secondary attack rate. In *Encyclopedia
    of Biostatistics*. Wiley.
    """
    if contacts <= 0:
        raise ValueError("contacts must be positive")
    if secondary_cases < 0 or secondary_cases > contacts:
        raise ValueError("secondary_cases must be in [0, contacts]")

    sar = secondary_cases / contacts
    ci_lo, ci_hi = stats.binom.interval(confidence, contacts, sar)
    ci_lo = ci_lo / contacts
    ci_hi = ci_hi / contacts

    return ESRes(
        measure="SAR",
        estimate=float(sar),
        ci_lower=float(ci_lo),
        ci_upper=float(ci_hi),
        n=contacts,
        extra={"secondary_cases": secondary_cases},
    )


secrt = secondary_attack_rate


def cheatsheet() -> str:
    return "secondary_attack_rate({}) -> Secondary attack rate with exact CI."
