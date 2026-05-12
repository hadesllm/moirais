# morie.fn -- function file (hadesllm/morie)
"""Cohort study risk ratio with log-based CI."""

from __future__ import annotations

import numpy as np
import scipy.stats as stats

from ._containers import ESRes


def cohort_risk_ratio(
    a: int,
    b: int,
    c: int,
    d: int,
    confidence: float = 0.95,
) -> ESRes:
    """Risk ratio from a cohort study (2x2 table).

    .. math::

        RR = \\frac{a / (a+b)}{c / (c+d)}

    Parameters
    ----------
    a : int
        Exposed with outcome.
    b : int
        Exposed without outcome.
    c : int
        Unexposed with outcome.
    d : int
        Unexposed without outcome.
    confidence : float, default 0.95
        Confidence level.

    Returns
    -------
    ESRes

    References
    ----------
    Rothman, K. J. et al. (2008). *Modern Epidemiology*, 3rd ed.
    Lippincott Williams & Wilkins, Ch. 15.
    """
    n1 = a + b
    n0 = c + d
    if n1 <= 0 or n0 <= 0:
        raise ValueError("Group sizes must be positive")
    if a < 0 or b < 0 or c < 0 or d < 0:
        raise ValueError("Cell counts must be non-negative")
    if c == 0:
        raise ValueError("Cannot compute RR when unexposed outcome count is 0")

    r1 = a / n1
    r0 = c / n0
    rr = r1 / r0 if r0 > 0 else np.inf

    se_ln = np.sqrt(1 / a - 1 / n1 + 1 / c - 1 / n0) if a > 0 else np.inf
    z = stats.norm.ppf((1 + confidence) / 2)
    ln_rr = np.log(rr) if rr > 0 and np.isfinite(rr) else np.inf

    ci_lo = np.exp(ln_rr - z * se_ln)
    ci_hi = np.exp(ln_rr + z * se_ln)

    return ESRes(
        measure="RR_cohort",
        estimate=float(rr),
        se=float(se_ln),
        ci_lower=float(ci_lo),
        ci_upper=float(ci_hi),
        n=n1 + n0,
        extra={"risk_exposed": float(r1), "risk_unexposed": float(r0)},
    )


cohrt = cohort_risk_ratio


def cheatsheet() -> str:
    return "cohort_risk_ratio({}) -> Cohort study risk ratio with CI."
