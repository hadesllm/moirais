# morie.fn — function file (hadesllm/morie)
"""Hazard rate from life table."""

from __future__ import annotations

import numpy as np

from ._containers import ESRes


def hazard_rate(
    deaths: list[int] | np.ndarray,
    person_years: list[float] | np.ndarray,
    confidence: float = 0.95,
) -> ESRes:
    """Compute age-specific hazard rates from life table data.

    .. math::

        h_i = \\frac{d_i}{PY_i}

    Parameters
    ----------
    deaths : array-like of int
        Deaths per interval.
    person_years : array-like of float
        Person-years at risk per interval.
    confidence : float, default 0.95
        Confidence level for Poisson-based CI.

    Returns
    -------
    ESRes
        estimate is overall hazard; extra has per-interval hazards and CIs.

    References
    ----------
    Kalbfleisch, J. D. & Prentice, R. L. (2002). *The Statistical
    Analysis of Failure Time Data*, 2nd ed. Wiley, Ch. 1.
    """
    import scipy.stats as stats

    d = np.asarray(deaths, dtype=float)
    py = np.asarray(person_years, dtype=float)

    if len(d) != len(py):
        raise ValueError("deaths and person_years must match")
    if np.any(py <= 0):
        raise ValueError("person_years must be positive")

    h = d / py
    overall_h = float(np.sum(d) / np.sum(py))

    alpha = 1 - confidence
    ci_lo = []
    ci_hi = []
    for i in range(len(d)):
        if d[i] == 0:
            lo = 0.0
            hi = stats.chi2.ppf(1 - alpha / 2, 2) / (2 * py[i])
        else:
            lo = stats.chi2.ppf(alpha / 2, 2 * d[i]) / (2 * py[i])
            hi = stats.chi2.ppf(1 - alpha / 2, 2 * (d[i] + 1)) / (2 * py[i])
        ci_lo.append(float(lo))
        ci_hi.append(float(hi))

    return ESRes(
        measure="hazard_rate",
        estimate=overall_h,
        n=int(np.sum(d)),
        extra={
            "rates": h.tolist(),
            "ci_lower": ci_lo,
            "ci_upper": ci_hi,
        },
    )


hzdrt = hazard_rate


def cheatsheet() -> str:
    return "hazard_rate({}) -> Age-specific hazard rates from life table data."
