# moirais.fn — function file (hadesllm/moirais)
"""Case fatality rate (CFR) with Wilson confidence interval."""

import numpy as np
import scipy.stats as stats

from ._containers import ESRes


def case_fatality_rate(
    deaths: int,
    cases: int,
    confidence: float = 0.95,
) -> ESRes:
    """Case fatality rate with Wilson score confidence interval.

    .. math::

        CFR = \\frac{\\text{deaths}}{\\text{cases}}

    Parameters
    ----------
    deaths : int
        Number of deaths.
    cases : int
        Number of cases (must be > 0).
    confidence : float, default 0.95
        Confidence level for Wilson CI.

    Returns
    -------
    ESRes

    References
    ----------
    Wilson, E. B. (1927). Probable inference, the law of succession, and
    statistical inference. Journal of the American Statistical
    Association, 22(158), 209-212.
    """
    if cases <= 0:
        raise ValueError("cases must be positive")
    if deaths < 0 or deaths > cases:
        raise ValueError("deaths must be in [0, cases]")

    p = deaths / cases
    z = stats.norm.ppf((1 + confidence) / 2)
    denom = 1 + z**2 / cases
    centre = p + z**2 / (2 * cases)
    margin = z * np.sqrt(p * (1 - p) / cases + z**2 / (4 * cases**2))
    ci_lo = (centre - margin) / denom
    ci_hi = (centre + margin) / denom

    return ESRes(
        measure="CFR",
        estimate=float(p),
        ci_lower=float(ci_lo),
        ci_upper=float(ci_hi),
        n=cases,
    )


cfr = case_fatality_rate


def cheatsheet() -> str:
    return "case_fatality_rate({}) -> Case fatality rate (CFR) with Wilson confidence interval."
