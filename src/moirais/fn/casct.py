# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""Case-control odds ratio with Woolf CI."""

from __future__ import annotations

import numpy as np
import scipy.stats as stats

from ._containers import ESRes


def case_control_or(a: int, b: int, c: int, d: int, confidence: float = 0.95, cdf=None) -> ESRes:
    """Odds ratio from a case-control study (2x2 table).

    .. math::

        OR = \\frac{ad}{bc}

    With Woolf (log-based) confidence interval.

    Parameters
    ----------
    a : int
        Exposed cases.
    b : int
        Unexposed cases.
    c : int
        Exposed controls.
    d : int
        Unexposed controls.
    confidence : float, default 0.95
        Confidence level.

    Returns
    -------
    ESRes

    References
    ----------
    Woolf, B. (1955). On estimating the relation between blood group
    and disease. Annals of Human Genetics, 19(4), 251-253.
    """
    if any(x <= 0 for x in [a, b, c, d]):
        raise ValueError("All cell counts must be positive")

    or_val = (a * d) / (b * c)
    se_ln = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    z = stats.norm.ppf((1 + confidence) / 2)
    ln_or = np.log(or_val)

    ci_lo = np.exp(ln_or - z * se_ln)
    ci_hi = np.exp(ln_or + z * se_ln)

    chi2 = (a * d - b * c) ** 2 * (a + b + c + d) / ((a + b) * (c + d) * (a + c) * (b + d))
    p_val = 1 - stats.chi2.cdf(chi2, 1)

    return ESRes(
        measure="OR_cc",
        estimate=float(or_val),
        se=float(se_ln),
        ci_lower=float(ci_lo),
        ci_upper=float(ci_hi),
        n=a + b + c + d,
        extra={"chi2": float(chi2), "p_value": float(p_val)},
    )


casct = case_control_or


def cheatsheet() -> str:
    return "case_control_or({}) -> Case-control odds ratio with Woolf CI."
