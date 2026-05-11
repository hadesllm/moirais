# morie.fn — function file (hadesllm/morie)
"""Risk difference (attributable risk) effect size for a 2x2 table."""

import math

import scipy.stats as stats

from ._containers import ESRes


def risk_difference(
    a: int,
    b: int,
    c: int,
    d: int,
    confidence: float = 0.95,
) -> ESRes:
    """Risk difference (attributable risk) for a 2x2 table.

    RD = a/(a+b) - c/(c+d)

    Parameters
    ----------
    a, b, c, d : int
    confidence : float, default 0.95

    Returns
    -------
    ESRes
    """
    n1, n2 = a + b, c + d
    p1 = a / n1 if n1 > 0 else 0.0
    p2 = c / n2 if n2 > 0 else 0.0
    rd = p1 - p2
    se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2) if n1 > 0 and n2 > 0 else 0.0
    z = stats.norm.ppf((1 + confidence) / 2)
    return ESRes(
        measure="Risk difference",
        estimate=float(rd),
        ci_lower=float(rd - z * se),
        ci_upper=float(rd + z * se),
        se=float(se),
        n=n1 + n2,
    )


rd_2x2 = risk_difference


def cheatsheet() -> str:
    return "risk_difference({}) -> Risk difference (attributable risk) effect size for a 2x2 ta"
