# morie.fn -- function file (hadesllm/morie)
"""Risk ratio (relative risk) effect size for a 2x2 table."""

import math

import numpy as np
import scipy.stats as stats

from ._containers import ESRes


def risk_ratio(
    a: int,
    b: int,
    c: int,
    d: int,
    confidence: float = 0.95,
) -> ESRes:
    """Risk ratio (relative risk) for a 2x2 table.

    RR = [a/(a+b)] / [c/(c+d)]

    Parameters
    ----------
    a, b, c, d : int
    confidence : float, default 0.95

    Returns
    -------
    ESRes
    """
    p1 = a / (a + b) if (a + b) > 0 else 0.0
    p2 = c / (c + d) if (c + d) > 0 else 0.0
    rr = p1 / p2 if p2 > 0 else np.inf
    log_rr = math.log(rr) if rr > 0 and np.isfinite(rr) else 0.0
    se_log = math.sqrt(b / (a * (a + b)) + d / (c * (c + d))) if a > 0 and c > 0 else np.inf
    z = stats.norm.ppf((1 + confidence) / 2)
    return ESRes(
        measure="Risk ratio",
        estimate=float(rr),
        ci_lower=float(math.exp(log_rr - z * se_log)),
        ci_upper=float(math.exp(log_rr + z * se_log)),
        se=float(se_log),
        n=a + b + c + d,
    )


rr_2x2 = risk_ratio


def cheatsheet() -> str:
    return "risk_ratio({}) -> Risk ratio (relative risk) effect size for a 2x2 table."
