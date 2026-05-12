# morie.fn -- function file (hadesllm/morie)
"""Odds ratio effect size for a 2x2 table with CI."""

import math

import numpy as np
import scipy.stats as stats

from ._containers import ESRes


def odds_ratio(
    a: int,
    b: int,
    c: int,
    d: int,
    confidence: float = 0.95,
) -> ESRes:
    """Odds ratio for a 2x2 table [[a, b], [c, d]].

    OR = (a * d) / (b * c)

    Parameters
    ----------
    a, b, c, d : int
        Cell counts of the 2x2 table.
    confidence : float, default 0.95

    Returns
    -------
    ESRes
    """
    or_val = (a * d) / (b * c) if b * c > 0 else np.inf
    log_or = math.log(or_val) if or_val > 0 and np.isfinite(or_val) else 0.0
    se_log = math.sqrt(1 / max(a, 1) + 1 / max(b, 1) + 1 / max(c, 1) + 1 / max(d, 1))
    z = stats.norm.ppf((1 + confidence) / 2)
    return ESRes(
        measure="Odds ratio",
        estimate=float(or_val),
        ci_lower=float(math.exp(log_or - z * se_log)),
        ci_upper=float(math.exp(log_or + z * se_log)),
        se=float(se_log),
        n=a + b + c + d,
        extra={"log_or": log_or},
    )


or_2x2 = odds_ratio


def cheatsheet() -> str:
    return "odds_ratio({}) -> Odds ratio effect size for a 2x2 table with CI."
