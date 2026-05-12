# morie.fn -- function file (hadesllm/morie)
"""Number needed to treat (NNT) from a 2x2 table."""

import numpy as np

from ._containers import ESRes
from .rd_es import risk_difference


def number_needed_to_treat(
    a: int,
    b: int,
    c: int,
    d: int,
    confidence: float = 0.95,
) -> ESRes:
    """Number needed to treat (NNT).

    NNT = 1 / |RD|

    Parameters
    ----------
    a, b, c, d : int
    confidence : float, default 0.95

    Returns
    -------
    ESRes
    """
    rd_res = risk_difference(a, b, c, d, confidence)
    rd = rd_res.estimate
    nnt = 1 / abs(rd) if abs(rd) > 0 else np.inf
    ci_lo = 1 / abs(rd_res.ci_upper) if rd_res.ci_upper and abs(rd_res.ci_upper) > 0 else np.inf
    ci_hi = 1 / abs(rd_res.ci_lower) if rd_res.ci_lower and abs(rd_res.ci_lower) > 0 else np.inf
    return ESRes(
        measure="NNT",
        estimate=float(nnt),
        ci_lower=float(min(ci_lo, ci_hi)),
        ci_upper=float(max(ci_lo, ci_hi)),
        n=rd_res.n,
    )


nnt = number_needed_to_treat


def cheatsheet() -> str:
    return "number_needed_to_treat({}) -> Number needed to treat (NNT) from a 2x2 table."
