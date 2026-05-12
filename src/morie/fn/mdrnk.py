# morie.fn -- function file (hadesllm/morie)
"""Midrank computation for tied observations (Gibbons Ch 5.6.2).

Returns the midranks of x.  Tied values get the mean of the ranks
they would have received had ties been broken arbitrarily.  This
is scipy.stats.rankdata's default ``'average'`` method.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from ._richresult import RichResult

__all__ = ["midranks"]


def midranks(x):
    """Midrank vector with tie-correction summary.

    Parameters
    ----------
    x : array-like

    Returns
    -------
    RichResult with payload:
        midranks       : (n,) midrank array
        n              : sample size
        ties           : list of (value, multiplicity) for repeated values
        tie_correction : sum t_j^3 - t_j across tied groups (used in
                         e.g. Wilcoxon tie-corrected variance)
    """
    x = np.asarray(x, dtype=float).ravel()
    n = int(x.size)
    if n < 1:
        return RichResult(payload={
            "midranks": np.array([]),
            "n": 0, "ties": [], "tie_correction": 0.0,
            "method": "Midranks",
        })
    mr = stats.rankdata(x, method="average")
    # Compute tied groups
    vals, counts = np.unique(x, return_counts=True)
    ties = [(float(v), int(c)) for v, c in zip(vals, counts) if c > 1]
    tie_correction = float(sum(c ** 3 - c for c in counts if c > 1))
    return RichResult(payload={
        "midranks": mr,
        "n": n,
        "ties": ties,
        "tie_correction": tie_correction,
        "method": "Midranks (Gibbons 5.6.2)",
    })


def cheatsheet():
    return "mdrnk: Midranks for tied observations"


# CANONICAL TEST
# >>> midranks([1, 2, 2, 3])
# midranks = [1, 2.5, 2.5, 4]; ties = [(2, 2)]; tie_correction = 2^3 - 2 = 6
