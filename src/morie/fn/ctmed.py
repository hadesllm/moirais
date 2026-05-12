# morie.fn -- function file (hadesllm/morie)
"""Control-median test (Gibbons Ch 6.5).

Mood's two-sample median test: count how many of each sample lie
above the pooled-sample median, test the resulting 2x2 table.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from ._richresult import RichResult

__all__ = ["control_median_test"]


def control_median_test(x, y):
    """Mood's median test (two-sample).

    Parameters
    ----------
    x : array-like
        Control sample.
    y : array-like
        Treatment sample.

    Returns
    -------
    RichResult with payload:
        statistic   : chi-square statistic (Yates-corrected by scipy)
        p_value     : two-sided p-value
        df          : 1
        n           : m + n
        grand_median: pooled-sample median
        table       : 2x2 contingency table (rows: x,y; cols: above,below)
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    m, n = int(x.size), int(y.size)
    if m < 2 or n < 2:
        return RichResult(payload={
            "statistic": np.nan, "p_value": np.nan, "df": 1,
            "n": m + n, "grand_median": np.nan,
            "method": "Control-median (Mood's median) test",
        })
    stat, p, med, tbl = stats.median_test(x, y, ties="below")
    return RichResult(payload={
        "statistic": float(stat),
        "p_value": float(p),
        "df": 1,
        "n": m + n,
        "m": m,
        "n_y": n,
        "grand_median": float(med),
        "table": tbl.tolist(),
        "method": "Control-median (Mood's median) test",
    })


def cheatsheet():
    return "ctmed: Control-median (Mood's) test"


# CANONICAL TEST
# >>> control_median_test([1,2,3,4,5], [6,7,8,9,10])
# Clear separation: grand_median=5.5, p_value very small
