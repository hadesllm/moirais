# moirais.fn — function file (hadesllm/moirais)
"""Point-biserial correlation."""

from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats

from ._containers import ESRes


def point_biserial(
    x,
    y,
) -> ESRes:
    """Point-biserial correlation between a binary and continuous variable.

    Parameters
    ----------
    x : array-like
        Binary variable (0/1).
    y : array-like
        Continuous variable.

    Returns
    -------
    ESRes
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) != len(y):
        raise ValueError("x and y must have the same length.")
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 3:
        raise ValueError("Need at least 3 observations.")
    unique = np.unique(x)
    if not np.array_equal(np.sort(unique), np.array([0.0, 1.0])):
        raise ValueError("x must be binary (0/1).")

    r, pval = sp_stats.pointbiserialr(x, y)
    return ESRes(
        measure="point_biserial",
        estimate=float(r),
        n=n,
        extra={"p_value": float(pval)},
    )


ptbis = point_biserial


def cheatsheet() -> str:
    return "point_biserial(x, y) -> Point-biserial correlation."
