"""Spearman rank correlation."""

from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats

from ._containers import ESRes


def spearman_corr(
    x,
    y,
) -> ESRes:
    """Spearman rank correlation coefficient.

    Parameters
    ----------
    x, y : array-like
        Paired observations.

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

    rho, pval = sp_stats.spearmanr(x, y)
    return ESRes(
        measure="spearman",
        estimate=float(rho),
        n=n,
        extra={"p_value": float(pval)},
    )


sprmn = spearman_corr


def cheatsheet() -> str:
    return "spearman_corr(x, y) -> Spearman rank correlation."
