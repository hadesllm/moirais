"""Vargha-Delaney A statistic (non-parametric effect size)."""

from typing import Union

import numpy as np
import pandas as pd
import scipy.stats as stats

from ._containers import ESRes
from ._helpers import _arr, _bootstrap_ci


def vargha_delaney_a(
    x: Union[np.ndarray, pd.Series, list],
    y: Union[np.ndarray, pd.Series, list],
    confidence: float = 0.95,
) -> ESRes:
    """Vargha-Delaney A statistic.

    A = U / (n1 * n2) where U is the Mann-Whitney statistic.

    Parameters
    ----------
    x, y : array-like
    confidence : float, default 0.95

    Returns
    -------
    ESRes
    """
    x, y = _arr(x), _arr(y)
    u, _ = stats.mannwhitneyu(x, y, alternative="two-sided")
    nx, ny = len(x), len(y)
    a_val = u / (nx * ny) if nx * ny > 0 else 0.5
    se, ci_lo, ci_hi = _bootstrap_ci(
        lambda a, b: stats.mannwhitneyu(a, b, alternative="two-sided").statistic / (len(a) * len(b)),
        (x, y),
        confidence=confidence,
    )
    return ESRes(
        measure="Vargha-Delaney A",
        estimate=float(a_val),
        ci_lower=ci_lo,
        ci_upper=ci_hi,
        se=se,
        n=nx + ny,
    )


vda = vargha_delaney_a


def cheatsheet() -> str:
    return "vargha_delaney_a({}) -> Vargha-Delaney A statistic (non-parametric effect size)."
