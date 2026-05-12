# morie.fn -- function file (hadesllm/morie)
"""Rank-biserial correlation (Mann-Whitney based)."""

from typing import Union

import numpy as np
import pandas as pd
import scipy.stats as stats

from ._containers import ESRes
from ._helpers import _arr, _bootstrap_ci


def rank_biserial_correlation(
    x: Union[np.ndarray, pd.Series, list],
    y: Union[np.ndarray, pd.Series, list],
    confidence: float = 0.95,
) -> ESRes:
    """Rank-biserial correlation (matched rank version).

    r = 1 - 2U / (n1 * n2) where U is the Mann-Whitney statistic.

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
    r = 1 - 2 * u / (nx * ny) if nx * ny > 0 else 0.0
    se, ci_lo, ci_hi = _bootstrap_ci(
        lambda a, b: 1 - 2 * stats.mannwhitneyu(a, b, alternative="two-sided").statistic / (len(a) * len(b)),
        (x, y),
        confidence=confidence,
    )
    return ESRes(
        measure="Rank-biserial correlation",
        estimate=float(r),
        ci_lower=ci_lo,
        ci_upper=ci_hi,
        se=se,
        n=nx + ny,
    )


rbc = rank_biserial_correlation


def cheatsheet() -> str:
    return "rank_biserial_correlation({}) -> Rank-biserial correlation (Mann-Whitney based)."
