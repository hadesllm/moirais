# morie.fn — function file (hadesllm/morie)
"""Cramer's V for categorical association."""

import math
from typing import Union

import numpy as np
import scipy.stats as stats


def cramers_v(contingency_table: Union[list, np.ndarray]) -> float:
    """
    Cramer's V for categorical association (r x c contingency table).

    V = sqrt(chi^2 / (n * min(r-1, c-1)))

    Ranges from 0 (no association) to 1 (perfect association).
    Conventional benchmarks for df_min = 1: 0.10 small, 0.30 medium, 0.50 large.

    :param contingency_table: r x c contingency table (non-negative integers).
    :return: Cramer's V in [0, 1].
    :raises ValueError: If table has fewer than 2 rows or columns, or negative entries.

    References
    ----------
    Cramer, H. (1946). Mathematical Methods of Statistics. Princeton University Press.
    Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences (2nd ed.).
    """
    tbl = np.asarray(contingency_table, dtype=float)
    if tbl.ndim != 2 or tbl.shape[0] < 2 or tbl.shape[1] < 2:
        raise ValueError("contingency_table must be at least a 2x2 array.")
    if np.any(tbl < 0):
        raise ValueError("Table entries must be non-negative.")
    chi2_stat, _, _, _ = stats.chi2_contingency(tbl)
    n = tbl.sum()
    min_dim = min(tbl.shape[0] - 1, tbl.shape[1] - 1)
    denom = n * min_dim
    return float(math.sqrt(chi2_stat / denom)) if denom > 0 else 0.0


cramv = cramers_v


def cheatsheet() -> str:
    return "cramers_v({}) -> Cramer's V for categorical association."
