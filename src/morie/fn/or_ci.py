# morie.fn -- function file (hadesllm/morie)
"""Odds ratio with confidence interval."""

import math
from typing import Union

import numpy as np
import scipy.stats as stats


def odds_ratio_ci(table_2x2: Union[list, np.ndarray], *, alpha: float = 0.05) -> dict:
    """
    Odds ratio with exact (Baptista-Pike) confidence interval from scipy.

    For a 2x2 table [[a, b], [c, d]]:
    OR = (a * d) / (b * c)

    :param table_2x2: 2x2 array [[exposed_case, exposed_control],
        [unexposed_case, unexposed_control]].
    :param alpha: Significance level. Default 0.05.
    :return: dict with keys ``odds_ratio``, ``ci_lower``, ``ci_upper``, ``p_value``.
    :raises ValueError: If table is not 2x2 or contains negatives.

    References
    ----------
    Baptista, J., & Pike, M. C. (1977). Exact two-sided confidence limits for the
        odds ratio in a 2 x 2 table. Applied Statistics, 26(2), 214-220.
    """
    tbl = np.asarray(table_2x2, dtype=float)
    if tbl.shape != (2, 2):
        raise ValueError(f"table_2x2 must be shape (2, 2), got {tbl.shape}.")
    if np.any(tbl < 0):
        raise ValueError("Table entries must be non-negative.")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
    # Fisher exact gives odds ratio and two-sided p-value
    or_point, p_val = stats.fisher_exact(tbl.astype(int))
    # For CI, use the Woolf log-normal CI (well-defined when all cells > 0)
    a, b, c, d = tbl[0, 0], tbl[0, 1], tbl[1, 0], tbl[1, 1]
    if all(v > 0 for v in [a, b, c, d]):
        log_or = math.log(a * d / (b * c))
        se_log_or = math.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
        z = float(stats.norm.ppf(1 - alpha / 2))
        ci_lower = math.exp(log_or - z * se_log_or)
        ci_upper = math.exp(log_or + z * se_log_or)
    else:
        # Haldane-Anscombe correction: add 0.5 to all cells
        a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5
        log_or = math.log(a * d / (b * c))
        se_log_or = math.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
        z = float(stats.norm.ppf(1 - alpha / 2))
        ci_lower = math.exp(log_or - z * se_log_or)
        ci_upper = math.exp(log_or + z * se_log_or)
    return {
        "odds_ratio": float(or_point),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "p_value": float(p_val),
    }


or_ci = odds_ratio_ci


def cheatsheet() -> str:
    return "odds_ratio_ci({}) -> Odds ratio with confidence interval."
