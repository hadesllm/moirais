# morie.fn -- function file (hadesllm/morie)
"""Risk difference (absolute risk difference) with Newcombe's CI."""

import math
from typing import Union

import numpy as np
import scipy.stats as stats


def risk_difference_ci(table_2x2: Union[list, np.ndarray], *, alpha: float = 0.05) -> dict:
    """
    Risk difference (absolute risk difference, ARD) with Newcombe's CI.

    RD = p1 - p2 = a/(a+b) - c/(c+d)

    Uses the Newcombe (1998) score-based method, which has better coverage
    than the standard Wald interval, especially near the boundary.

    :param table_2x2: 2x2 array [[a, b], [c, d]].
    :param alpha: Significance level. Default 0.05.
    :return: dict with keys ``risk_difference``, ``ci_lower``, ``ci_upper``,
        ``p_value``.
    :raises ValueError: If table is not 2x2, contains negatives, or row totals are 0.

    References
    ----------
    Newcombe, R. G. (1998). Interval estimation for the difference between
        independent proportions: comparison of eleven methods. Statistics in
        Medicine, 17, 873-890.
    """
    tbl = np.asarray(table_2x2, dtype=float)
    if tbl.shape != (2, 2):
        raise ValueError(f"table_2x2 must be shape (2, 2), got {tbl.shape}.")
    if np.any(tbl < 0):
        raise ValueError("Table entries must be non-negative.")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
    a, b, c, d = tbl[0, 0], tbl[0, 1], tbl[1, 0], tbl[1, 1]
    n1 = a + b
    n2 = c + d
    if n1 == 0 or n2 == 0:
        raise ValueError("Row totals must be > 0.")
    p1 = a / n1
    p2 = c / n2
    rd = p1 - p2
    # Newcombe hybrid Wilson score CI for the difference
    z = float(stats.norm.ppf(1 - alpha / 2))

    # Wilson CI for each proportion
    def _wilson_bounds(x_count, n_total):
        denom = 1 + z**2 / n_total
        centre = (x_count / n_total + z**2 / (2 * n_total)) / denom
        half = z * math.sqrt((x_count / n_total) * (1 - x_count / n_total) / n_total + z**2 / (4 * n_total**2)) / denom
        return max(0.0, centre - half), min(1.0, centre + half)

    l1, u1 = _wilson_bounds(a, n1)
    l2, u2 = _wilson_bounds(c, n2)
    ci_lower = rd - z * math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    ci_upper = rd + z * math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    # Newcombe's refined bounds using Wilson limits
    ci_lower_nc = rd - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    ci_upper_nc = rd + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    # Two-sided z-test
    se_rd = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    z_stat = rd / se_rd if se_rd > 0 else float("nan")
    p_val = 2.0 * float(stats.norm.sf(abs(z_stat))) if math.isfinite(z_stat) else float("nan")
    return {
        "risk_difference": float(rd),
        "ci_lower": float(ci_lower_nc),
        "ci_upper": float(ci_upper_nc),
        "p_value": float(p_val),
    }


rd_ci = risk_difference_ci


def cheatsheet() -> str:
    return "risk_difference_ci({}) -> Risk difference (absolute risk difference) with Newcombe's C"
