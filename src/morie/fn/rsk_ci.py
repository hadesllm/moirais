# morie.fn -- function file (hadesllm/morie)
"""Risk ratio (relative risk) with log-normal Wald confidence interval."""

import math
from typing import Union

import numpy as np
import scipy.stats as stats
from ._richresult import RichResult


def risk_ratio_ci(table_2x2: Union[list, np.ndarray], *, alpha: float = 0.05) -> dict:
    """
    Risk ratio (relative risk) with log-normal Wald confidence interval.

    For a 2x2 table [[a, b], [c, d]] where rows are exposure and
    columns are outcome:
    RR = [a/(a+b)] / [c/(c+d)]

    :param table_2x2: 2x2 array [[exposed_outcome, exposed_no_outcome],
        [unexposed_outcome, unexposed_no_outcome]].
    :param alpha: Significance level. Default 0.05.
    :return: dict with keys ``risk_ratio``, ``ci_lower``, ``ci_upper``, ``p_value``.
    :raises ValueError: If table is not 2x2 or contains negatives.

    References
    ----------
    Rothman, K. J., Greenland, S., & Lash, T. L. (2008). Modern Epidemiology
        (3rd ed.). Lippincott Williams & Wilkins. (Chapter 4.)
    """
    tbl = np.asarray(table_2x2, dtype=float)
    if tbl.shape != (2, 2):
        raise ValueError(f"table_2x2 must be shape (2, 2), got {tbl.shape}.")
    if np.any(tbl < 0):
        raise ValueError("Table entries must be non-negative.")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
    a, b, c, d = tbl[0, 0], tbl[0, 1], tbl[1, 0], tbl[1, 1]
    n1 = a + b  # exposed
    n2 = c + d  # unexposed
    if n1 == 0 or n2 == 0:
        raise ValueError("Row totals must be > 0.")
    p1 = a / n1
    p2 = c / n2
    if p2 == 0:
        return RichResult(payload={"risk_ratio": float("inf"), "ci_lower": float("nan"), "ci_upper": float("nan"), "p_value": float("nan")})
    rr = p1 / p2
    log_rr = math.log(rr)
    # SE(log RR) = sqrt((1-p1)/(a) + (1-p2)/(c)) via delta method
    se_log_rr = math.sqrt((1 - p1) / a + (1 - p2) / c) if (a > 0 and c > 0) else float("inf")
    z = float(stats.norm.ppf(1 - alpha / 2))
    ci_lower = math.exp(log_rr - z * se_log_rr)
    ci_upper = math.exp(log_rr + z * se_log_rr)
    z_stat = log_rr / se_log_rr if se_log_rr > 0 else float("nan")
    p_val = 2.0 * float(stats.norm.sf(abs(z_stat))) if math.isfinite(z_stat) else float("nan")
    return {
        "risk_ratio": float(rr),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "p_value": float(p_val),
    }


rsk_ci = risk_ratio_ci


def cheatsheet() -> str:
    return "risk_ratio_ci({}) -> Risk ratio (relative risk) with log-normal Wald confidence i"
