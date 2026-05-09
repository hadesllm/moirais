# moirais.fn — function file (hadesllm/moirais)
"""Incidence rate ratio (IRR) with asymptotic log-normal confidence interval."""

import math

import scipy.stats as stats


def rate_ratio_ci(n1: int, t1: float, n2: int, t2: float, *, alpha: float = 0.05) -> dict:
    """
    Incidence rate ratio (IRR) with asymptotic log-normal confidence interval.

    IRR = (n1/t1) / (n2/t2)

    The CI uses the delta method on the log scale:
    SE(log IRR) = sqrt(1/n1 + 1/n2) (assuming Poisson counts).

    :param n1: Event count in group 1 (>= 0).
    :param t1: Person-time in group 1 (> 0).
    :param n2: Event count in group 2 (>= 0).
    :param t2: Person-time in group 2 (> 0).
    :param alpha: Significance level. Default 0.05.
    :return: dict with keys ``irr``, ``log_irr``, ``se_log_irr``, ``ci_lower``,
        ``ci_upper``, ``p_value``.
    :raises ValueError: If t1 <= 0, t2 <= 0, or n1/n2 < 0.

    References
    ----------
    Rothman, K. J., Greenland, S., & Lash, T. L. (2008). Modern Epidemiology
        (3rd ed.). Lippincott Williams & Wilkins. (Chapter 14.)
    """
    if t1 <= 0 or t2 <= 0:
        raise ValueError("Person-time values t1 and t2 must be > 0.")
    if n1 < 0 or n2 < 0:
        raise ValueError("Event counts n1 and n2 must be >= 0.")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
    r1 = n1 / t1
    r2 = n2 / t2
    irr = r1 / r2 if r2 > 0 else float("inf")
    log_irr = math.log(irr) if irr > 0 else float("-inf")
    # SE of log IRR under Poisson: sqrt(1/n1 + 1/n2)
    se_log = math.sqrt(1.0 / n1 + 1.0 / n2) if (n1 > 0 and n2 > 0) else float("inf")
    z = float(stats.norm.ppf(1 - alpha / 2))
    ci_lower = math.exp(log_irr - z * se_log) if math.isfinite(log_irr) else 0.0
    ci_upper = math.exp(log_irr + z * se_log) if math.isfinite(log_irr) else float("inf")
    # Two-sided Wald z-test on log scale
    z_stat = log_irr / se_log if se_log > 0 else float("nan")
    p_val = 2.0 * float(stats.norm.sf(abs(z_stat))) if math.isfinite(z_stat) else float("nan")
    return {
        "irr": irr,
        "log_irr": log_irr,
        "se_log_irr": se_log,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value": p_val,
    }


rr_ci = rate_ratio_ci


def cheatsheet() -> str:
    return "rate_ratio_ci({}) -> Incidence rate ratio (IRR) with asymptotic log-normal confid"
