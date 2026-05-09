# moirais.fn — function file (hadesllm/moirais)
"""
medsv.py - Median survival time with confidence interval.

The median survival time is the smallest t such that S(t) <= 0.5.
The confidence interval uses the Brookmeyer-Crowley method based on
inverting the log-rank test.

Reference: Brookmeyer, R. & Crowley, J. (1982). A confidence interval for
the median survival time. Biometrics, 38(1), 29-41.
"""

__all__ = ["medsv"]

import numpy as np


def medsv(
    time: np.ndarray,
    event: np.ndarray,
    alpha: float = 0.05,
    quantile: float = 0.5,
) -> dict:
    """
    Estimate median (or other quantile) survival time with confidence interval.

    Uses the Kaplan-Meier estimator with the Brookmeyer-Crowley CI, which
    is the set of values m such that a log-rank-type test of H0: median = m
    is not rejected at level alpha.

    Parameters
    ----------
    time : np.ndarray, shape (n,)
        Observed survival/censoring times (> 0).
    event : np.ndarray, shape (n,)
        Event indicators (1 = event, 0 = censored).
    alpha : float, optional
        Significance level. Default 0.05.
    quantile : float, optional
        Survival quantile to estimate (0.5 = median). Must be in (0, 1).
        The corresponding survival probability is 1 - quantile.

    Returns
    -------
    dict
        estimate : float
            Estimated survival time at the specified quantile.
        ci_lower : float
            Lower confidence bound (Brookmeyer-Crowley).
        ci_upper : float
            Upper confidence bound.
        km_survival : float
            KM survival value at the estimated quantile time.
        n_events : int
            Total number of events.
        n_total : int
            Total sample size.

    Raises
    ------
    ValueError
        If quantile is outside (0, 1) or inputs are invalid.

    References
    ----------
    Brookmeyer, R. & Crowley, J. (1982). Biometrics, 38(1), 29-41.
    Kaplan, E.L. & Meier, P. (1958). Journal of the American Statistical
    Association, 53(282), 457-481.
    """
    from scipy import stats as _stats

    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=float)
    if np.any(time <= 0):
        raise ValueError("All times must be strictly positive.")
    if not np.all(np.isin(event, [0, 1])):
        raise ValueError("event must be binary (0/1).")
    if not 0 < quantile < 1:
        raise ValueError("quantile must be in (0, 1).")

    target_survival = 1.0 - quantile
    n = len(time)
    n_events = int(event.sum())

    order = np.argsort(time)
    t_s = time[order]
    e_s = event[order]

    event_times = np.unique(t_s[e_s == 1])
    km_t = [0.0]
    km_s = [1.0]
    S = 1.0
    gw_sum = 0.0
    km_se = [0.0]  # SE at t=0 is 0

    for t_j in event_times:
        n_r = np.sum(t_s >= t_j)
        n_e = np.sum((t_s == t_j) & (e_s == 1))
        S *= (1 - n_e / n_r)
        if n_r > n_e:
            gw_sum += n_e / (n_r * (n_r - n_e))
        km_t.append(float(t_j))
        km_s.append(float(S))
        km_se.append(float(S * np.sqrt(gw_sum)))

    km_t = np.array(km_t)
    km_s = np.array(km_s)
    km_se = np.array(km_se)

    # Estimate: first time S(t) <= target_survival
    idx_below = np.where(km_s <= target_survival)[0]
    if len(idx_below) == 0:
        estimate = np.nan
        s_at_est = float(km_s[-1])
    else:
        estimate = float(km_t[idx_below[0]])
        s_at_est = float(km_s[idx_below[0]])

    # Brookmeyer-Crowley CI: invert the test using the log-log-transformed CI
    # The CI is [L, U] where L = min t: S(t) >= lower_bound and
    #                         U = max t: S(t) >= upper_bound
    # where lower/upper bounds come from:
    # lower_bound = S(t) * exp(z * se / S(t) / |log S(t)|)
    # upper_bound = S(t) * exp(-z * se / S(t) / |log S(t)|)
    # on the log-log scale.
    z = float(_stats.norm.ppf(1 - alpha / 2))

    eps = 1e-15
    # Log-log CI for each time point
    with np.errstate(divide="ignore", invalid="ignore"):
        log_s = np.log(np.maximum(km_s, eps))
        log_neg_log_s = np.log(np.maximum(-log_s, eps))
        se_ll = np.where(
            (km_s > eps) & (-log_s > eps),
            km_se / (km_s * np.maximum(-log_s, eps)),
            0.0,
        )
    ci_lo_curve = np.clip(np.exp(-np.exp(log_neg_log_s + z * se_ll)), 0.0, 1.0)
    ci_hi_curve = np.clip(np.exp(-np.exp(log_neg_log_s - z * se_ll)), 0.0, 1.0)

    # CI bounds for quantile: find times where CI band crosses target_survival
    lo_cross = np.where(ci_hi_curve <= target_survival)[0]
    hi_cross = np.where(ci_lo_curve <= target_survival)[0]
    ci_lower = float(km_t[lo_cross[0]]) if len(lo_cross) > 0 else np.nan
    ci_upper = float(km_t[hi_cross[0]]) if len(hi_cross) > 0 else np.nan

    return {
        "estimate": estimate,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "km_survival": s_at_est,
        "n_events": n_events,
        "n_total": n,
    }
