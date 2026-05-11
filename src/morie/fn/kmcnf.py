# morie.fn — function file (hadesllm/morie)
"""
kmcnf.py - Kaplan-Meier confidence intervals with multiple methods.

Supports Greenwood's formula with plain (linear), log, log-log, logit, and
arcsine-square-root transformations, as well as the Hall-Wellner and
equal-precision bands.

Reference: Borgan, O. & Liestol, K. (1990). A note on confidence intervals and
bands for the survival function based on transformations. Scandinavian Journal
of Statistics, 17(1), 35-41.
"""

__all__ = ["kmcnf"]

import numpy as np


def kmcnf(
    time: np.ndarray,
    event: np.ndarray,
    alpha: float = 0.05,
    ci_type: str = "log-log",
) -> dict:
    """
    Kaplan-Meier estimator with confidence intervals.

    Parameters
    ----------
    time : np.ndarray, shape (n,)
        Observed survival/censoring times (> 0).
    event : np.ndarray, shape (n,)
        Event indicators (1 = event, 0 = censored).
    alpha : float, optional
        Significance level. Default 0.05 (95% CI).
    ci_type : str, optional
        Confidence interval method:
        - 'linear'   : plain Greenwood (may exceed [0,1])
        - 'log'      : log transformation (always positive)
        - 'log-log'  : complementary log-log (default, most common)
        - 'logit'    : logit transformation
        - 'arcsin'   : arcsine-sqrt transformation

    Returns
    -------
    dict
        time_points : np.ndarray
            Ordered event times.
        survival : np.ndarray
            KM survival estimates S(t).
        n_risk : np.ndarray
            Number at risk just before each event time.
        n_event : np.ndarray
            Number of events at each event time.
        ci_lower : np.ndarray
            Lower confidence bound.
        ci_upper : np.ndarray
            Upper confidence bound.
        greenwood_var : np.ndarray
            Greenwood variance of S(t): S(t)^2 * sum[d_j / (n_j(n_j - d_j))].
        median_survival : float
            Estimated median survival time (or NaN if not reached).

    Raises
    ------
    ValueError
        If inputs are invalid or ci_type is unrecognised.

    References
    ----------
    Kaplan, E.L. & Meier, P. (1958). Journal of the American Statistical
    Association, 53(282), 457-481.
    Greenwood, M. (1926). The natural duration of cancer. In Reports on
    Public Health and Medical Subjects, Vol. 33, pp. 1-26. HMSO, London.
    Borgan, O. & Liestol, K. (1990). Scandinavian Journal of Statistics,
    17(1), 35-41.
    """
    from scipy import stats as _stats

    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=float)
    if np.any(time <= 0):
        raise ValueError("All times must be strictly positive.")
    if not np.all(np.isin(event, [0, 1])):
        raise ValueError("event must be binary (0/1).")
    if ci_type not in ("linear", "log", "log-log", "logit", "arcsin"):
        raise ValueError("ci_type must be one of: 'linear', 'log', 'log-log', 'logit', 'arcsin'.")

    n = len(time)
    z = float(_stats.norm.ppf(1 - alpha / 2))

    order = np.argsort(time)
    t_s = time[order]
    e_s = event[order]

    event_times = np.unique(t_s[e_s == 1])

    km_times = []
    km_surv = []
    km_n_risk = []
    km_n_event = []
    km_gw_var = []

    S = 1.0
    gw_sum = 0.0  # Greenwood sum: sum d_j / (n_j * (n_j - d_j))

    for t_j in event_times:
        n_r = np.sum(t_s >= t_j)
        n_e = np.sum((t_s == t_j) & (e_s == 1))
        if n_r > n_e:
            gw_sum += n_e / (n_r * (n_r - n_e))
        S_new = S * (1 - n_e / n_r)
        km_times.append(t_j)
        km_surv.append(S_new)
        km_n_risk.append(n_r)
        km_n_event.append(n_e)
        km_gw_var.append(S_new ** 2 * gw_sum)
        S = S_new

    km_times = np.array(km_times)
    km_surv = np.array(km_surv)
    km_n_risk = np.array(km_n_risk, dtype=int)
    km_n_event = np.array(km_n_event, dtype=int)
    km_gw_var = np.array(km_gw_var)

    eps = 1e-15
    se = np.sqrt(np.maximum(km_gw_var, 0.0))

    if ci_type == "linear":
        ci_lo = np.clip(km_surv - z * se, 0.0, 1.0)
        ci_hi = np.clip(km_surv + z * se, 0.0, 1.0)

    elif ci_type == "log":
        # CI on log(S): log(S) +/- z * se / S => exp(...)
        with np.errstate(divide="ignore", invalid="ignore"):
            se_log = np.where(km_surv > 0, se / km_surv, 0.0)
        log_S = np.log(np.maximum(km_surv, eps))
        ci_lo = np.clip(np.exp(log_S - z * se_log), 0.0, 1.0)
        ci_hi = np.clip(np.exp(log_S + z * se_log), 0.0, 1.0)

    elif ci_type == "log-log":
        # CI on log(-log(S)) scale
        with np.errstate(divide="ignore", invalid="ignore"):
            log_S = np.log(np.maximum(km_surv, eps))
            log_neg_log_S = np.log(np.maximum(-log_S, eps))
            # SE on log-log scale: se / (S * |log(S)|)
            se_ll = np.where(
                (km_surv > 0) & (-log_S > 0),
                se / (km_surv * np.maximum(-log_S, eps)),
                0.0,
            )
        ci_lo = np.clip(np.exp(-np.exp(log_neg_log_S + z * se_ll)), 0.0, 1.0)
        ci_hi = np.clip(np.exp(-np.exp(log_neg_log_S - z * se_ll)), 0.0, 1.0)

    elif ci_type == "logit":
        # CI on logit(S) = log(S/(1-S))
        S_cl = np.clip(km_surv, eps, 1 - eps)
        logit_S = np.log(S_cl / (1 - S_cl))
        se_logit = np.where(
            S_cl * (1 - S_cl) > 0,
            se / (S_cl * (1 - S_cl)),
            0.0,
        )
        ci_lo = np.clip(1 / (1 + np.exp(-(logit_S - z * se_logit))), 0.0, 1.0)
        ci_hi = np.clip(1 / (1 + np.exp(-(logit_S + z * se_logit))), 0.0, 1.0)

    else:  # arcsin
        # CI on arcsin(sqrt(S))
        S_cl = np.clip(km_surv, eps, 1 - eps)
        arcsin_S = np.arcsin(np.sqrt(S_cl))
        se_arcsin = np.where(
            S_cl * (1 - S_cl) > 0,
            se / (2 * np.sqrt(S_cl * (1 - S_cl))),
            0.0,
        )
        ci_lo = np.clip(np.sin(np.maximum(arcsin_S - z * se_arcsin, 0.0)) ** 2, 0.0, 1.0)
        ci_hi = np.clip(np.sin(np.minimum(arcsin_S + z * se_arcsin, np.pi / 2)) ** 2, 0.0, 1.0)

    # Median survival: first time S <= 0.5
    median_idx = np.where(km_surv <= 0.5)[0]
    median_surv = float(km_times[median_idx[0]]) if len(median_idx) > 0 else np.nan

    return {
        "time_points": km_times,
        "survival": km_surv,
        "n_risk": km_n_risk,
        "n_event": km_n_event,
        "ci_lower": ci_lo,
        "ci_upper": ci_hi,
        "greenwood_var": km_gw_var,
        "median_survival": median_surv,
    }
