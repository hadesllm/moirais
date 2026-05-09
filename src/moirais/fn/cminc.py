# moirais.fn — function file (hadesllm/moirais)
"""
cminc.py - Cumulative incidence function (CIF) via the Aalen-Johansen estimator
for competing risks data.

Reference: Aalen, O.O. & Johansen, S. (1978). An empirical transition matrix
for non-homogeneous Markov chains based on censored observations. Scandinavian
Journal of Statistics, 5(3), 141-150.
"""

__all__ = ["cminc"]

import numpy as np


def cminc(
    time: np.ndarray,
    event: np.ndarray,
    cause: int = 1,
) -> dict:
    """
    Estimate the cumulative incidence function (CIF) for a specific cause
    in a competing-risks framework using the Aalen-Johansen estimator.

    The CIF for cause k at time t is:
        F_k(t) = sum_{t_j <= t} S(t_j^-) * h_k(t_j)
    where S(t^-) is the overall (all-cause) survival just before t_j, and
    h_k(t_j) = d_kj / n_j is the cause-specific hazard increment.

    Parameters
    ----------
    time : np.ndarray, shape (n,)
        Observed event/censoring times (> 0).
    event : np.ndarray, shape (n,)
        Cause indicators. 0 = censored; positive integers = competing causes.
    cause : int, optional
        The cause of interest for CIF estimation. Default 1.

    Returns
    -------
    dict
        time_points : np.ndarray
            Unique event times at which CIF is evaluated.
        cif : np.ndarray
            Estimated CIF values at each time point.
        overall_survival : np.ndarray
            Estimated overall (all-cause) Kaplan-Meier survival S(t^-).
        se : np.ndarray
            Standard errors of the CIF (Gray's method).
        ci_lower : np.ndarray
            95% lower confidence bound (log-log transformation).
        ci_upper : np.ndarray
            95% upper confidence bound.

    Raises
    ------
    ValueError
        If cause not found in event indicators or inputs are invalid.

    References
    ----------
    Aalen, O.O. & Johansen, S. (1978). Scandinavian Journal of Statistics,
    5(3), 141-150.
    Gray, R.J. (1988). Journal of the American Statistical Association,
    83(403), 1198-1206.
    """
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    if np.any(time <= 0):
        raise ValueError("All times must be strictly positive.")
    if cause not in event:
        raise ValueError(f"Cause {cause} not found in event array.")

    n = len(time)
    # Unique event times across all causes
    all_event_times = np.unique(time[event > 0])

    # Overall (all-cause) KM survival S(t^-)
    # Use standard KM with event = any cause
    overall_event = (event > 0).astype(float)
    order = np.argsort(time)
    t_s = time[order]
    e_s = overall_event[order]

    # KM survival at event times
    km_times = []
    km_surv = []
    S = 1.0
    idx = 0
    for t_j in all_event_times:
        # Risk set size just before t_j
        n_risk = np.sum(t_s >= t_j)
        n_event_all = np.sum((t_s == t_j) & (e_s == 1))
        km_times.append(t_j)
        km_surv.append(S)  # S(t^-), survival just before t_j
        S *= (1 - n_event_all / n_risk) if n_risk > 0 else 1.0

    km_times = np.array(km_times)
    km_surv = np.array(km_surv)  # S(t_j^-)

    # Cause-specific hazard increments for cause k at each event time
    cause_hazard = np.zeros(len(all_event_times))
    for i, t_j in enumerate(all_event_times):
        n_risk = np.sum(time >= t_j)
        n_cause_k = np.sum((time == t_j) & (event == cause))
        cause_hazard[i] = n_cause_k / n_risk if n_risk > 0 else 0.0

    # CIF: F_k(t) = cumsum of S(t_j^-) * dH_k(t_j)
    cif = np.cumsum(km_surv * cause_hazard)

    # Variance estimation (Gray, 1988): simplified formula
    # Var(F_k(t)) ~ sum_{j: t_j<=t} S(t_j^-)^2 * h_k(t_j) * (1 - h_k(t_j)) / n_j
    var_cif = np.zeros(len(all_event_times))
    for i, t_j in enumerate(all_event_times):
        n_risk = np.sum(time >= t_j)
        n_cause_k = np.sum((time == t_j) & (event == cause))
        h_k = n_cause_k / n_risk if n_risk > 0 else 0.0
        var_cif[i] = km_surv[i] ** 2 * h_k * (1 - h_k) / max(n_risk, 1)
    se = np.sqrt(np.cumsum(var_cif))

    # 95% CI via log transformation on CIF: CI on log(F_k) scale.
    # Lower CI = F_k * exp(-z * se / F_k)
    # Upper CI = F_k * exp(+z * se / F_k)
    alpha = 0.05
    z = 1.959963985  # qnorm(0.975)
    eps = 1e-15
    with np.errstate(divide="ignore", invalid="ignore"):
        se_rel = np.where(cif > eps, se / cif, 0.0)
        ci_lo = cif * np.exp(-z * se_rel)
        ci_hi = cif * np.exp(z * se_rel)
        # Clip to [0, 1]
        ci_lo = np.clip(np.where(np.isfinite(ci_lo), ci_lo, 0.0), 0.0, 1.0)
        ci_hi = np.clip(np.where(np.isfinite(ci_hi), ci_hi, 1.0), 0.0, 1.0)

    return {
        "time_points": all_event_times,
        "cif": cif,
        "overall_survival": km_surv,
        "se": se,
        "ci_lower": ci_lo,
        "ci_upper": ci_hi,
    }
