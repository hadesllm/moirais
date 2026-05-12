# morie.fn -- function file (hadesllm/morie)
"""Kaplan-Meier survival estimator."""

from __future__ import annotations

import numpy as np
from scipy import stats

from ._containers import SurvivalResult


def kaplan_meier(time, event) -> SurvivalResult:
    """Kaplan-Meier product-limit estimator.

    Parameters
    ----------
    time : array-like
        Observed times (positive).
    event : array-like
        Event indicator (1 = event, 0 = censored).

    Returns
    -------
    SurvivalResult
    """
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    mask = np.isfinite(time)
    time, event = time[mask], event[mask]
    order = np.argsort(time)
    time, event = time[order], event[order]
    unique_times = np.unique(time[event == 1])
    surv = np.ones(len(unique_times) + 1)
    times_out = np.concatenate([[0], unique_times])
    # Greenwood variance accumulator
    var_log = np.zeros(len(times_out))

    for i, t in enumerate(unique_times):
        n_at_risk = int(np.sum(time >= t))
        d = int(np.sum((time == t) & (event == 1)))
        surv[i + 1] = surv[i] * (1 - d / n_at_risk) if n_at_risk > 0 else surv[i]
        if n_at_risk > 0 and n_at_risk > d:
            var_log[i + 1] = var_log[i] + d / (n_at_risk * (n_at_risk - d))
        else:
            var_log[i + 1] = var_log[i]

    se = surv * np.sqrt(var_log)
    z = stats.norm.ppf(0.975)
    median_idx = np.searchsorted(-surv, -0.5)
    med = float(times_out[min(median_idx, len(times_out) - 1)])
    return SurvivalResult(
        name="Kaplan-Meier",
        times=times_out,
        survival=surv,
        ci_lower=np.maximum(surv - z * se, 0),
        ci_upper=np.minimum(surv + z * se, 1),
        median_survival=med,
        n_events=int(event.sum()),
        n_censored=int((event == 0).sum()),
    )


km = kaplan_meier


def cheatsheet() -> str:
    return "kaplan_meier({}) -> Kaplan-Meier survival estimator."
