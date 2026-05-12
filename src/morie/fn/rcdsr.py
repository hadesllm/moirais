# morie.fn -- function file (hadesllm/morie)
"""Kaplan-Meier survival curve for time-to-recidivism."""

from __future__ import annotations

import numpy as np

from morie.fn._containers import DescriptiveResult


def recidivism_survival(
    times: np.ndarray,
    events: np.ndarray,
) -> DescriptiveResult:
    """Kaplan-Meier survival curve for time-to-recidivism.

    Parameters
    ----------
    times : ndarray
        Time to event or censoring.
    events : ndarray
        Event indicator (1 = recidivism, 0 = censored).

    Returns
    -------
    DescriptiveResult
        extra contains times, survival, ci_lower, ci_upper.
    """
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=float)
    order = np.argsort(times)
    t_sorted = times[order]
    e_sorted = events[order]
    unique_t = np.unique(t_sorted[e_sorted == 1])
    surv = np.ones(len(unique_t))
    se = np.zeros(len(unique_t))
    n_at_risk = len(times)
    cum_surv = 1.0
    cum_var = 0.0
    for i, t in enumerate(unique_t):
        n_events = np.sum((t_sorted == t) & (e_sorted == 1))
        n_censor = np.sum((t_sorted == t) & (e_sorted == 0))
        if n_at_risk > 0 and n_at_risk > n_events:
            cum_surv *= 1.0 - n_events / n_at_risk
            cum_var += n_events / (n_at_risk * (n_at_risk - n_events)) if n_at_risk > n_events else 0
        surv[i] = cum_surv
        se[i] = cum_surv * np.sqrt(cum_var)
        n_at_risk -= n_events + n_censor
    ci_lo = np.maximum(surv - 1.96 * se, 0)
    ci_hi = np.minimum(surv + 1.96 * se, 1)
    return DescriptiveResult(
        name="recidivism_survival",
        value=float(surv[-1]) if len(surv) > 0 else 1.0,
        extra={
            "times": unique_t.tolist(),
            "survival": surv.tolist(),
            "ci_lower": ci_lo.tolist(),
            "ci_upper": ci_hi.tolist(),
        },
    )


rcdsr = recidivism_survival


def cheatsheet() -> str:
    return "recidivism_survival({}) -> Kaplan-Meier survival curve for time-to-recidivism."
