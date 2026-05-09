# moirais.fn — function file (hadesllm/moirais)
"""Competing risks analysis for recidivism."""

from __future__ import annotations

import numpy as np

from moirais.fn._containers import DescriptiveResult


def recidivism_competing(
    times: np.ndarray,
    event_types: np.ndarray,
) -> DescriptiveResult:
    """Competing risks: recidivism vs death vs other exit.

    Computes cumulative incidence for each event type using the
    Aalen-Johansen estimator (non-parametric).

    Parameters
    ----------
    times : ndarray
        Time to event or censoring.
    event_types : ndarray
        Event type: 0 = censored, 1 = recidivism, 2 = death, etc.

    Returns
    -------
    DescriptiveResult
        extra contains cumulative incidence per event type.
    """
    times = np.asarray(times, dtype=float)
    event_types = np.asarray(event_types, dtype=int)
    unique_types = sorted(set(event_types) - {0})
    order = np.argsort(times)
    t_s = times[order]
    e_s = event_types[order]
    unique_t = np.sort(np.unique(t_s))
    n = len(times)
    cif = {k: np.zeros(len(unique_t)) for k in unique_types}
    surv = 1.0
    n_risk = n
    for i, t in enumerate(unique_t):
        mask = t_s == t
        d_total = np.sum(e_s[mask] > 0)
        for k in unique_types:
            d_k = np.sum(e_s[mask] == k)
            if n_risk > 0:
                cif[k][i] = (cif[k][i - 1] if i > 0 else 0) + surv * d_k / n_risk
            else:
                cif[k][i] = cif[k][i - 1] if i > 0 else 0
        if n_risk > 0:
            surv *= 1.0 - d_total / n_risk
        n_risk -= np.sum(mask)
    extra = {f"cif_type_{k}": cif[k].tolist() for k in unique_types}
    extra["times"] = unique_t.tolist()
    extra["event_types"] = unique_types
    return DescriptiveResult(name="recidivism_competing", value=None, extra=extra)


rcdcm = recidivism_competing


def cheatsheet() -> str:
    return "recidivism_competing({}) -> Competing risks analysis for recidivism."
