# morie.fn -- function file (hadesllm/morie)
"""Competing risks: cause-specific hazard."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def competing_risks(
    times: np.ndarray | list,
    event_types: np.ndarray | list,
    *,
    cause: int = 1,
) -> DescriptiveResult:
    """
    Compute cause-specific cumulative incidence (competing risks).

    Uses the Aalen-Johansen estimator for cause-specific CIF.

    Parameters
    ----------
    times : array-like
        Event/censoring times.
    event_types : array-like
        Event type indicators (0 = censored, 1..K = causes).
    cause : int
        Cause of interest.

    Returns
    -------
    DescriptiveResult
        extra has 'times', 'cif', 'n_events'.

    References
    ----------
    Aalen, O. O., & Johansen, S. (1978). An empirical transition
    matrix for non-homogeneous Markov chains based on censored
    observations. *Scand J Stat*, 5, 141-150.
    """
    t = np.asarray(times, dtype=float)
    ev = np.asarray(event_types, dtype=int)
    if len(t) != len(ev):
        raise ValueError("times and event_types must match.")

    n = len(t)
    order = np.argsort(t)
    t_sorted = t[order]
    ev_sorted = ev[order]

    unique_times = np.unique(t_sorted[ev_sorted > 0])
    cif = np.zeros(len(unique_times))
    surv = 1.0

    for i, ut in enumerate(unique_times):
        at_risk = np.sum(t_sorted >= ut)
        if at_risk == 0:
            continue
        d_cause = np.sum((t_sorted == ut) & (ev_sorted == cause))
        d_all = np.sum((t_sorted == ut) & (ev_sorted > 0))
        cif[i] = cif[i - 1] if i > 0 else 0.0
        cif[i] += surv * d_cause / at_risk
        surv *= 1 - d_all / at_risk

    n_events = int(np.sum(ev == cause))

    return DescriptiveResult(
        name="competing_risks",
        value=float(cif[-1]) if len(cif) > 0 else 0.0,
        extra={
            "times": unique_times.tolist(),
            "cif": cif.tolist(),
            "n_events": n_events,
            "cause": cause,
        },
    )


cmprs = competing_risks


def cheatsheet() -> str:
    return "competing_risks({}) -> Competing risks: cause-specific hazard."
