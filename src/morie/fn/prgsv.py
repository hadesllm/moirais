# morie.fn -- function file (hadesllm/morie)
"""Program time-to-completion survival analysis."""

from __future__ import annotations

import numpy as np

from morie.fn._containers import DescriptiveResult


def program_survival(
    times: np.ndarray | list[float],
    completed: np.ndarray | list[int],
) -> DescriptiveResult:
    """Kaplan-Meier-style time-to-completion for correctional programs.

    Parameters
    ----------
    times : array-like
        Time in program (weeks/months).
    completed : array-like
        Event indicator (1 = completed, 0 = censored/dropped).

    Returns
    -------
    DescriptiveResult
    """
    t = np.asarray(times, dtype=float)
    e = np.asarray(completed, dtype=int)
    if len(t) != len(e) or len(t) == 0:
        raise ValueError("times and completed must be non-empty and same length")
    order = np.argsort(t)
    t_sorted = t[order]
    e_sorted = e[order]

    unique_times = np.unique(t_sorted[e_sorted == 1])
    surv = 1.0
    km_times = [0.0]
    km_surv = [1.0]
    for ut in unique_times:
        at_risk = np.sum(t_sorted >= ut)
        events = np.sum((t_sorted == ut) & (e_sorted == 1))
        if at_risk > 0:
            surv *= 1 - events / at_risk
        km_times.append(float(ut))
        km_surv.append(surv)

    median_idx = next((i for i, s in enumerate(km_surv) if s <= 0.5), None)
    median_time = km_times[median_idx] if median_idx is not None else float("nan")
    return DescriptiveResult(
        name="program_survival",
        value=float(median_time),
        extra={
            "km_times": km_times,
            "km_survival": km_surv,
            "completion_rate": float(np.mean(e)),
            "n": len(t),
            "n_events": int(np.sum(e)),
        },
    )


prgsv = program_survival


def cheatsheet() -> str:
    return "program_survival({}) -> Program time-to-completion survival analysis."
