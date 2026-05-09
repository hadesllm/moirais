"""Ideal point trajectories over time."""

from __future__ import annotations

from ._containers import DescriptiveResult


def trace_over_time(X_sessions, time_labels) -> DescriptiveResult:
    """Track legislator ideal points across sessions.

    .. epigraph:: "Change is good, right?" -- Walter White, Breaking Bad
    """
    import numpy as np

    trajectories = []
    for i, X in enumerate(X_sessions):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        trajectories.append(X)
    n_sessions = len(trajectories)
    n_leg = trajectories[0].shape[0] if trajectories else 0
    shifts = []
    if n_sessions >= 2:
        for t in range(1, n_sessions):
            diff = np.linalg.norm(trajectories[t] - trajectories[t - 1], axis=1)
            shifts.append(float(np.mean(diff)))
    return DescriptiveResult(
        name="trace_over_time",
        value=float(np.mean(shifts)) if shifts else 0.0,
        extra={
            "n_sessions": n_sessions,
            "n_legislators": n_leg,
            "mean_shifts": shifts,
            "time_labels": list(time_labels),
        },
    )


tmtrc = trace_over_time


def cheatsheet() -> str:
    return "trace_over_time({}) -> Ideal point trajectories over time."
