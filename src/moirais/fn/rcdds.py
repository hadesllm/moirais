# moirais.fn — function file (hadesllm/moirais)
"""Desistance curve: probability of stopping offending by time."""

from __future__ import annotations

import numpy as np

from moirais.fn._containers import DescriptiveResult


def recidivism_desistance(
    recidivism_times: np.ndarray,
    *,
    max_time: float | None = None,
    n_points: int = 50,
) -> DescriptiveResult:
    """Desistance curve: complement of recidivism CDF over time.

    Parameters
    ----------
    recidivism_times : ndarray
        Observed times to recidivism (positive only, no censored).
    max_time : float, optional
        Maximum time horizon. Defaults to max observed.
    n_points : int
        Number of evaluation points.

    Returns
    -------
    DescriptiveResult
        extra: times, desistance_prob.
    """
    t = np.asarray(recidivism_times, dtype=float)
    t = t[t > 0]
    if max_time is None:
        max_time = float(np.max(t)) if len(t) > 0 else 1.0
    eval_times = np.linspace(0, max_time, n_points)
    n = len(t)
    desist = np.array([1.0 - np.sum(t <= et) / n for et in eval_times]) if n > 0 else np.ones(n_points)
    return DescriptiveResult(
        name="recidivism_desistance",
        value=float(desist[-1]) if len(desist) > 0 else 1.0,
        extra={"times": eval_times.tolist(), "desistance_prob": desist.tolist()},
    )


rcdds = recidivism_desistance


def cheatsheet() -> str:
    return "recidivism_desistance({}) -> Desistance curve: probability of stopping offending by time."
