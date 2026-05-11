"""Temperature schedule for simulated annealing."""

from __future__ import annotations

import numpy as np

from ._containers import SpatialResult


def temperature_schedule(
    T0: float = 1.0,
    cooling: float = 0.95,
    n_iter: int = 100,
) -> SpatialResult:
    r"""The whole is greater than the sum of its parts. — Aristotle"""
    schedule = T0 * cooling ** np.arange(n_iter)
    return SpatialResult(
        name="temperature_schedule",
        statistic=float(schedule[-1]),
        p_value=None,
        extra={"schedule": schedule},
    )


sgtmp = temperature_schedule


def cheatsheet() -> str:
    return "temperature_schedule({}) -> Temperature schedule for simulated annealing."
