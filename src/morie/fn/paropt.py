"""Identify the Pareto-optimal front from a set of multi-objective."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def pareto_optimize(
    objectives: np.ndarray,
    *,
    minimize: list[bool] | None = None,
) -> DescriptiveResult:
    """Identify the Pareto-optimal front from a set of multi-objective
    evaluations.

    Parameters
    ----------
    objectives : np.ndarray
        (n x m) matrix where each row is an evaluation of *m* objectives.
    minimize : list of bool or None
        Per-objective direction. True = minimize, False = maximize.
        Default: all minimize.

    Returns
    -------
    DescriptiveResult
        ``value`` dict with ``pareto_mask`` (bool array, n), ``pareto_front``
        (k x m), ``n_pareto``, ``hypervolume`` (2D only, otherwise None).
    """
    O = np.asarray(objectives, dtype=float)
    if O.ndim != 2:
        raise ValueError("objectives must be 2D (n x m)")
    n, m = O.shape

    if minimize is None:
        minimize = [True] * m
    if len(minimize) != m:
        raise ValueError("minimize must have length m")

    obj = O.copy()
    for j in range(m):
        if not minimize[j]:
            obj[:, j] = -obj[:, j]

    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_pareto[i]:
            continue
        for j in range(n):
            if i == j or not is_pareto[j]:
                continue
            if np.all(obj[j] <= obj[i]) and np.any(obj[j] < obj[i]):
                is_pareto[i] = False
                break

    front = O[is_pareto]
    n_pareto = int(is_pareto.sum())

    hv = None
    if m == 2 and n_pareto > 0:
        sorted_front = front[np.argsort(front[:, 0])]
        ref = np.array([obj[:, 0].max() + 1, obj[:, 1].max() + 1])
        hv = 0.0
        for i in range(len(sorted_front)):
            x_lo = sorted_front[i, 0] if minimize[0] else -sorted_front[i, 0]
            y_lo = sorted_front[i, 1] if minimize[1] else -sorted_front[i, 1]
            x_hi = (
                ref[0]
                if i == len(sorted_front) - 1
                else (sorted_front[i + 1, 0] if minimize[0] else -sorted_front[i + 1, 0])
            )
            hv += abs(x_hi - x_lo) * abs(ref[1] - y_lo)
        hv = float(hv)

    return DescriptiveResult(
        name="pareto_optimize",
        value={
            "pareto_mask": is_pareto,
            "pareto_front": front,
            "n_pareto": n_pareto,
            "hypervolume": hv,
        },
        extra={"n": n, "m": m},
    )


paropt = pareto_optimize


def cheatsheet() -> str:
    return 'paropt() -> Identify the Pareto-optimal front from a set of multi-objective'
