# moirais.fn — function file (hadesllm/moirais)
"""Logistic map iterations."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def logistic_map(
    r: float,
    x0: float = 0.5,
    n_iter: int = 100,
) -> DescriptiveResult:
    """Iterate the logistic map x_{n+1} = r * x_n * (1 - x_n).

    Parameters
    ----------
    r : float
        Growth rate parameter (interesting range: 2.5 to 4.0).
    x0 : float
        Initial condition in (0, 1).
    n_iter : int
        Number of iterations.

    Returns
    -------
    DescriptiveResult
    """
    if not 0 < x0 < 1:
        raise ValueError("x0 must be in (0, 1).")

    xs = np.empty(n_iter + 1)
    xs[0] = x0
    for i in range(n_iter):
        xs[i + 1] = r * xs[i] * (1 - xs[i])

    tail = xs[n_iter // 2 :]
    unique_approx = len(np.unique(np.round(tail, 6)))

    return DescriptiveResult(
        name="logistic_map",
        value=float(xs[-1]),
        extra={
            "r": r,
            "x0": x0,
            "n_iter": n_iter,
            "mean_tail": float(tail.mean()),
            "std_tail": float(tail.std()),
            "unique_attractors_approx": unique_approx,
            "trajectory_last10": xs[-10:].tolist(),
        },
    )


chaos = logistic_map


def cheatsheet() -> str:
    return "logistic_map({}) -> Logistic map iterations."
