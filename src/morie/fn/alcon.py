# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Algorithm convergence diagnostics."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "No man ever steps in the same river twice. -- Heraclitus"


def algorithm_convergence(cost_history, **kwargs) -> DescriptiveResult:
    """Convergence diagnostics for an iterative algorithm.

    Checks whether the cost history is monotonically decreasing
    and estimates the relative improvement.

    Parameters
    ----------
    cost_history : array-like
        Sequence of cost/loss values per iteration.

    Returns
    -------
    DescriptiveResult
    """
    c = np.asarray(cost_history, dtype=float)
    if len(c) < 2:
        raise ValueError("Need at least 2 cost values.")
    diffs = np.diff(c)
    monotonic = bool(np.all(diffs <= 0))
    rel_change = float(np.abs(diffs[-1]) / (np.abs(c[-2]) + 1e-15))
    total_reduction = float(c[0] - c[-1])
    return DescriptiveResult(
        name="algorithm_convergence",
        value=rel_change,
        extra={
            "monotonic": monotonic,
            "final_cost": float(c[-1]),
            "initial_cost": float(c[0]),
            "total_reduction": total_reduction,
            "rel_change_last": rel_change,
            "n_iterations": len(c),
        },
    )


alcon = algorithm_convergence


def cheatsheet() -> str:
    return "algorithm_convergence({}) -> Algorithm convergence diagnostics."
