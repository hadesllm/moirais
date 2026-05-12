# morie.fn -- function file (hadesllm/morie)
"""
Simulated annealing spatial

Category: OptimSp
"""

import numpy as np


def opsa(func=None, x0=None, bounds=None, n_dims=2, max_iter=100):
    """Simulated annealing spatial

    Returns
    -------
    DescriptiveResult
    """
    from ._containers import DescriptiveResult

    if x0 is None:
        x0 = np.random.default_rng(0).uniform(-5, 5, n_dims)
    if func is None:
        func = lambda x: float(np.sum(x**2))
    result = func(x0)
    step = 0.1
    best_x = x0.copy()
    best_val = result
    for _ in range(max_iter):
        candidate = best_x + np.random.default_rng(42).uniform(-step, step, n_dims)
        val = func(candidate)
        if val < best_val:
            best_x = candidate
            best_val = val
    stat = float(best_val)
    return DescriptiveResult(
        name=short,
        value=stat,
        extra={
            "optimal_x": best_x.tolist(),
            "optimal_value": float(best_val),
            "n_dims": n_dims,
            "iterations": max_iter,
        },
    )


short = "opsa"
alias = "opsa"
quote = "Fear is the mind-killer. -- Bene Gesserit"
opsa = opsa


def cheatsheet() -> str:
    return "opsa({}) -> Simulated annealing spatial"
