# moirais.fn — function file (hadesllm/moirais)
"""Permutation test. 'The unexamined life is not worth living. -- Socrates'"""
from __future__ import annotations

import numpy as np

from ._containers import TestResult


def permutation_test(
    x,
    y,
    *,
    n_perm: int = 9999,
    seed: int = 42,
    alternative: str = "two-sided",
) -> TestResult:
    """Permutation test for difference in means."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x, y = x[np.isfinite(x)], y[np.isfinite(y)]
    obs_diff = float(x.mean() - y.mean())
    combined = np.concatenate([x, y])
    n_x = len(x)
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        p = rng.permutation(combined)
        perm_diff = p[:n_x].mean() - p[n_x:].mean()
        if alternative == "two-sided":
            if abs(perm_diff) >= abs(obs_diff):
                count += 1
        elif alternative == "greater":
            if perm_diff >= obs_diff:
                count += 1
        else:
            if perm_diff <= obs_diff:
                count += 1
    p_val = (count + 1) / (n_perm + 1)
    return TestResult(
        test_name="Permutation",
        statistic=obs_diff,
        p_value=float(p_val),
        n=len(combined),
        method=f"Permutation test ({n_perm} permutations, {alternative})",
    )


perm = permutation_test


def cheatsheet() -> str:
    return "permutation_test({}) -> Permutation test. 'The Force is what gives a Jedi his power."
