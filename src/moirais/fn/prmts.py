# moirais.fn — function file (hadesllm/moirais)
"""Permutation test (two-sample)."""

from __future__ import annotations

import numpy as np

from ._containers import TestResult


def permutation_test_two(
    x,
    y,
    *,
    statistic: str = "mean",
    n_perm: int = 9999,
    seed: int = 42,
    alternative: str = "two-sided",
) -> TestResult:
    """Exact-style permutation test for two independent samples.

    Parameters
    ----------
    x, y : array-like
        Two independent samples.
    statistic : str
        ``"mean"`` or ``"median"``.
    n_perm : int
        Number of permutations.
    seed : int
        RNG seed.
    alternative : str
        ``"two-sided"``, ``"greater"``, or ``"less"``.

    Returns
    -------
    TestResult
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 1 or len(y) < 1:
        raise ValueError("Both samples must be non-empty.")

    stat_fn = np.mean if statistic == "mean" else np.median
    obs = stat_fn(x) - stat_fn(y)
    combined = np.concatenate([x, y])
    n1 = len(x)
    rng = np.random.default_rng(seed)

    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(combined)
        d = stat_fn(perm[:n1]) - stat_fn(perm[n1:])
        if alternative == "two-sided":
            if abs(d) >= abs(obs):
                count += 1
        elif alternative == "greater":
            if d >= obs:
                count += 1
        else:
            if d <= obs:
                count += 1

    pval = (count + 1) / (n_perm + 1)

    return TestResult(
        test_name="Permutation test (two-sample)",
        statistic=float(obs),
        p_value=float(pval),
        n=len(x) + len(y),
        method=f"statistic={statistic}, alternative={alternative}",
        extra={"n_perm": n_perm},
    )


prmts = permutation_test_two


def cheatsheet() -> str:
    return "permutation_test_two(x, y) -> Two-sample permutation test."
