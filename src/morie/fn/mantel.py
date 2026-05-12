# morie.fn -- function file (hadesllm/morie)
"""Mantel test (matrix correlation)."""

from __future__ import annotations

import numpy as np

from ._containers import TestResult


def mantel_test(
    D1: np.ndarray,
    D2: np.ndarray,
    *,
    n_perm: int = 999,
    seed: int | None = None,
) -> TestResult:
    """Mantel test: correlation between two distance matrices.

    Parameters
    ----------
    D1, D2 : (n, n) symmetric distance matrices
    n_perm : int
        Permutations for p-value.
    seed : int, optional

    Returns
    -------
    TestResult
    """
    D1 = np.asarray(D1, dtype=float)
    D2 = np.asarray(D2, dtype=float)
    n = D1.shape[0]
    if D1.shape != D2.shape or D1.shape[0] != D1.shape[1]:
        raise ValueError("D1 and D2 must be square and same size.")

    idx = np.triu_indices(n, k=1)
    d1 = D1[idx]
    d2 = D2[idx]
    r_obs = float(np.corrcoef(d1, d2)[0, 1])

    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(n)
        D2p = D2[np.ix_(perm, perm)]
        r_perm = np.corrcoef(d1, D2p[idx])[0, 1]
        if r_perm >= r_obs:
            count += 1

    p = (count + 1) / (n_perm + 1)

    return TestResult(
        test_name="Mantel",
        statistic=r_obs,
        p_value=float(p),
        method="Mantel permutation test",
        n=n,
        extra={"n_perm": n_perm},
    )


mantel = mantel_test


def cheatsheet() -> str:
    return "mantel_test({}) -> Mantel test (matrix correlation)."
