# moirais.fn — function file (hadesllm/moirais)
"""Jonckheere-Terpstra test. 'Rebellions are built on hope. -- Jyn Erso'"""

from __future__ import annotations

import numpy as np
from scipy import stats

from ._containers import TestResult


def jonckheere_terpstra(*groups: np.ndarray) -> TestResult:
    """Jonckheere-Terpstra test for ordered alternatives.

    Tests H1: theta_1 <= theta_2 <= ... <= theta_k (at least one strict)
    against H0: all theta_i equal.  Computes the J statistic as the
    sum of Mann-Whitney U counts across ordered pairs.

    :param groups: Two or more 1-D arrays in hypothesised order.
    :return: TestResult with J statistic and normal-approximation p-value.
    """
    if len(groups) < 2:
        raise ValueError("Need at least 2 groups.")
    arrays = [np.asarray(g, dtype=float).ravel() for g in groups]
    k = len(arrays)
    ns = [len(a) for a in arrays]
    N = sum(ns)

    J = 0.0
    for i in range(k - 1):
        for j in range(i + 1, k):
            for xi in arrays[i]:
                for yj in arrays[j]:
                    if yj > xi:
                        J += 1.0
                    elif yj == xi:
                        J += 0.5

    E_J = (N**2 - sum(ni**2 for ni in ns)) / 4.0
    Var_J = (N**2 * (2 * N + 3) - sum(ni**2 * (2 * ni + 3) for ni in ns)) / 72.0
    z = (J - E_J) / (np.sqrt(Var_J) + 1e-12)
    pval = float(1.0 - stats.norm.cdf(z))

    return TestResult(
        test_name="Jonckheere-Terpstra",
        statistic=float(J),
        p_value=pval,
        method="JT ordered alternatives (normal approx)",
        n=N,
        extra={"z": float(z), "E_J": E_J, "Var_J": Var_J, "k": k, "ns": ns},
    )


jnckh = jonckheere_terpstra


def cheatsheet() -> str:
    return "jonckheere_terpstra({}) -> Jonckheere-Terpstra test. 'Rebellions are built on hope. -- "
