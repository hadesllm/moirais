# morie.fn -- function file (hadesllm/morie)
"""Jonckheere-Terpstra test for ordered alternatives."""

from typing import Union

import numpy as np
import scipy.stats as stats

from ._containers import TestResult


def jonckheere_terpstra_test(
    *groups: Union[list, np.ndarray],
) -> TestResult:
    r"""
    Jonckheere-Terpstra test for ordered alternatives.

    Tests H0: all *k* groups come from the same distribution against
    H1: there is an a-priori ordering
    :math:`\\theta_1 \\le \\theta_2 \\le \\cdots \\le \\theta_k` (with at
    least one strict inequality).

    .. math::

        J = \\sum_{i < j} U_{ij}

    where :math:`U_{ij}` is the Mann-Whitney U statistic comparing group *i*
    to group *j*.

    :param groups: Two or more array-like groups in hypothesised order.
    :return: TestResult with J statistic, z, p_value.
    :raises ValueError: If fewer than 2 groups provided.

    References
    ----------
    Jonckheere, A. R. (1954). A distribution-free k-sample test against
        ordered alternatives. Biometrika, 41(1-2), 133-145.
    """
    if len(groups) < 2:
        raise ValueError("Jonckheere-Terpstra test requires at least 2 groups.")

    arrays = [np.asarray(g, dtype=float) for g in groups]
    k = len(arrays)
    ns = [len(a) for a in arrays]
    N = sum(ns)

    # Compute J = sum of U_{ij} for i < j
    J = 0.0
    for i in range(k):
        for j in range(i + 1, k):
            for xi in arrays[i]:
                for xj in arrays[j]:
                    if xj > xi:
                        J += 1.0
                    elif xj == xi:
                        J += 0.5

    # Expected value and variance under H0
    mu_J = (N**2 - sum(ni**2 for ni in ns)) / 4.0

    # Variance (no ties formula)
    sum_ni2 = sum(ni**2 for ni in ns)
    sum_ni3 = sum(ni**3 for ni in ns)

    var_num1 = N**2 * (2 * N + 3) - sum(ni**2 * (2 * ni + 3) for ni in ns)
    var_J = var_num1 / 72.0

    if var_J <= 0:
        return TestResult(
            test_name="Jonckheere-Terpstra",
            statistic=float(J),
            p_value=float("nan"),
            method="Jonckheere-Terpstra test (degenerate variance)",
            n=N,
            extra={"k": k},
        )

    z = (J - mu_J) / np.sqrt(var_J)
    # One-sided: large J supports ordered alternative
    p_value = float(stats.norm.sf(z))

    return TestResult(
        test_name="Jonckheere-Terpstra",
        statistic=float(J),
        p_value=p_value,
        method="Jonckheere-Terpstra test",
        n=N,
        extra={"z": float(z), "k": k, "expected_J": mu_J},
    )


jt = jonckheere_terpstra_test


def cheatsheet() -> str:
    return "jonckheere_terpstra_test({}) -> Jonckheere-Terpstra test for ordered alternatives."
