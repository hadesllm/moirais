# morie.fn -- function file (hadesllm/morie)
"""Chinese restaurant process and partition probability. 'Understand structure, you must.'"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.special import loggamma


def chinese_restaurant_process(
    n: int,
    *,
    alpha: float = 1.0,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Simulate a Chinese Restaurant Process (CRP).

    :param n: Number of customers.
    :param alpha: Concentration parameter.
    :param seed: Random seed.
    :return: Dictionary with table_assignments, n_tables, table_sizes.

    References
    ----------
    Aldous, D. J. (1985). *Lecture Notes in Mathematics*, 1117, 1--198.
    """
    rng = np.random.default_rng(seed)
    assignments = np.zeros(n, dtype=int)
    table_counts = {}
    next_table = 0

    for i in range(n):
        probs = []
        tables = []
        for t, count in table_counts.items():
            probs.append(count)
            tables.append(t)
        probs.append(alpha)
        tables.append(next_table if next_table not in table_counts else max(table_counts.keys()) + 1)

        probs = np.array(probs, dtype=float)
        probs /= np.sum(probs)
        chosen = tables[rng.choice(len(tables), p=probs)]
        assignments[i] = chosen

        if chosen in table_counts:
            table_counts[chosen] += 1
        else:
            table_counts[chosen] = 1
            next_table = max(table_counts.keys()) + 1

    n_tables = len(table_counts)
    table_sizes = sorted(table_counts.values(), reverse=True)

    return {
        "assignments": assignments.tolist(),
        "n_tables": n_tables,
        "table_sizes": table_sizes,
        "alpha": alpha,
        "n": n,
    }


def crp_partition_prob(partition: list[list[int]], alpha: float = 1.0) -> float:
    r"""
    Compute log-probability of a partition under CRP.

    .. math::
        \log P(\text{partition} | \alpha) = \log \Gamma(\alpha) + K \log \alpha
        - \log \Gamma(\alpha + n) + \sum_k \log(n_k - 1)!
    """
    all_items = set()
    for cluster in partition:
        all_items.update(cluster)

    n = len(all_items)
    K = len(partition)

    log_prob = loggamma(alpha) + K * np.log(alpha + 1e-8) - loggamma(alpha + n)

    for cluster in partition:
        n_k = len(cluster)
        if n_k > 0:
            log_prob += loggamma(n_k)

    return float(log_prob)


crppr = crp_partition_prob


def cheatsheet() -> str:
    return "crppr(partition, alpha=1.0) -> log CRP partition probability"
