# moirais.fn — function file (hadesllm/moirais)
"""Ewens sampling formula for random partitions. 'Mutate, alleles do.'"""

from __future__ import annotations

import numpy as np


def ewens_partition(
    n: int,
    theta: float = 1.0,
    rng: np.random.Generator | None = None,
) -> dict:
    r"""
    Sample a random partition under the Ewens sampling formula.

    The Ewens formula gives the probability of a partition with
    multiplicities :math:`(m_1, m_2, \ldots)` under a neutral model:

    .. math::

        P(m_1, \ldots, m_n | \theta) \propto \theta^K \prod_{j=1}^{n} j^{m_j} m_j!

    :param n: Number of items.
    :param theta: Ewens parameter (theta > 0).
    :param rng: Random number generator.
    :return: Dictionary with 'partition', 'n_classes', 'class_sizes', 'theta'.
    """
    if rng is None:
        rng = np.random.default_rng()

    if n <= 0 or theta <= 0:
        raise ValueError("n and theta must be > 0")

    # Chinese Restaurant Process (equivalent to Ewens for n items)
    # Generate using CRP stick-breaking
    partition = {}
    assignments = np.zeros(n, dtype=int)
    table_idx = 0

    for i in range(n):
        # Probability of joining existing table k proportional to n_k
        # Probability of new table proportional to theta
        existing_sizes = np.array([partition.get(t, 0) for t in range(table_idx)])
        probs = np.append(existing_sizes, theta)
        probs = probs / np.sum(probs)

        chosen = rng.choice(len(probs), p=probs)

        if chosen < table_idx:
            # Join existing table
            assignments[i] = chosen
            partition[chosen] = partition.get(chosen, 0) + 1
        else:
            # New table
            assignments[i] = table_idx
            partition[table_idx] = 1
            table_idx += 1

    # Convert to partition representation
    partition_list = [[] for _ in range(table_idx)]
    for i, t in enumerate(assignments):
        partition_list[t].append(i)

    class_sizes = np.array([len(c) for c in partition_list if len(c) > 0], dtype=int)

    return {
        "partition": partition_list,
        "n_classes": len(partition_list),
        "class_sizes": class_sizes,
        "theta": theta,
        "n": n,
    }


ewens = ewens_partition


def cheatsheet() -> str:
    return "ewens(n, theta=1.0) -> Ewens sampling formula random partition"
