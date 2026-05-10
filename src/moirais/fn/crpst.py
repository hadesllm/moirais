# moirais.fn — function file (hadesllm/moirais)
"""Chinese restaurant process simulation. 'What we cannot speak about we must pass over in silence. -- Wittgenstein'"""
from __future__ import annotations

import numpy as np


def chinese_restaurant_process(
    n: int = 100,
    alpha: float = 1.0,
    rng: np.random.Generator | None = None,
) -> dict:
    r"""
    Simulate the Chinese restaurant process (CRP).

    The CRP is a stochastic process generating random partitions. Customers
    arrive sequentially; customer i sits at table k with probability
    proportional to the number of occupied tables, or at a new table with
    probability proportional to :math:`\alpha`.

    .. math::

        P(\text{customer } i \text{ sits at table } k) \propto
        \begin{cases}
        n_k & \text{if table } k \text{ occupied} \\
        \alpha & \text{if new table}
        \end{cases}

    :param n: Number of customers. Default 100.
    :type n: int
    :param alpha: Concentration parameter (alpha > 0). Default 1.0.
    :type alpha: float
    :param rng: Random number generator. If None, creates new generator.
    :type rng: np.random.Generator | None
    :return: Dictionary with keys: 'table_assignments', 'table_counts', 'n_tables'.
    :rtype: dict

    References
    ----------
    Aldous D.J. (1985). Exchangeability and related topics. In *École d'Été de
    Probabilités de Saint-Flour*, Springer, Berlin.
    """
    if rng is None:
        rng = np.random.default_rng()

    if n <= 0:
        raise ValueError(f"n must be > 0, got {n}")
    if alpha <= 0:
        raise ValueError(f"alpha must be > 0, got {alpha}")

    table_assignments = np.zeros(n, dtype=int)
    table_counts = {}

    # First customer sits at table 0
    table_assignments[0] = 0
    table_counts[0] = 1

    for i in range(1, n):
        # Compute probabilities for existing tables
        probs = np.array([table_counts[k] for k in range(len(table_counts))], dtype=float)
        # Add probability of new table
        probs = np.append(probs, alpha)
        # Normalize
        probs /= np.sum(probs)

        # Sample table assignment
        table_idx = rng.choice(len(probs), p=probs)

        if table_idx < len(table_counts):
            # Existing table
            table_assignments[i] = table_idx
            table_counts[table_idx] += 1
        else:
            # New table
            new_table = max(table_counts.keys()) + 1
            table_assignments[i] = new_table
            table_counts[new_table] = 1

    n_tables = len(table_counts)
    table_counts_array = np.array([table_counts[k] for k in range(n_tables)], dtype=int)

    return {
        "table_assignments": table_assignments,
        "table_counts": table_counts_array,
        "n_tables": n_tables,
        "alpha": alpha,
    }


crpst = chinese_restaurant_process


def cheatsheet() -> str:
    return "chinese_restaurant_process(n=100, alpha=1.0) -> CRP table assignments"
