# morie.fn -- function file (hadesllm/morie)
"""Simulate the Indian Buffet Process (IBP) for feature allocation."""

from __future__ import annotations

import numpy as np


def indian_buffet_process(
    n: int = 100,
    alpha: float = 1.0,
    rng: np.random.Generator | None = None,
) -> dict:
    r"""
    Simulate the Indian Buffet Process (IBP) for feature allocation.

    The IBP generates sparse binary feature matrices where rows are customers
    and columns are dishes (features). Customer i samples dishes:
    - Each occupied dish k with probability :math:`m_k / i`
    - A Poisson(:math:`\alpha / i`) number of new dishes

    .. math::

        m_k^{(i)} \sim \text{Bernoulli}(m_k / i), \quad
        K_i^+ \sim \text{Poisson}(\alpha / i)

    :param n: Number of customers (rows). Default 100.
    :type n: int
    :param alpha: Concentration parameter (alpha > 0). Default 1.0.
    :type alpha: float
    :param rng: Random number generator. If None, creates new generator.
    :type rng: np.random.Generator | None
    :return: Dictionary with keys: 'features' (binary matrix, n x K), 'dish_counts' (K,).
    :rtype: dict

    References
    ----------
    Griffiths T.L., Ghahramani Z. (2011). The Indian buffet process:
    An introduction and review. *J. Mach. Learn. Res.*, 12, 1185-1224.
    """
    if rng is None:
        rng = np.random.default_rng()

    if n <= 0:
        raise ValueError(f"n must be > 0, got {n}")
    if alpha <= 0:
        raise ValueError(f"alpha must be > 0, got {alpha}")

    features = []
    dish_counts = []

    for i in range(n):
        # Sample existing dishes
        z_i = []
        for k, m_k in enumerate(dish_counts):
            if rng.uniform() < m_k / (i + 1.0):
                z_i.append(1)
            else:
                z_i.append(0)

        # Sample new dishes
        k_new = rng.poisson(alpha / (i + 1.0))
        z_i.extend([1] * k_new)

        # Update dish counts
        for k in range(len(z_i)):
            if k >= len(dish_counts):
                dish_counts.append(0)
            dish_counts[k] += z_i[k]

        features.append(z_i)

    # Pad to rectangular array
    max_k = max(len(row) for row in features) if features else 0
    features_array = np.zeros((n, max_k), dtype=int)
    for i, z_i in enumerate(features):
        features_array[i, :len(z_i)] = z_i

    dish_counts = np.array(dish_counts, dtype=int)

    return {
        "features": features_array,
        "dish_counts": dish_counts,
        "n_customers": n,
        "n_dishes": len(dish_counts),
        "alpha": alpha,
    }


ibprc = indian_buffet_process


def cheatsheet() -> str:
    return "indian_buffet_process(n=100, alpha=1.0) -> IBP feature allocation matrix"
