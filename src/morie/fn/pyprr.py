# morie.fn -- function file (hadesllm/morie)
"""Pitman-Yor process: two-parameter generalization of DP. 'You have power over your mind -- not outside events. Realize this, and you will find strength. -- Marcus Aurelius'"""

from __future__ import annotations

import numpy as np


def pitman_yor_process(
    alpha: float = 1.0,
    discount: float = 0.5,
    n_clusters: int | None = None,
    cutoff: float = 1e-6,
    rng: np.random.Generator | None = None,
) -> dict:
    r"""
    Generate a Pitman-Yor process, a two-parameter generalization of DP.

    The Pitman-Yor (or Poisson-Dirichlet) process has stick-breaking weights:

    .. math::

        V_k \sim \text{Beta}(1 - d, \alpha + kd), \quad
        w_k = V_k \prod_{j < k} (1 - V_j)

    where :math:`d \in [0, 1)` is the discount and :math:`\alpha > -d` is the strength.

    When :math:`d = 0`, recovers the Dirichlet process. When :math:`d > 0`,
    produces a power-law tail in cluster sizes (rich-get-richer).

    :param alpha: Strength parameter (alpha > -discount). Default 1.0.
    :type alpha: float
    :param discount: Discount parameter in [0, 1). Default 0.5.
    :type discount: float
    :param n_clusters: Maximum number of clusters (default: stop at cutoff).
    :type n_clusters: int | None
    :param cutoff: Weight threshold below which to stop. Default 1e-6.
    :type cutoff: float
    :param rng: Random number generator. If None, creates new generator.
    :type rng: np.random.Generator | None
    :return: Dictionary with keys: 'weights', 'log_weights', 'n_clusters', 'alpha', 'discount'.
    :rtype: dict

    References
    ----------
    Pitman J., Yor M. (1997). The two-parameter Poisson-Dirichlet distribution
    derived from a stable subordinator. *Ann. Probab.*, 25(2), 855-900.
    """
    if rng is None:
        rng = np.random.default_rng()

    if discount < 0 or discount >= 1:
        raise ValueError(f"discount must be in [0, 1), got {discount}")
    if alpha <= -discount:
        raise ValueError(f"alpha must be > -discount, got alpha={alpha}, discount={discount}")

    weights = []
    log_weights = []
    prod_one_minus_v = 1.0
    k = 0

    while True:
        if n_clusters is not None and k >= n_clusters:
            break

        # Beta(1 - discount, alpha + k * discount)
        a = 1.0 - discount
        b = alpha + k * discount
        v_k = rng.beta(a, b)
        w_k = v_k * prod_one_minus_v
        weights.append(w_k)
        log_weights.append(np.log(w_k) if w_k > 0 else -np.inf)

        prod_one_minus_v *= (1.0 - v_k)
        k += 1

        if w_k < cutoff or prod_one_minus_v < cutoff:
            break

    weights = np.asarray(weights, dtype=float)
    log_weights = np.asarray(log_weights, dtype=float)

    return {
        "weights": weights,
        "log_weights": log_weights,
        "n_clusters": len(weights),
        "alpha": alpha,
        "discount": discount,
    }


pyprr = pitman_yor_process


def cheatsheet() -> str:
    return "pitman_yor_process(alpha=1.0, discount=0.5) -> Pitman-Yor stick-breaking"
