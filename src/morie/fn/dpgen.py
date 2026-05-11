# morie.fn — function file (hadesllm/morie)
"""Dirichlet process stick-breaking construction. 'It's dangerous to go alone.' -- Old Man"""

from __future__ import annotations

import numpy as np


def dirichlet_process_gen(
    alpha: float = 1.0,
    n_clusters: int | None = None,
    cutoff: float = 1e-6,
    rng: np.random.Generator | None = None,
) -> dict:
    """
    Generate a Dirichlet process via stick-breaking (GEM distribution).

    The stick-breaking construction generates weights :math:`w_k` as:

    .. math::

        V_k \\sim \\text{Beta}(1, \\alpha), \\quad
        w_k = V_k \\prod_{j < k} (1 - V_j)

    where the process is parameterized by concentration :math:`\\alpha`.

    :param alpha: Concentration parameter (alpha > 0). Default 1.0.
    :type alpha: float
    :param n_clusters: Maximum number of clusters before stopping (default: stop at cutoff).
    :type n_clusters: int | None
    :param cutoff: Weight threshold below which to stop (default 1e-6).
    :type cutoff: float
    :param rng: Random number generator. If None, creates new generator.
    :type rng: np.random.Generator | None
    :return: Dictionary with keys: 'weights', 'log_weights', 'n_clusters', 'alpha'.
    :rtype: dict

    References
    ----------
    Sethuraman J. (1994). A constructive definition of Dirichlet priors.
    *Statistica Sinica*, 4(2), 639-650.
    """
    if rng is None:
        rng = np.random.default_rng()

    if alpha <= 0:
        raise ValueError(f"alpha must be > 0, got {alpha}")
    if cutoff <= 0 or cutoff >= 1:
        raise ValueError(f"cutoff must be in (0, 1), got {cutoff}")

    weights = []
    log_weights = []
    prod_one_minus_v = 1.0
    k = 0

    while True:
        if n_clusters is not None and k >= n_clusters:
            break

        v_k = rng.beta(1.0, alpha)
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
    }


dpgen = dirichlet_process_gen


def cheatsheet() -> str:
    return "dirichlet_process_gen(alpha=1.0, n_clusters=None, cutoff=1e-6) -> DP via stick-breaking"
