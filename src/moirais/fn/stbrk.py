"""It is the mark of an educated mind to entertain a thought without accepting it. — Aristotle"""

from __future__ import annotations

import numpy as np


def stick_breaking_weights(
    alpha: float = 1.0,
    n_components: int | None = None,
    cutoff: float = 1e-6,
    rng: np.random.Generator | None = None,
) -> dict:
    r"""
    Generate stick-breaking weights (Griffiths-Engen-McCloskey distribution).

    The GEM(:math:`\alpha`) distribution generates weights via:

    .. math::

        V_k \sim \text{Beta}(1, \alpha), \quad
        w_k = V_k \prod_{j < k} (1 - V_j)

    These weights sum to 1 and define a discrete probability distribution
    commonly used as the base for nonparametric priors.

    :param alpha: Concentration parameter (alpha > 0). Default 1.0.
    :type alpha: float
    :param n_components: Maximum number of components (default: stop at cutoff).
    :type n_components: int | None
    :param cutoff: Weight threshold below which to stop. Default 1e-6.
    :type cutoff: float
    :param rng: Random number generator. If None, creates new generator.
    :type rng: np.random.Generator | None
    :return: Dictionary with keys: 'weights', 'log_weights', 'n_components'.
    :rtype: dict

    References
    ----------
    Griffiths T.L., Engen S., McCloskey J.W. (1969). On the distribution of
    the first eigenvalue spacing. *Ann. Math. Statist.*, 40(1), 262-266.
    """
    if rng is None:
        rng = np.random.default_rng()

    if alpha <= 0:
        raise ValueError(f"alpha must be > 0, got {alpha}")

    weights = []
    prod_one_minus_v = 1.0
    k = 0

    while True:
        if n_components is not None and k >= n_components:
            break

        v_k = rng.beta(1.0, alpha)
        w_k = v_k * prod_one_minus_v
        weights.append(w_k)

        prod_one_minus_v *= (1.0 - v_k)
        k += 1

        if w_k < cutoff or prod_one_minus_v < cutoff:
            break

    weights = np.asarray(weights, dtype=float)
    log_weights = np.log(weights)

    return {
        "weights": weights,
        "log_weights": log_weights,
        "n_components": len(weights),
        "alpha": alpha,
        "sum_weights": float(np.sum(weights)),
    }


stbrk = stick_breaking_weights


def cheatsheet() -> str:
    return "stick_breaking_weights(alpha=1.0) -> GEM weights for discrete prior"
