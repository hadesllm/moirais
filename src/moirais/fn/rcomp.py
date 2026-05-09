# moirais.fn — function file (hadesllm/moirais)
"""Rademacher complexity computation."""

from __future__ import annotations

import numpy as np


def rademacher_complexity(
    X: np.ndarray,
    *,
    n_draws: int = 1000,
    seed: int = 42,
) -> dict:
    r"""
    Empirical Rademacher complexity of a dataset (linear function class).

    For a sample :math:`S = \{x_1, \ldots, x_n\}` and a function class
    :math:`\mathcal{F}`, the empirical Rademacher complexity is:

    .. math::

        \hat{\mathcal{R}}_n(\mathcal{F})
        = E_\sigma\!\left[\sup_{f \in \mathcal{F}}
        \frac{1}{n}\sum_{i=1}^n \sigma_i f(x_i)\right]

    For the linear class :math:`\{x \mapsto w^\top x : \|w\| \le 1\}`,
    this simplifies to:

    .. math::

        \hat{\mathcal{R}}_n
        = \frac{1}{n} E_\sigma\!\left[\left\|\sum_i \sigma_i x_i\right\|\right]

    :param X: 2-D array of shape (n, d) or 1-D array of shape (n,).
    :param n_draws: Number of Rademacher draws for Monte Carlo estimate.
        Default 1000.
    :param seed: Random seed. Default 42.
    :return: dict with ``rademacher_complexity``, ``std``, ``n``, ``d``.
    :raises ValueError: If X is empty.

    References
    ----------
    Bartlett, P.L. & Mendelson, S. (2002). Rademacher and Gaussian
        complexities: Risk bounds and structural results. *JMLR*, 3,
        463--482.
    Kosorok, M.R. (2008). *Introduction to Empirical Processes and
        Semiparametric Inference*, Sec. 8.1. Springer.
    """
    X = np.asarray(X, dtype=float)
    if X.size == 0:
        raise ValueError("X must be non-empty.")
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    n, d = X.shape
    rng = np.random.default_rng(seed)

    vals = np.empty(n_draws)
    for k in range(n_draws):
        sigma = rng.choice([-1.0, 1.0], size=n)
        weighted = sigma[:, None] * X
        vals[k] = np.linalg.norm(weighted.sum(axis=0)) / n

    return {
        "rademacher_complexity": float(np.mean(vals)),
        "std": float(np.std(vals, ddof=1)),
        "n": n,
        "d": d,
        "n_draws": n_draws,
        "method": "Empirical Rademacher complexity (linear class)",
    }


rcomp = rademacher_complexity


def cheatsheet() -> str:
    return "rademacher_complexity({X}) -> Empirical Rademacher complexity."
