# morie.fn -- function file (hadesllm/morie)
"""Gaussian complexity computation."""

from __future__ import annotations

import numpy as np


def gaussian_complexity(
    X: np.ndarray,
    *,
    n_draws: int = 1000,
    seed: int = 42,
) -> dict:
    r"""
    Empirical Gaussian complexity of a dataset (linear function class).

    The Gaussian complexity replaces Rademacher variables with standard
    Gaussian variables :math:`g_i \sim N(0,1)`:

    .. math::

        \hat{\mathcal{G}}_n(\mathcal{F})
        = E_g\!\left[\sup_{f \in \mathcal{F}}
        \frac{1}{n}\sum_{i=1}^n g_i f(x_i)\right]

    For the linear class :math:`\{x \mapsto w^\top x : \|w\| \le 1\}`:

    .. math::

        \hat{\mathcal{G}}_n
        = \frac{1}{n} E_g\!\left[\left\|\sum_i g_i x_i\right\|\right]

    Gaussian and Rademacher complexities are related by:

    .. math::

        \mathcal{R}_n \le \sqrt{\frac{\pi}{2}}\,\mathcal{G}_n

    :param X: 2-D array (n, d) or 1-D array (n,).
    :param n_draws: Number of Gaussian draws. Default 1000.
    :param seed: Random seed. Default 42.
    :return: dict with ``gaussian_complexity``, ``std``,
        ``rademacher_upper`` (via sqrt(pi/2) relation), ``n``, ``d``.
    :raises ValueError: If X is empty.

    References
    ----------
    Bartlett, P.L. & Mendelson, S. (2002). Rademacher and Gaussian
        complexities. *JMLR*, 3, 463--482.
    Kosorok, M.R. (2008). *Introduction to Empirical Processes and
        Semiparametric Inference*, Sec. 8.2. Springer.
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
        g = rng.standard_normal(n)
        weighted = g[:, None] * X
        vals[k] = np.linalg.norm(weighted.sum(axis=0)) / n

    gc = float(np.mean(vals))

    return {
        "gaussian_complexity": gc,
        "std": float(np.std(vals, ddof=1)),
        "rademacher_upper": gc * np.sqrt(np.pi / 2.0),
        "n": n,
        "d": d,
        "n_draws": n_draws,
        "method": "Empirical Gaussian complexity (linear class)",
    }


gcplx = gaussian_complexity


def cheatsheet() -> str:
    return "gaussian_complexity({X}) -> Gaussian complexity computation."
