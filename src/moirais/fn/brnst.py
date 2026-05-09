# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""Bernstein polynomial density estimation. 'Powerful, Bernstein is.'"""

from __future__ import annotations

import numpy as np
from scipy.special import comb


def bernstein_density_estimate(
    x: np.ndarray,
    n_basis: int = 10,
    x_eval: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> dict:
    r"""
    Nonparametric density estimation via Bernstein polynomial basis.

    Approximates density as:

    .. math::

        \hat{f}(x) = \sum_{k=0}^{n} \hat{\theta}_k B_{k,n}(x)

    where :math:`B_{k,n}(x) = \binom{n}{k} x^k (1-x)^{n-k}` are Bernstein basis polynomials.

    :param x: (n,) observations in [0, 1].
    :param n_basis: Number of basis functions.
    :param x_eval: Evaluation points. If None, uses uniform grid.
    :param rng: Random number generator.
    :return: Dictionary with 'x_eval', 'density', 'coefficients'.
    """
    if rng is None:
        rng = np.random.default_rng()

    x = np.asarray(x, dtype=float).ravel()
    n = len(x)

    # Rescale to [0, 1] if necessary
    x_min, x_max = np.min(x), np.max(x)
    x_scaled = (x - x_min) / (x_max - x_min + 1e-8)

    if x_eval is None:
        x_eval = np.linspace(0, 1, 100)
    else:
        x_eval = np.asarray(x_eval, dtype=float).ravel()
        x_eval_scaled = (x_eval - x_min) / (x_max - x_min + 1e-8)
        x_eval_scaled = np.clip(x_eval_scaled, 0, 1)
        x_eval = x_eval_scaled

    # Build Bernstein basis matrix
    B = np.zeros((len(x_eval), n_basis + 1))
    for k in range(n_basis + 1):
        B[:, k] = comb(n_basis, k) * (x_eval ** k) * ((1 - x_eval) ** (n_basis - k))

    # Fit coefficients by LS (non-negative)
    B_train = np.zeros((n, n_basis + 1))
    for k in range(n_basis + 1):
        B_train[:, k] = comb(n_basis, k) * (x_scaled ** k) * ((1 - x_scaled) ** (n_basis - k))

    # Non-negative LS: minimize ||By - theta||^2 subject to theta >= 0
    from scipy.optimize import nnls
    theta, _ = nnls(B_train, np.ones(n))
    theta /= np.sum(theta)  # Normalize

    density = B @ theta

    return {
        "x_eval": x_eval,
        "density": density,
        "coefficients": theta,
        "n_basis": n_basis,
        "n_obs": n,
    }


brnst = bernstein_density_estimate


def cheatsheet() -> str:
    return "bernstein_density_estimate(x, n_basis=10) -> Bernstein polynomial density"
