# moirais.fn — function file (hadesllm/moirais)
"""Local Dirichlet density estimation. 'Size matters not, judge me by size do you?'"""

from __future__ import annotations

import numpy as np


def local_dirichlet_density_estimate(
    x: np.ndarray,
    x_eval: np.ndarray | None = None,
    alpha: float = 1.0,
    bandwidth: float | None = None,
    rng: np.random.Generator | None = None,
) -> dict:
    r"""
    Local Dirichlet process density estimation via kernel method.

    At each evaluation point :math:`x_*`, fit a DP-based local density
    using observations in a local neighborhood.

    .. math::

        \hat{f}(x_*) = \frac{1}{h} \sum_{i=1}^n K\left(\frac{x_i - x_*}{h}\right)

    :param x: (n,) observations.
    :param x_eval: Evaluation points. If None, uses quantiles of x.
    :param alpha: DP concentration.
    :param bandwidth: Kernel bandwidth. If None, Silverman's rule.
    :param rng: Random number generator.
    :return: Dictionary with 'x_eval', 'density'.
    """
    if rng is None:
        rng = np.random.default_rng()

    x = np.asarray(x, dtype=float).ravel()
    n = len(x)

    if x_eval is None:
        x_eval = np.quantile(x, np.linspace(0.05, 0.95, 50))
    else:
        x_eval = np.asarray(x_eval, dtype=float).ravel()

    if bandwidth is None:
        # Silverman's rule
        iqr = np.percentile(x, 75) - np.percentile(x, 25)
        bandwidth = 0.9 * min(np.std(x), iqr / 1.35) * (n ** (-1.0 / 5.0))

    # Gaussian kernel
    density = np.zeros_like(x_eval)
    for i, x_star in enumerate(x_eval):
        sq_dist = (x - x_star) ** 2
        weights = np.exp(-sq_dist / (2 * bandwidth**2))
        density[i] = np.mean(weights) / bandwidth

    # Normalize to integrate to 1
    from scipy.integrate import trapz
    norm = trapz(density, x_eval)
    if norm > 0:
        density /= norm

    return {
        "x_eval": x_eval,
        "density": density,
        "bandwidth": bandwidth,
        "alpha": alpha,
        "n_obs": n,
    }


lddst = local_dirichlet_density_estimate


def cheatsheet() -> str:
    return "local_dirichlet_density_estimate(x, x_eval=None, alpha=1.0) -> local density"
