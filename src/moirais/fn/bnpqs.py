# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""Bayesian nonparametric quantile estimation. 'Quantiles, understand them you must.'"""

from __future__ import annotations

import numpy as np


def bayesian_nonparametric_quantiles(x: np.ndarray, quantiles: np.ndarray | None = None, alpha: float = 1.0, n_boot: int = 1000, rng: np.random.Generator | None = None, cdf=None) -> dict:
    r"""
    Bayesian nonparametric quantile estimation via Bayesian bootstrap.

    Uses Dirichlet process prior on the CDF to estimate quantiles.

    .. math::

        F^* | x \sim \mathcal{DP}(n, F_n), \quad q_p = F^{*-1}(p)

    :param x: (n,) observations.
    :param quantiles: Quantile levels to estimate (default: [0.25, 0.5, 0.75]).
    :param alpha: DP concentration (unused in standard BB, kept for interface).
    :param n_boot: Number of bootstrap samples.
    :param rng: Random number generator.
    :return: Dictionary with quantile estimates and credible intervals.
    """
    if rng is None:
        rng = np.random.default_rng()

    x = np.asarray(x, dtype=float).ravel()
    n = len(x)

    if quantiles is None:
        quantiles = np.array([0.25, 0.5, 0.75])
    else:
        quantiles = np.asarray(quantiles, dtype=float).ravel()

    if np.any((quantiles < 0) | (quantiles > 1)):
        raise ValueError("quantiles must be in [0, 1]")

    # Bayesian bootstrap quantiles
    quantile_samples = {q: [] for q in quantiles}

    for _ in range(n_boot):
        # Draw Dirichlet weights
        weights = rng.exponential(1.0, size=n)
        weights /= np.sum(weights)

        # Weighted empirical quantiles
        x_sorted = np.sort(x)
        cdf = np.cumsum(weights[np.argsort(x)])

        for q in quantiles:
            idx = np.searchsorted(cdf, q, side='left')
            idx = min(idx, n - 1)
            quantile_samples[q].append(x_sorted[idx])

    # Summary statistics
    result = {
        "quantile_estimates": {},
        "credible_intervals": {},
        "samples": quantile_samples,
        "alpha": alpha,
        "n_obs": n,
        "n_bootstrap": n_boot,
    }

    for q in quantiles:
        samples = np.array(quantile_samples[q])
        result["quantile_estimates"][float(q)] = float(np.mean(samples))
        result["credible_intervals"][float(q)] = (
            float(np.percentile(samples, 2.5)),
            float(np.percentile(samples, 97.5))
        )

    return result


bnpqs = bayesian_nonparametric_quantiles


def cheatsheet() -> str:
    return "bayesian_nonparametric_quantiles(x, quantiles=None) -> BNP quantile estimates"
