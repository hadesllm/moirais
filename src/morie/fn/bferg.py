# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Ferguson's Bayesian bootstrap. 'The greatest teacher, failure is.'"""

from __future__ import annotations

import numpy as np


def ferguson_bayesian_bootstrap(
    x: np.ndarray,
    n_boot: int = 1000,
    rng: np.random.Generator | None = None,
) -> dict:
    r"""
    Perform Ferguson's Bayesian bootstrap.

    Given observations :math:`x_1, \ldots, x_n`, the Bayesian bootstrap
    draws samples from the posterior of the empirical CDF under a DP prior:

    .. math::

        F^* | x \sim \mathcal{DP}(n, F_n)

    where :math:`F_n` is the empirical CDF. Equivalent to resampling via
    random weights that follow Dirichlet(:math:`1, \ldots, 1`).

    :param x: (n,) observations.
    :type x: np.ndarray
    :param n_boot: Number of bootstrap samples. Default 1000.
    :type n_boot: int
    :param rng: Random number generator. If None, creates new generator.
    :type rng: np.random.Generator | None
    :return: Dictionary with keys: 'bootstrap_samples', 'posterior_mean', 'posterior_std'.
    :rtype: dict

    References
    ----------
    Ferguson T.S. (1973). A Bayesian analysis of some nonparametric problems.
    *Ann. Statist.*, 1(2), 209-230.
    """
    if rng is None:
        rng = np.random.default_rng()

    x = np.asarray(x, dtype=float).ravel()
    n = len(x)

    if n <= 0:
        raise ValueError(f"x must have at least 1 element, got {len(x)}")

    bootstrap_samples = []
    posterior_samples = []

    for _ in range(n_boot):
        # Draw random weights from Dirichlet(1, 1, ..., 1)
        # This is equivalent to sorting exponential(1) variates
        weights = rng.exponential(1.0, size=n)
        weights /= np.sum(weights)

        # Resample from x with these weights
        sample = rng.choice(x, size=n, p=weights, replace=True)
        bootstrap_samples.append(sample)

        # Estimate posterior mean from this sample
        posterior_samples.append(float(np.mean(sample)))

    bootstrap_samples = np.array(bootstrap_samples, dtype=float)
    posterior_samples = np.array(posterior_samples, dtype=float)

    posterior_mean = float(np.mean(posterior_samples))
    posterior_std = float(np.std(posterior_samples, ddof=1))

    return {
        "bootstrap_samples": bootstrap_samples,
        "posterior_samples": posterior_samples,
        "posterior_mean": posterior_mean,
        "posterior_std": posterior_std,
        "n_bootstrap": n_boot,
        "n_obs": n,
    }


bferg = ferguson_bayesian_bootstrap


def cheatsheet() -> str:
    return "ferguson_bayesian_bootstrap(x, n_boot=1000) -> Ferguson BB posterior"
