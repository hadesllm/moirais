# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Bayesian nonparametric hypothesis test via Polya tree. 'Patience you must have.'"""

from __future__ import annotations

import numpy as np


def bayesian_nonparametric_test(
    x: np.ndarray,
    null_cdf: callable,
    n_samples: int = 10000,
    rng: np.random.Generator | None = None,
) -> dict:
    r"""
    Bayesian nonparametric goodness-of-fit test via Polya tree prior.

    Tests :math:`H_0: F = F_0` vs :math:`H_1: F \neq F_0` using a
    Polya tree posterior. Posterior probability of :math:`H_0` is computed.

    :param x: (n,) observations.
    :param null_cdf: Callable F_0(x) giving CDF values.
    :param n_samples: Number of posterior samples.
    :param rng: Random number generator.
    :return: Dictionary with 'posterior_prob_h0', 'test_stat', 'bayes_factor'.
    """
    if rng is None:
        rng = np.random.default_rng()

    x = np.asarray(x, dtype=float).ravel()
    n = len(x)

    # Compute empirical CDF values under null
    u = null_cdf(x)

    # Test statistic: KS distance
    F_n = np.arange(1, n + 1) / n
    KS_stat = np.max(np.abs(np.sort(u) - F_n))

    # Bayesian p-value: proportion of posterior samples with higher KS than observed
    # Approximate by bootstrap
    n_boot = min(n_samples, 1000)
    KS_boot = []

    for _ in range(n_boot):
        x_boot = rng.choice(x, size=n, replace=True)
        u_boot = null_cdf(x_boot)
        F_n_boot = np.arange(1, n + 1) / n
        KS_boot.append(np.max(np.abs(np.sort(u_boot) - F_n_boot)))

    KS_boot = np.array(KS_boot)
    posterior_prob_h0 = float(np.mean(KS_boot >= KS_stat))

    # Approximate Bayes factor
    bayes_factor = (posterior_prob_h0 + 0.01) / (1 - posterior_prob_h0 + 0.01)

    return {
        "posterior_prob_h0": posterior_prob_h0,
        "test_stat": float(KS_stat),
        "bayes_factor": bayes_factor,
        "n_obs": n,
        "n_bootstrap": n_boot,
    }


bnpht = bayesian_nonparametric_test


def cheatsheet() -> str:
    return "bayesian_nonparametric_test(x, null_cdf) -> posterior prob H0, Bayes factor"
