# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""Bayesian bootstrap (Rubin). 'That is why you fail.'"""

from __future__ import annotations

import numpy as np


def bayesian_bootstrap(
    x: np.ndarray,
    n_boot: int = 1000,
    statistic: callable | None = None,
    rng: np.random.Generator | None = None,
) -> dict:
    r"""
    Rubin's Bayesian bootstrap: Bayesian inference without parametric models.

    The Bayesian bootstrap applies random Dirichlet(:math:`1, \ldots, 1`) weights
    to the empirical distribution. For a statistic :math:`T(\cdot)`:

    .. math::

        T(\widehat{F}^b) \quad \text{where} \quad
        \widehat{F}^b \text{ weights } (x_1, \ldots, x_n) \text{ by } (w_1^b, \ldots, w_n^b)

    This provides a Bayesian posterior without assuming a parametric model.

    :param x: (n,) observations.
    :type x: np.ndarray
    :param n_boot: Number of bootstrap resamples. Default 1000.
    :type n_boot: int
    :param statistic: Callable that computes a statistic from data.
                      If None, defaults to mean. Default None.
    :type statistic: callable | None
    :param rng: Random number generator. If None, creates new generator.
    :type rng: np.random.Generator | None
    :return: Dictionary with keys: 'statistic_samples', 'posterior_mean',
             'posterior_std', 'ci_lower', 'ci_upper' (95% credible interval).
    :rtype: dict

    References
    ----------
    Rubin D.B. (1981). The Bayesian bootstrap. *Ann. Statist.*, 9(1), 130-134.
    """
    if rng is None:
        rng = np.random.default_rng()

    x = np.asarray(x, dtype=float).ravel()
    n = len(x)

    if n <= 0:
        raise ValueError(f"x must have at least 1 element, got {len(x)}")

    if statistic is None:
        statistic = lambda data: float(np.mean(data))

    statistic_samples = []

    for _ in range(n_boot):
        # Draw weights from Dirichlet(1, ..., 1)
        weights = rng.exponential(1.0, size=n)
        weights /= np.sum(weights)

        # Compute statistic with weighted data
        weighted_sample = rng.choice(x, size=n, p=weights, replace=True)
        stat_value = statistic(weighted_sample)
        statistic_samples.append(stat_value)

    statistic_samples = np.array(statistic_samples, dtype=float)
    posterior_mean = float(np.mean(statistic_samples))
    posterior_std = float(np.std(statistic_samples, ddof=1))
    ci_lower = float(np.percentile(statistic_samples, 2.5))
    ci_upper = float(np.percentile(statistic_samples, 97.5))

    return {
        "statistic_samples": statistic_samples,
        "posterior_mean": posterior_mean,
        "posterior_std": posterior_std,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "n_bootstrap": n_boot,
        "n_obs": n,
    }


bbstr = bayesian_bootstrap


def cheatsheet() -> str:
    return "bayesian_bootstrap(x, n_boot=1000) -> Rubin BB posterior distribution"
