# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Non-parametric percentile bootstrap confidence intervals."""

from collections.abc import Callable

import numpy as np
import pandas as pd


def bootstrap_ci(
    estimation_func: Callable,
    data: pd.DataFrame,
    n_iterations: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """
    Compute non-parametric percentile bootstrap confidence intervals.

    A fixed random seed is used so that results are **reproducible** across
    calls with the same inputs.  The default seed (42) is arbitrary; pass a
    different ``seed`` value if you need independent bootstrap draws in a
    simulation study.

    The percentile bootstrap is used rather than the BCa or studentised
    bootstrap because it requires only a point-estimate function and makes
    no symmetry assumptions.  For highly skewed sampling distributions,
    consider the BCa bootstrap (Efron & Tibshirani, 1993, Chapter 14).

    :param estimation_func: A callable that takes a DataFrame sample and
        returns a numeric point estimate.
    :type estimation_func: Callable
    :param data: The pandas DataFrame representing the sample population.
    :type data: pandas.DataFrame
    :param n_iterations: Number of bootstrap iterations to perform,
        defaults to 1000.
    :type n_iterations: int, optional
    :param alpha: The significance level for the interval (e.g. 0.05 for a
        95% CI), defaults to 0.05.
    :type alpha: float, optional
    :param seed: Random seed for reproducibility.  Default 42.
    :type seed: int, optional
    :return: A tuple ``(lower, upper)`` containing the percentile bootstrap
        confidence interval bounds.
    :rtype: tuple[float, float]

    References
    ----------
    Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*.
    Chapman & Hall/CRC. https://doi.org/10.1201/9780429246593
    """
    # Set the random seed BEFORE any stochastic operation so that all
    # bootstrap draws are deterministic given the same (data, seed) pair.
    np.random.seed(seed)

    estimates = []
    n = len(data)
    for _ in range(n_iterations):
        sample = data.sample(n=n, replace=True)
        estimates.append(estimation_func(sample))

    lower = float(np.percentile(estimates, (alpha / 2) * 100))
    upper = float(np.percentile(estimates, (1 - alpha / 2) * 100))
    return lower, upper


boot = bootstrap_ci


def cheatsheet() -> str:
    return "bootstrap_ci({}) -> Non-parametric percentile bootstrap confidence intervals."
