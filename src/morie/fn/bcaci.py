# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""BCa bootstrap confidence interval. 'The art of doing mathematics consists in finding that special case which contains all the germs of generality. — David Hilbert'"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy import stats as _st

from ._containers import DescriptiveResult


def bca_ci(data: np.ndarray, stat_fn: Callable, n_boot: int = 2000, alpha: float = 0.05, seed: int = 42, cdf=None) -> DescriptiveResult:
    """
    Bias-corrected and accelerated (BCa) bootstrap confidence interval.

    Adjusts percentile endpoints for bias and skewness of the bootstrap
    distribution.

    :param data: 1-D data array.
    :type data: numpy.ndarray
    :param stat_fn: Function that takes a 1-D array and returns a scalar.
    :type stat_fn: Callable
    :param n_boot: Number of bootstrap replicates. Default 2000.
    :type n_boot: int
    :param alpha: Significance level (two-sided). Default 0.05.
    :type alpha: float
    :param seed: Random seed. Default 42.
    :type seed: int
    :return: DescriptiveResult with CI bounds.
    :rtype: DescriptiveResult

    References
    ----------
    Efron B. (1987). Better bootstrap confidence intervals. *Journal of
    the American Statistical Association*, 82(397), 171-185.
    """
    data = np.asarray(data, dtype=float).ravel()
    n = len(data)
    rng = np.random.default_rng(seed)
    theta_hat = float(stat_fn(data))
    boot_stats = np.empty(n_boot)
    for i in range(n_boot):
        sample = data[rng.integers(0, n, size=n)]
        boot_stats[i] = stat_fn(sample)
    z0 = _st.norm.ppf(np.mean(boot_stats < theta_hat))
    jackknife = np.empty(n)
    for i in range(n):
        jack_sample = np.concatenate([data[:i], data[i + 1 :]])
        jackknife[i] = stat_fn(jack_sample)
    jack_mean = jackknife.mean()
    diff = jack_mean - jackknife
    a = float(np.sum(diff**3) / (6.0 * np.sum(diff**2) ** 1.5)) if np.sum(diff**2) > 0 else 0.0
    z_lo = _st.norm.ppf(alpha / 2)
    z_hi = _st.norm.ppf(1 - alpha / 2)
    a1 = _st.norm.cdf(z0 + (z0 + z_lo) / (1 - a * (z0 + z_lo)))
    a2 = _st.norm.cdf(z0 + (z0 + z_hi) / (1 - a * (z0 + z_hi)))
    sorted_boot = np.sort(boot_stats)
    lo = float(sorted_boot[max(0, int(np.floor(a1 * n_boot)))])
    hi = float(sorted_boot[min(n_boot - 1, int(np.ceil(a2 * n_boot)) - 1)])
    return DescriptiveResult(
        name="bca_ci",
        value=theta_hat,
        extra={
            "lower": lo,
            "upper": hi,
            "estimate": theta_hat,
            "bias_correction": float(z0),
            "acceleration": a,
            "alpha": alpha,
        },
    )


bcaci = bca_ci


def cheatsheet() -> str:
    return "bca_ci({}) -> BCa bootstrap confidence interval. 'Do. Or do not. There is "
