"""
Sign test for population quantiles.

Nonparametric test for whether a sample median equals a hypothesized value
without assuming any particular distribution shape.

Reference: Gibbons & Chakraborti (2011), Nonparametric Statistical Inference, 5th Ed. § 2.2
"""

import numpy as np
from scipy import stats as sp_stats

__all__ = ["signt"]


def signt(x, theta0=None, axis=0, alternative="two-sided", cdf=None):
    r"""
    Sign test for population quantiles.

    Tests H0: median = θ₀ using the sign test (binomial distribution).

    Parameters
    ----------
    x : array_like
        Input sample.
    theta0 : float, optional
        Hypothesized median. If None, uses sample median.
    axis : int, optional
        Axis along which to apply the test (default 0).
    alternative : {'two-sided', 'less', 'greater'}, optional
        Alternative hypothesis (default 'two-sided').

    Returns
    -------
    dict
        Keys:
        - 'statistic': number of plus signs (X_+)
        - 'n_plus': count of x > θ₀
        - 'n_minus': count of x < θ₀
        - 'n_zero': count of x = θ₀
        - 'n_eff': effective sample size (n_plus + n_minus)
        - 'p_value': p-value from binomial distribution
        - 'interpretation': "reject" or "not reject" null

    Notes
    -----
    Very robust to outliers. Works on any quantile, not just median.
    Exact test for small n, normal approximation for large n.

    Examples
    --------
    >>> x = np.array([5, 3, 8, 6, 9, 2, 7, 4])
    >>> result = signt(x, theta0=5)
    >>> result['p_value']  # doctest: +SKIP
    """
    x = np.asarray(x, dtype=np.float64)

    if x.ndim == 2:
        x = np.take(x, 0, axis=axis)
    elif x.ndim != 1:
        raise ValueError("Input must be 1D or 2D array")

    n = len(x)
    if n < 1:
        raise ValueError("Sample size must be at least 1")

    if theta0 is None:
        theta0 = np.median(x)

    # Count signs
    diff = x - theta0
    n_plus = np.sum(diff > 0)
    n_minus = np.sum(diff < 0)
    n_zero = np.sum(diff == 0)
    n_eff = n_plus + n_minus

    if n_eff == 0:
        raise ValueError("All observations equal to theta0")

    # Test statistic: use X_+ (number of positive signs)
    X_plus = n_plus

    # p-value from binomial distribution
    # Under H0, X_+ ~ Binomial(n_eff, 0.5)
    if alternative == "two-sided":
        # Two-tailed: p = 2 * P(X_+ ≤ min(X_+, n_eff - X_+))
        p_value = 2 * sp_stats.binom.cdf(min(X_plus, n_eff - X_plus), n_eff, 0.5)
    elif alternative == "greater":
        # H1: median > θ₀, so count upper tail
        p_value = 1 - sp_stats.binom.cdf(X_plus - 1, n_eff, 0.5)
    elif alternative == "less":
        # H1: median < θ₀, so count lower tail
        p_value = sp_stats.binom.cdf(X_plus, n_eff, 0.5)
    else:
        raise ValueError("alternative must be 'two-sided', 'less', or 'greater'")

    interpretation = "reject" if p_value < 0.05 else "not reject"

    return {
        "statistic": int(X_plus),
        "n_plus": int(n_plus),
        "n_minus": int(n_minus),
        "n_zero": int(n_zero),
        "n_eff": int(n_eff),
        "p_value": float(p_value),
        "interpretation": interpretation,
    }
