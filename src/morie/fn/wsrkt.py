"""
Wilcoxon signed-rank test.

Tests whether a sample median equals a hypothesized value (one-sample) or
whether two paired samples have the same distribution.

Reference: Gibbons & Chakraborti (2011), Nonparametric Statistical Inference, 5th Ed. § 2.3
"""

import numpy as np
from scipy import stats as sp_stats

__all__ = ["wsrkt"]


def wsrkt(x, y=None, theta0=None, axis=0, alternative="two-sided", cdf=None):
    r"""
    Wilcoxon signed-rank test (one- or paired-sample).

    Tests H0: median(X) = θ₀ (one-sample) or
           H0: median(X - Y) = 0 (paired samples).

    Parameters
    ----------
    x : array_like
        First sample or differences.
    y : array_like, optional
        Second sample (for paired test). If None, one-sample test.
    theta0 : float, optional
        Hypothesized median (one-sample). Default 0.
    axis : int, optional
        Axis along which to apply the test (default 0).
    alternative : {'two-sided', 'less', 'greater'}, optional
        Alternative hypothesis (default 'two-sided').

    Returns
    -------
    dict
        Keys:
        - 'statistic': T (sum of positive ranks)
        - 'n_pos': count of positive differences
        - 'n_neg': count of negative differences
        - 'n_zero': count of zero differences
        - 'n_eff': effective sample size
        - 'p_value': p-value
        - 'interpretation': "reject" or "not reject" null

    Notes
    -----
    More powerful than sign test. Robust to outliers.
    Uses normal approximation for large n with ties correction.

    Examples
    --------
    >>> x = np.array([5, 3, 8, 6, 9, 2, 7, 4])
    >>> result = wsrkt(x, theta0=5)
    >>> result['p_value']  # doctest: +SKIP
    """
    x = np.asarray(x, dtype=np.float64)

    if x.ndim == 2:
        x = np.take(x, 0, axis=axis)
    elif x.ndim != 1:
        raise ValueError("Input must be 1D or 2D array")

    # Paired or one-sample?
    if y is not None:
        y = np.asarray(y, dtype=np.float64)
        if y.ndim == 2:
            y = np.take(y, 0, axis=axis)
        if len(x) != len(y):
            raise ValueError("x and y must have same length for paired test")
        diff = x - y
    else:
        if theta0 is None:
            theta0 = 0
        diff = x - theta0

    n = len(diff)
    if n < 1:
        raise ValueError("Sample size must be at least 1")

    # Remove zeros
    nonzero_diff = diff[diff != 0]
    n_eff = len(nonzero_diff)
    n_zero = n - n_eff

    if n_eff == 0:
        if y is not None:
            return {
                "statistic": 0.0,
                "n_pos": 0,
                "n_neg": 0,
                "n_zero": int(n_zero),
                "n_eff": 0,
                "p_value": 1.0,
                "interpretation": "not reject",
            }
        raise ValueError("All observations equal to theta0")

    # Get absolute values and ranks
    abs_diff = np.abs(nonzero_diff)
    ranks = sp_stats.rankdata(abs_diff)

    # Count positive/negative
    n_pos = np.sum(nonzero_diff > 0)
    n_neg = np.sum(nonzero_diff < 0)

    # T+ = sum of ranks for positive differences
    T_plus = np.sum(ranks[nonzero_diff > 0])

    # Expected value and variance under null
    E_T = n_eff * (n_eff + 1) / 4

    # Variance (with ties correction)
    # First, handle ties
    unique_vals = np.unique(abs_diff)
    ties_correction = 0
    for val in unique_vals:
        t = np.sum(abs_diff == val)
        if t > 1:
            ties_correction += t * (t - 1) * (t + 1)

    Var_T = n_eff * (n_eff + 1) * (2 * n_eff + 1) / 24 - ties_correction / 48

    # Standardized statistic
    z_stat = (T_plus - E_T) / np.sqrt(Var_T)

    # p-value
    if alternative == "two-sided":
        p_value = 2 * (1 - sp_stats.norm.cdf(np.abs(z_stat)))
    elif alternative == "greater":
        p_value = 1 - sp_stats.norm.cdf(z_stat)
    elif alternative == "less":
        p_value = sp_stats.norm.cdf(z_stat)
    else:
        raise ValueError("alternative must be 'two-sided', 'less', or 'greater'")

    interpretation = "reject" if p_value < 0.05 else "not reject"

    return {
        "statistic": float(T_plus),
        "n_pos": int(n_pos),
        "n_neg": int(n_neg),
        "n_zero": int(n_zero),
        "n_eff": int(n_eff),
        "p_value": float(p_value),
        "interpretation": interpretation,
    }
