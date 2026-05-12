# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""
Anderson-Darling test for goodness of fit.

Tests whether a sample comes from a specified distribution. More powerful
than Kolmogorov-Smirnov for detecting departures in tails.

Reference: Gibbons & Chakraborti (2011), Nonparametric Statistical Inference, 5th Ed. § 4.4
"""

import numpy as np
from scipy import stats as sp_stats

__all__ = ["ander"]


def ander(x, dist="norm", dist_params=None, axis=0, cdf=None):
    r"""
    Anderson-Darling test for goodness of fit.

    Tests H0: sample comes from specified distribution using a
    weighted version of the KS statistic with more weight on tails.

    Parameters
    ----------
    x : array_like
        Input sample.
    dist : str, optional
        Distribution ('norm', 'uniform', 'expon'). Default 'norm'.
    dist_params : dict, optional
        Distribution parameters. If None, estimated from data.
    axis : int, optional
        Axis along which to apply the test (default 0).

    Returns
    -------
    dict
        Keys:
        - 'statistic': A² test statistic
        - 'p_value': p-value (approximation)
        - 'critical_value': critical value at α=0.05
        - 'interpretation': "reject" or "not reject" null

    Notes
    -----
    More powerful than K-S test. Uses critical values from tables.
    More sensitive to deviations in tails.

    Examples
    --------
    >>> x = np.random.default_rng(42).standard_normal(100)
    >>> result = ander(x, dist='norm')
    >>> result['p_value']  # doctest: +SKIP
    """
    x = np.asarray(x, dtype=np.float64)

    if x.ndim == 2:
        x = np.take(x, 0, axis=axis)
    elif x.ndim != 1:
        raise ValueError("Input must be 1D or 2D array")

    n = len(x)
    if n < 2:
        raise ValueError("Sample size must be at least 2")

    # Get distribution
    if dist == "norm":
        if dist_params is None:
            loc = np.mean(x)
            scale = np.std(x, ddof=1)
        else:
            loc = dist_params.get("loc", np.mean(x))
            scale = dist_params.get("scale", np.std(x, ddof=1))
        F = sp_stats.norm(loc=loc, scale=scale).cdf
    elif dist == "uniform":
        if dist_params is None:
            loc = np.min(x)
            scale = np.max(x) - np.min(x)
        else:
            loc = dist_params.get("loc", np.min(x))
            scale = dist_params.get("scale", np.max(x) - np.min(x))
        F = sp_stats.uniform(loc=loc, scale=scale).cdf
    elif dist == "expon":
        if dist_params is None:
            loc = 0
            scale = np.mean(x)
        else:
            loc = dist_params.get("loc", 0)
            scale = dist_params.get("scale", np.mean(x))
        F = sp_stats.expon(loc=loc, scale=scale).cdf
    else:
        raise ValueError(f"Unknown distribution: {dist}")

    # Sort data
    x_sorted = np.sort(x)

    # Compute CDF values
    F_vals = F(x_sorted)

    # Ensure values are in (0, 1)
    F_vals = np.clip(F_vals, 1e-10, 1 - 1e-10)

    # Anderson-Darling statistic
    # A² = -n - (1/n) * Σ[(2i - 1) * (ln(F_i) + ln(1 - F_{n+1-i}))]
    i = np.arange(1, n + 1)
    A_squared = -n - (1 / n) * np.sum((2 * i - 1) * (np.log(F_vals) + np.log(1 - F_vals[::-1])))

    # Critical values (α = 0.05) for normal distribution
    # For other distributions, use normal approximation
    if dist == "norm":
        critical_value = 0.752
    else:
        critical_value = 0.752

    # p-value (approximation)
    if A_squared < 0.576:
        p_value = 1.0
    elif A_squared <= 0.656:
        p_value = 1.0 - 0.2 * (A_squared - 0.576)
    elif A_squared <= 5.1:
        p_value = np.exp(-1.062 * A_squared + 0.4 * np.log(A_squared))
    else:
        p_value = 0.0

    interpretation = "reject" if A_squared > critical_value else "not reject"

    return {
        "statistic": float(A_squared),
        "p_value": float(p_value),
        "critical_value": float(critical_value),
        "interpretation": interpretation,
    }
