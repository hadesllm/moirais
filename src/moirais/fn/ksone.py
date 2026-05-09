# moirais.fn — function file (hadesllm/moirais)
"""
Kolmogorov-Smirnov one-sample test.

Tests whether a sample is consistent with a specified continuous distribution
by comparing the empirical and theoretical cumulative distribution functions.

Reference: Gibbons & Chakraborti (2011), Nonparametric Statistical Inference, 5th Ed. § 4.2
"""

import numpy as np
from scipy import stats as sp_stats

__all__ = ["ksone"]


def ksone(x, dist="norm", dist_params=None, axis=0, cdf=None):
    r"""
    Kolmogorov-Smirnov one-sample test.

    Tests H0: F(x) = F0(x) for all x, where F0 is a specified distribution.

    Parameters
    ----------
    x : array_like
        Input sample.
    dist : str, optional
        Distribution to test against ('norm', 'uniform', 'expon'). Default 'norm'.
    dist_params : dict, optional
        Parameters for the distribution. For 'norm': {'loc': mean, 'scale': sd}.
        If None, estimated from data.
    axis : int, optional
        Axis along which to apply the test (default 0).

    Returns
    -------
    dict
        Keys:
        - 'statistic': D = max|F(x) - F0(x)|
        - 'p_value': two-tailed p-value
        - 'critical_value': critical value at α=0.05
        - 'interpretation': "reject" or "not reject" null

    Notes
    -----
    Uses Kolmogorov-Smirnov distribution. When parameters are estimated from data,
    p-values are conservative (slightly too large).

    Examples
    --------
    >>> x = np.random.default_rng(42).standard_normal(100)
    >>> result = ksone(x, dist='norm')
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
        F0 = sp_stats.norm(loc=loc, scale=scale).cdf
    elif dist == "uniform":
        if dist_params is None:
            loc = np.min(x)
            scale = np.max(x) - np.min(x)
        else:
            loc = dist_params.get("loc", np.min(x))
            scale = dist_params.get("scale", np.max(x) - np.min(x))
        F0 = sp_stats.uniform(loc=loc, scale=scale).cdf
    elif dist == "expon":
        if dist_params is None:
            loc = 0
            scale = np.mean(x)
        else:
            loc = dist_params.get("loc", 0)
            scale = dist_params.get("scale", np.mean(x))
        F0 = sp_stats.expon(loc=loc, scale=scale).cdf
    else:
        raise ValueError(f"Unknown distribution: {dist}")

    # Sort data
    x_sorted = np.sort(x)

    # Empirical CDF: F(x_i^-) = (i-1)/n, F(x_i) = i/n
    # KS statistic uses max of both
    i = np.arange(1, n + 1)
    F_empirical_before = (i - 1) / n
    F_empirical_after = i / n
    F_theoretical = F0(x_sorted)

    # D+ = max(F_empirical_after - F_theoretical)
    # D- = max(F_theoretical - F_empirical_before)
    D_plus = np.max(F_empirical_after - F_theoretical)
    D_minus = np.max(F_theoretical - F_empirical_before)
    D = max(D_plus, D_minus)

    # Approximate p-value using Kolmogorov-Smirnov distribution
    # For large n, use normal approximation: sqrt(n) * D ~ KS
    statistic_scaled = np.sqrt(n) * D

    # Critical value for α = 0.05 (two-tailed): 1.36
    critical_value = 1.36 / np.sqrt(n)

    # p-value from KS distribution (approximation)
    p_value = 2 * np.exp(-2 * statistic_scaled**2) if statistic_scaled > 0 else 1.0
    p_value = min(p_value, 1.0)

    interpretation = "reject" if critical_value < D else "not reject"

    return {
        "statistic": float(D),
        "p_value": float(p_value),
        "critical_value": float(critical_value),
        "interpretation": interpretation,
    }
