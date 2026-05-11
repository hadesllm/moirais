# morie.fn — function file (hadesllm/morie)
"""
Lilliefors test for normality.

Tests whether a sample comes from a normal distribution when parameters
are estimated from the data (less conservative than K-S one-sample).

Reference: Gibbons & Chakraborti (2011), Nonparametric Statistical Inference, 5th Ed. § 4.3
"""

import numpy as np
from scipy import stats as sp_stats

__all__ = ["lilef"]


def lilef(x, axis=0, cdf=None):
    r"""
    Lilliefors test for normality.

    Modification of Kolmogorov-Smirnov test using critical values adjusted
    for parameter estimation from data.

    Parameters
    ----------
    x : array_like
        Input sample.
    axis : int, optional
        Axis along which to apply the test (default 0).

    Returns
    -------
    dict
        Keys:
        - 'statistic': D = max|F_n(x) - Φ((x-μ̂)/σ̂)|
        - 'p_value': p-value (approximation)
        - 'critical_value': critical value at α=0.05
        - 'interpretation': "reject" or "not reject" null
        - 'mean': estimated mean
        - 'std': estimated standard deviation

    Notes
    -----
    Uses Lilliefors critical values, which are more appropriate than K-S
    values when parameters are estimated from the sample.

    Examples
    --------
    >>> x = np.random.default_rng(42).standard_normal(100)
    >>> result = lilef(x)
    >>> result['p_value']  # doctest: +SKIP
    """
    x = np.asarray(x, dtype=np.float64)

    if x.ndim == 2:
        x = np.take(x, 0, axis=axis)
    elif x.ndim != 1:
        raise ValueError("Input must be 1D or 2D array")

    n = len(x)
    if n < 4:
        raise ValueError("Sample size must be at least 4 for Lilliefors test")

    # Estimate parameters
    mean = np.mean(x)
    std = np.std(x, ddof=1)

    # Standardize
    z = (x - mean) / std

    # Compute empirical CDF
    z_sorted = np.sort(z)
    i = np.arange(1, n + 1)
    F_empirical_before = (i - 1) / n
    F_empirical_after = i / n

    # Theoretical CDF (standard normal)
    F_theoretical = sp_stats.norm.cdf(z_sorted)

    # D+ = max(F_empirical - F_theoretical)
    # D- = max(F_theoretical - F_empirical_before)
    D_plus = np.max(F_empirical_after - F_theoretical)
    D_minus = np.max(F_theoretical - F_empirical_before)
    D = max(D_plus, D_minus)

    # Lilliefors critical values (α = 0.05)
    # Approximate formula: critical ≈ 0.886 / sqrt(n)
    critical_value = 0.886 / np.sqrt(n)

    # p-value (very approximate)
    # More refined formula from Lilliefors
    T = D * (np.sqrt(n) - 0.01 + 0.85 / np.sqrt(n))
    if T <= 0.775:
        p_value = 1.0
    elif T <= 0.819:
        p_value = 1.0 - T**2
    elif T <= 5.0:
        p_value = np.exp(-1.23 / T**2)
    else:
        p_value = 0.0

    interpretation = "reject" if critical_value < D else "not reject"

    return {
        "statistic": float(D),
        "p_value": float(p_value),
        "critical_value": float(critical_value),
        "interpretation": interpretation,
        "mean": float(mean),
        "std": float(std),
    }
