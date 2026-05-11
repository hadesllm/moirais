# morie.fn — function file (hadesllm/morie)
"""
Kolmogorov-Smirnov two-sample test.

Tests whether two independent samples come from the same continuous distribution
by comparing their empirical cumulative distribution functions.

Reference: Gibbons & Chakraborti (2011), Nonparametric Statistical Inference, 5th Ed. § 5.2
"""

import numpy as np

__all__ = ["kstwo"]


def kstwo(x, y, axis=0):
    r"""
    Kolmogorov-Smirnov two-sample test.

    Tests H0: F_X(t) = F_Y(t) for all t.

    Parameters
    ----------
    x, y : array_like
        Two independent samples.
    axis : int, optional
        Axis along which to apply the test (default 0).

    Returns
    -------
    dict
        Keys:
        - 'statistic': D = max|F_X(t) - F_Y(t)|
        - 'p_value': two-tailed p-value
        - 'critical_value': critical value at α=0.05
        - 'interpretation': "reject" or "not reject" null
        - 'n1': size of first sample
        - 'n2': size of second sample

    Notes
    -----
    Uses Kolmogorov-Smirnov distribution for two-sample case.
    Test is exact for discrete distributions, approximate for continuous.

    Examples
    --------
    >>> x = np.random.default_rng(42).standard_normal(50)
    >>> y = np.random.default_rng(43).standard_normal(50)
    >>> result = kstwo(x, y)
    >>> result['p_value']  # doctest: +SKIP
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if x.ndim == 2:
        x = np.take(x, 0, axis=axis)
    if y.ndim == 2:
        y = np.take(y, 0, axis=axis)

    n1 = len(x)
    n2 = len(y)

    if n1 < 1 or n2 < 1:
        raise ValueError("Both samples must have at least 1 observation")

    # Combine and sort
    combined = np.concatenate([x, y])
    combined_sorted = np.sort(combined)

    # Compute empirical CDFs
    F_X = np.zeros(len(combined_sorted))
    F_Y = np.zeros(len(combined_sorted))

    for i, val in enumerate(combined_sorted):
        F_X[i] = np.sum(x <= val) / n1
        F_Y[i] = np.sum(y <= val) / n2

    # KS statistic
    D = np.max(np.abs(F_X - F_Y))

    # Critical value for α = 0.05 (two-tailed)
    # Critical value ≈ sqrt(-ln(0.025/2)) / sqrt(n1*n2/(n1+n2))
    critical_value = 1.358 * np.sqrt((n1 + n2) / (n1 * n2))

    # p-value from KS distribution
    # For two-sample case: use Kolmogorov distribution
    statistic_scaled = D * np.sqrt(n1 * n2 / (n1 + n2))
    p_value = 2 * np.exp(-2 * statistic_scaled**2) if statistic_scaled > 0 else 1.0
    p_value = min(p_value, 1.0)

    interpretation = "reject" if critical_value < D else "not reject"

    return {
        "statistic": float(D),
        "p_value": float(p_value),
        "critical_value": float(critical_value),
        "interpretation": interpretation,
        "n1": int(n1),
        "n2": int(n2),
    }
