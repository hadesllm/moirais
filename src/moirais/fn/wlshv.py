"""
Walsh averages for confidence intervals in location inference.

Computes all pairwise averages (x_i + x_j) / 2 for creating nonparametric
confidence intervals on the population median.

Reference: Gibbons & Chakraborti (2011), Nonparametric Statistical Inference, 5th Ed. § 2.4
"""

import numpy as np

__all__ = ["wlshv"]


def wlshv(x, axis=0):
    r"""
    Compute Walsh averages for confidence intervals on the median.

    Computes all pairwise averages W_ij = (X_i + X_j) / 2 for i ≤ j.

    Parameters
    ----------
    x : array_like
        Input sample.
    axis : int, optional
        Axis along which to apply (default 0).

    Returns
    -------
    dict
        Keys:
        - 'walsh_averages': sorted array of all Walsh averages
        - 'count': total number of averages
        - 'ci_lower_idx': index for approximate 95% CI lower bound
        - 'ci_upper_idx': index for approximate 95% CI upper bound
        - 'ci_lower': lower bound of CI
        - 'ci_upper': upper bound of CI

    Notes
    -----
    Total number of Walsh averages = n(n+1)/2 where n is sample size.
    Used in conjunction with Wilcoxon signed-rank test for CI construction.

    Examples
    --------
    >>> x = np.array([1, 2, 3, 4, 5])
    >>> result = wlshv(x)
    >>> result['count']
    15
    """
    x = np.asarray(x, dtype=np.float64)

    if x.ndim == 2:
        x = np.take(x, 0, axis=axis)
    elif x.ndim != 1:
        raise ValueError("Input must be 1D or 2D array")

    n = len(x)
    if n < 2:
        raise ValueError("Sample size must be at least 2")

    # Compute all Walsh averages W_ij = (x_i + x_j) / 2 for i ≤ j
    walsh = []
    for i in range(n):
        for j in range(i, n):
            walsh.append((x[i] + x[j]) / 2)

    walsh = np.array(walsh)
    walsh_sorted = np.sort(walsh)
    count = len(walsh_sorted)

    # Indices for approximate 95% confidence interval
    # Using Wilcoxon signed-rank critical value for α = 0.05
    # CI endpoints at positions d and count - d + 1
    # where d = critical value from Wilcoxon distribution

    # For large n, approximate d using normal:
    # d ≈ (n(n+1)/4) - z_{α/2} * sqrt(n(n+1)(2n+1)/24)
    if n < 20:
        # Use small-sample critical values (from tables)
        # This is approximate
        d = int(np.ceil(n * (n + 1) / 4 - 1.96 * np.sqrt(n * (n + 1) * (2 * n + 1) / 24)))
    else:
        d = int(np.ceil(n * (n + 1) / 4 - 1.96 * np.sqrt(n * (n + 1) * (2 * n + 1) / 24)))

    d = max(0, min(d, count - 1))

    ci_lower_idx = d
    ci_upper_idx = count - d - 1

    ci_lower = walsh_sorted[ci_lower_idx] if ci_lower_idx < count else walsh_sorted[0]
    ci_upper = walsh_sorted[ci_upper_idx] if ci_upper_idx >= 0 else walsh_sorted[-1]

    return {
        "walsh_averages": walsh_sorted,
        "count": int(count),
        "ci_lower_idx": int(ci_lower_idx),
        "ci_upper_idx": int(ci_upper_idx),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
    }
