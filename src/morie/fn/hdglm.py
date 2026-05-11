# morie.fn — function file (hadesllm/morie)
"""
Hodges-Lehmann point estimator for population location.

A robust, nonparametric point estimator of the population median based on
the median of all pairwise averages.

Reference: Gibbons & Chakraborti (2011), Nonparametric Statistical Inference, 5th Ed. § 2.4
"""

import numpy as np

__all__ = ["hdglm"]


def hdglm(x, axis=0):
    r"""
    Hodges-Lehmann point estimator for population location.

    Computes the median of all Walsh averages (X_i + X_j) / 2.
    This is a robust, nonparametric estimate of the population median.

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
        - 'estimate': Hodges-Lehmann point estimator
        - 'walsh_count': number of Walsh averages
        - 'lower_ci': approximate 95% CI lower bound
        - 'upper_ci': approximate 95% CI upper bound

    Notes
    -----
    More robust than the sample mean. Breakdown point = 0.29.
    Pairs well with Wilcoxon signed-rank test.

    Examples
    --------
    >>> x = np.array([1, 2, 3, 4, 5, 100])
    >>> result = hdglm(x)
    >>> result['estimate']  # doctest: +SKIP
    3.0
    """
    x = np.asarray(x, dtype=np.float64)

    if x.ndim == 2:
        x = np.take(x, 0, axis=axis)
    elif x.ndim != 1:
        raise ValueError("Input must be 1D or 2D array")

    n = len(x)
    if n < 2:
        raise ValueError("Sample size must be at least 2")

    # Compute all Walsh averages
    walsh = []
    for i in range(n):
        for j in range(i, n):
            walsh.append((x[i] + x[j]) / 2)

    walsh = np.array(walsh)
    walsh_sorted = np.sort(walsh)
    count = len(walsh_sorted)

    # Hodges-Lehmann estimator is the median of Walsh averages
    hl_estimate = np.median(walsh_sorted)

    # Approximate 95% CI using Wilcoxon signed-rank critical value
    d = int(np.ceil(n * (n + 1) / 4 - 1.96 * np.sqrt(n * (n + 1) * (2 * n + 1) / 24)))
    d = max(0, min(d, count - 1))

    ci_lower_idx = d
    ci_upper_idx = count - d - 1

    ci_lower = walsh_sorted[ci_lower_idx] if ci_lower_idx < count else walsh_sorted[0]
    ci_upper = walsh_sorted[ci_upper_idx] if ci_upper_idx >= 0 else walsh_sorted[-1]

    return {
        "estimate": float(hl_estimate),
        "walsh_count": int(count),
        "lower_ci": float(ci_lower),
        "upper_ci": float(ci_upper),
    }
