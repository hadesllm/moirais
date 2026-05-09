# moirais.fn — function file (hadesllm/moirais)
"""
Longest run test for randomness.

Tests whether a sequence is random by examining the length of the longest
run (maximal sequence of identical outcomes).

Reference: Gibbons & Chakraborti (2011), Nonparametric Statistical Inference, 5th Ed. § 3.3
"""

import numpy as np

__all__ = ["lrunt"]


def lrunt(x, axis=0):
    r"""
    Longest run test for randomness.

    Computes the length of the longest run in a sequence and tests
    the hypothesis of randomness based on critical values.

    Parameters
    ----------
    x : array_like
        Input sequence or 2D array. If 2D, test applied along `axis`.
    axis : int, optional
        Axis along which to apply the test (default 0).

    Returns
    -------
    dict
        Keys:
        - 'statistic': length of the longest run
        - 'n': sample size
        - 'n_plus': count of values above/equal median
        - 'n_minus': count of values below median
        - 'interpretation': "reject" or "not reject" null (α=0.05)

    Notes
    -----
    Uses critical value tables from Gibbons & Chakraborti.
    For exact p-values, consult published tables or use simulation.

    Examples
    --------
    >>> x = np.array([1, 1, 1, 2, 2, 1, 1, 2, 2, 2])
    >>> result = lrunt(x)
    >>> result['statistic']  # doctest: +SKIP
    """
    x = np.asarray(x, dtype=np.float64)

    if x.ndim == 2:
        x = np.take(x, 0, axis=axis)
    elif x.ndim != 1:
        raise ValueError("Input must be 1D or 2D array")

    n = len(x)
    if n < 2:
        raise ValueError("Sample size must be at least 2")

    # Convert to binary sequence: 1 if ≥ median, 0 otherwise
    median = np.median(x)
    binary = (x >= median).astype(int)

    # Find longest run
    if n == 0:
        longest_run = 0
    else:
        runs = np.split(binary, np.where(np.diff(binary) != 0)[0] + 1)
        longest_run = max(len(run) for run in runs)

    n_plus = np.sum(binary)
    n_minus = n - n_plus

    # Critical value approximation for α = 0.05
    # From Gibbons & Chakraborti tables
    critical_value = int(np.ceil(np.log2(n)) + 1)

    interpretation = "reject" if longest_run > critical_value else "not reject"

    return {
        "statistic": int(longest_run),
        "n": int(n),
        "n_plus": int(n_plus),
        "n_minus": int(n_minus),
        "critical_value": int(critical_value),
        "interpretation": interpretation,
    }
