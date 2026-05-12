# morie.fn -- function file (hadesllm/morie)
"""
Wald-Wolfowitz runs test for randomness.

Tests whether a sequence is random by counting the number of runs
(maximal sequences of identical outcomes above or below median).

Reference: Gibbons & Chakraborti (2011), Nonparametric Statistical Inference, 5th Ed. § 3.2
"""

import numpy as np
from scipy import stats

__all__ = ["runsq"]


def runsq(x, axis=0, cdf=None):
    r"""
    Wald-Wolfowitz runs test for randomness.

    Tests the null hypothesis that a sequence was generated randomly
    by examining the number of runs (streaks) above/below the median.

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
        - 'statistic': observed number of runs
        - 'n_plus': count of values above median
        - 'n_minus': count of values at/below median
        - 'expected_runs': E[R] under null
        - 'variance': Var(R) under null
        - 'z_stat': standardized test statistic
        - 'p_value': two-tailed p-value (normal approximation)
        - 'interpretation': "random" or "not random" at α=0.05

    Notes
    -----
    For large samples (n ≥ 30), uses normal approximation.
    Small-sample exact tables not included; falls back to normal.

    Examples
    --------
    >>> x = np.array([5, 2, 8, 3, 9, 1, 7, 4, 6])
    >>> result = runsq(x)
    >>> result['p_value']  # doctest: +SKIP
    """
    x = np.asarray(x, dtype=np.float64)

    # Flatten if 2D
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

    # Count runs
    runs = 1 + np.sum(np.diff(binary) != 0)

    n_plus = np.sum(binary)
    n_minus = n - n_plus

    E_R = 1 + 2 * n_plus * n_minus / n
    Var_R = 2 * n_plus * n_minus * (2 * n_plus * n_minus - n) / (n**2 * (n - 1))

    if Var_R <= 0 or n_plus == 0 or n_minus == 0:
        z_stat = float("nan")
        p_value = 0.0
    else:
        z_stat = (runs - E_R) / np.sqrt(Var_R)
        p_value = 2 * (1 - stats.norm.cdf(np.abs(z_stat)))

    interpretation = "random" if p_value > 0.05 else "not random"

    return {
        "statistic": int(runs),
        "n_plus": int(n_plus),
        "n_minus": int(n_minus),
        "expected_runs": float(E_R),
        "variance": float(Var_R),
        "z_stat": float(z_stat),
        "p_value": float(p_value),
        "interpretation": interpretation,
    }
