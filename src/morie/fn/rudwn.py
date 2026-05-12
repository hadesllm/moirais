# morie.fn -- function file (hadesllm/morie)
"""
Runs up and down test for randomness.

Tests whether a sequence is random by counting sequences of increases
and decreases (runs up/down).

Reference: Gibbons & Chakraborti (2011), Nonparametric Statistical Inference, 5th Ed. § 3.4
"""

import numpy as np
from scipy import stats

__all__ = ["rudwn"]


def rudwn(x, axis=0, cdf=None):
    r"""
    Runs up and down test for randomness.

    Tests randomness by counting runs in the sequence of increases/decreases.
    A run is a maximal sequence of consecutive ups or consecutive downs.

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
        - 'statistic': observed number of runs up/down
        - 'expected_runs': E[R] under null
        - 'variance': Var(R) under null
        - 'z_stat': standardized test statistic
        - 'p_value': two-tailed p-value
        - 'interpretation': "random" or "not random" at α=0.05

    Notes
    -----
    Uses normal approximation. More powerful than signs test for detecting
    certain alternative hypotheses.

    Examples
    --------
    >>> x = np.array([5, 2, 8, 3, 9, 1, 7])
    >>> result = rudwn(x)
    >>> result['statistic']  # doctest: +SKIP
    """
    x = np.asarray(x, dtype=np.float64)

    if x.ndim == 2:
        x = np.take(x, 0, axis=axis)
    elif x.ndim != 1:
        raise ValueError("Input must be 1D or 2D array")

    n = len(x)
    if n < 3:
        raise ValueError("Sample size must be at least 3")

    # Compute differences: 1 if x[i+1] > x[i], 0 otherwise
    diffs = np.diff(x) > 0
    diffs = diffs.astype(int)

    # Count runs
    if len(diffs) > 0:
        runs = 1 + np.sum(np.diff(diffs) != 0)
    else:
        runs = 0

    # Expected value and variance
    # E[R] = (2n - 1) / 3
    # Var(R) = (16n - 29) / 90
    E_R = (2 * n - 1) / 3
    Var_R = (16 * n - 29) / 90

    # Standardized test statistic
    z_stat = (runs - E_R) / np.sqrt(Var_R)

    # Two-tailed p-value
    p_value = 2 * (1 - stats.norm.cdf(np.abs(z_stat)))

    interpretation = "random" if p_value > 0.05 else "not random"

    return {
        "statistic": int(runs),
        "expected_runs": float(E_R),
        "variance": float(Var_R),
        "z_stat": float(z_stat),
        "p_value": float(p_value),
        "interpretation": interpretation,
    }
