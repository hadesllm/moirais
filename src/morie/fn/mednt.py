# morie.fn — function file (hadesllm/morie)
"""
Median test (two-sample).

Tests whether two independent samples have the same median using a
contingency-table approach based on pooled median.

Reference: Gibbons & Chakraborti (2011), Nonparametric Statistical Inference, 5th Ed. § 5.4
"""

import numpy as np
from scipy import stats as sp_stats

__all__ = ["mednt"]


def mednt(x, y, axis=0):
    r"""
    Median test for two independent samples.

    Tests H0: the two samples have the same median using 2×2 contingency table
    of observations above/below pooled median.

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
        - 'statistic': chi-square test statistic
        - 'p_value': p-value from chi-square distribution
        - 'n1_above': count in x above pooled median
        - 'n1_below': count in x below pooled median
        - 'n2_above': count in y above pooled median
        - 'n2_below': count in y below pooled median
        - 'interpretation': "reject" or "not reject" null

    Notes
    -----
    Less powerful than Mann-Whitney but tests broader null (any location difference).
    Uses 2×2 contingency table with chi-square distribution (df=1).

    Examples
    --------
    >>> x = np.array([1, 2, 3, 4, 5])
    >>> y = np.array([3, 4, 5, 6, 7])
    >>> result = mednt(x, y)
    >>> result['p_value']  # doctest: +SKIP
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if x.ndim == 2:
        x = np.take(x, 0, axis=axis)
    if y.ndim == 2:
        y = np.take(y, 0, axis=axis)

    n_x = len(x)
    n_y = len(y)

    if n_x < 1 or n_y < 1:
        raise ValueError("Both samples must have at least 1 observation")

    # Pooled median
    pooled = np.concatenate([x, y])
    med_pooled = np.median(pooled)

    # Contingency table
    n1_above = np.sum(x > med_pooled)
    n1_below = np.sum(x < med_pooled)
    n2_above = np.sum(y > med_pooled)
    n2_below = np.sum(y < med_pooled)

    # Ignore ties at median
    # Contingency table: 2x2
    # Above | Below
    # ------+------
    #  n1a  | n1b
    #  n2a  | n2b

    contingency = np.array([[n1_above, n1_below], [n2_above, n2_below]])

    # Chi-square test
    chi2, p_value, dof, expected_freq = sp_stats.chi2_contingency(contingency)

    interpretation = "reject" if p_value < 0.05 else "not reject"

    return {
        "statistic": float(chi2),
        "p_value": float(p_value),
        "n1_above": int(n1_above),
        "n1_below": int(n1_below),
        "n2_above": int(n2_above),
        "n2_below": int(n2_below),
        "interpretation": interpretation,
    }
