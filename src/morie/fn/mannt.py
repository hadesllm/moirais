# morie.fn — function file (hadesllm/morie)
"""
Mann-Whitney U test for two independent samples.

Nonparametric test of whether two independent samples come from populations
with the same distribution, with special focus on location shifts.

Reference: Gibbons & Chakraborti (2011), Nonparametric Statistical Inference, 5th Ed. § 5.3
"""

import numpy as np
from scipy import stats as sp_stats

__all__ = ["mannt"]


def mannt(x, y, axis=0, alternative="two-sided", cdf=None):
    r"""
    Mann-Whitney U test for two independent samples.

    Tests H0: F_X(t) = F_Y(t) for all t (or equivalently, P(X > Y) = 0.5).

    Parameters
    ----------
    x, y : array_like
        Two independent samples.
    axis : int, optional
        Axis along which to apply the test (default 0).
    alternative : {'two-sided', 'less', 'greater'}, optional
        Alternative hypothesis (default 'two-sided').

    Returns
    -------
    dict
        Keys:
        - 'statistic': U statistic (minimum of U_x and U_y)
        - 'U_x': Mann-Whitney statistic for x
        - 'U_y': Mann-Whitney statistic for y
        - 'n_x': size of x
        - 'n_y': size of y
        - 'p_value': p-value from normal approximation
        - 'interpretation': "reject" or "not reject" null

    Notes
    -----
    Equivalent to Wilcoxon rank-sum test for two independent samples.
    Uses normal approximation with continuity correction.
    Handles ties correctly.

    Examples
    --------
    >>> x = np.array([1, 3, 5, 7, 9])
    >>> y = np.array([2, 4, 6, 8, 10])
    >>> result = mannt(x, y)
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

    # Combine and rank
    combined = np.concatenate([x, y])
    ranks = sp_stats.rankdata(combined)

    # Ranks for x and y
    ranks_x = ranks[:n_x]
    ranks_y = ranks[n_x:]

    # Sum of ranks
    R_x = np.sum(ranks_x)
    R_y = np.sum(ranks_y)

    # U statistics
    U_x = n_x * n_y + n_x * (n_x + 1) / 2 - R_x
    U_y = n_x * n_y + n_y * (n_y + 1) / 2 - R_y

    U = min(U_x, U_y)
    E_U = n_x * n_y / 2

    unique_vals = np.unique(combined)
    ties_correction = 0
    for val in unique_vals:
        t = np.sum(combined == val)
        if t > 1:
            ties_correction += t * (t - 1) * (t + 1)

    Var_U = (n_x * n_y * (n_x + n_y + 1)) / 12 - ties_correction / 12

    # U_x = n_x*n_y + n_x*(n_x+1)/2 - R_x; large U_x <=> x ranks LOW <=> x < y.
    # So "greater" (x > y) rejects when U_x is SMALL relative to E_U.
    if alternative == "two-sided":
        z_stat = (U - E_U - 0.5) / np.sqrt(Var_U)
        p_value = 2 * (1 - sp_stats.norm.cdf(np.abs(z_stat)))
    elif alternative == "greater":
        z_stat = (U_x - E_U + 0.5) / np.sqrt(Var_U)
        p_value = sp_stats.norm.cdf(z_stat)
    elif alternative == "less":
        z_stat = (U_x - E_U - 0.5) / np.sqrt(Var_U)
        p_value = 1 - sp_stats.norm.cdf(z_stat)
    else:
        raise ValueError("alternative must be 'two-sided', 'less', or 'greater'")

    interpretation = "reject" if p_value < 0.05 else "not reject"

    return {
        "statistic": float(U),
        "U_x": float(U_x),
        "U_y": float(U_y),
        "n_x": int(n_x),
        "n_y": int(n_y),
        "p_value": float(p_value),
        "interpretation": interpretation,
    }
