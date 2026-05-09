# moirais.fn — function file (hadesllm/moirais)
"""Levene's test for equality of variances."""

import numpy as np
import scipy.stats as stats


def levene_test(*groups) -> dict:
    """
    Levene's test for equality of variances.

    Tests H0: sigma_1^2 = sigma_2^2 = ... = sigma_k^2.
    Uses the median (Brown-Forsythe variant) for robustness to non-normality.

    :param groups: Two or more 1-D array-like group samples.
    :return: dict with keys ``W``, ``df_between``, ``df_within``, ``p_value``.
    :raises ValueError: If fewer than 2 groups are provided.

    References
    ----------
    Levene, H. (1960). Robust tests for equality of variances. In I. Olkin (Ed.),
        Contributions to Probability and Statistics. Stanford University Press.
    Brown, M. B., & Forsythe, A. B. (1974). Robust tests for equality of
        variances. Journal of the American Statistical Association, 69, 364-367.
    """
    if len(groups) < 2:
        raise ValueError("levene_test requires at least 2 groups.")
    arrays = [np.asarray(g, dtype=float) for g in groups]
    w_stat, p_val = stats.levene(*arrays, center="median")
    k = len(arrays)
    N = sum(len(a) for a in arrays)
    return {
        "W": float(w_stat),
        "df_between": k - 1,
        "df_within": N - k,
        "p_value": float(p_val),
        "method": "Levene test (Brown-Forsythe variant)",
    }


levene = levene_test


def cheatsheet() -> str:
    return "levene_test({}) -> Levene's test for equality of variances."
