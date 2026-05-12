# morie.fn -- function file (hadesllm/morie)
"""Kruskal-Wallis H-test (non-parametric one-way ANOVA)."""

import numpy as np
import scipy.stats as stats


def kruskal_wallis_test(*groups) -> dict:
    """
    Kruskal-Wallis H-test (non-parametric one-way ANOVA).

    A rank-based alternative to one-way ANOVA that does not assume normality.

    :param groups: Two or more 1-D array-like group samples.
    :return: dict with keys ``H``, ``df``, ``p_value``.
    :raises ValueError: If fewer than 2 groups are provided.

    References
    ----------
    Kruskal, W. H., & Wallis, W. A. (1952). Use of ranks in one-criterion
        variance analysis. Journal of the American Statistical Association, 47.
    """
    if len(groups) < 2:
        raise ValueError("kruskal_wallis_test requires at least 2 groups.")
    arrays = [np.asarray(g, dtype=float) for g in groups]
    h_stat, p_val = stats.kruskal(*arrays)
    return {
        "H": float(h_stat),
        "df": len(groups) - 1,
        "p_value": float(p_val),
        "method": "Kruskal-Wallis H-test",
    }


kw = kruskal_wallis_test


def cheatsheet() -> str:
    return "kruskal_wallis_test({}) -> Kruskal-Wallis H-test (non-parametric one-way ANOVA)."
