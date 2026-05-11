# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""One-way ANOVA F-test."""

import numpy as np
import scipy.stats as stats


def anova_one_way(*groups) -> dict:
    """
    One-way ANOVA F-test.

    Tests H0: mu_1 = mu_2 = ... = mu_k. Assumes independent observations,
    normality within each group, and homoscedasticity.

    :param groups: Two or more 1-D array-like group samples.
    :return: dict with keys ``F``, ``df_between``, ``df_within``, ``p_value``,
        ``eta_squared``.
    :raises ValueError: If fewer than 2 groups are provided.

    References
    ----------
    Fisher, R. A. (1925). Statistical Methods for Research Workers. Oliver & Boyd.
    Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences (2nd ed.).
    """
    if len(groups) < 2:
        raise ValueError("anova_one_way requires at least 2 groups.")
    arrays = [np.asarray(g, dtype=float) for g in groups]
    for i, a in enumerate(arrays):
        if len(a) < 1:
            raise ValueError(f"Group {i} is empty.")
    f_stat, p_val = stats.f_oneway(*arrays)
    k = len(arrays)
    N = sum(len(a) for a in arrays)
    df_between = k - 1
    df_within = N - k
    # Eta-squared: SS_between / SS_total
    grand_mean = np.mean(np.concatenate(arrays))
    ss_between = sum(len(a) * (np.mean(a) - grand_mean) ** 2 for a in arrays)
    ss_within = sum(np.sum((a - np.mean(a)) ** 2) for a in arrays)
    ss_total = ss_between + ss_within
    eta_sq = float(ss_between / ss_total) if ss_total > 0 else 0.0
    return {
        "F": float(f_stat),
        "df_between": df_between,
        "df_within": df_within,
        "p_value": float(p_val),
        "eta_squared": eta_sq,
        "method": "One-way ANOVA",
    }


anova = anova_one_way


def cheatsheet() -> str:
    return "anova_one_way({}) -> One-way ANOVA F-test."
