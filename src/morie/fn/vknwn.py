"""Known-groups validity: do scores differ across expected groups?"""

from __future__ import annotations

import numpy as np
from scipy import stats as sp


def validity_known_groups(
    scores: np.ndarray,
    groups: np.ndarray,
) -> dict:
    """Known-groups (discriminative) validity.

    Tests whether scale scores differ significantly across groups that
    are expected to differ on the construct.  Uses independent t-test
    for 2 groups and one-way ANOVA F-test for 3+ groups.

    Parameters
    ----------
    scores : array-like
        Scale scores.
    groups : array-like
        Group membership labels.

    Returns
    -------
    dict
        Keys: ``test``, ``statistic``, ``p_value``, ``effect_size``,
        ``group_means``, ``group_ns``, ``n_groups``.

    References
    ----------
    Hattie, J., & Cooksey, R. W. (1984). Procedures for assessing the
    validities of tests using the "known-groups" method. *Applied
    Psychological Measurement*, 8(3), 295--305.
    """
    s = np.asarray(scores, dtype=np.float64).ravel()
    g = np.asarray(groups).ravel()
    mask = np.isfinite(s)
    s, g = s[mask], g[mask]

    unique_groups = np.unique(g)
    k = len(unique_groups)
    group_data = [s[g == grp] for grp in unique_groups]
    group_means = {str(grp): float(np.mean(gd)) for grp, gd in zip(unique_groups, group_data)}
    group_ns = {str(grp): len(gd) for grp, gd in zip(unique_groups, group_data)}

    if k < 2:
        return {
            "test": "none",
            "statistic": np.nan,
            "p_value": np.nan,
            "effect_size": np.nan,
            "group_means": group_means,
            "group_ns": group_ns,
            "n_groups": k,
        }

    if k == 2:
        stat, p = sp.ttest_ind(group_data[0], group_data[1])
        # Cohen's d
        n1, n2 = len(group_data[0]), len(group_data[1])
        pooled_std = np.sqrt(
            ((n1 - 1) * np.var(group_data[0], ddof=1) + (n2 - 1) * np.var(group_data[1], ddof=1)) / (n1 + n2 - 2)
        )
        es = float((np.mean(group_data[0]) - np.mean(group_data[1])) / pooled_std) if pooled_std > 0 else np.nan
        return {
            "test": "t-test",
            "statistic": float(stat),
            "p_value": float(p),
            "effect_size": es,
            "effect_size_type": "Cohen_d",
            "group_means": group_means,
            "group_ns": group_ns,
            "n_groups": k,
        }
    else:
        stat, p = sp.f_oneway(*group_data)
        # Eta-squared
        grand_mean = np.mean(s)
        ss_between = sum(len(gd) * (np.mean(gd) - grand_mean) ** 2 for gd in group_data)
        ss_total = np.sum((s - grand_mean) ** 2)
        eta2 = float(ss_between / ss_total) if ss_total > 0 else np.nan
        return {
            "test": "ANOVA",
            "statistic": float(stat),
            "p_value": float(p),
            "effect_size": eta2,
            "effect_size_type": "eta_squared",
            "group_means": group_means,
            "group_ns": group_ns,
            "n_groups": k,
        }


def cheatsheet() -> str:
    return "validity_known_groups({}) -> Known-groups validity: do scores differ across expected grou"
