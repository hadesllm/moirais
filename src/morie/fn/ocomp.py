# morie.fn — function file (hadesllm/morie)
"""OTIS group comparison — t-test/ANOVA + effect size."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


def otis_group_compare(
    df: pd.DataFrame,
    *,
    metric_col: str = "Y",
    group_col: str = "gender",
) -> dict:
    """Compare groups on a metric using t-test (2 groups) or ANOVA (3+).

    Returns the test statistic, p-value, and Cohen's d or eta-squared.

    Parameters
    ----------
    df : DataFrame
        Correctional records.
    metric_col : str
        Numeric metric column.
    group_col : str
        Grouping column.

    Returns
    -------
    dict
        Keys: ``test``, ``statistic``, ``p_value``, ``effect_size``,
        ``effect_measure``, ``group_means``, ``n_groups``.
    """
    groups = [g[metric_col].dropna().values for _, g in df.groupby(group_col)]
    group_means = df.groupby(group_col)[metric_col].mean().to_dict()
    n_groups = len(groups)

    if n_groups < 2:
        return {
            "test": "none",
            "statistic": float("nan"),
            "p_value": float("nan"),
            "effect_size": float("nan"),
            "effect_measure": "none",
            "group_means": group_means,
            "n_groups": n_groups,
        }

    if n_groups == 2:
        stat, p = sp_stats.ttest_ind(groups[0], groups[1], equal_var=False)
        n1, n2 = len(groups[0]), len(groups[1])
        pooled_std = np.sqrt(
            ((n1 - 1) * groups[0].std(ddof=1) ** 2 + (n2 - 1) * groups[1].std(ddof=1) ** 2) / (n1 + n2 - 2)
        )
        d = (groups[0].mean() - groups[1].mean()) / pooled_std if pooled_std > 0 else 0.0
        return {
            "test": "welch_t",
            "statistic": float(stat),
            "p_value": float(p),
            "effect_size": float(d),
            "effect_measure": "cohens_d",
            "group_means": group_means,
            "n_groups": n_groups,
        }

    stat, p = sp_stats.f_oneway(*groups)
    grand_mean = df[metric_col].mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    ss_total = float(((df[metric_col] - grand_mean) ** 2).sum())
    eta2 = ss_between / ss_total if ss_total > 0 else 0.0
    return {
        "test": "anova_f",
        "statistic": float(stat),
        "p_value": float(p),
        "effect_size": float(eta2),
        "effect_measure": "eta_squared",
        "group_means": group_means,
        "n_groups": n_groups,
    }


def cheatsheet() -> str:
    return "otis_group_compare({}) -> OTIS group comparison — t-test/ANOVA + effect size."
