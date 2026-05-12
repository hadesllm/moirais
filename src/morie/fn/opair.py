# morie.fn -- function file (hadesllm/morie)
"""Pairwise group comparisons for OTIS correctional data."""

from __future__ import annotations

import pandas as pd
from scipy import stats


def otis_pairwise_compare(
    df: pd.DataFrame,
    *,
    metric: str = "Y",
    group_col: str = "region",
    method: str = "t",
) -> pd.DataFrame:
    """All pairwise group comparisons for a numeric metric.

    Computes pairwise t-tests (or Mann-Whitney) between every pair
    of groups with Bonferroni correction.

    Parameters
    ----------
    df : DataFrame
        Data with metric and group columns.
    metric : str
        Numeric column to compare.
    group_col : str
        Categorical column defining groups.
    method : str
        ``"t"`` for Welch t-test, ``"mw"`` for Mann-Whitney U.

    Returns
    -------
    DataFrame
        Columns: group_1, group_2, diff, statistic, pval, pval_adj, n1, n2.
    """
    data = df[[metric, group_col]].dropna()
    groups = sorted(data[group_col].unique())
    n_comparisons = len(groups) * (len(groups) - 1) // 2

    results = []
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            v1 = data.loc[data[group_col] == groups[i], metric].values
            v2 = data.loc[data[group_col] == groups[j], metric].values

            if len(v1) < 2 or len(v2) < 2:
                continue

            diff = float(v1.mean() - v2.mean())

            if method == "mw":
                stat, pval = stats.mannwhitneyu(v1, v2, alternative="two-sided")
            else:
                stat, pval = stats.ttest_ind(v1, v2, equal_var=False)

            pval_adj = min(float(pval) * n_comparisons, 1.0)

            results.append(
                {
                    "group_1": groups[i],
                    "group_2": groups[j],
                    "diff": round(diff, 4),
                    "statistic": round(float(stat), 4),
                    "pval": round(float(pval), 4),
                    "pval_adj": round(pval_adj, 4),
                    "n1": len(v1),
                    "n2": len(v2),
                }
            )

    return pd.DataFrame(results)


def cheatsheet() -> str:
    return "otis_pairwise_compare({}) -> Pairwise group comparisons for OTIS correctional data."
