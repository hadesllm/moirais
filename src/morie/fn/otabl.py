# morie.fn — function file (hadesllm/morie)
"""Table 1 (baseline characteristics by group) for OTIS data."""

from __future__ import annotations

import pandas as pd
from scipy import stats


def otis_table1(
    df: pd.DataFrame,
    *,
    group_col: str = "D",
) -> pd.DataFrame:
    """Generate Table 1: baseline characteristics by treatment group.

    Numeric columns: mean (sd) per group + t-test p-value.
    Categorical columns: n (%) per group + chi-squared p-value.

    Parameters
    ----------
    df : DataFrame
        Data with a binary group column and baseline characteristics.
    group_col : str
        Column that defines the two groups (e.g. treatment indicator).

    Returns
    -------
    DataFrame
        Columns: variable, group_0, group_1, pval, test.
    """
    groups = sorted(df[group_col].dropna().unique())
    if len(groups) < 2:
        return pd.DataFrame(columns=["variable", "group_0", "group_1", "pval", "test"])

    g0 = df[df[group_col] == groups[0]]
    g1 = df[df[group_col] == groups[1]]

    rows = []
    for col in df.columns:
        if col == group_col:
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            v0 = g0[col].dropna()
            v1 = g1[col].dropna()
            if len(v0) < 2 or len(v1) < 2:
                continue
            m0 = f"{v0.mean():.2f} ({v0.std():.2f})"
            m1 = f"{v1.mean():.2f} ({v1.std():.2f})"
            _, pval = stats.ttest_ind(v0, v1, equal_var=False)
            rows.append(
                {
                    "variable": col,
                    "group_0": m0,
                    "group_1": m1,
                    "pval": round(float(pval), 4),
                    "test": "t-test",
                }
            )
        else:
            ct = pd.crosstab(df[col], df[group_col])
            if ct.shape[0] < 2 or ct.shape[1] < 2:
                continue
            chi2, pval, _, _ = stats.chi2_contingency(ct)
            n0 = len(g0)
            n1 = len(g1)
            levels = ct.index.tolist()
            for lev in levels:
                c0 = int(ct.loc[lev, groups[0]]) if groups[0] in ct.columns else 0
                c1 = int(ct.loc[lev, groups[1]]) if groups[1] in ct.columns else 0
                rows.append(
                    {
                        "variable": f"{col}={lev}",
                        "group_0": f"{c0} ({100 * c0 / n0:.1f}%)" if n0 > 0 else "0",
                        "group_1": f"{c1} ({100 * c1 / n1:.1f}%)" if n1 > 0 else "0",
                        "pval": round(float(pval), 4),
                        "test": "chi2",
                    }
                )

    return pd.DataFrame(rows)


def cheatsheet() -> str:
    return "otis_table1({}) -> Table 1 (baseline characteristics by group) for OTIS data."
