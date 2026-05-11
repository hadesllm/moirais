# morie.fn — function file (hadesllm/morie)
"""OTIS proportion test across groups."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


def otis_proportions(
    df: pd.DataFrame,
    *,
    col: str = "D",
    group_col: str = "region",
) -> dict:
    """Chi-square test of proportions for a binary column across groups.

    Parameters
    ----------
    df : DataFrame
        Correctional records.
    col : str
        Binary column (1 = positive).
    group_col : str
        Grouping column.

    Returns
    -------
    dict
        Keys: ``chi2``, ``p_value``, ``dof``, ``group_proportions`` (dict),
        ``overall_proportion``.
    """
    grp = df.groupby(group_col)[col].agg(n="count", positive="sum")
    observed = np.array([grp["positive"].values, (grp["n"] - grp["positive"]).values])
    chi2, p, dof, _ = sp_stats.chi2_contingency(observed)
    props = (grp["positive"] / grp["n"]).to_dict()
    overall = float(df[col].mean())
    return {
        "chi2": float(chi2),
        "p_value": float(p),
        "dof": int(dof),
        "group_proportions": props,
        "overall_proportion": overall,
    }


def cheatsheet() -> str:
    return "otis_proportions({}) -> OTIS proportion test across groups."
