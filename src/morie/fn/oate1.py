# morie.fn — function file (hadesllm/morie)
"""Simple ATE by region (difference in means) for OTIS data."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def otis_ate_region(df: pd.DataFrame, cdf=None, *, outcome: str = "Y", treatment: str = "D", region_col: str = "region") -> pd.DataFrame:
    """Estimate naive ATE (difference in means) within each region.

    Parameters
    ----------
    df : DataFrame
        Data with outcome, treatment, and region columns.
    outcome, treatment, region_col : str
        Column names.

    Returns
    -------
    DataFrame
        Columns: region, ate, se, pval, ci_lower, ci_upper, n1, n0.
    """
    results = []
    for region, grp in df.groupby(region_col):
        data = grp[[outcome, treatment]].dropna()
        treated = data.loc[data[treatment] == 1, outcome]
        control = data.loc[data[treatment] == 0, outcome]

        if len(treated) < 2 or len(control) < 2:
            continue

        ate = float(treated.mean() - control.mean())
        se = float(np.sqrt(treated.var() / len(treated) + control.var() / len(control)))
        z = ate / se if se > 0 else 0.0
        pval = float(2 * (1 - stats.norm.cdf(abs(z))))

        results.append(
            {
                "region": region,
                "ate": ate,
                "se": se,
                "pval": pval,
                "ci_lower": ate - 1.96 * se,
                "ci_upper": ate + 1.96 * se,
                "n1": len(treated),
                "n0": len(control),
            }
        )

    return pd.DataFrame(results)


def cheatsheet() -> str:
    return "otis_ate_region({}) -> Simple ATE by region (difference in means) for OTIS data."
