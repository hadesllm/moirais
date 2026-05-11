# morie.fn — function file (hadesllm/morie)
"""GATE by age group for OTIS correctional data."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def otis_gate_age(df: pd.DataFrame, cdf=None, *, outcome: str = "Y", treatment: str = "D", age_col: str = "age_group") -> pd.DataFrame:
    """Estimate Group Average Treatment Effect (GATE) by age group.

    Simple difference-in-means within each age stratum, with
    heteroskedasticity-robust standard errors.

    Parameters
    ----------
    df : DataFrame
        Data with outcome, treatment, and age group columns.
    outcome, treatment, age_col : str
        Column names.

    Returns
    -------
    DataFrame
        Columns: age_group, gate, se, pval, ci_lower, ci_upper, n1, n0.
    """
    results = []
    for grp_name, grp in df.groupby(age_col):
        data = grp[[outcome, treatment]].dropna()
        t1 = data.loc[data[treatment] == 1, outcome]
        t0 = data.loc[data[treatment] == 0, outcome]

        if len(t1) < 2 or len(t0) < 2:
            continue

        gate = float(t1.mean() - t0.mean())
        se = float(np.sqrt(t1.var() / len(t1) + t0.var() / len(t0)))
        z = gate / se if se > 0 else 0.0
        pval = float(2 * (1 - stats.norm.cdf(abs(z))))

        results.append(
            {
                "age_group": grp_name,
                "gate": gate,
                "se": se,
                "pval": pval,
                "ci_lower": gate - 1.96 * se,
                "ci_upper": gate + 1.96 * se,
                "n1": len(t1),
                "n0": len(t0),
            }
        )

    return pd.DataFrame(results)


def cheatsheet() -> str:
    return "otis_gate_age({}) -> GATE by age group for OTIS correctional data."
