# morie.fn — function file (hadesllm/morie)
"""Demographic profile per age group for OTIS correctional data."""

from __future__ import annotations

import pandas as pd


def otis_demo_age(
    df: pd.DataFrame,
    *,
    age_col: str = "age_group",
    gender_col: str = "gender",
    region_col: str = "region",
    outcome_col: str = "Y",
) -> pd.DataFrame:
    """Demographic profile summary per age group.

    Parameters
    ----------
    df : DataFrame
        Data with age, gender, region, and outcome columns.
    age_col, gender_col, region_col, outcome_col : str
        Column names.

    Returns
    -------
    DataFrame
        One row per age group with N, gender split, region distribution,
        and outcome mean/sd.
    """
    results = []
    for age, grp in df.groupby(age_col):
        row: dict = {"age_group": age, "n": len(grp)}

        if gender_col in grp.columns:
            for gen, pct in grp[gender_col].value_counts(normalize=True).items():
                row[f"pct_{gen}"] = round(float(pct), 4)

        if region_col in grp.columns:
            for reg, pct in grp[region_col].value_counts(normalize=True).items():
                row[f"pct_{reg}"] = round(float(pct), 4)

        if outcome_col in grp.columns:
            row["outcome_mean"] = round(float(grp[outcome_col].mean()), 4)
            row["outcome_sd"] = round(float(grp[outcome_col].std()), 4)

        results.append(row)

    return pd.DataFrame(results)


def cheatsheet() -> str:
    return "otis_demo_age({}) -> Demographic profile per age group for OTIS correctional data"
