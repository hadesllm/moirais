# moirais.fn — function file (hadesllm/moirais)
"""Demographic profile per gender for OTIS correctional data."""

from __future__ import annotations

import pandas as pd


def otis_demo_gender(
    df: pd.DataFrame,
    *,
    gender_col: str = "gender",
    age_col: str = "age_group",
    region_col: str = "region",
    outcome_col: str = "Y",
) -> pd.DataFrame:
    """Demographic profile summary per gender.

    Parameters
    ----------
    df : DataFrame
        Data with gender, age, region, and outcome columns.
    gender_col, age_col, region_col, outcome_col : str
        Column names.

    Returns
    -------
    DataFrame
        One row per gender with N, age distribution, region distribution,
        and outcome mean/sd.
    """
    results = []
    for gen, grp in df.groupby(gender_col):
        row: dict = {"gender": gen, "n": len(grp)}

        if age_col in grp.columns:
            for age, pct in grp[age_col].value_counts(normalize=True).items():
                row[f"pct_{age}"] = round(float(pct), 4)

        if region_col in grp.columns:
            for reg, pct in grp[region_col].value_counts(normalize=True).items():
                row[f"pct_{reg}"] = round(float(pct), 4)

        if outcome_col in grp.columns:
            row["outcome_mean"] = round(float(grp[outcome_col].mean()), 4)
            row["outcome_sd"] = round(float(grp[outcome_col].std()), 4)

        results.append(row)

    return pd.DataFrame(results)


def cheatsheet() -> str:
    return "otis_demo_gender({}) -> Demographic profile per gender for OTIS correctional data."
