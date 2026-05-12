# morie.fn -- function file (hadesllm/morie)
"""Demographic profile per fiscal year for OTIS correctional data."""

from __future__ import annotations

import pandas as pd


def otis_demo_year(
    df: pd.DataFrame,
    *,
    year_col: str = "end_fiscal_year",
    region_col: str = "region",
    gender_col: str = "gender",
    outcome_col: str = "Y",
) -> pd.DataFrame:
    """Demographic profile summary per fiscal year.

    Parameters
    ----------
    df : DataFrame
        Data with year, region, gender, and outcome columns.
    year_col, region_col, gender_col, outcome_col : str
        Column names.

    Returns
    -------
    DataFrame
        One row per year with N, region distribution, gender split,
        and outcome mean/sd.
    """
    results = []
    for year, grp in df.groupby(year_col):
        row: dict = {"year": year, "n": len(grp)}

        if region_col in grp.columns:
            for reg, pct in grp[region_col].value_counts(normalize=True).items():
                row[f"pct_{reg}"] = round(float(pct), 4)

        if gender_col in grp.columns:
            for gen, pct in grp[gender_col].value_counts(normalize=True).items():
                row[f"pct_{gen}"] = round(float(pct), 4)

        if outcome_col in grp.columns:
            row["outcome_mean"] = round(float(grp[outcome_col].mean()), 4)
            row["outcome_sd"] = round(float(grp[outcome_col].std()), 4)

        results.append(row)

    return pd.DataFrame(results)


def cheatsheet() -> str:
    return "otis_demo_year({}) -> Demographic profile per fiscal year for OTIS correctional da"
