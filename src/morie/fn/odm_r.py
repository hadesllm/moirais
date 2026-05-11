# morie.fn — function file (hadesllm/morie)
"""Demographic profile per region for OTIS correctional data."""

from __future__ import annotations

import pandas as pd


def otis_demo_region(
    df: pd.DataFrame,
    *,
    region_col: str = "region",
    age_col: str = "age_group",
    gender_col: str = "gender",
    alert_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Demographic profile summary per region.

    Computes N, age distribution, gender distribution, and alert
    prevalence for each region.

    Parameters
    ----------
    df : DataFrame
        Data with region, age, gender, and alert columns.
    region_col, age_col, gender_col : str
        Column names.
    alert_cols : list of str, optional
        Alert indicator columns. Defaults to mental health, suicide risk, watch.

    Returns
    -------
    DataFrame
        One row per region with demographic summary columns.
    """
    if alert_cols is None:
        alert_cols = [
            c for c in ["alert_mental_health", "alert_suicide_risk", "alert_suicide_watch"] if c in df.columns
        ]

    results = []
    for region, grp in df.groupby(region_col):
        row: dict = {"region": region, "n": len(grp)}

        # Age distribution
        if age_col in grp.columns:
            age_counts = grp[age_col].value_counts(normalize=True)
            for age, pct in age_counts.items():
                row[f"pct_{age}"] = round(float(pct), 4)

        # Gender distribution
        if gender_col in grp.columns:
            gen_counts = grp[gender_col].value_counts(normalize=True)
            for gen, pct in gen_counts.items():
                row[f"pct_{gen}"] = round(float(pct), 4)

        # Alert prevalence
        for ac in alert_cols:
            if ac in grp.columns:
                row[f"prev_{ac}"] = round(float(grp[ac].mean()), 4)

        results.append(row)

    return pd.DataFrame(results)


def cheatsheet() -> str:
    return "otis_demo_region({}) -> Demographic profile per region for OTIS correctional data."
