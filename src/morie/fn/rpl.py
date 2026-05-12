# morie.fn -- function file (hadesllm/morie)
"""Regional placement analysis by age group."""

from __future__ import annotations

import pandas as pd

from morie.fn._containers import RplRes
from morie.fn._otis_const import REGIONS


def rplace(
    df: pd.DataFrame,
    year: int,
    sex: str | None = None,
    *,
    id_col: str = "unique_individual_id",
    age_col: str = "age_category",
    region_col: str = "region_at_time_of_placement",
    year_col: str = "end_fiscal_year",
    gender_col: str = "gender",
) -> RplRes:
    """Regional placement analysis by age group.

    Computes count matrix and proportion matrix of placements across
    regions, stratified by age group and optionally by sex.

    Parameters
    ----------
    df : DataFrame
        Correctional placement data.
    year : int
        Fiscal year to analyze.
    sex : str, optional
        Filter by gender (e.g. ``"Male"`` or ``"Female"``).
    id_col : str
        Column with unique individual identifiers.
    age_col : str
        Column with age category labels.
    region_col : str
        Column with region at time of placement.
    year_col : str
        Column with fiscal year.
    gender_col : str
        Column with gender labels.

    Returns
    -------
    RplRes
        Counts and proportions matrices with metadata.
    """
    mask = (df[year_col] == year) & df[age_col].notna() & df[region_col].notna()
    sub = df[mask].copy()
    if sex is not None:
        sub = sub[sub[gender_col] == sex]

    counts = sub.groupby([age_col, region_col])[id_col].nunique().reset_index()
    counts.columns = ["age", "region", "n"]

    pivot = counts.pivot_table(index="age", columns="region", values="n", fill_value=0)
    for r in REGIONS:
        if r not in pivot.columns:
            pivot[r] = 0
    pivot = pivot[REGIONS]

    row_sums = pivot.sum(axis=1)
    props = pivot.div(row_sums, axis=0).fillna(0)

    return RplRes(counts=pivot, props=props, year=year, sex=sex)


def cheatsheet() -> str:
    return "rplace({}) -> Regional placement analysis by age group."
