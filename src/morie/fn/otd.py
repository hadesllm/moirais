# morie.fn -- function file (hadesllm/morie)
"""OTIS descriptive statistics suite."""

from __future__ import annotations

from typing import Any

import pandas as pd


def otdesc(
    df: pd.DataFrame,
    *,
    id_col: str = "unique_individual_id",
    year_col: str = "end_fiscal_year",
) -> dict[str, Any]:
    """Full OTIS descriptive statistics suite.

    Parameters
    ----------
    df : DataFrame
        Correctional placement data.
    id_col : str
        Unique individual identifier column.
    year_col : str
        Fiscal year column.

    Returns
    -------
    dict
        Keys: n_total, n_records, n_by_year, n_by_region, n_by_age,
        n_by_gender, placement_dist.
    """
    result: dict[str, Any] = {}
    result["n_total"] = df[id_col].nunique()
    result["n_records"] = len(df)

    result["n_by_year"] = df.groupby(year_col)[id_col].nunique().reset_index().rename(columns={id_col: "n"})

    if "region_at_time_of_placement" in df.columns:
        result["n_by_region"] = (
            df.groupby("region_at_time_of_placement")[id_col].nunique().reset_index().rename(columns={id_col: "n"})
        )

    if "age_category" in df.columns:
        result["n_by_age"] = df.groupby("age_category")[id_col].nunique().reset_index().rename(columns={id_col: "n"})

    if "gender" in df.columns:
        result["n_by_gender"] = df.groupby("gender")[id_col].nunique().reset_index().rename(columns={id_col: "n"})

    freq = df.groupby(id_col).size().reset_index(name="n_placements")
    result["placement_dist"] = freq["n_placements"].describe().to_dict()

    return result


def cheatsheet() -> str:
    return "otdesc({}) -> OTIS descriptive statistics suite."
