"""Confine yourself to the present. -- Marcus Aurelius"""

from __future__ import annotations

import pandas as pd

from ._containers import DescriptiveResult


def sw_film_summary(
    df: pd.DataFrame,
    title_col: str = "title",
    year_col: str = "release_year",
) -> DescriptiveResult:
    """Summary statistics for Star Wars film data.

    Parameters
    ----------
    df : pd.DataFrame
    title_col, year_col : str

    Returns
    -------
    DescriptiveResult
    """
    result: dict = {"count": len(df)}
    if year_col in df.columns:
        years = pd.to_numeric(df[year_col], errors="coerce").dropna()
        result["earliest"] = int(years.min()) if len(years) > 0 else None
        result["latest"] = int(years.max()) if len(years) > 0 else None
        result["span_years"] = int(years.max() - years.min()) if len(years) > 1 else 0
    if title_col in df.columns:
        result["titles"] = df[title_col].tolist()
    return DescriptiveResult(name="SW film summary", value=result)


swflm = sw_film_summary


def cheatsheet() -> str:
    return "Confine yourself to the present. -- Marcus Aurelius"
