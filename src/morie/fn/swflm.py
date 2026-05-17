"""Summary statistics for Solar System mission data."""

from __future__ import annotations

import pandas as pd

from ._containers import DescriptiveResult


def solar_mission_summary(
    df: pd.DataFrame,
    name_col: str = "name",
    year_col: str = "launch_year",
) -> DescriptiveResult:
    """Summary statistics for Solar System mission data.

    Parameters
    ----------
    df : pd.DataFrame
    name_col, year_col : str

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
    if name_col in df.columns:
        result["missions"] = df[name_col].tolist()
    return DescriptiveResult(name="Solar System mission summary", value=result)


swflm = solar_mission_summary


def cheatsheet() -> str:
    return 'swflm() -> Summary statistics for Solar System mission data'
