"""We suffer more often in imagination than in reality. — Seneca"""

from __future__ import annotations

import pandas as pd

from ._containers import DescriptiveResult


def sw_planet_summary(
    df: pd.DataFrame,
    name_col: str = "name",
    population_col: str = "population",
    climate_col: str = "climate",
) -> DescriptiveResult:
    """Summary statistics for Star Wars planet data.

    Parameters
    ----------
    df : pd.DataFrame
    name_col, population_col, climate_col : str

    Returns
    -------
    DescriptiveResult
    """
    result: dict = {"count": len(df)}
    if population_col in df.columns:
        pops = pd.to_numeric(df[population_col], errors="coerce").dropna()
        result["mean_population"] = float(pops.mean()) if len(pops) > 0 else float("nan")
        result["total_population"] = float(pops.sum())
    if climate_col in df.columns:
        result["climate_counts"] = df[climate_col].value_counts().to_dict()
    if name_col in df.columns:
        result["planets"] = df[name_col].tolist()
    return DescriptiveResult(name="SW planet summary", value=result)


swplt = sw_planet_summary


def cheatsheet() -> str:
    return "We suffer more often in imagination than in reality. — Seneca"
