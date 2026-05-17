"""Summary statistics for Solar System orbital data."""

from __future__ import annotations

import pandas as pd

from ._containers import DescriptiveResult


def solar_orbit_summary(
    df: pd.DataFrame,
    name_col: str = "name",
    period_col: str = "orbital_period_days",
    moons_col: str = "moons",
) -> DescriptiveResult:
    """Summary statistics for Solar System orbital data.

    Parameters
    ----------
    df : pd.DataFrame
    name_col, period_col, moons_col : str

    Returns
    -------
    DescriptiveResult
    """
    result: dict = {"count": len(df)}
    if period_col in df.columns:
        periods = pd.to_numeric(df[period_col], errors="coerce").dropna()
        result["mean_period_days"] = float(periods.mean()) if len(periods) > 0 else float("nan")
        result["min_period_days"] = float(periods.min()) if len(periods) > 0 else float("nan")
        result["max_period_days"] = float(periods.max()) if len(periods) > 0 else float("nan")
    if moons_col in df.columns:
        moons = pd.to_numeric(df[moons_col], errors="coerce").dropna()
        result["total_moons"] = int(moons.sum())
    if name_col in df.columns:
        result["bodies"] = df[name_col].tolist()
    return DescriptiveResult(name="Solar System orbit summary", value=result)


swplt = solar_orbit_summary


def cheatsheet() -> str:
    return 'swplt() -> Summary statistics for Solar System orbital data'
