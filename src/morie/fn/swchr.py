"""Summary statistics for Solar System body data."""

from __future__ import annotations

import pandas as pd

from ._containers import DescriptiveResult


def solar_body_summary(
    df: pd.DataFrame,
    name_col: str = "name",
    mass_col: str = "mass_earths",
    radius_col: str = "radius_km",
) -> DescriptiveResult:
    """Summary statistics for Solar System body data.

    Parameters
    ----------
    df : pd.DataFrame
    name_col, mass_col, radius_col : str

    Returns
    -------
    DescriptiveResult
    """
    result: dict = {"count": len(df)}
    for col_name, col_key in [(mass_col, "mass"), (radius_col, "radius")]:
        if col_name in df.columns:
            vals = pd.to_numeric(df[col_name], errors="coerce").dropna()
            result[f"mean_{col_key}"] = float(vals.mean()) if len(vals) > 0 else float("nan")
            result[f"std_{col_key}"] = float(vals.std()) if len(vals) > 1 else 0.0
            result[f"min_{col_key}"] = float(vals.min()) if len(vals) > 0 else float("nan")
            result[f"max_{col_key}"] = float(vals.max()) if len(vals) > 0 else float("nan")
    if name_col in df.columns:
        result["bodies"] = df[name_col].tolist()
    return DescriptiveResult(name="Solar System body summary", value=result)


swchr = solar_body_summary


def cheatsheet() -> str:
    return 'swchr() -> Summary statistics for Solar System body data'
