"""The art of doing mathematics consists in finding that special case which contains all the germs of generality. — David Hilbert"""

from __future__ import annotations

import pandas as pd

from ._containers import DescriptiveResult


def sw_character_summary(
    df: pd.DataFrame,
    name_col: str = "name",
    height_col: str = "height",
    mass_col: str = "mass",
) -> DescriptiveResult:
    """Summary statistics for Star Wars character data.

    Parameters
    ----------
    df : pd.DataFrame
    name_col, height_col, mass_col : str

    Returns
    -------
    DescriptiveResult
    """
    result: dict = {"count": len(df)}
    for col_name, col_key in [(height_col, "height"), (mass_col, "mass")]:
        if col_name in df.columns:
            vals = pd.to_numeric(df[col_name], errors="coerce").dropna()
            result[f"mean_{col_key}"] = float(vals.mean()) if len(vals) > 0 else float("nan")
            result[f"std_{col_key}"] = float(vals.std()) if len(vals) > 1 else 0.0
            result[f"min_{col_key}"] = float(vals.min()) if len(vals) > 0 else float("nan")
            result[f"max_{col_key}"] = float(vals.max()) if len(vals) > 0 else float("nan")
    if name_col in df.columns:
        result["characters"] = df[name_col].tolist()
    return DescriptiveResult(name="SW character summary", value=result)


swchr = sw_character_summary


def cheatsheet() -> str:
    return "The art of doing mathematics consists in finding that special case which contains all the germs of generality. — David Hilbert"
