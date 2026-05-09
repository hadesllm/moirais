"""Compare crime rates across divisions."""

from __future__ import annotations

import pandas as pd

from moirais.fn._containers import DescriptiveResult


def tps_division_compare(
    df: pd.DataFrame,
    *,
    division_col: str = "division",
    rate_col: str = "crime_rate",
) -> DescriptiveResult:
    """Compare crime rates across police divisions.

    Parameters
    ----------
    df : DataFrame
    division_col : str
    rate_col : str

    Returns
    -------
    DescriptiveResult
    """
    if division_col not in df.columns or rate_col not in df.columns:
        raise ValueError("Required columns not found")
    grouped = df.groupby(division_col)[rate_col].agg(["mean", "std", "count"])
    ranked = grouped.sort_values("mean", ascending=False)
    return DescriptiveResult(
        name="division_comparison",
        value=float(grouped["mean"].std()) if len(grouped) > 1 else 0.0,
        extra={
            "division_stats": grouped.to_dict("index"),
            "highest": ranked.index[0],
            "lowest": ranked.index[-1],
            "n_divisions": len(grouped),
            "cv": float(grouped["mean"].std() / grouped["mean"].mean()) if grouped["mean"].mean() > 0 else 0.0,
        },
    )


tpsdiv = tps_division_compare


def cheatsheet() -> str:
    return "tps_division_compare({}) -> Compare crime rates across divisions."
