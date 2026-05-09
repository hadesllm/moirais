"""Neighbourhood-level crime profile."""

from __future__ import annotations

import pandas as pd

from moirais.fn._containers import DescriptiveResult


def tps_neighborhood(
    df: pd.DataFrame,
    *,
    neighborhood_col: str = "neighbourhood",
    crime_col: str = "crime_type",
) -> DescriptiveResult:
    """Profile crime by neighbourhood.

    Parameters
    ----------
    df : DataFrame
    neighborhood_col : str
    crime_col : str

    Returns
    -------
    DescriptiveResult
    """
    if neighborhood_col not in df.columns:
        raise ValueError(f"Column '{neighborhood_col}' not found")
    counts = df.groupby(neighborhood_col).size()
    top5 = counts.nlargest(5).to_dict()
    return DescriptiveResult(
        name="neighbourhood_profile",
        value=float(len(counts)),
        extra={
            "total_incidents": int(counts.sum()),
            "n_neighbourhoods": len(counts),
            "top5": top5,
            "mean_per_neighbourhood": float(counts.mean()),
        },
    )


tpsnb = tps_neighborhood


def cheatsheet() -> str:
    return "tps_neighborhood({}) -> Neighbourhood-level crime profile."
