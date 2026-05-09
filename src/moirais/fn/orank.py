# moirais.fn — function file (hadesllm/moirais)
"""OTIS rank regions by metric."""

from __future__ import annotations

import pandas as pd


def otis_rank_regions(
    df: pd.DataFrame,
    *,
    metric_col: str = "Y",
    region_col: str = "region",
) -> pd.DataFrame:
    """Rank regions by mean value of a metric (descending).

    Parameters
    ----------
    df : DataFrame
        Correctional records.
    metric_col : str
        Numeric metric column.
    region_col : str
        Region column.

    Returns
    -------
    DataFrame
        Columns: ``[region_col, 'mean', 'std', 'n', 'rank']``.
    """
    grp = df.groupby(region_col)[metric_col].agg(mean="mean", std="std", n="count").reset_index()
    grp = grp.sort_values("mean", ascending=False).reset_index(drop=True)
    grp["rank"] = range(1, len(grp) + 1)
    return grp


def cheatsheet() -> str:
    return "otis_rank_regions({}) -> OTIS rank regions by metric."
