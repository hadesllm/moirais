"""Regional volatility / movement metric."""

from __future__ import annotations

import pandas as pd

from moirais.fn._containers import VolRes


def volat(
    df: pd.DataFrame,
    *,
    id_col: str = "unique_individual_id",
    year_col: str = "end_fiscal_year",
    regA_col: str = "region_at_time_of_placement",
    regB_col: str = "region_most_recent_placement",
) -> VolRes:
    """Regional volatility / movement metric.

    Counts the number of distinct regions an individual was placed in
    (combining initial and most recent placement regions).

    Parameters
    ----------
    df : DataFrame
        Correctional placement data.
    id_col : str
        Unique individual identifier column.
    year_col : str
        Fiscal year column.
    regA_col : str
        Region at time of placement column.
    regB_col : str
        Region of most recent placement column.

    Returns
    -------
    VolRes
        Per-person volatility data with mean and median.
    """

    def _count_regions(group: pd.DataFrame) -> int:
        regions = set(group[regA_col].dropna()) | set(group[regB_col].dropna())
        return len(regions)

    vm = df.groupby([id_col, year_col]).apply(_count_regions, include_groups=False).reset_index()
    vm.columns = [id_col, year_col, "vm"]

    return VolRes(
        data=vm,
        mean=float(vm["vm"].mean()),
        median=float(vm["vm"].median()),
    )


def cheatsheet() -> str:
    return "volat({}) -> Regional volatility / movement metric."
