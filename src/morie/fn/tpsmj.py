"""Major crime indicators summary."""

from __future__ import annotations

import pandas as pd

from morie.fn._containers import DescriptiveResult


def tps_major_crime(
    df: pd.DataFrame,
    *,
    mci_col: str = "mci_category",
    count_col: str = "count",
) -> DescriptiveResult:
    """Summarise Major Crime Indicators (MCI).

    Parameters
    ----------
    df : DataFrame
    mci_col : str
        Column with MCI category (Assault, Break and Enter, Auto Theft, Robbery, Theft Over).
    count_col : str
        Column with counts.

    Returns
    -------
    DescriptiveResult
    """
    if mci_col not in df.columns:
        raise ValueError(f"Column '{mci_col}' not found")
    if count_col in df.columns:
        grouped = df.groupby(mci_col)[count_col].sum()
    else:
        grouped = df[mci_col].value_counts()
    total = grouped.sum()
    return DescriptiveResult(
        name="major_crime_indicators",
        value=float(total),
        extra={
            "categories": grouped.to_dict(),
            "proportions": (grouped / total).to_dict() if total > 0 else {},
            "n_categories": len(grouped),
        },
    )


tpsmj = tps_major_crime


def cheatsheet() -> str:
    return "tps_major_crime({}) -> Major crime indicators summary."
