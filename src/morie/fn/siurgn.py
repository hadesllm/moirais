"""SIU cases by geographic region."""

from __future__ import annotations

import pandas as pd

from morie.fn._containers import DescriptiveResult


def siu_by_region(
    df: pd.DataFrame,
    *,
    region_col: str = "region",
) -> DescriptiveResult:
    """Analyse SIU cases by geographic region.

    Parameters
    ----------
    df : DataFrame
    region_col : str

    Returns
    -------
    DescriptiveResult
    """
    if region_col not in df.columns:
        raise ValueError(f"Column '{region_col}' not found")
    counts = df[region_col].value_counts()
    total = counts.sum()
    return DescriptiveResult(
        name="siu_by_region",
        value=float(total),
        extra={
            "counts": counts.to_dict(),
            "proportions": (counts / total).to_dict() if total > 0 else {},
            "n_regions": len(counts),
        },
    )


siurgn = siu_by_region


def cheatsheet() -> str:
    return "siu_by_region({}) -> SIU cases by geographic region."
