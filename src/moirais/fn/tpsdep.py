"""Officer deployment analysis by division."""

from __future__ import annotations

import pandas as pd

from moirais.fn._containers import DescriptiveResult


def tps_deployment(
    df: pd.DataFrame,
    *,
    division_col: str = "division",
    count_col: str = "officer_count",
) -> DescriptiveResult:
    """Analyse officer deployment across divisions.

    Parameters
    ----------
    df : DataFrame
        Deployment data.
    division_col : str
        Column with division identifiers.
    count_col : str
        Column with officer counts.

    Returns
    -------
    DescriptiveResult
    """
    if division_col not in df.columns:
        raise ValueError(f"Column '{division_col}' not found")
    if count_col not in df.columns:
        raise ValueError(f"Column '{count_col}' not found")
    grouped = df.groupby(division_col)[count_col].sum()
    total = grouped.sum()
    props = (grouped / total).to_dict() if total > 0 else {}
    return DescriptiveResult(
        name="deployment_analysis",
        value=float(total),
        extra={
            "divisions": grouped.to_dict(),
            "proportions": props,
            "n_divisions": len(grouped),
            "mean_per_division": float(grouped.mean()),
            "std_per_division": float(grouped.std()) if len(grouped) > 1 else 0.0,
        },
    )


tpsdep = tps_deployment


def cheatsheet() -> str:
    return "tps_deployment({}) -> Officer deployment analysis by division."
