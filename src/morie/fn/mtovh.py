# morie.fn -- function file (hadesllm/morie)
"""Crash rates by vehicle type."""

from __future__ import annotations

import pandas as pd

from morie.fn._containers import DescriptiveResult


def mto_vehicle_type(
    df: pd.DataFrame,
    *,
    vehicle_col: str = "vehicle_type",
    crash_col: str = "n_crashes",
) -> DescriptiveResult:
    """Compute crash rates by vehicle type.

    Parameters
    ----------
    df : DataFrame
    vehicle_col, crash_col : str

    Returns
    -------
    DescriptiveResult
    """
    if vehicle_col not in df.columns:
        raise ValueError(f"Column '{vehicle_col}' not found")
    if crash_col in df.columns:
        grouped = df.groupby(vehicle_col)[crash_col].sum()
    else:
        grouped = df[vehicle_col].value_counts()
    total = grouped.sum()
    return DescriptiveResult(
        name="vehicle_type_crashes",
        value=float(total),
        extra={"by_type": grouped.to_dict(), "proportions": (grouped / total).to_dict() if total > 0 else {}},
    )


mtovh = mto_vehicle_type


def cheatsheet() -> str:
    return "mto_vehicle_type({}) -> Crash rates by vehicle type."
