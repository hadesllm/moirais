# morie.fn -- function file (hadesllm/morie)
"""Driver risk profile by age/gender."""

from __future__ import annotations

import pandas as pd

from morie.fn._containers import DescriptiveResult


def mto_driver_risk(
    df: pd.DataFrame,
    *,
    age_col: str = "age_group",
    gender_col: str = "gender",
    crash_col: str = "crash",
) -> DescriptiveResult:
    """Compute crash rates by driver age group and gender.

    Parameters
    ----------
    df : DataFrame
    age_col, gender_col, crash_col : str

    Returns
    -------
    DescriptiveResult
    """
    for c in [age_col, gender_col, crash_col]:
        if c not in df.columns:
            raise ValueError(f"Column '{c}' not found")
    rates = df.groupby([age_col, gender_col])[crash_col].mean()
    return DescriptiveResult(
        name="driver_risk_profile",
        value=float(df[crash_col].mean()),
        extra={"group_rates": rates.to_dict(), "n": len(df)},
    )


mtodr = mto_driver_risk


def cheatsheet() -> str:
    return "mto_driver_risk({}) -> Driver risk profile by age/gender."
