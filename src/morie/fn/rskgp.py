# morie.fn -- function file (hadesllm/morie)
"""Profile of risk groups by demographics and offenses."""

from __future__ import annotations

import pandas as pd

from morie.fn._containers import DescriptiveResult


def risk_group_profile(
    df: pd.DataFrame,
    *,
    risk_col: str = "risk_level",
    profile_cols: list[str] | None = None,
) -> DescriptiveResult:
    """Profile of risk groups across covariates.

    Parameters
    ----------
    df : DataFrame
    risk_col : str
        Risk level column.
    profile_cols : list of str, optional
        Columns to profile. If None, uses all non-risk columns.

    Returns
    -------
    DescriptiveResult
        value is DataFrame of group profiles.
    """
    if profile_cols is None:
        profile_cols = [c for c in df.columns if c != risk_col]
    profiles = {}
    for g in df[risk_col].unique():
        sub = df[df[risk_col] == g]
        prof = {}
        for col in profile_cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                prof[col] = {"mean": float(sub[col].mean()), "std": float(sub[col].std())}
            else:
                prof[col] = sub[col].value_counts(normalize=True).to_dict()
        profiles[str(g)] = prof
    return DescriptiveResult(
        name="risk_group_profile",
        value=None,
        extra={"profiles": profiles, "n_groups": len(profiles)},
    )


rskgp = risk_group_profile


def cheatsheet() -> str:
    return "risk_group_profile({}) -> Profile of risk groups by demographics and offenses."
