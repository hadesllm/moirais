"""Demographics in SIU cases."""

from __future__ import annotations

import pandas as pd

from moirais.fn._containers import DescriptiveResult


def siu_demographics(
    df: pd.DataFrame,
    *,
    age_col: str = "age",
    gender_col: str = "gender",
    race_col: str = "race",
) -> DescriptiveResult:
    """Analyse demographics of affected persons in SIU cases.

    Parameters
    ----------
    df : DataFrame
    age_col, gender_col, race_col : str

    Returns
    -------
    DescriptiveResult
    """
    extra = {"n": len(df)}
    if age_col in df.columns:
        ages = pd.to_numeric(df[age_col], errors="coerce").dropna()
        if len(ages) > 0:
            extra["mean_age"] = float(ages.mean())
            extra["median_age"] = float(ages.median())
    if gender_col in df.columns:
        extra["gender_dist"] = df[gender_col].value_counts().to_dict()
    if race_col in df.columns:
        extra["race_dist"] = df[race_col].value_counts().to_dict()
    return DescriptiveResult(name="siu_demographics", value=float(len(df)), extra=extra)


siudem = siu_demographics


def cheatsheet() -> str:
    return "siu_demographics({}) -> Demographics in SIU cases."
