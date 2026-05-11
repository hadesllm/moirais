# morie.fn — function file (hadesllm/morie)
"""Youth court special metrics (YCJA compliance)."""

from __future__ import annotations

import pandas as pd

from morie.fn._containers import DescriptiveResult


def court_youth(
    df: pd.DataFrame,
    *,
    age_col: str = "age",
    sentence_col: str = "sentence_type",
    days_col: str = "days_to_disposition",
) -> DescriptiveResult:
    """Analyse youth court metrics under Youth Criminal Justice Act.

    Parameters
    ----------
    df : DataFrame
    age_col : str
    sentence_col : str
    days_col : str

    Returns
    -------
    DescriptiveResult
    """
    extra = {"n": len(df)}
    if age_col in df.columns:
        ages = pd.to_numeric(df[age_col], errors="coerce").dropna()
        extra["mean_age"] = float(ages.mean()) if len(ages) > 0 else None
    if sentence_col in df.columns:
        extra["sentence_dist"] = df[sentence_col].value_counts().to_dict()
        extra["pct_custodial"] = float(df[sentence_col].str.lower().str.contains("custod", na=False).mean())
    if days_col in df.columns:
        days = pd.to_numeric(df[days_col], errors="coerce").dropna()
        if len(days) > 0:
            extra["median_days"] = float(days.median())
            extra["pct_over_180"] = float((days > 180).mean())
    return DescriptiveResult(name="youth_court", value=float(len(df)), extra=extra)


crtya = court_youth


def cheatsheet() -> str:
    return "court_youth({}) -> Youth court special metrics (YCJA compliance)."
