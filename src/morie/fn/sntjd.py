"""Judicial variation in sentencing."""

from __future__ import annotations

import numpy as np
import pandas as pd

from morie.fn._containers import DescriptiveResult


def sentence_judicial(
    df: pd.DataFrame,
    *,
    judge_col: str = "judge_id",
    sentence_col: str = "sentence_days",
) -> DescriptiveResult:
    """Judicial variation: ICC and variance decomposition.

    Parameters
    ----------
    df : DataFrame
    judge_col : str
    sentence_col : str

    Returns
    -------
    DescriptiveResult
        extra: icc, between_var, within_var.
    """
    grand_mean = float(df[sentence_col].mean())
    groups = df.groupby(judge_col)[sentence_col]
    group_means = groups.mean()
    group_counts = groups.count()
    k = len(group_means)
    between_var = float(np.sum(group_counts * (group_means - grand_mean) ** 2) / max(k - 1, 1))
    within_var = float(groups.var().mean()) if k > 0 else 0.0
    icc = between_var / (between_var + within_var) if (between_var + within_var) > 0 else 0.0
    return DescriptiveResult(
        name="sentence_judicial",
        value=float(icc),
        extra={
            "icc": float(icc),
            "between_var": between_var,
            "within_var": within_var,
            "n_judges": k,
            "grand_mean": grand_mean,
        },
    )


sntjd = sentence_judicial


def cheatsheet() -> str:
    return "sentence_judicial({}) -> Judicial variation in sentencing."
