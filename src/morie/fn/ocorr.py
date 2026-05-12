# morie.fn -- function file (hadesllm/morie)
"""Correlation matrix for numeric columns in OTIS data."""

from __future__ import annotations

import pandas as pd


def otis_correlation(
    df: pd.DataFrame,
    *,
    cols: list[str] | None = None,
    method: str = "pearson",
) -> pd.DataFrame:
    """Compute correlation matrix for numeric columns.

    Parameters
    ----------
    df : DataFrame
        Data with numeric columns.
    cols : list of str, optional
        Columns to include. Defaults to all numeric columns.
    method : str
        Correlation method: ``"pearson"``, ``"spearman"``, or ``"kendall"``.

    Returns
    -------
    DataFrame
        Symmetric correlation matrix.
    """
    if cols is not None:
        data = df[cols].select_dtypes(include="number")
    else:
        data = df.select_dtypes(include="number")

    return data.corr(method=method)


def cheatsheet() -> str:
    return "otis_correlation({}) -> Correlation matrix for numeric columns in OTIS data."
