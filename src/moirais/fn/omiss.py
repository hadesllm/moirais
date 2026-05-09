# moirais.fn — function file (hadesllm/moirais)
"""Missing data report for OTIS correctional data."""

from __future__ import annotations

import pandas as pd


def otis_missing_report(
    df: pd.DataFrame,
    *,
    sort_by: str = "pct_missing",
) -> pd.DataFrame:
    """Generate a missing-data report for every column.

    Parameters
    ----------
    df : DataFrame
        Any tabular data.
    sort_by : str
        Sort column: ``"pct_missing"`` (default) or ``"n_missing"``.

    Returns
    -------
    DataFrame
        Columns: column, n_total, n_missing, pct_missing, dtype.
    """
    rows = []
    for col in df.columns:
        n_total = len(df)
        n_miss = int(df[col].isna().sum())
        pct_miss = round(n_miss / n_total, 4) if n_total > 0 else 0.0
        rows.append(
            {
                "column": col,
                "n_total": n_total,
                "n_missing": n_miss,
                "pct_missing": pct_miss,
                "dtype": str(df[col].dtype),
            }
        )

    result = pd.DataFrame(rows)
    if sort_by in result.columns:
        result = result.sort_values(sort_by, ascending=False).reset_index(drop=True)
    return result


def cheatsheet() -> str:
    return "otis_missing_report({}) -> Missing data report for OTIS correctional data."
