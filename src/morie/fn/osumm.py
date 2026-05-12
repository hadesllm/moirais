# morie.fn -- function file (hadesllm/morie)
"""Full summary table for OTIS correctional data."""

from __future__ import annotations

import numpy as np
import pandas as pd


def otis_summary_table(
    df: pd.DataFrame,
    *,
    numeric_stats: list[str] | None = None,
) -> pd.DataFrame:
    """Comprehensive summary table: numeric and categorical columns.

    For numeric columns: N, mean, sd, min, Q1, median, Q3, max, missing.
    For categorical columns: N, n_unique, mode, mode_pct, missing.

    Parameters
    ----------
    df : DataFrame
        Any tabular data.
    numeric_stats : list of str, optional
        Stats to compute for numeric columns.
        Defaults to ["count", "mean", "std", "min", "25%", "50%", "75%", "max"].

    Returns
    -------
    DataFrame
        One row per column with summary statistics.
    """
    rows = []
    for col in df.columns:
        s = df[col]
        n_miss = int(s.isna().sum())

        if pd.api.types.is_numeric_dtype(s):
            valid = s.dropna()
            rows.append(
                {
                    "column": col,
                    "dtype": "numeric",
                    "n": len(valid),
                    "missing": n_miss,
                    "mean": round(float(valid.mean()), 4) if len(valid) > 0 else np.nan,
                    "sd": round(float(valid.std()), 4) if len(valid) > 1 else np.nan,
                    "min": float(valid.min()) if len(valid) > 0 else np.nan,
                    "q1": float(valid.quantile(0.25)) if len(valid) > 0 else np.nan,
                    "median": float(valid.median()) if len(valid) > 0 else np.nan,
                    "q3": float(valid.quantile(0.75)) if len(valid) > 0 else np.nan,
                    "max": float(valid.max()) if len(valid) > 0 else np.nan,
                }
            )
        else:
            valid = s.dropna()
            mode_val = valid.mode().iloc[0] if len(valid) > 0 else None
            mode_pct = float((valid == mode_val).mean()) if mode_val is not None else np.nan
            rows.append(
                {
                    "column": col,
                    "dtype": "categorical",
                    "n": len(valid),
                    "missing": n_miss,
                    "n_unique": int(valid.nunique()),
                    "mode": mode_val,
                    "mode_pct": round(mode_pct, 4),
                }
            )

    return pd.DataFrame(rows)


def cheatsheet() -> str:
    return "otis_summary_table({}) -> Full summary table for OTIS correctional data."
