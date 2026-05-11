# morie.fn — function file (hadesllm/morie)
"""Histogram bin counts (no plotting) for OTIS correctional data."""

from __future__ import annotations

import numpy as np
import pandas as pd


def otis_histogram_data(
    df: pd.DataFrame,
    *,
    col: str = "sentence_days",
    bins: int = 20,
) -> pd.DataFrame:
    """Compute histogram bin counts for a numeric column.

    Parameters
    ----------
    df : DataFrame
        Data with the target numeric column.
    col : str
        Numeric column to histogram.
    bins : int
        Number of bins.

    Returns
    -------
    DataFrame
        Columns: bin_lo, bin_hi, count, density.
    """
    values = df[col].dropna().values.astype(np.float64)
    counts, edges = np.histogram(values, bins=bins)
    total = len(values)
    widths = np.diff(edges)

    rows = []
    for i in range(len(counts)):
        density = float(counts[i] / (total * widths[i])) if total > 0 and widths[i] > 0 else 0.0
        rows.append(
            {
                "bin_lo": round(float(edges[i]), 4),
                "bin_hi": round(float(edges[i + 1]), 4),
                "count": int(counts[i]),
                "density": round(density, 6),
            }
        )

    return pd.DataFrame(rows)


def cheatsheet() -> str:
    return "otis_histogram_data({}) -> Histogram bin counts (no plotting) for OTIS correctional dat"
