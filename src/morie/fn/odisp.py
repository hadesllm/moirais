# morie.fn — function file (hadesllm/morie)
"""OTIS disparity index — max/min group mean ratio."""

from __future__ import annotations

import pandas as pd


def otis_disparity_index(
    df: pd.DataFrame,
    *,
    metric_col: str = "Y",
    group_col: str = "gender",
) -> dict:
    """Disparity ratio: max group mean / min group mean.

    A value of 1.0 indicates no disparity. Higher values indicate
    greater disparity between the highest and lowest group means.

    Parameters
    ----------
    df : DataFrame
        Correctional records.
    metric_col : str
        Numeric metric column.
    group_col : str
        Grouping column.

    Returns
    -------
    dict
        Keys: ``disparity_index``, ``max_group``, ``min_group``,
        ``max_mean``, ``min_mean``, ``group_means``.
    """
    means = df.groupby(group_col)[metric_col].mean()
    group_means = means.to_dict()
    if len(means) < 2 or means.min() == 0:
        return {
            "disparity_index": float("nan"),
            "max_group": None,
            "min_group": None,
            "max_mean": float("nan"),
            "min_mean": float("nan"),
            "group_means": group_means,
        }
    max_g = means.idxmax()
    min_g = means.idxmin()
    # Use absolute values to handle metrics that can be negative
    abs_means = means.abs()
    idx = float(abs_means.max() / abs_means.min()) if abs_means.min() > 0 else float("nan")
    return {
        "disparity_index": idx,
        "max_group": max_g,
        "min_group": min_g,
        "max_mean": float(means[max_g]),
        "min_mean": float(means[min_g]),
        "group_means": group_means,
    }


def cheatsheet() -> str:
    return "otis_disparity_index({}) -> OTIS disparity index — max/min group mean ratio."
