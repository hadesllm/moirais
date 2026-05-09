"""Sentence disparity between groups."""

from __future__ import annotations

import numpy as np
import pandas as pd

from moirais.fn._otis_const import DEFAULT_COLS


def sentence_disparity(
    df: pd.DataFrame,
    *,
    sent_col: str = DEFAULT_COLS["sentence"],
    group_col: str = DEFAULT_COLS["gender"],
) -> dict:
    """Sentence disparity: ratio of median sentence between groups.

    For each pair of groups, computes median_A / median_B. Returns the
    maximum disparity ratio and summary statistics.

    Parameters
    ----------
    df : DataFrame
        Dataset with sentence and group columns.
    sent_col : str
        Column with sentence length (days).
    group_col : str
        Column with group labels.

    Returns
    -------
    dict
        max_ratio, min_median_group, max_median_group, medians (dict),
        n_per_group (dict).
    """
    tmp = df[[group_col, sent_col]].dropna()
    grouped = tmp.groupby(group_col)[sent_col].agg(["median", "count"])

    medians = grouped["median"].to_dict()
    counts = grouped["count"].to_dict()

    if len(medians) < 2:
        return {
            "max_ratio": np.nan,
            "min_median_group": None,
            "max_median_group": None,
            "medians": medians,
            "n_per_group": counts,
        }

    min_g = min(medians, key=medians.get)
    max_g = max(medians, key=medians.get)
    min_val = medians[min_g]
    max_val = medians[max_g]
    ratio = max_val / min_val if min_val > 0 else np.inf

    return {
        "max_ratio": float(ratio),
        "min_median_group": min_g,
        "max_median_group": max_g,
        "medians": {k: float(v) for k, v in medians.items()},
        "n_per_group": {k: int(v) for k, v in counts.items()},
    }


sntdp = sentence_disparity


def cheatsheet() -> str:
    return "sentence_disparity({}) -> Sentence disparity between groups."
