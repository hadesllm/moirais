# morie.fn -- function file (hadesllm/morie)
"""Custody gender parity index."""

from __future__ import annotations

import pandas as pd
from ._richresult import RichResult


def custody_gender_parity(
    df: pd.DataFrame,
    *,
    gender_col: str = "gender",
    metric_col: str = "sentence_days",
) -> dict:
    """Gender parity index for a metric (min group mean / max group mean).

    A value of 1.0 indicates perfect parity. Values below 1.0 indicate
    disparity between gender groups.

    Parameters
    ----------
    df : DataFrame
        Correctional records.
    gender_col : str
        Gender column.
    metric_col : str
        Numeric metric column.

    Returns
    -------
    dict
        Keys: ``parity_index``, ``group_means`` (dict), ``n_groups``.
    """
    means = df.groupby(gender_col)[metric_col].mean()
    group_means = means.to_dict()
    if len(means) < 2 or means.max() == 0:
        parity = 1.0
    else:
        parity = float(means.min() / means.max())
    return RichResult(payload={"parity_index": parity, "group_means": group_means, "n_groups": len(means)})


def cheatsheet() -> str:
    return "custody_gender_parity({}) -> Custody gender parity index."
