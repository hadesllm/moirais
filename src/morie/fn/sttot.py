"""Correlation of each subscale score with total score."""

from __future__ import annotations

import numpy as np
import pandas as pd

from morie.fn._mapq_const import SUBSCALES
from ._richresult import RichResult


def subscale_total_corr(
    data: pd.DataFrame,
    *,
    subscales: dict[str, list[str]] | None = None,
) -> dict:
    """Pearson correlation of each subscale mean score with the total score.

    The total score is the mean of all items across all subscales.

    Parameters
    ----------
    data : DataFrame
        Item response data.
    subscales : dict, optional
        Subscale name -> list of item column names. Default: MAPQ.

    Returns
    -------
    dict
        Subscale name -> correlation with total.
    """
    subs = subscales if subscales is not None else SUBSCALES

    # All items
    all_items = []
    for items in subs.values():
        all_items.extend([c for c in items if c in data.columns])

    if not all_items:
        return RichResult(payload={})

    total = data[all_items].mean(axis=1).to_numpy()

    result = {}
    for name, items in subs.items():
        cols = [c for c in items if c in data.columns]
        if not cols:
            continue
        sub_score = data[cols].mean(axis=1).to_numpy()
        # Use corrected correlation: remove subscale items from total
        rest_items = [c for c in all_items if c not in cols]
        if rest_items:
            rest_total = data[rest_items].mean(axis=1).to_numpy()
            r = float(np.corrcoef(sub_score, rest_total)[0, 1])
        else:
            r = 1.0
        result[name] = r

    return result


def cheatsheet() -> str:
    return "subscale_total_corr({}) -> Correlation of each subscale score with total score."
