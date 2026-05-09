"""Normative table (mean, sd, percentiles) per subscale."""

from __future__ import annotations

import numpy as np
import pandas as pd

from moirais.fn._mapq_const import SUBSCALES


def subscale_norms(
    data: pd.DataFrame,
    *,
    subscales: dict[str, list[str]] | None = None,
    percentiles: list[int] | None = None,
) -> pd.DataFrame:
    """Normative statistics per subscale.

    Parameters
    ----------
    data : DataFrame
        Item response data.
    subscales : dict, optional
        Subscale name -> list of item column names. Default: MAPQ.
    percentiles : list of int, optional
        Percentile values to compute. Default: [5, 25, 50, 75, 95].

    Returns
    -------
    DataFrame
        Columns: subscale, n, mean, sd, min, max, plus one column
        per percentile (e.g., p5, p25, p50, p75, p95).
    """
    subs = subscales if subscales is not None else SUBSCALES
    pcts = percentiles if percentiles is not None else [5, 25, 50, 75, 95]

    rows = []
    for name, items in subs.items():
        cols = [c for c in items if c in data.columns]
        if not cols:
            continue
        scores = data[cols].mean(axis=1).dropna()
        row = {
            "subscale": name,
            "n": len(scores),
            "mean": float(scores.mean()),
            "sd": float(scores.std(ddof=1)),
            "min": float(scores.min()),
            "max": float(scores.max()),
        }
        for p in pcts:
            row[f"p{p}"] = float(np.percentile(scores, p))
        rows.append(row)

    return pd.DataFrame(rows)


def cheatsheet() -> str:
    return "subscale_norms({}) -> Normative table (mean, sd, percentiles) per subscale."
