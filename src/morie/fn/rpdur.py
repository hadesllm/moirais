# morie.fn — function file (hadesllm/morie)
"""Placement duration statistics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._otis_const import DEFAULT_COLS


def rplace_duration(
    df: pd.DataFrame,
    *,
    id_col: str = DEFAULT_COLS["id"],
    start_col: str = "placement_start_date",
    end_col: str = "placement_end_date",
) -> pd.DataFrame:
    """Compute descriptive statistics for placement duration in days.

    Duration is ``end_col - start_col`` in days for each placement record.
    Both columns are coerced to datetime.

    Parameters
    ----------
    df : DataFrame
        Correctional placement data with start and end dates.
    id_col : str
        Column with unique individual identifiers.
    start_col : str
        Column with placement start date.
    end_col : str
        Column with placement end date.

    Returns
    -------
    DataFrame
        Single-row summary with ``n_placements``, ``mean_days``,
        ``median_days``, ``sd_days``, ``min_days``, ``max_days``.
    """
    tmp = df[[id_col, start_col, end_col]].dropna(subset=[start_col, end_col]).copy()
    tmp[start_col] = pd.to_datetime(tmp[start_col])
    tmp[end_col] = pd.to_datetime(tmp[end_col])
    tmp["duration_days"] = (tmp[end_col] - tmp[start_col]).dt.days

    dur = tmp["duration_days"].dropna()
    dur = dur[dur >= 0]

    return pd.DataFrame(
        {
            "n_placements": [len(dur)],
            "mean_days": [float(np.mean(dur)) if len(dur) else np.nan],
            "median_days": [float(np.median(dur)) if len(dur) else np.nan],
            "sd_days": [float(np.std(dur, ddof=1)) if len(dur) > 1 else np.nan],
            "min_days": [float(np.min(dur)) if len(dur) else np.nan],
            "max_days": [float(np.max(dur)) if len(dur) else np.nan],
        }
    )


rpdur = rplace_duration


def cheatsheet() -> str:
    return "rplace_duration({}) -> Placement duration statistics."
