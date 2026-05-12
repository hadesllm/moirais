# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Alert complexity index per individual."""

from __future__ import annotations

import pandas as pd

from morie.fn._otis_const import ALERT_COMBOS, DEFAULT_COLS


def alcmpx(
    df: pd.DataFrame,
    *,
    id_col: str = DEFAULT_COLS["id"],
    year_col: str = DEFAULT_COLS["year"],
    alert_mh_col: str = DEFAULT_COLS["alert_mh"],
    alert_sr_col: str = DEFAULT_COLS["alert_sr"],
    alert_sw_col: str = DEFAULT_COLS["alert_sw"],
) -> pd.DataFrame:
    """Alert complexity index per individual.

    Counts the number of distinct alert-state combinations (a1--a8)
    each individual exhibits across all observed years. Higher values
    indicate more variable alert trajectories.

    Parameters
    ----------
    df : DataFrame
        Correctional placement data.
    id_col : str
        Unique individual identifier column.
    year_col : str
        Fiscal year column.
    alert_mh_col, alert_sr_col, alert_sw_col : str
        Alert indicator columns.

    Returns
    -------
    DataFrame
        Columns: id, complexity (int, 1--8), n_years.
    """
    data = df.copy()
    acols = [alert_mh_col, alert_sr_col, alert_sw_col]
    for c in acols:
        if pd.api.types.is_string_dtype(data[c]):
            data[c] = (data[c].str.lower() == "yes").astype(int)

    def _encode(row: pd.Series) -> str:
        vals = (int(row[acols[0]]), int(row[acols[1]]), int(row[acols[2]]))
        for label, combo in ALERT_COMBOS.items():
            if combo == vals:
                return label
        return "a8"

    data["_state"] = data.apply(_encode, axis=1)

    result = (
        data.groupby(id_col)
        .agg(
            complexity=("_state", "nunique"),
            n_years=(year_col, "nunique"),
        )
        .reset_index()
    )
    result.columns = ["id", "complexity", "n_years"]
    return result


short = alcmpx


def cheatsheet() -> str:
    return "alcmpx({}) -> Alert complexity index per individual."
