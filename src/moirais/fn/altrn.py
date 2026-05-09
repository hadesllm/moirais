# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""Alert-state transition matrix."""

from __future__ import annotations

import numpy as np
import pandas as pd

from moirais.fn._otis_const import ALERT_COMBOS, DEFAULT_COLS


def altrans(
    df: pd.DataFrame,
    *,
    id_col: str = DEFAULT_COLS["id"],
    year_col: str = DEFAULT_COLS["year"],
    alert_mh_col: str = DEFAULT_COLS["alert_mh"],
    alert_sr_col: str = DEFAULT_COLS["alert_sr"],
    alert_sw_col: str = DEFAULT_COLS["alert_sw"],
) -> pd.DataFrame:
    """Alert-state transition matrix.

    Builds an 8x8 transition probability matrix P(state_{t+1} | state_t)
    for alert-state combinations a1--a8.

    Parameters
    ----------
    df : DataFrame
        Correctional placement data, multiple rows per individual over years.
    id_col : str
        Unique individual identifier column.
    year_col : str
        Fiscal year column.
    alert_mh_col, alert_sr_col, alert_sw_col : str
        Alert indicator columns.

    Returns
    -------
    DataFrame
        8x8 matrix indexed and columned by a1--a8, values are transition
        probabilities.
    """
    data = df.copy()
    acols = [alert_mh_col, alert_sr_col, alert_sw_col]
    for c in acols:
        if pd.api.types.is_string_dtype(data[c]):
            data[c] = (data[c].str.lower() == "yes").astype(int)

    # Encode alert state
    def _encode(row: pd.Series) -> str:
        vals = (int(row[acols[0]]), int(row[acols[1]]), int(row[acols[2]]))
        for label, combo in ALERT_COMBOS.items():
            if combo == vals:
                return label
        return "a8"

    data["_state"] = data.apply(_encode, axis=1)
    data = data.sort_values([id_col, year_col])

    states = [f"a{i}" for i in range(1, 9)]
    counts = np.zeros((8, 8), dtype=int)

    for _, grp in data.groupby(id_col):
        seq = grp["_state"].tolist()
        for i in range(len(seq) - 1):
            r = states.index(seq[i])
            c = states.index(seq[i + 1])
            counts[r, c] += 1

    row_sums = counts.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        probs = np.where(row_sums > 0, counts / row_sums, 0.0)

    return pd.DataFrame(probs, index=states, columns=states)


short = altrans


def cheatsheet() -> str:
    return "altrans({}) -> Alert-state transition matrix."
