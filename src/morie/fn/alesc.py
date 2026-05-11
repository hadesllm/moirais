# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Alert escalation patterns."""

from __future__ import annotations

import pandas as pd

from morie.fn._otis_const import DEFAULT_COLS


def alescl(
    df: pd.DataFrame,
    *,
    id_col: str = DEFAULT_COLS["id"],
    year_col: str = DEFAULT_COLS["year"],
    alert_mh_col: str = DEFAULT_COLS["alert_mh"],
    alert_sr_col: str = DEFAULT_COLS["alert_sr"],
    alert_sw_col: str = DEFAULT_COLS["alert_sw"],
) -> pd.DataFrame:
    """Alert escalation patterns.

    Identifies individuals whose total number of active alerts
    (0--3) strictly increased over consecutive years (e.g. 1 -> 2 -> 3).
    Returns per-individual escalation trajectories.

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
        Columns: id, trajectory (list of alert counts), escalated (bool).
    """
    data = df.copy()
    acols = [alert_mh_col, alert_sr_col, alert_sw_col]
    for c in acols:
        if pd.api.types.is_string_dtype(data[c]):
            data[c] = (data[c].str.lower() == "yes").astype(int)

    data["_n_alerts"] = data[acols].sum(axis=1).astype(int)

    # Take max alert count per person-year
    yearly = data.groupby([id_col, year_col])["_n_alerts"].max().reset_index().sort_values([id_col, year_col])

    rows = []
    for pid, grp in yearly.groupby(id_col):
        traj = grp["_n_alerts"].tolist()
        escalated = all(traj[i] < traj[i + 1] for i in range(len(traj) - 1)) and len(traj) > 1
        rows.append({"id": pid, "trajectory": traj, "escalated": escalated})

    return pd.DataFrame(rows)


short = alescl


def cheatsheet() -> str:
    return "alescl({}) -> Alert escalation patterns."
