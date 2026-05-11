# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Alert-state combination encoding."""

from __future__ import annotations

import pandas as pd

from morie.fn._containers import AstRes


def astcmb(
    df: pd.DataFrame,
    *,
    alert_cols: tuple[str, ...] = (
        "mental_health_alert",
        "suicide_risk_alert",
        "suicide_watch_alert",
    ),
    id_col: str = "unique_individual_id",
    year_col: str = "end_fiscal_year",
    np_col: str = "number_of_placements",
) -> AstRes:
    """Alert-state combination encoding.

    Encodes 3 binary alert indicators into 8 possible combinations
    (a1--a8) and computes a complexity index (count of distinct
    combinations per person-year).

    Parameters
    ----------
    df : DataFrame
        Must contain alert columns (Yes/No or 1/0).
    alert_cols : tuple of str
        Names of the three alert columns.
    id_col : str
        Unique individual identifier column.
    year_col : str
        Fiscal year column.
    np_col : str
        Number of placements column (unused but accepted for API compat).

    Returns
    -------
    AstRes
        Per-person alert data and complexity summary.
    """
    data = df.copy()

    # Binarize if needed
    for col in alert_cols:
        if pd.api.types.is_string_dtype(data[col]):
            data[col] = (data[col].str.lower() == "yes").astype(int)
        elif pd.api.types.is_numeric_dtype(data[col]):
            data[col] = data[col].astype(int)
        else:
            data[col] = data[col].astype(str).str.lower().eq("yes").astype(int)

    a, b, c = alert_cols
    data["a1"] = ((data[a] == 1) & (data[b] == 0) & (data[c] == 0)).astype(int)
    data["a2"] = ((data[a] == 0) & (data[b] == 1) & (data[c] == 0)).astype(int)
    data["a3"] = ((data[a] == 0) & (data[b] == 0) & (data[c] == 1)).astype(int)
    data["a4"] = ((data[a] == 1) & (data[b] == 1) & (data[c] == 0)).astype(int)
    data["a5"] = ((data[a] == 0) & (data[b] == 1) & (data[c] == 1)).astype(int)
    data["a6"] = ((data[a] == 1) & (data[b] == 0) & (data[c] == 1)).astype(int)
    data["a7"] = ((data[a] == 1) & (data[b] == 1) & (data[c] == 1)).astype(int)
    data["a8"] = ((data[a] == 0) & (data[b] == 0) & (data[c] == 0)).astype(int)

    acols = [f"a{i}" for i in range(1, 9)]
    grouped = data.groupby([id_col, year_col])[acols].sum().reset_index()

    # Complexity = number of distinct alert states observed
    grouped["ac"] = (grouped[acols] > 0).sum(axis=1)

    summary = grouped.groupby("ac").size().reset_index(name="n_persons")
    summary = summary.sort_values("ac", ascending=False)

    return AstRes(data=grouped, summary=summary)


def cheatsheet() -> str:
    return "astcmb({}) -> Alert-state combination encoding."
