# moirais.fn — function file (hadesllm/moirais)
"""Full demographic cross-tabulation."""

from __future__ import annotations

import pandas as pd

from moirais.fn._containers import DescriptiveResult


def otis_demo_cross(
    df: pd.DataFrame,
    *,
    row_col: str = "region",
    col_col: str = "age_group",
    layer_col: str = "gender",
    id_col: str = "person_id",
) -> DescriptiveResult:
    """Full cross-tabulation (region x age x gender).

    Parameters
    ----------
    df : DataFrame
    row_col, col_col, layer_col, id_col : str

    Returns
    -------
    DescriptiveResult
        value is a multi-level DataFrame.
    """
    ct = df.groupby([row_col, col_col, layer_col]).size().reset_index(name="count")
    pivot = ct.pivot_table(index=[row_col, layer_col], columns=col_col, values="count", fill_value=0)
    return DescriptiveResult(name="otis_demo_cross", value=pivot)


odm_c = otis_demo_cross


def cheatsheet() -> str:
    return "otis_demo_cross({}) -> Full demographic cross-tabulation."
