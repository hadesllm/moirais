# morie.fn — function file (hadesllm/morie)
"""DIF by gender via Mantel-Haenszel."""

from __future__ import annotations

import numpy as np
import pandas as pd

from morie.fn._containers import DIFResult
from morie.fn._helpers import _validate_df


def difgn(
    data: pd.DataFrame,
    *,
    gender_col: str = "gender",
    item_cols: list[str] | None = None,
    ref_group: int | str = 0,
    n_strata: int = 5,
    alpha: float = 0.05,
) -> DIFResult:
    """DIF detection by gender using the Mantel-Haenszel procedure.

    Convenience wrapper around difmh that extracts the gender grouping
    variable and item columns from a DataFrame.

    Parameters
    ----------
    data : DataFrame
        Must contain a gender column and item response columns.
    gender_col : str
        Column name for gender grouping variable (default "gender").
    item_cols : list[str], optional
        Columns to test for DIF.  If None, uses all numeric columns
        except the gender column.
    ref_group : int or str
        Value in gender_col identifying the reference group (default 0).
    n_strata : int
        Number of score strata (default 5).
    alpha : float
        Significance level (default 0.05).

    Returns
    -------
    DIFResult
        method="MH", group_var set to gender_col.
    """
    _validate_df(data, gender_col)

    group = data[gender_col].to_numpy()

    if item_cols is None:
        item_cols = [c for c in data.columns if c != gender_col and pd.api.types.is_numeric_dtype(data[c])]

    if len(item_cols) < 2:
        raise ValueError("Need at least 2 item columns for DIF analysis.")

    item_data = data[item_cols].to_numpy(dtype=np.float64)

    from morie.fn.difmh import difmh

    result = difmh(
        item_data,
        group,
        item_names=item_cols,
        n_strata=n_strata,
        alpha=alpha,
        ref_group=ref_group,
    )
    result.group_var = gender_col
    return result


gender_dif = difgn


def cheatsheet() -> str:
    return "difgn({}) -> DIF by gender via Mantel-Haenszel."
