# moirais.fn — function file (hadesllm/moirais)
"""DIF by age group via Mantel-Haenszel."""

from __future__ import annotations

import numpy as np
import pandas as pd

from moirais.fn._containers import DIFResult
from moirais.fn._helpers import _validate_df


def difag(
    data: pd.DataFrame,
    *,
    age_col: str = "age_group",
    item_cols: list[str] | None = None,
    ref_group: int | str = 0,
    n_strata: int = 5,
    alpha: float = 0.05,
) -> DIFResult:
    """DIF detection by age group using the Mantel-Haenszel procedure.

    Convenience wrapper around difmh that extracts the age grouping
    variable.  If more than two age groups exist, the function
    dichotomizes by comparing the reference group against all others.

    Parameters
    ----------
    data : DataFrame
        Must contain an age group column and item response columns.
    age_col : str
        Column name for age grouping variable (default "age_group").
    item_cols : list[str], optional
        Columns to test for DIF.  If None, uses all numeric columns
        except the age column.
    ref_group : int or str
        Value in age_col identifying the reference group (default 0).
    n_strata : int
        Number of score strata (default 5).
    alpha : float
        Significance level (default 0.05).

    Returns
    -------
    DIFResult
        method="MH", group_var set to age_col.
    """
    _validate_df(data, age_col)

    group_raw = data[age_col].to_numpy()

    # Dichotomize if more than 2 groups: ref vs. all others
    unique_groups = np.unique(group_raw[pd.notna(group_raw)])
    if len(unique_groups) > 2:
        group = np.where(group_raw == ref_group, 0, 1)
    elif len(unique_groups) == 2:
        group = group_raw
    else:
        raise ValueError(f"Need at least 2 age groups, got {len(unique_groups)}.")

    if item_cols is None:
        item_cols = [c for c in data.columns if c != age_col and pd.api.types.is_numeric_dtype(data[c])]

    if len(item_cols) < 2:
        raise ValueError("Need at least 2 item columns for DIF analysis.")

    item_data = data[item_cols].to_numpy(dtype=np.float64)

    from moirais.fn.difmh import difmh

    result = difmh(
        item_data,
        group,
        item_names=item_cols,
        n_strata=n_strata,
        alpha=alpha,
        ref_group=0 if len(unique_groups) > 2 else ref_group,
    )
    result.group_var = age_col
    return result


age_dif = difag


def cheatsheet() -> str:
    return "difag({}) -> DIF by age group via Mantel-Haenszel."
