# morie.fn — function file (hadesllm/morie)
"""Missing data pattern analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd


def missing_data_patterns(
    data: pd.DataFrame,
    *,
    columns: list[str] | None = None,
) -> dict:
    """
    Analyse missing-data patterns in a DataFrame.

    Returns per-column missingness fractions, per-row missingness fractions,
    a pattern matrix (unique row-wise indicator patterns with counts), and a
    boolean flag indicating whether the pattern is monotone.

    A missing-data pattern is **monotone** when, after reordering columns by
    their missingness fraction (ascending), each column's observed set is a
    superset of the next column's observed set.  Monotone patterns allow
    simpler sequential imputation (Little & Rubin, 2019, Section 4.2).

    :param data: Input DataFrame (may contain NaN).
    :type data: pandas.DataFrame
    :param columns: Subset of columns to analyse.  Default: all columns.
    :type columns: list[str] or None
    :return: Dictionary with keys ``per_col`` (Series), ``per_row`` (Series),
        ``patterns`` (DataFrame with binary indicator columns plus ``count``),
        and ``is_monotone`` (bool).
    :rtype: dict

    References
    ----------
    Little, R. J. A., & Rubin, D. B. (2019). *Statistical Analysis with
    Missing Data* (3rd ed.). Wiley. https://doi.org/10.1002/9781119482260
    """
    if columns is not None:
        data = data[columns]

    n, p = data.shape
    if n == 0 or p == 0:
        raise ValueError("DataFrame must have at least one row and one column.")

    # Per-column fraction missing
    per_col = data.isna().mean()

    # Per-row fraction missing
    per_row = data.isna().mean(axis=1)

    # Unique patterns: 1 = missing, 0 = observed
    indicator = data.isna().astype(int)
    pattern_strings = indicator.apply(lambda row: tuple(row), axis=1)
    pattern_counts = pattern_strings.value_counts().reset_index()
    pattern_counts.columns = ["pattern", "count"]

    # Expand tuple back to columns
    pattern_df = pd.DataFrame(
        pattern_counts["pattern"].tolist(),
        columns=data.columns,
    )
    pattern_df["count"] = pattern_counts["count"].values
    pattern_df = pattern_df.sort_values("count", ascending=False).reset_index(drop=True)

    # Monotone check: sort columns by missingness fraction ascending.
    sorted_cols = per_col.sort_values().index.tolist()
    sorted_indicator = indicator[sorted_cols].values  # n x p
    is_monotone = True
    for j in range(1, len(sorted_cols)):
        # For monotone: wherever col j is observed, col j-1 must also be observed
        obs_j = sorted_indicator[:, j] == 0
        obs_prev = sorted_indicator[:, j - 1] == 0
        if not np.all(obs_prev[obs_j]):
            is_monotone = False
            break

    return {
        "per_col": per_col,
        "per_row": per_row,
        "patterns": pattern_df,
        "is_monotone": is_monotone,
    }


md_pat = missing_data_patterns


def cheatsheet() -> str:
    return "missing_data_patterns({}) -> Missing data pattern analysis."
