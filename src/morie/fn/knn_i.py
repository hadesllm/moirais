# morie.fn — function file (hadesllm/morie)
"""KNN imputation for missing data."""

from __future__ import annotations

import numpy as np
import pandas as pd


def knn_impute(
    data: pd.DataFrame,
    *,
    columns: list[str] | None = None,
    k: int = 5,
) -> pd.DataFrame:
    """
    K-nearest-neighbour imputation for numeric missing data.

    For each row containing missing values, identifies the *k* nearest
    complete-case neighbours using Euclidean distance computed over the
    mutually observed columns.  Missing values are imputed as the
    inverse-distance-weighted mean of the neighbours' values.

    If fewer than *k* complete neighbours share observed columns with a
    row, all available complete neighbours are used.  If no neighbours are
    available (all rows miss the same column), the column mean is used as
    fallback.

    Only numeric columns are imputed; non-numeric columns are returned
    unchanged.

    :param data: Input DataFrame (may contain NaN in numeric columns).
    :type data: pandas.DataFrame
    :param columns: Columns to impute.  Default: all numeric columns with
        missing values.
    :type columns: list[str] or None
    :param k: Number of nearest neighbours.  Default 5.
    :type k: int
    :return: Completed DataFrame with NaN values filled.
    :rtype: pandas.DataFrame
    :raises ValueError: If *k* < 1 or no missing values found.

    References
    ----------
    Troyanskaya, O., Cantor, M., Sherlock, G., Brown, P., Hastie, T.,
    Tibshirani, R., Botstein, D., & Altman, R. B. (2001). Missing value
    estimation methods for DNA microarrays. *Bioinformatics*, 17(6),
    520--525. https://doi.org/10.1093/bioinformatics/17.6.520
    """
    if k < 1:
        raise ValueError("k must be >= 1.")

    result = data.copy()
    numeric_cols = result.select_dtypes(include=[np.number]).columns.tolist()

    if columns is not None:
        target_cols = [c for c in columns if c in numeric_cols and result[c].isna().any()]
    else:
        target_cols = [c for c in numeric_cols if result[c].isna().any()]

    if not target_cols:
        raise ValueError("No missing numeric values found to impute.")

    values = result[numeric_cols].values.astype(float)
    col_idx = {c: i for i, c in enumerate(numeric_cols)}
    target_idxs = [col_idx[c] for c in target_cols]
    col_means = np.nanmean(values, axis=0)

    n = values.shape[0]

    for row_i in range(n):
        missing_cols = [j for j in target_idxs if np.isnan(values[row_i, j])]
        if not missing_cols:
            continue

        # Observed columns for this row
        obs_cols = [j for j in range(values.shape[1]) if not np.isnan(values[row_i, j])]
        if not obs_cols:
            # No observed values at all — fill with column means
            for j in missing_cols:
                values[row_i, j] = col_means[j]
            continue

        # Find complete-case neighbours on the observed columns
        distances = []
        for row_j in range(n):
            if row_j == row_i:
                continue
            # Neighbour must have values in both the observed cols and the missing cols
            if any(np.isnan(values[row_j, j]) for j in obs_cols):
                continue
            if any(np.isnan(values[row_j, j]) for j in missing_cols):
                continue
            d = np.sqrt(np.sum((values[row_i, obs_cols] - values[row_j, obs_cols]) ** 2))
            distances.append((row_j, d))

        if not distances:
            # Fallback: column mean
            for j in missing_cols:
                values[row_i, j] = col_means[j]
            continue

        distances.sort(key=lambda x: x[1])
        top_k = distances[:k]

        # Inverse-distance weighted mean
        for j in missing_cols:
            neighbor_vals = np.array([values[nb, j] for nb, _ in top_k])
            neighbor_dists = np.array([d for _, d in top_k])

            if np.all(neighbor_dists == 0):
                # All neighbours are identical to this row
                values[row_i, j] = np.mean(neighbor_vals)
            else:
                weights = 1.0 / np.maximum(neighbor_dists, 1e-12)
                values[row_i, j] = np.average(neighbor_vals, weights=weights)

    result[numeric_cols] = values
    return result


knn_i = knn_impute


def cheatsheet() -> str:
    return "knn_impute({}) -> KNN imputation for missing data."
