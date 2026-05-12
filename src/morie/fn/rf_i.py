# morie.fn -- function file (hadesllm/morie)
"""Iterative OLS regression imputation (simplified random-forest style)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def rf_impute(
    data: pd.DataFrame,
    *,
    columns: list[str] | None = None,
    n_iter: int = 5,
) -> pd.DataFrame:
    """
    Iterative regression imputation using OLS (random-forest style).

    For each column with missing data, fit an OLS model on the remaining
    columns using complete cases, then predict the missing values.  This is
    repeated for *n_iter* cycles so that imputed values in predictor columns
    can improve downstream predictions.

    This is a deterministic, single-imputation analogue of the iterative
    random-forest approach in missForest (Stekhoven & Buhlmann, 2012).
    Using ``numpy.linalg.lstsq`` instead of a random forest keeps the
    implementation dependency-free.

    Only numeric columns are imputed; non-numeric columns are returned
    unchanged.

    :param data: Input DataFrame (may contain NaN in numeric columns).
    :type data: pandas.DataFrame
    :param columns: Columns to impute.  Default: all numeric columns with
        missing values.
    :type columns: list[str] or None
    :param n_iter: Number of iteration cycles.  Default 5.
    :type n_iter: int
    :return: Completed DataFrame with NaN values filled.
    :rtype: pandas.DataFrame
    :raises ValueError: If no missing numeric values are found.

    References
    ----------
    Stekhoven, D. J., & Buhlmann, P. (2012). MissForest -- non-parametric
    missing value imputation for mixed-type data. *Bioinformatics*, 28(1),
    112--118. https://doi.org/10.1093/bioinformatics/btr597
    """
    result = data.copy()
    numeric_cols = result.select_dtypes(include=[np.number]).columns.tolist()

    if columns is not None:
        target_cols = [c for c in columns if c in numeric_cols and result[c].isna().any()]
    else:
        target_cols = [c for c in numeric_cols if result[c].isna().any()]

    if not target_cols:
        raise ValueError("No missing numeric values found to impute.")

    # Initial fill with column means
    col_means = result[numeric_cols].mean()
    for col in target_cols:
        result[col] = result[col].fillna(col_means[col])

    original_missing = {col: data[col].isna() for col in target_cols}

    for _cycle in range(n_iter):
        # Sort columns by amount of missingness (fewest first)
        sorted_cols = sorted(target_cols, key=lambda c: original_missing[c].sum())

        for col in sorted_cols:
            missing_mask = original_missing[col]
            if not missing_mask.any():
                continue

            predictors = [c for c in numeric_cols if c != col]
            if not predictors:
                continue

            obs_mask = ~missing_mask
            X_obs = result.loc[obs_mask, predictors].values
            y_obs = result.loc[obs_mask, col].values
            X_mis = result.loc[missing_mask, predictors].values

            n_obs = X_obs.shape[0]
            if n_obs < 2:
                continue

            # Add intercept
            X_obs_i = np.column_stack([np.ones(n_obs), X_obs])
            X_mis_i = np.column_stack([np.ones(X_mis.shape[0]), X_mis])

            try:
                beta, _, _, _ = np.linalg.lstsq(X_obs_i, y_obs, rcond=None)
            except np.linalg.LinAlgError:
                continue

            y_pred = X_mis_i @ beta
            result.loc[missing_mask, col] = y_pred

    return result


rf_i = rf_impute


def cheatsheet() -> str:
    return "rf_impute({}) -> Iterative OLS regression imputation (simplified random-fores"
