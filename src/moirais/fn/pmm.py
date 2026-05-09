# moirais.fn — function file (hadesllm/moirais)
"""Predictive mean matching imputation."""

from __future__ import annotations

import numpy as np
import pandas as pd


def pmm_impute(
    data: pd.DataFrame,
    *,
    columns: list[str] | None = None,
    k: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Impute missing values using Predictive Mean Matching (PMM).

    For each variable with missing data:

    1. Fit an OLS model on complete cases using the other variables as
       predictors.
    2. Compute predicted values for **all** rows (observed and missing).
    3. For each missing value, find the *k* observed rows whose predicted
       values are closest to the missing row's predicted value.
    4. Randomly select one of these *k* donors and use its **observed**
       value as the imputed value.

    PMM preserves the distribution of observed values (no out-of-range
    imputations) and is semi-parametric: the model is used only for
    matching, not for generating the imputed value itself.

    Only numeric columns are imputed; non-numeric columns are returned
    unchanged.

    :param data: Input DataFrame (may contain NaN in numeric columns).
    :type data: pandas.DataFrame
    :param columns: Columns to impute.  Default: all numeric columns with
        missing values.
    :type columns: list[str] or None
    :param k: Number of candidate donors to match.  Default 5.
    :type k: int
    :param seed: Random seed for reproducibility.
    :type seed: int
    :return: Completed DataFrame with NaN values filled via PMM.
    :rtype: pandas.DataFrame
    :raises ValueError: If *k* < 1 or no missing values found.

    References
    ----------
    Little, R. J. A. (1988). Missing-data adjustments in large surveys.
    *Journal of Business & Economic Statistics*, 6(3), 287--296.
    https://doi.org/10.1080/07350015.1988.10509663

    van Buuren, S. (2018). *Flexible Imputation of Missing Data* (2nd ed.).
    CRC Press. https://doi.org/10.1201/9780429492259
    """
    if k < 1:
        raise ValueError("k must be >= 1.")

    rng = np.random.default_rng(seed)
    result = data.copy()
    numeric_cols = result.select_dtypes(include=[np.number]).columns.tolist()

    if columns is not None:
        target_cols = [c for c in columns if c in numeric_cols and result[c].isna().any()]
    else:
        target_cols = [c for c in numeric_cols if result[c].isna().any()]

    if not target_cols:
        raise ValueError("No missing numeric values found to impute.")

    # Initial mean-fill for predictors (so OLS works on full rows)
    filled = result.copy()
    for col in target_cols:
        filled[col] = filled[col].fillna(filled[col].mean())

    original_missing = {col: data[col].isna() for col in target_cols}

    for col in target_cols:
        missing_mask = original_missing[col]
        obs_mask = ~missing_mask

        predictors = [c for c in numeric_cols if c != col]
        if not predictors:
            continue

        n_obs = obs_mask.sum()
        if n_obs < 2:
            continue

        X_all = filled[predictors].values
        y_obs = data.loc[obs_mask, col].values.astype(float)

        # Add intercept
        X_all_i = np.column_stack([np.ones(len(X_all)), X_all])
        X_obs_i = X_all_i[obs_mask]

        try:
            beta, _, _, _ = np.linalg.lstsq(X_obs_i, y_obs, rcond=None)
        except np.linalg.LinAlgError:
            continue

        # Predicted values for all rows
        y_hat_all = X_all_i @ beta
        y_hat_obs = y_hat_all[obs_mask]
        y_hat_mis = y_hat_all[missing_mask]

        obs_indices = np.where(obs_mask.values)[0]

        for i, y_pred in enumerate(y_hat_mis):
            # Distances between this missing row's prediction and all observed predictions
            dists = np.abs(y_hat_obs - y_pred)
            # Find k nearest
            k_actual = min(k, len(dists))
            nearest_idx = np.argpartition(dists, k_actual - 1)[:k_actual]
            # Pick one donor at random
            donor_local = rng.choice(nearest_idx)
            donor_row = obs_indices[donor_local]
            result.iloc[np.where(missing_mask.values)[0][i], result.columns.get_loc(col)] = data.iloc[donor_row][col]

    return result


pmm = pmm_impute


def cheatsheet() -> str:
    return "pmm_impute({}) -> Predictive mean matching imputation."
