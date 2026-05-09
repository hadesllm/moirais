# moirais.fn — function file (hadesllm/moirais)
"""Multiple Imputation by Chained Equations (MICE)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def mice_impute(
    data: pd.DataFrame,
    *,
    columns: list[str] | None = None,
    n_iter: int = 10,
    m: int = 5,
    seed: int = 42,
) -> list[pd.DataFrame]:
    """
    Multiple Imputation by Chained Equations (MICE).

    For each of *m* imputations, missing values are filled by iterating
    through each variable with missing data and regressing it on all other
    variables using OLS (for numeric) or mode-based donor (for categorical).
    The procedure repeats for *n_iter* cycles to allow the conditional
    distributions to stabilise.

    Only columns listed in *columns* (default: all) are imputed.  Numeric
    columns use OLS with Bayesian proper imputation noise; categorical columns
    use the most frequent value among predicted donors.

    :param data: Input DataFrame (may contain NaN).
    :type data: pandas.DataFrame
    :param columns: Columns to impute.  Default: all columns with missing.
    :type columns: list[str] or None
    :param n_iter: Number of chained-equation cycles per imputation.
    :type n_iter: int
    :param m: Number of multiply-imputed datasets to produce.
    :type m: int
    :param seed: Random seed for reproducibility.
    :type seed: int
    :return: List of *m* completed DataFrames.
    :rtype: list[pandas.DataFrame]
    :raises ValueError: If no missing values are found.

    References
    ----------
    van Buuren, S., & Groothuis-Oudshoorn, K. (2011). mice: Multivariate
    Imputation by Chained Equations in R. *Journal of Statistical Software*,
    45(3), 1--67. https://doi.org/10.18637/jss.v045.i03

    Rubin, D. B. (1987). *Multiple Imputation for Nonresponse in Surveys*.
    Wiley. https://doi.org/10.1002/9780470316696
    """
    rng = np.random.default_rng(seed)
    df = data.copy()

    if columns is not None:
        target_cols = [c for c in columns if df[c].isna().any()]
    else:
        target_cols = [c for c in df.columns if df[c].isna().any()]

    if not target_cols:
        raise ValueError("No missing values found in the specified columns.")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    all_cols = df.columns.tolist()

    results: list[pd.DataFrame] = []

    for imp in range(m):
        filled = df.copy()
        # Initial fill: mean for numeric, mode for categorical
        for col in target_cols:
            if col in numeric_cols:
                filled[col] = filled[col].fillna(filled[col].mean())
            else:
                mode_val = filled[col].mode()
                if len(mode_val) > 0:
                    filled[col] = filled[col].fillna(mode_val.iloc[0])

        for _cycle in range(n_iter):
            for col in target_cols:
                missing_mask = df[col].isna()
                if not missing_mask.any():
                    continue

                predictors = [c for c in all_cols if c != col]
                # Build design matrix from filled data (numeric predictors only)
                pred_numeric = [c for c in predictors if c in numeric_cols]
                if not pred_numeric:
                    continue

                if col in numeric_cols:
                    # OLS imputation with proper noise
                    obs_mask = ~missing_mask
                    X_obs = filled.loc[obs_mask, pred_numeric].values
                    y_obs = filled.loc[obs_mask, col].values
                    X_mis = filled.loc[missing_mask, pred_numeric].values

                    n_obs, k = X_obs.shape
                    # Add intercept
                    X_obs_i = np.column_stack([np.ones(n_obs), X_obs])
                    X_mis_i = np.column_stack([np.ones(X_mis.shape[0]), X_mis])

                    # OLS fit
                    try:
                        beta, residuals, _, _ = np.linalg.lstsq(X_obs_i, y_obs, rcond=None)
                    except np.linalg.LinAlgError:
                        continue

                    y_pred = X_mis_i @ beta

                    # Residual variance for proper imputation
                    if n_obs > k + 1:
                        resid = y_obs - X_obs_i @ beta
                        sigma2 = np.sum(resid**2) / (n_obs - k - 1)
                    else:
                        sigma2 = 1.0

                    # Add noise for proper imputation (Bayesian draw)
                    noise = rng.normal(0, np.sqrt(sigma2), size=len(y_pred))
                    filled.loc[missing_mask, col] = y_pred + noise
                else:
                    # Categorical: mode of observed
                    obs_vals = filled.loc[~missing_mask, col]
                    if len(obs_vals) > 0:
                        mode_val = obs_vals.mode()
                        if len(mode_val) > 0:
                            filled.loc[missing_mask, col] = mode_val.iloc[0]

        results.append(filled)

    return results


mice = mice_impute


def cheatsheet() -> str:
    return "mice_impute({}) -> Multiple Imputation by Chained Equations (MICE)."
