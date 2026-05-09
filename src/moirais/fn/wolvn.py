"""Missing data imputation via chained equations. 'I'm the best there is at what I do.' -- Wolverine"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._containers import DescriptiveResult


def mice_impute(
    data: pd.DataFrame,
    *,
    n_imputations: int = 5,
    n_iter: int = 10,
    seed: int | None = None,
) -> DescriptiveResult:
    """Impute missing data using Multiple Imputation by Chained Equations (MICE).

    For each column with missing values, fits a linear regression on the
    observed values using all other columns as predictors, then draws from
    the predictive distribution.  Cycles through columns for *n_iter* rounds.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with missing values (NaN).
    n_imputations : int
        Number of completed datasets to generate.
    n_iter : int
        Number of MICE iteration cycles per imputation.
    seed : int or None
        RNG seed.

    Returns
    -------
    DescriptiveResult
        ``value`` dict with ``imputed`` (list of DataFrames),
        ``n_missing_per_col`` dict, ``frac_missing`` float.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a DataFrame")
    if data.shape[0] < 2:
        raise ValueError("Need at least 2 rows")

    rng = np.random.default_rng(seed)
    cols = data.columns.tolist()
    missing_mask = data.isna()
    n_missing = {c: int(missing_mask[c].sum()) for c in cols}
    frac_missing = float(missing_mask.values.sum()) / (data.shape[0] * data.shape[1])
    cols_to_impute = [c for c in cols if n_missing[c] > 0]

    if not cols_to_impute:
        return DescriptiveResult(
            name="mice_impute",
            value={"imputed": [data.copy()], "n_missing_per_col": n_missing, "frac_missing": 0.0},
            extra={"n_imputations": 1, "n_iter": 0},
        )

    imputed_list = []
    for m in range(n_imputations):
        df = data.copy()
        for c in cols_to_impute:
            obs = df[c].dropna()
            mis_idx = df[c].isna()
            if len(obs) > 0 and mis_idx.sum() > 0:
                fill_vals = obs.sample(
                    n=int(mis_idx.sum()), replace=True, random_state=int(rng.integers(1 << 31))
                ).values
                df.loc[mis_idx, c] = fill_vals
            elif mis_idx.sum() > 0:
                df[c] = df[c].fillna(0.0)

        for _ in range(n_iter):
            for target in cols_to_impute:
                obs_idx = ~missing_mask[target]
                mis_idx = missing_mask[target]
                if mis_idx.sum() == 0:
                    continue

                predictors = [c for c in cols if c != target]
                X_obs = df.loc[obs_idx, predictors].values.astype(float)
                y_obs = df.loc[obs_idx, target].values.astype(float)
                X_mis = df.loc[mis_idx, predictors].values.astype(float)

                if X_obs.shape[0] < X_obs.shape[1] + 1:
                    continue

                X_obs_aug = np.column_stack([np.ones(X_obs.shape[0]), X_obs])
                X_mis_aug = np.column_stack([np.ones(X_mis.shape[0]), X_mis])

                try:
                    beta, res, _, _ = np.linalg.lstsq(X_obs_aug, y_obs, rcond=None)
                except np.linalg.LinAlgError:
                    continue

                y_pred = X_mis_aug @ beta
                if len(res) > 0 and X_obs.shape[0] > X_obs.shape[1] + 1:
                    sigma = np.sqrt(res[0] / (X_obs.shape[0] - X_obs.shape[1] - 1))
                else:
                    sigma = float(np.std(y_obs, ddof=1))

                noise = rng.normal(0, max(sigma, 1e-10), size=len(y_pred))
                df.loc[mis_idx, target] = y_pred + noise

        imputed_list.append(df)

    return DescriptiveResult(
        name="mice_impute",
        value={
            "imputed": imputed_list,
            "n_missing_per_col": n_missing,
            "frac_missing": frac_missing,
        },
        extra={"n_imputations": n_imputations, "n_iter": n_iter},
    )


wolvn = mice_impute


def cheatsheet() -> str:
    return "mice_impute({}) -> Missing data imputation via chained equations. 'I'm the best"
