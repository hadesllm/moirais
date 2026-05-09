"""Standardised regression coefficients (beta weights)."""

from typing import Union

import numpy as np
import pandas as pd


def standardized_coefficients(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
) -> pd.DataFrame:
    """Compute standardised regression coefficients (beta weights).

    Standardises X and y to zero mean and unit variance before OLS.

    Parameters
    ----------
    X : array-like or DataFrame
        Predictor matrix (n x p).
    y : array-like or Series
        Outcome variable.

    Returns
    -------
    DataFrame
        Columns: variable, beta, se, t, p_value.
    """
    import statsmodels.api as sm

    if isinstance(X, pd.DataFrame):
        names = X.columns.tolist()
        X_arr = X.values.astype(np.float64)
    else:
        X_arr = np.asarray(X, dtype=np.float64)
        names = [f"x{i}" for i in range(X_arr.shape[1])]
    y_arr = np.asarray(y, dtype=np.float64).ravel()

    # Standardise
    X_std = (X_arr - X_arr.mean(axis=0)) / (X_arr.std(axis=0, ddof=1) + 1e-15)
    y_std = (y_arr - y_arr.mean()) / (y_arr.std(ddof=1) + 1e-15)

    model = sm.OLS(y_std, sm.add_constant(X_std)).fit()
    # Skip constant (index 0)
    results = []
    for i, name in enumerate(names):
        results.append(
            {
                "variable": name,
                "beta": float(model.params[i + 1]),
                "se": float(model.bse[i + 1]),
                "t": float(model.tvalues[i + 1]),
                "p_value": float(model.pvalues[i + 1]),
            }
        )
    return pd.DataFrame(results)


stdbeta = standardized_coefficients


def cheatsheet() -> str:
    return "standardized_coefficients({}) -> Standardised regression coefficients (beta weights)."
