# morie.fn — function file (hadesllm/morie)
"""K-fold cross-validation. 'Let the past die. Kill it, if you have to.' -- Kylo Ren"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._containers import DescriptiveResult
from ._helpers import _validate_df


def kfold_cv(
    data: pd.DataFrame,
    *,
    y: str = "y",
    x: list[str] | str = "x",
    k: int = 5,
    seed: int = 42,
) -> DescriptiveResult:
    """K-fold cross-validation for OLS. Returns mean MSE across folds."""
    _validate_df(data, y)
    if isinstance(x, str):
        x = [x]
    _validate_df(data, *x)
    df = data[[y, *x]].dropna()
    Y = df[y].to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(df)), df[x].to_numpy(dtype=float)])
    n = len(df)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    fold_size = n // k
    mse_folds = []
    for i in range(k):
        test_idx = idx[i * fold_size : (i + 1) * fold_size] if i < k - 1 else idx[i * fold_size :]
        train_idx = np.setdiff1d(idx, test_idx)
        X_train, Y_train = X[train_idx], Y[train_idx]
        X_test, Y_test = X[test_idx], Y[test_idx]
        beta = np.linalg.lstsq(X_train, Y_train, rcond=None)[0]
        pred = X_test @ beta
        mse_folds.append(float(np.mean((Y_test - pred) ** 2)))
    mean_mse = float(np.mean(mse_folds))
    se_mse = float(np.std(mse_folds, ddof=1) / np.sqrt(k))
    return DescriptiveResult(
        name=f"{k}-fold CV",
        value=mean_mse,
        extra={"mean_mse": mean_mse, "se_mse": se_mse, "fold_mses": mse_folds, "k": k},
    )


kylo = kfold_cv


def cheatsheet() -> str:
    return "kfold_cv({}) -> K-fold cross-validation. 'Let the past die. Kill it, if you "
