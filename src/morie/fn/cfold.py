# morie.fn -- function file (hadesllm/morie)
"""Cross-fitting (K-fold sample splitting for DML)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._helpers import _validate_df


def cross_fit(
    data: pd.DataFrame,
    *,
    y: str = "outcome",
    t: str = "treatment",
    covariates: list[str] | None = None,
    n_folds: int = 5,
    seed: int = 42,
) -> dict:
    r"""K-fold cross-fitting for Double Machine Learning.

    Splits the sample into K folds. For each fold k:

    1. Train nuisance models on all folds except k.
    2. Predict on fold k to get :math:`\hat{m}(X_i)` and :math:`\hat{e}(X_i)`.
    3. Compute the Neyman-orthogonal score on fold k.

    The DML estimator is:

    .. math::

        \hat{\theta} = \frac{1}{n}\sum_{i=1}^{n}
        \frac{(Y_i - \hat{m}(X_i))(T_i - \hat{e}(X_i))}{
        \frac{1}{n}\sum_j (T_j - \hat{e}(X_j))^2}

    Uses ridge regression for nuisance estimation (no sklearn).

    Parameters
    ----------
    data : pd.DataFrame
    y, t : str
        Outcome and treatment columns.
    covariates : list[str]
        Covariate columns.
    n_folds : int
        Number of folds (default 5).
    seed : int
        Random seed for fold assignment.

    Returns
    -------
    dict
        Keys: 'theta', 'se', 'scores', 'fold_estimates'.

    References
    ----------
    Chernozhukov, V., et al. (2018). Double/debiased machine learning.
    *The Econometrics Journal*, 21(1), C1-C68.
    """
    if covariates is None or len(covariates) == 0:
        raise ValueError("covariates required for cross-fitting")
    _validate_df(data, y, t, *covariates)
    df = data[[y, t] + covariates].dropna().reset_index(drop=True)
    n = len(df)

    Y = df[y].to_numpy(dtype=float)
    T = df[t].to_numpy(dtype=float)
    X = df[covariates].to_numpy(dtype=float)

    rng = np.random.default_rng(seed)
    fold_ids = rng.integers(0, n_folds, size=n)
    perm = rng.permutation(n)
    fold_ids = np.zeros(n, dtype=int)
    fold_size = n // n_folds
    for k in range(n_folds):
        start = k * fold_size
        end = (k + 1) * fold_size if k < n_folds - 1 else n
        fold_ids[perm[start:end]] = k

    def _ridge(X_train, y_train, X_test, lam=1.0):
        p = X_train.shape[1]
        XtX = X_train.T @ X_train + lam * np.eye(p)
        Xty = X_train.T @ y_train
        beta = np.linalg.solve(XtX, Xty)
        return X_test @ beta

    m_hat = np.zeros(n)
    e_hat = np.zeros(n)
    fold_estimates = []

    for k in range(n_folds):
        train_mask = fold_ids != k
        test_mask = fold_ids == k

        X_train = X[train_mask]
        X_test = X[test_mask]

        m_hat[test_mask] = _ridge(X_train, Y[train_mask], X_test)
        e_hat[test_mask] = _ridge(X_train, T[train_mask], X_test)

    V_resid = T - e_hat
    Y_resid = Y - m_hat
    denom = float(np.mean(V_resid ** 2))
    theta = float(np.mean(Y_resid * V_resid) / denom) if denom > 1e-15 else float("nan")

    scores = V_resid * (Y_resid - theta * V_resid) / denom
    se = float(np.std(scores, ddof=1) / np.sqrt(n))

    for k in range(n_folds):
        mask = fold_ids == k
        v_k = V_resid[mask]
        yr_k = Y_resid[mask]
        d_k = float(np.mean(v_k ** 2))
        if d_k > 1e-15:
            fold_estimates.append(float(np.mean(yr_k * v_k) / d_k))
        else:
            fold_estimates.append(float("nan"))

    return {
        "theta": theta,
        "se": se,
        "scores": scores,
        "fold_estimates": fold_estimates,
        "n_folds": n_folds,
        "n": n,
    }


cfold = cross_fit


def cheatsheet() -> str:
    return "cross_fit({}) -> Cross-fitting (K-fold for DML)."
