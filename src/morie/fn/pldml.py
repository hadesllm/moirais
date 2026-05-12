# morie.fn -- function file (hadesllm/morie)
"""Partially linear DML (double/debiased ML)."""

from __future__ import annotations

import numpy as np
from scipy import stats


def pldml(
    y: np.ndarray,
    X: np.ndarray,
    Z: np.ndarray,
    *,
    n_folds: int = 5,
    n_basis: int | None = None,
    seed: int | None = None,
) -> dict:
    r"""
    Partially linear model via double/debiased machine learning.

    Uses cross-fitting with series regression as the first stage
    (Chernozhukov et al. 2018):

    1. Split data into K folds.
    2. For each fold, estimate nuisance functions :math:`E[Y|Z]` and
       :math:`E[X|Z]` on the complement.
    3. Construct residuals and run OLS.

    Parameters
    ----------
    y : np.ndarray
        Response (n,).
    X : np.ndarray
        Treatment/linear covariates (n, p).
    Z : np.ndarray
        Controls/nonparametric covariate (n,) or (n, q).
    n_folds : int
        Number of cross-fitting folds.
    n_basis : int or None
        Sieve basis dimension per Z column.
    seed : int or None
        RNG seed for fold assignment.

    Returns
    -------
    dict
        ``beta``, ``se``, ``t_stat``, ``pval``, ``n_folds``, ``n_obs``.

    References
    ----------
    Chernozhukov, V. et al. (2018). Double/debiased ML for treatment and
        structural parameters. Econometrics J., 21, C1-C68.
    Horowitz (2009). Ch 3.
    """
    y = np.asarray(y, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    Z = np.asarray(Z, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)
    n, p = X.shape
    q = Z.shape[1]
    if y.shape[0] != n or Z.shape[0] != n:
        raise ValueError("y, X, Z must have same n.")
    if n < 20:
        raise ValueError("Need at least 20 observations for cross-fitting.")

    if n_basis is None:
        n_basis = min(n // (2 * n_folds), 8)
    n_basis = max(1, n_basis)

    rng = np.random.default_rng(seed)
    fold_ids = rng.integers(0, n_folds, size=n)

    def _build_basis(z_col, nb):
        lo, hi = z_col.min(), z_col.max()
        rng_z = hi - lo if hi > lo else 1.0
        zs = (z_col - lo) / rng_z
        return np.column_stack([zs**k for k in range(1, nb + 1)])

    y_resid = np.zeros(n)
    X_resid = np.zeros((n, p))

    for f in range(n_folds):
        train = fold_ids != f
        test = fold_ids == f
        if test.sum() == 0:
            continue

        B_train_parts = [_build_basis(Z[train, j], n_basis) for j in range(q)]
        B_train = np.column_stack(B_train_parts) if q > 0 else np.ones((train.sum(), 1))
        B_test_parts = [_build_basis(Z[test, j], n_basis) for j in range(q)]
        B_test = np.column_stack(B_test_parts) if q > 0 else np.ones((test.sum(), 1))

        BtB = B_train.T @ B_train + 1e-6 * np.eye(B_train.shape[1])

        gamma_y = np.linalg.solve(BtB, B_train.T @ y[train])
        y_resid[test] = y[test] - B_test @ gamma_y

        for j in range(p):
            gamma_x = np.linalg.solve(BtB, B_train.T @ X[train, j])
            X_resid[test, j] = X[test, j] - B_test @ gamma_x

    XtX = X_resid.T @ X_resid
    try:
        beta = np.linalg.solve(XtX, X_resid.T @ y_resid)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(XtX, X_resid.T @ y_resid, rcond=None)[0]

    eps = y_resid - X_resid @ beta
    meat = np.zeros((p, p))
    for i in range(n):
        v = X_resid[i, :] * eps[i]
        meat += np.outer(v, v)
    meat /= n

    try:
        XtX_inv = np.linalg.inv(XtX / n)
    except np.linalg.LinAlgError:
        XtX_inv = np.linalg.pinv(XtX / n)

    cov = (XtX_inv @ meat @ XtX_inv) / n
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    t_stat = beta / np.where(se > 0, se, np.inf)
    pval = 2 * stats.norm.sf(np.abs(t_stat))

    return {
        "beta": beta.tolist(),
        "se": se.tolist(),
        "t_stat": t_stat.tolist(),
        "pval": pval.tolist(),
        "n_folds": n_folds,
        "n_obs": n,
    }


pldml_fn = pldml


def cheatsheet() -> str:
    return "pldml({y, X, Z}) -> Partially linear double/debiased ML."
