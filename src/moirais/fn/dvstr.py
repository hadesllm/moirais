# moirais.fn — function file (hadesllm/moirais)
"""Ensemble stacking. 'Devastator, merge!' -- Scrapper"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def ensemble_stack(
    base_predictions: np.ndarray,
    y_true: np.ndarray,
    *,
    meta_learner: str = "ridge",
    alpha: float = 1.0,
    cv_folds: int = 5,
    seed: int = 42,
) -> DescriptiveResult:
    """Stacked generalization (Wolpert, 1992) ensemble.

    Fits a meta-learner on top of base model predictions using
    cross-validated out-of-fold predictions.

    Parameters
    ----------
    base_predictions : ndarray of shape (n_samples, n_models)
        Predictions from base models.
    y_true : ndarray of shape (n_samples,)
        True target values.
    meta_learner : str
        Meta-learner type: 'ridge' or 'ols'.
    alpha : float
        Ridge regularization strength (ignored if ols).
    cv_folds : int
        Number of CV folds for generating meta-features.
    seed : int
        Random seed.

    Returns
    -------
    DescriptiveResult
        With ``value`` = stacked predictions and ``extra`` containing
        meta-learner weights and CV RMSE.
    """
    X = np.asarray(base_predictions, dtype=float)
    y = np.asarray(y_true, dtype=float).ravel()
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, k = X.shape
    if len(y) != n:
        raise ValueError("y length must match number of samples")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    fold_size = n // cv_folds
    meta_preds = np.zeros(n)
    oof_preds = np.zeros((n, k))

    for fold in range(cv_folds):
        start = fold * fold_size
        end = start + fold_size if fold < cv_folds - 1 else n
        val_idx = indices[start:end]
        train_idx = np.concatenate([indices[:start], indices[end:]])

        X_tr, y_tr = X[train_idx], y[train_idx]

        if meta_learner == "ridge":
            XtX = X_tr.T @ X_tr + alpha * np.eye(k)
            Xty = X_tr.T @ y_tr
            w = np.linalg.solve(XtX, Xty)
        else:
            w = np.linalg.lstsq(X_tr, y_tr, rcond=None)[0]

        meta_preds[val_idx] = X[val_idx] @ w
        oof_preds[val_idx] = X[val_idx]

    if meta_learner == "ridge":
        XtX = X.T @ X + alpha * np.eye(k)
        Xty = X.T @ y
        final_w = np.linalg.solve(XtX, Xty)
    else:
        final_w = np.linalg.lstsq(X, y, rcond=None)[0]

    stacked = X @ final_w
    cv_rmse = float(np.sqrt(np.mean((meta_preds - y) ** 2)))

    return DescriptiveResult(
        name="ensemble_stack",
        value=stacked,
        extra={"weights": final_w, "cv_rmse": cv_rmse, "n_models": k, "meta_learner": meta_learner, "alpha": alpha},
    )


dvstr = ensemble_stack


def cheatsheet() -> str:
    return "ensemble_stack({}) -> Ensemble stacking. 'Devastator, merge!' -- Scrapper"
