# moirais.fn — function file (hadesllm/moirais)
"""Model selection via cross-validated loss."""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["mdsel"]


def mdsel(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_folds: int = 5,
    models: list | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    r"""
    Select the best model via cross-validated squared error loss.

    Compares multiple candidate models using V-fold CV and selects the
    one with minimum cross-validated risk. Reports one-standard-error
    rule selection as well.

    :param X: Feature matrix, shape (n, p).
    :param y: Response vector, shape (n,).
    :param n_folds: Number of CV folds. Default 5.
    :param models: List of (name, fit_predict_func) tuples. Each function
        has signature func(X_train, y_train, X_val) -> predictions.
        Default: intercept-only, OLS, Ridge(1), Ridge(10).
    :param seed: Random seed.
    :return: Dict with ``selected``, ``selected_1se``, ``cv_risks``,
        ``cv_ses``, ``model_names``, ``n``, ``n_folds``.
    :raises ValueError: If arrays are empty.

    References
    ----------
    Kosorok, M.R. (2008). Ch. 22. Springer.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    n, p = X.shape
    if n == 0:
        raise ValueError("Arrays must be non-empty.")

    def _intercept(X_tr, y_tr, X_val):
        return np.full(len(X_val), np.mean(y_tr))

    def _ols(X_tr, y_tr, X_val):
        Xa = np.column_stack([np.ones(len(X_tr)), X_tr])
        beta = np.linalg.lstsq(Xa, y_tr, rcond=None)[0]
        return np.column_stack([np.ones(len(X_val)), X_val]) @ beta

    def _ridge(lam):
        def fn(X_tr, y_tr, X_val):
            Xa = np.column_stack([np.ones(len(X_tr)), X_tr])
            d = Xa.shape[1]
            beta = np.linalg.solve(Xa.T @ Xa + lam * np.eye(d), Xa.T @ y_tr)
            return np.column_stack([np.ones(len(X_val)), X_val]) @ beta
        return fn

    if models is None:
        models = [
            ("Intercept", _intercept),
            ("OLS", _ols),
            ("Ridge(1)", _ridge(1.0)),
            ("Ridge(10)", _ridge(10.0)),
        ]

    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    folds = np.array_split(idx, n_folds)

    n_models = len(models)
    model_names = [m[0] for m in models]
    fold_risks = np.zeros((n_models, n_folds))

    for fold_idx in range(n_folds):
        val_idx = folds[fold_idx]
        train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != fold_idx])
        for m, (_, fn) in enumerate(models):
            pred = fn(X[train_idx], y[train_idx], X[val_idx])
            fold_risks[m, fold_idx] = float(np.mean((y[val_idx] - pred) ** 2))

    cv_risks = np.mean(fold_risks, axis=1)
    cv_ses = np.std(fold_risks, axis=1, ddof=1) / np.sqrt(n_folds)

    selected = int(np.argmin(cv_risks))

    threshold = cv_risks[selected] + cv_ses[selected]
    candidates_1se = np.where(cv_risks <= threshold)[0]
    selected_1se = int(candidates_1se[0]) if len(candidates_1se) > 0 else selected

    return {
        "selected": selected,
        "selected_1se": selected_1se,
        "cv_risks": cv_risks,
        "cv_ses": cv_ses,
        "model_names": model_names,
        "n": n,
        "n_folds": n_folds,
    }


def cheatsheet() -> str:
    return "mdsel(X, y) -> Model selection via cross-validated loss."
