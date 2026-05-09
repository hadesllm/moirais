"""Super learner (ensemble via cross-validation)."""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["sllrn"]


def sllrn(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_folds: int = 5,
    alpha: float = 0.05,
    seed: int | None = None,
) -> dict[str, Any]:
    r"""
    Super learner: optimal ensemble combination via cross-validated risk.

    Combines multiple base learners (OLS, ridge, kernel regression) by
    minimizing cross-validated risk over convex combination weights.

    .. math::

        \hat\alpha = \arg\min_{\alpha \in \Delta} \sum_{i=1}^n
        \left(Y_i - \sum_k \alpha_k \hat{f}_k^{(-i)}(X_i)\right)^2

    :param X: Feature matrix, shape (n, p).
    :param y: Response vector, shape (n,).
    :param n_folds: Number of CV folds. Default 5.
    :param alpha: Significance level. Default 0.05.
    :param seed: Random seed.
    :return: Dict with ``weights``, ``cv_risks``, ``learner_names``,
        ``ensemble_risk``, ``n``, ``n_folds``.
    :raises ValueError: If arrays are empty.

    References
    ----------
    Kosorok, M.R. (2008). Ch. 21-22. Springer.
    van der Laan, Polley, Hubbard (2007). Super Learner. *Stat Appl Genet Mol Biol*.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    n, p = X.shape
    if n == 0:
        raise ValueError("Arrays must be non-empty.")

    rng = np.random.default_rng(seed)

    def _ols_predict(X_tr, y_tr, X_val):
        X_aug = np.column_stack([np.ones(len(X_tr)), X_tr])
        beta = np.linalg.lstsq(X_aug, y_tr, rcond=None)[0]
        return np.column_stack([np.ones(len(X_val)), X_val]) @ beta

    def _ridge_predict(X_tr, y_tr, X_val, lam=1.0):
        X_aug = np.column_stack([np.ones(len(X_tr)), X_tr])
        d = X_aug.shape[1]
        beta = np.linalg.solve(X_aug.T @ X_aug + lam * np.eye(d), X_aug.T @ y_tr)
        return np.column_stack([np.ones(len(X_val)), X_val]) @ beta

    def _mean_predict(X_tr, y_tr, X_val):
        return np.full(len(X_val), np.mean(y_tr))

    learners = [
        ("OLS", _ols_predict),
        ("Ridge", lambda Xtr, ytr, Xv: _ridge_predict(Xtr, ytr, Xv, 1.0)),
        ("Mean", _mean_predict),
    ]
    n_learners = len(learners)
    learner_names = [name for name, _ in learners]

    idx = rng.permutation(n)
    folds = np.array_split(idx, n_folds)

    Z = np.zeros((n, n_learners))
    for fold_idx in range(n_folds):
        val_idx = folds[fold_idx]
        train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != fold_idx])
        for k, (_, predict_fn) in enumerate(learners):
            Z[val_idx, k] = predict_fn(X[train_idx], y[train_idx], X[val_idx])

    from scipy.optimize import minimize
    def cv_loss(w):
        pred = Z @ w
        return float(np.mean((y - pred) ** 2))

    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    bounds = [(0, 1)] * n_learners
    w0 = np.ones(n_learners) / n_learners
    res = minimize(cv_loss, w0, bounds=bounds, constraints=constraints, method="SLSQP")
    weights = res.x

    cv_risks = np.array([float(np.mean((y - Z[:, k]) ** 2)) for k in range(n_learners)])
    ensemble_risk = float(res.fun)

    return {
        "weights": weights,
        "cv_risks": cv_risks,
        "learner_names": learner_names,
        "ensemble_risk": ensemble_risk,
        "n": n,
        "n_folds": n_folds,
    }


def cheatsheet() -> str:
    return "sllrn(X, y) -> Super learner ensemble via CV."
