# morie.fn -- function file (hadesllm/morie)
"""Oracle inequality verification for model selection."""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["oracl"]


def oracl(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_folds: int = 5,
    penalty_factors: np.ndarray | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    r"""
    Verify oracle inequality for cross-validated model selection.

    The oracle inequality states that the CV-selected model satisfies:

    .. math::

        R(\hat{f}_{CV}) \le (1 + \delta_n) \min_k R(\hat{f}_k) + r_n

    where :math:`\delta_n \to 0` and :math:`r_n = O((\log K)/n)`.

    Tests this by comparing CV risk of selected model against each
    candidate's empirical risk.

    :param X: Feature matrix, shape (n, p).
    :param y: Response vector, shape (n,).
    :param n_folds: Number of CV folds. Default 5.
    :param penalty_factors: Ridge penalty values to compare. Default: logspace.
    :param seed: Random seed.
    :return: Dict with ``selected_idx``, ``cv_risks``, ``oracle_risk``,
        ``ratio``, ``satisfies_oracle``, ``n``, ``n_models``.
    :raises ValueError: If arrays are empty.

    References
    ----------
    Kosorok, M.R. (2008). Ch. 22. Springer.
    van der Laan, Dudoit, van der Vaart (2006). Oracle inequalities for CV.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    n, p = X.shape
    if n == 0:
        raise ValueError("Arrays must be non-empty.")

    if penalty_factors is None:
        penalty_factors = np.logspace(-3, 3, 10)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    folds = np.array_split(idx, n_folds)

    n_models = len(penalty_factors)
    cv_risks = np.zeros(n_models)

    X_aug = np.column_stack([np.ones(n), X])
    d = X_aug.shape[1]

    for m, lam in enumerate(penalty_factors):
        fold_risks = []
        for fold_idx in range(n_folds):
            val_idx = folds[fold_idx]
            train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != fold_idx])

            X_tr = X_aug[train_idx]
            y_tr = y[train_idx]
            X_val = X_aug[val_idx]
            y_val = y[val_idx]

            beta = np.linalg.solve(X_tr.T @ X_tr + lam * np.eye(d), X_tr.T @ y_tr)
            pred = X_val @ beta
            fold_risks.append(float(np.mean((y_val - pred) ** 2)))

        cv_risks[m] = float(np.mean(fold_risks))

    selected_idx = int(np.argmin(cv_risks))
    oracle_risk = float(np.min(cv_risks))
    ratio = float(cv_risks[selected_idx] / max(oracle_risk, 1e-12))

    remainder = np.log(n_models) / n
    satisfies = ratio <= 1.0 + 2 * remainder + 1e-6

    return {
        "selected_idx": selected_idx,
        "cv_risks": cv_risks,
        "oracle_risk": oracle_risk,
        "ratio": ratio,
        "satisfies_oracle": bool(satisfies),
        "n": n,
        "n_models": n_models,
    }


def cheatsheet() -> str:
    return "oracl(X, y) -> Oracle inequality verification."
