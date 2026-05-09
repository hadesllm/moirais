# moirais.fn — function file (hadesllm/moirais)
"""Gradient boosting machine (simplified, pure NumPy)."""

from __future__ import annotations

from typing import Any, Union

import numpy as np
from ._richresult import RichResult


def gradient_boosting(
    X: Union[np.ndarray, Any],
    y: Union[np.ndarray, Any],
    *,
    n_estimators: int = 100,
    lr: float = 0.1,
    max_depth: int = 3,
    random_state: int = 42,
) -> dict[str, Any]:
    """Simplified gradient boosting for binary classification.

    Sequentially fits decision stumps to residuals of the log-loss.
    Pure NumPy — no external ML library required.

    Parameters
    ----------
    X : array-like of shape (n, p)
        Feature matrix.
    y : array-like of shape (n,)
        Binary labels (0/1).
    n_estimators : int
        Number of boosting iterations (default 100).
    lr : float
        Learning rate / shrinkage (default 0.1).
    max_depth : int
        Not used for stumps; reserved for compatibility.
    random_state : int
        Random seed.

    Returns
    -------
    dict
        predictions (n,), feature_importance (p,).

    References
    ----------
    Friedman, J. H. (2001). Greedy function approximation: A gradient boosting
        machine. *The Annals of Statistics*, 29(5), 1189-1232.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of rows.")

    rng = np.random.default_rng(random_state)
    n, p = X.shape
    F = np.full(n, _logit(np.clip(y.mean(), 1e-6, 1 - 1e-6)))
    importance = np.zeros(p)

    for _ in range(n_estimators):
        prob = _sigmoid(F)
        residuals = y - prob

        # Fit a stump
        best_feat, best_thresh, val_left, val_right, gain = _fit_stump(
            X,
            residuals,
            rng,
        )
        mask = X[:, best_feat] <= best_thresh
        update = np.where(mask, val_left, val_right)
        F += lr * update
        importance[best_feat] += gain

    imp_sum = importance.sum()
    if imp_sum > 0:
        importance /= imp_sum

    preds = (_sigmoid(F) >= 0.5).astype(float)
    return RichResult(payload={"predictions": preds, "feature_importance": importance})


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def _logit(p: float) -> float:
    return float(np.log(p / (1 - p)))


def _fit_stump(
    X: np.ndarray,
    residuals: np.ndarray,
    rng: np.random.Generator,
) -> tuple[int, float, float, float, float]:
    n, p = X.shape
    best_loss = float("inf")
    best = (0, 0.0, 0.0, 0.0, 0.0)
    for j in range(p):
        vals = np.unique(X[:, j])
        if len(vals) > 20:
            vals = rng.choice(vals, 20, replace=False)
        for t in vals:
            mask = X[:, j] <= t
            nl, nr = mask.sum(), (~mask).sum()
            if nl == 0 or nr == 0:
                continue
            vl = residuals[mask].mean()
            vr = residuals[~mask].mean()
            loss = np.sum((residuals - np.where(mask, vl, vr)) ** 2)
            if loss < best_loss:
                gain = best_loss - loss if best_loss < float("inf") else 0.0
                best_loss = loss
                best = (j, float(t), float(vl), float(vr), max(gain, 0.0))
    return best


gbm = gradient_boosting


def cheatsheet() -> str:
    return "gradient_boosting({}) -> Gradient boosting machine (simplified, pure NumPy)."
