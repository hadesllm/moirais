"""XGBoost / gradient boosting classifier wrapper."""

from __future__ import annotations

from typing import Any, Union

import numpy as np
from ._richresult import RichResult


def xgb_classify(
    X: Union[np.ndarray, Any],
    y: Union[np.ndarray, Any],
    *,
    n_estimators: int = 100,
    max_depth: int = 3,
    learning_rate: float = 0.1,
    random_state: int = 42,
) -> dict[str, Any]:
    """XGBoost-style gradient boosting classifier.

    Tries ``xgboost.XGBClassifier`` first, then falls back to
    ``sklearn.ensemble.GradientBoostingClassifier``, then to a
    simplified pure-NumPy sequential stump ensemble.

    Parameters
    ----------
    X : array-like of shape (n, p)
        Feature matrix.
    y : array-like of shape (n,)
        Binary labels (0/1).
    n_estimators : int
        Number of boosting rounds (default 100).
    max_depth : int
        Maximum tree depth (default 3).
    learning_rate : float
        Step-size shrinkage (default 0.1).
    random_state : int
        Random seed.

    Returns
    -------
    dict
        predictions, feature_importance, accuracy.

    References
    ----------
    Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System.
        *KDD '16*, 785-794. doi:10.1145/2939672.2939785
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of rows.")

    # Try xgboost
    try:
        from xgboost import XGBClassifier

        model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            use_label_encoder=False,
            eval_metric="logloss",
        )
        model.fit(X, y)
        preds = model.predict(X)
        imp = model.feature_importances_
        acc = float(np.mean(preds == y))
        return RichResult(payload={"predictions": preds, "feature_importance": imp, "accuracy": acc})
    except ImportError:
        pass

    # Try sklearn
    try:
        from sklearn.ensemble import GradientBoostingClassifier

        model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
        )
        model.fit(X, y)
        preds = model.predict(X)
        imp = model.feature_importances_
        acc = float(np.mean(preds == y))
        return RichResult(payload={"predictions": preds, "feature_importance": imp, "accuracy": acc})
    except ImportError:
        pass

    # Pure-numpy stump boosting
    rng = np.random.default_rng(random_state)
    n, p = X.shape
    preds_raw = np.zeros(n)
    importances = np.zeros(p)

    for _ in range(n_estimators):
        residuals = y - _sigmoid(preds_raw)
        best_feat, best_thresh, best_val_left, best_val_right = _best_stump(
            X,
            residuals,
            rng,
        )
        mask = X[:, best_feat] <= best_thresh
        update = np.where(mask, best_val_left, best_val_right)
        preds_raw += learning_rate * update
        importances[best_feat] += 1.0

    importances /= importances.sum() + 1e-12
    preds = (_sigmoid(preds_raw) >= 0.5).astype(float)
    acc = float(np.mean(preds == y))
    return RichResult(payload={"predictions": preds, "feature_importance": importances, "accuracy": acc})


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def _best_stump(
    X: np.ndarray,
    residuals: np.ndarray,
    rng: np.random.Generator,
) -> tuple[int, float, float, float]:
    """Find the single-feature split that best reduces squared residuals."""
    n, p = X.shape
    best_loss = float("inf")
    best = (0, 0.0, 0.0, 0.0)
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
                best_loss = loss
                best = (j, float(t), float(vl), float(vr))
    return best


xgb = xgb_classify


def cheatsheet() -> str:
    return "xgb_classify({}) -> XGBoost / gradient boosting classifier wrapper."
