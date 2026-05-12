# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""CART decision tree (pure numpy, recursive splitting)."""

import numpy as np

from ._containers import DescriptiveResult
from ._richresult import RichResult


def _best_split(X, y, max_features=None):
    n, p = X.shape
    best_mse = np.inf
    best_feat, best_thr = 0, 0.0
    features = np.arange(p) if max_features is None else np.random.choice(p, min(max_features, p), replace=False)
    for j in features:
        thresholds = np.unique(X[:, j])
        for t in thresholds:
            left = y[X[:, j] <= t]
            right = y[X[:, j] > t]
            if len(left) == 0 or len(right) == 0:
                continue
            mse = len(left) * np.var(left) + len(right) * np.var(right)
            if mse < best_mse:
                best_mse = mse
                best_feat = j
                best_thr = t
    return best_feat, best_thr, best_mse


def _build_tree(X, y, depth, max_depth, min_samples):
    if depth >= max_depth or len(y) <= min_samples or np.var(y) < 1e-10:
        return RichResult(payload={"leaf": True, "value": float(np.mean(y))})
    feat, thr, mse = _best_split(X, y)
    left_mask = X[:, feat] <= thr
    right_mask = ~left_mask
    if left_mask.sum() == 0 or right_mask.sum() == 0:
        return RichResult(payload={"leaf": True, "value": float(np.mean(y))})
    return {
        "leaf": False,
        "feature": feat,
        "threshold": thr,
        "left": _build_tree(X[left_mask], y[left_mask], depth + 1, max_depth, min_samples),
        "right": _build_tree(X[right_mask], y[right_mask], depth + 1, max_depth, min_samples),
    }


def _predict_one(tree, x):
    if tree["leaf"]:
        return tree["value"]
    if x[tree["feature"]] <= tree["threshold"]:
        return _predict_one(tree["left"], x)
    return _predict_one(tree["right"], x)


def decision_tree(X: np.ndarray, y: np.ndarray, max_depth: int = 5, min_samples: int = 2) -> DescriptiveResult:
    """
    CART decision tree for regression (pure numpy).

    :param X: (n, p) feature matrix.
    :param y: (n,) target.
    :param max_depth: Maximum tree depth.
    :param min_samples: Minimum samples to split.
    :return: DescriptiveResult with predictions and R-squared.

    References
    ----------
    Breiman L et al. (1984). Classification and Regression Trees.
    Wadsworth.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    tree = _build_tree(X, y, 0, max_depth, min_samples)
    preds = np.array([_predict_one(tree, x) for x in X])
    ss_res = np.sum((y - preds) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return DescriptiveResult(
        name="decision_tree",
        value=r2,
        extra={"tree": tree, "predictions": preds, "r_squared": r2, "max_depth": max_depth, "n": len(y)},
    )


cart = decision_tree


def cheatsheet() -> str:
    return "_best_split({}) -> CART decision tree (pure numpy, recursive splitting)."
